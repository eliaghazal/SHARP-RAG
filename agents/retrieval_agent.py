"""Retrieval agent V2 — pass-aware retrieval with adaptive TOP_K,
suggested-query reuse, HyDE broadening, and cross-pass ID-level deduplication.

Pass schedule (driven by retry_count BEFORE this node runs):
  pass 0 (retry_count == 0): TOP_K = 3 — narrow, low-noise over raw sub_queries.
  pass 1 (retry_count == 1): TOP_K = 5 — fire critic's suggested_queries verbatim.
  pass 2 (retry_count == 2): TOP_K = 7 — HyDE-style broadening over missing facts.

Deduplication is two-level:
  1. Cross-pass: seen_doc_ids (persisted in state) filters out IDs already returned
     in a previous pass, forcing the retriever to surface genuinely new evidence.
  2. Within-pass: seen_texts guards against exact-text duplicates within a single pass.

New docs are APPENDED to the prior retrieved_docs (so hop-1 evidence from pass 0 is
not discarded when pass 1 fetches hop-2 evidence); the combined list is then re-sorted
by relevance_score descending.

This agent does NOT touch retry_count — the critique agent is the sole owner.
"""

import os
from typing import Any

from dotenv import load_dotenv

from core.vector_store import VectorStore
from core.groq_client import call_groq, get_client

load_dotenv()

# Adaptive TOP_K schedule keyed by retry_count before this pass.
_TOP_K_SCHEDULE: dict[int, int] = {0: 3, 1: 5, 2: 7}

# HyDE prompt: asks the model for a hypothetical passage that embeds near real answer docs.
_HYDE_SYSTEM = (
    "You are a concise knowledge source. Given a question about a specific fact, "
    "write exactly ONE sentence that states the answer as if it were a factual passage "
    "in an encyclopedia. Do not hedge; write it as established fact."
)
_HYDE_USER = "Write a one-sentence hypothetical factual passage that would answer: {fact}"


def run_retrieval_agent(state: dict, vector_store: VectorStore) -> dict:
    """Retrieve top-K docs per query, deduplicate, accumulate, and sort by relevance.

    Reads (from state):
        sub_queries                 — original decomposed queries (always present)
        critique_feedback           — free-text explanation from the critic
        critique_suggested_queries  — concrete entity-named retry queries (pass 1)
        critique_missing_information — missing facts for HyDE broadening (pass 2)
        retry_count                 — determines which pass logic and TOP_K to use
        retrieved_docs              — prior docs accumulated across passes
        seen_doc_ids                — cross-pass dedup set (list in state)

    Writes (returned dict):
        retrieved_docs              — prior docs + new unique docs, sorted by relevance
        seen_doc_ids                — updated with newly returned IDs
        reasoning_trace             — appended with a trace entry for this pass
    """
    sub_queries: list[str] = state.get("sub_queries", [])
    retry_count: int = state.get("retry_count", 0)
    critique_feedback: str = state.get("critique_feedback", "")
    suggested_queries: list[str] = state.get("critique_suggested_queries") or []
    missing_information: list[str] = state.get("critique_missing_information") or []

    # Prior accumulated docs and cross-pass seen-ID set.
    prior_docs: list[dict] = state.get("retrieved_docs") or []
    seen_doc_ids: set[str] = set(state.get("seen_doc_ids") or [])

    # ------------------------------------------------------------------
    # Adaptive TOP_K
    # ------------------------------------------------------------------
    top_k: int = _TOP_K_SCHEDULE.get(retry_count, 7)

    # ------------------------------------------------------------------
    # Query set per pass
    # ------------------------------------------------------------------
    if retry_count == 0:
        # Pass 0: raw sub_queries — narrow, low-noise first pass.
        queries = list(sub_queries)

    elif retry_count == 1:
        # Pass 1: use critic's suggested_queries verbatim if available.
        # They are already concrete, entity-named questions — embedding them directly
        # avoids the "Find information about: <failure description>" pollution.
        if suggested_queries:
            queries = list(suggested_queries)
        else:
            # Parse failure fallback: use feedback but strip the old boilerplate prefix.
            fallback_text = critique_feedback.strip()
            prefix = "Find information about:"
            if fallback_text.lower().startswith(prefix.lower()):
                fallback_text = fallback_text[len(prefix):].strip()
            queries = list(sub_queries)
            if fallback_text:
                queries.append(fallback_text)

    else:
        # Pass 2: HyDE-style broadening — generate a hypothetical passage for each
        # missing fact and use the passage as the retrieval query so it embeds near
        # real answer documents.  Falls back to sub_queries + suggested_queries if
        # no missing_information is available or the LLM call fails.
        hyde_queries = _build_hyde_queries(missing_information)
        if hyde_queries:
            queries = hyde_queries
            # Also include the original question as an anchor for recall.
            question: str = state.get("question", "")
            if question:
                queries.append(question)
        else:
            # Fallback: broaden by combining sub_queries with any suggested_queries.
            queries = list(sub_queries) + [q for q in suggested_queries if q]

    # ------------------------------------------------------------------
    # Retrieval with two-level deduplication
    # ------------------------------------------------------------------
    # Within-pass guard against exact-text duplicates.
    seen_texts: set[str] = {d["text"] for d in prior_docs}

    new_docs: list[dict] = []
    for query in queries:
        if not query.strip():
            continue
        results = vector_store.search(query, n_results=top_k)
        for doc in results:
            doc_id: str = doc.get("id", "")
            doc_text: str = doc.get("text", "")

            # Cross-pass dedup: skip any doc whose ID was seen in a prior pass.
            if doc_id and doc_id in seen_doc_ids:
                continue
            # Within-pass dedup: skip exact-text duplicates.
            if doc_text in seen_texts:
                continue

            seen_texts.add(doc_text)
            new_docs.append(doc)

    # Update the cross-pass seen-ID set with newly retrieved IDs.
    for doc in new_docs:
        doc_id = doc.get("id", "")
        if doc_id:
            seen_doc_ids.add(doc_id)

    # ------------------------------------------------------------------
    # Accumulate new docs onto prior docs, then sort combined list.
    # ------------------------------------------------------------------
    all_docs = prior_docs + new_docs
    all_docs.sort(key=lambda d: d.get("relevance_score", 0.0), reverse=True)

    trace_entry = (
        f"[RetrievalAgent] pass={retry_count} | TOP_K={top_k} | "
        f"queries={len(queries)} | new_docs={len(new_docs)} | "
        f"total_docs={len(all_docs)} | seen_ids={len(seen_doc_ids)}"
    )

    return {
        "retrieved_docs": all_docs,
        "seen_doc_ids": sorted(seen_doc_ids),   # list for JSON/LangGraph serialisation
        "retry_count": retry_count,              # pass through unchanged — critic owns this
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


# --------------------------------------------------------------------------
# HyDE helper
# --------------------------------------------------------------------------

def _build_hyde_queries(missing_facts: list[str]) -> list[str]:
    """Generate hypothetical factual passages for each missing fact via Groq.

    Each generated passage embeds near real answer documents, giving the
    retriever a better anchor than the abstract description of a missing fact.

    Returns an empty list if GROQ_API_KEY is absent, missing_facts is empty,
    or the LLM call fails — the caller falls back to a simple broadening strategy.
    """
    if not missing_facts:
        return []

    if not os.environ.get("GROQ_API_KEY"):
        return []

    client = get_client()
    hyde_passages: list[str] = []

    for fact in missing_facts:
        if not fact.strip():
            continue
        try:
            response = call_groq(
                client=client,
                system_prompt=_HYDE_SYSTEM,
                user_message=_HYDE_USER.format(fact=fact),
                max_tokens=80,
                temperature=0.3,
                max_retries=2,
                caller="RetrievalAgent/HyDE",
            )
            passage = response.choices[0].message.content.strip()
            if passage:
                hyde_passages.append(passage)
        except Exception as exc:
            # Log but do not raise — fall through to fallback in caller.
            print(f"[RetrievalAgent] HyDE call failed for fact '{fact[:60]}': {exc}")

    return hyde_passages
