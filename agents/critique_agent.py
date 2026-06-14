"""Critique agent — evaluates whether retrieved evidence is sufficient."""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from core.groq_client import call_groq, get_client

load_dotenv()

CRITIQUE_SYSTEM_PROMPT = (
    "You are a strict evidence evaluator in a multi-hop retrieval-augmented QA system. "
    "You are given a QUESTION, its SUB-QUERIES, and a numbered set of RETRIEVED EVIDENCE "
    "chunks. Decide whether the evidence is sufficient to answer the question ACCURATELY "
    "and COMPLETELY, and if not, say PRECISELY what is missing and how to retrieve it.\n\n"
    "You must reason about EACH hop: for a multi-hop question, every intermediate fact "
    "(e.g. the bridge entity AND its target property) must be present in the evidence. "
    "Evidence that answers only the first hop is INSUFFICIENT.\n\n"
    "VERDICTS:\n"
    '- "sufficient": every fact required to answer the question (all hops) is explicitly '
    "present in the evidence.\n"
    '- "insufficient": one or more required facts are absent from the evidence.\n'
    '- "contradictory": the evidence contains directly conflicting statements about a fact '
    "required to answer the question.\n\n"
    "You MUST identify WHAT is missing, not merely THAT something is missing. For each "
    "missing fact, write a concrete, self-contained search query (naming the entity) that "
    "would retrieve it — these queries will be fed DIRECTLY into the retriever, so phrase "
    "them as real questions, NOT as descriptions of failure (write \"When was the Eiffel "
    "Tower completed?\", never \"the documents do not mention the completion date\").\n\n"
    "CONFIDENCE: a float in [0,1] for how certain you are in the verdict. Use >=0.9 only "
    "when the evidence very clearly does or does not contain the needed facts.\n\n"
    "OUTPUT FORMAT (STRICT): Output ONLY a single valid JSON object, no markdown fences, "
    "no prose before or after. Schema:\n"
    "{\n"
    '  "verdict": "sufficient" | "insufficient" | "contradictory",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "missing_information": [<short string naming each missing fact>],\n'
    '  "suggested_queries": [<one concrete search query per missing fact>],\n'
    '  "reasoning": "<one sentence explaining the verdict>"\n'
    "}\n\n"
    'If verdict is "sufficient", set missing_information and suggested_queries to empty '
    'arrays []. If "contradictory", put the conflicting fact in missing_information and a '
    "disambiguating query in suggested_queries. Keep missing_information and "
    "suggested_queries the SAME length and aligned by index.\n\n"
    "EXAMPLE (insufficient):\n"
    '{"verdict":"insufficient","confidence":0.88,"missing_information":["nationality of '
    'director Luc Besson"],"suggested_queries":["What is the nationality of film director '
    'Luc Besson?"],"reasoning":"Evidence identifies Luc Besson as the director but never '
    'states his nationality, so the second hop is unanswerable."}\n\n'
    "EXAMPLE (sufficient):\n"
    '{"verdict":"sufficient","confidence":0.94,"missing_information":[],"suggested_queries"'
    ':[],"reasoning":"Both the founding years of Arthur\'s Magazine (1844) and First for '
    'Women (1989) are present, which fully answers the comparison."}'
)

CRITIQUE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Sub-queries:\n{sub_queries_block}\n\n"
    "Retrieved evidence ({n_docs} chunks):\n"
    "{evidence_block}\n\n"
    "Return ONLY the JSON object."
)


def run_critique_agent(state: dict) -> dict:
    """Evaluate the retrieved docs and set critique_result + critique_feedback.

    Reads ``state["question"]``, ``state["sub_queries"]``, and
    ``state["retrieved_docs"]``.
    Writes ``state["critique_result"]``, ``state["critique_feedback"]``,
    ``state["critique_confidence"]``, ``state["critique_suggested_queries"]``,
    and increments ``state["retry_count"]`` whenever verdict != 'sufficient'.
    Also appends to ``state["reasoning_trace"]``.
    """
    question: str = state["question"]
    sub_queries: list[str] = state.get("sub_queries", [])
    retrieved_docs: list[dict] = state.get("retrieved_docs", [])
    retry_count: int = state.get("retry_count", 0)

    if not retrieved_docs:
        # Nothing was retrieved — treat as insufficient immediately.
        return {
            "critique_result": "insufficient",
            "critique_feedback": "No documents were retrieved.",
            "critique_confidence": 1.0,
            "critique_suggested_queries": sub_queries,
            "critique_missing_information": [],
            "retry_count": retry_count + 1,
            "reasoning_trace": state.get("reasoning_trace", [])
            + ["[CritiqueAgent] No docs retrieved — marking insufficient."],
        }

    sub_queries_block = "\n".join(sub_queries) if sub_queries else question
    evidence_block = _build_evidence_block(retrieved_docs)
    user_message = CRITIQUE_USER_TEMPLATE.format(
        question=question,
        sub_queries_block=sub_queries_block,
        n_docs=len(retrieved_docs),
        evidence_block=evidence_block,
    )

    client = get_client()
    response = call_groq(
        client=client,
        system_prompt=CRITIQUE_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.1,
        max_tokens=400,
        caller="CritiqueAgent",
    )

    raw: str = response.choices[0].message.content.strip()
    parsed = _parse_critique_response(raw, sub_queries)

    verdict_raw = parsed["verdict"]
    confidence: float = parsed["confidence"]
    missing_information: list[str] = parsed["missing_information"]
    suggested_queries: list[str] = parsed["suggested_queries"]
    reasoning: str = parsed["reasoning"]

    # Normalise the verdict to one of three canonical values.
    if verdict_raw == "sufficient":
        critique_result = "sufficient"
    elif verdict_raw == "contradictory":
        critique_result = "contradictory"
    else:
        critique_result = "insufficient"

    # Increment retry counter whenever verdict is not sufficient.
    # The router enforces the cap — this agent is the sole owner of retry_count.
    new_retry_count = retry_count
    if critique_result != "sufficient":
        new_retry_count = retry_count + 1

    trace_entry = (
        f"[CritiqueAgent] Verdict='{critique_result}' | "
        f"confidence={confidence:.2f} | "
        f"retry_count={new_retry_count} | Feedback: {reasoning}"
    )

    return {
        "critique_result": critique_result,
        "critique_feedback": reasoning,
        "critique_confidence": confidence,
        "critique_suggested_queries": suggested_queries,
        "critique_missing_information": missing_information,
        "retry_count": new_retry_count,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_critique_response(raw: str, fallback_queries: list[str]) -> dict:
    """Parse the LLM JSON response with fenced and regex fallbacks.

    Returns a dict with keys: verdict, confidence, missing_information,
    suggested_queries, reasoning.
    """
    # Strip ```json ... ``` fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    # Attempt 1: direct json.loads.
    try:
        obj = json.loads(text)
        return _normalise_parsed(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: regex-extract the first {...} span.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return _normalise_parsed(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: safe default so the loop degrades gracefully.
    return {
        "verdict": "insufficient",
        "confidence": 0.5,
        "missing_information": [],
        "suggested_queries": list(fallback_queries),
        "reasoning": raw[:200],
    }


def _normalise_parsed(obj: dict) -> dict:
    """Ensure all expected keys are present and typed correctly."""
    verdict = str(obj.get("verdict", "insufficient")).lower().strip()
    confidence = float(obj.get("confidence", 0.5))
    missing_information = [str(x) for x in obj.get("missing_information", [])]
    suggested_queries = [str(x) for x in obj.get("suggested_queries", [])]
    reasoning = str(obj.get("reasoning", ""))

    # Guarantee index alignment — pad the shorter list with empty strings.
    max_len = max(len(missing_information), len(suggested_queries))
    missing_information += [""] * (max_len - len(missing_information))
    suggested_queries += [""] * (max_len - len(suggested_queries))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "missing_information": missing_information,
        "suggested_queries": suggested_queries,
        "reasoning": reasoning,
    }


def _build_evidence_block(docs: list[dict]) -> str:
    """Format retrieved docs as a numbered block for the critique prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "unknown")
        score = doc.get("relevance_score", 0.0)
        text = doc.get("text", "").strip()
        parts.append(f"[{i}] (source={source}, relevance={score:.3f})\n{text}")
    return "\n\n".join(parts)


