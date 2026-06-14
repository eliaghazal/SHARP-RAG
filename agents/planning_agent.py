"""Planning agent — decomposes a complex question into focused sub-queries."""

import os
import re
from typing import Any

from dotenv import load_dotenv

from core.groq_client import call_groq, get_client

load_dotenv()

PLANNING_SYSTEM_PROMPT = (
    "You are a question-decomposition specialist in a multi-hop retrieval-augmented QA system "
    "(corpus: HotpotQA over ChromaDB). Your ONLY job is to break a complex question into 2-3 "
    "focused, COMPLEMENTARY sub-queries that an embedding-based retriever can answer independently. "
    "You never answer the question yourself.\n\n"
    "Think step by step INTERNALLY, then output ONLY the final sub-queries.\n\n"
    "DECOMPOSITION STRATEGY BY QUESTION TYPE:\n"
    "1. BRIDGE questions (\"Who directed the film that starred X?\"): the FIRST sub-query must find "
    "the bridge entity (the film), the SECOND must ask about the target property of that entity "
    "(its director). Do not collapse both hops into one query.\n"
    "2. COMPARISON questions (\"Which is older, A or B?\"): produce ONE sub-query per entity asking "
    "for the SAME attribute (e.g. \"When was A founded?\" / \"When was B founded?\"). Never compare "
    "inside a single sub-query.\n"
    "3. YES/NO questions (\"Are A and B from the same country?\"): decompose into the underlying "
    "FACTS that decide the answer — one sub-query per entity for the attribute being compared "
    "(e.g. \"What country is A from?\" / \"What country is B from?\"). Do NOT produce a yes/no "
    "sub-query; produce the fact-finding queries whose answers let a reasoner decide yes or no.\n\n"
    "HARD RULES:\n"
    "- Produce 2-3 sub-queries. Use 2 unless the question genuinely has three independent facts.\n"
    "- Sub-queries must be COMPLEMENTARY, not overlapping: each must target a DIFFERENT entity or "
    "a DIFFERENT attribute. If two sub-queries would retrieve the same documents, merge or rewrite them.\n"
    "- Maximize vocabulary DIVERSITY: do not copy the original question's wording verbatim into "
    "every sub-query. Rephrase using synonyms and the specific entity names so the embeddings "
    "spread across the corpus.\n"
    "- Each sub-query must be a complete, self-contained question that names its entity explicitly "
    "(no pronouns like \"it\" / \"they\" / \"that film\").\n"
    "- Never answer the question. Never add commentary, numbering, bullets, or labels.\n\n"
    "OUTPUT FORMAT (STRICT):\n"
    "First write a single line starting with \"THINKING:\" containing your brief reasoning about "
    "the question type and the hops.\n"
    "Then write a line containing exactly \"QUERIES:\".\n"
    "Then write each sub-query on its own line, one per line, with NO numbering, NO bullets, "
    "NO preamble, NO blank lines, and nothing after the last query.\n\n"
    "EXAMPLE A (bridge — GOOD):\n"
    "Question: \"What is the nationality of the director of the 1994 film Léon?\"\n"
    "THINKING: Bridge question. Hop 1 find the director of Léon (1994); hop 2 find that director's nationality.\n"
    "QUERIES:\n"
    "Who directed the 1994 film Léon?\n"
    "What is the nationality of director Luc Besson?\n\n"
    "EXAMPLE A (bridge — BAD, do NOT do this):\n"
    "What is the nationality of the director of the 1994 film Léon?\n"
    "(reason it is bad: this is the original question copied verbatim; it collapses both hops and "
    "gives the retriever no diversity.)\n\n"
    "EXAMPLE B (comparison — GOOD):\n"
    "Question: \"Which magazine was started first, Arthur's Magazine or First for Women?\"\n"
    "THINKING: Comparison of founding dates. One date sub-query per magazine.\n"
    "QUERIES:\n"
    "When was Arthur's Magazine first published?\n"
    "When was First for Women magazine first published?\n\n"
    "EXAMPLE B (comparison — BAD):\n"
    "Which was published first, Arthur's Magazine or First for Women?\n"
    "(bad: comparison kept inside one query; retriever cannot resolve a relative comparison.)\n\n"
    "EXAMPLE C (yes/no — GOOD):\n"
    "Question: \"Are both Kurram Garhi and Trogwood located in the same country?\"\n"
    "THINKING: Yes/no resolved by comparing the country of each place. One country sub-query per entity. "
    "No yes/no sub-query.\n"
    "QUERIES:\n"
    "In which country is Kurram Garhi located?\n"
    "In which country is Trogwood located?\n\n"
    "EXAMPLE C (yes/no — BAD):\n"
    "Are Kurram Garhi and Trogwood in the same country?\n"
    "(bad: a yes/no sub-query retrieves nothing useful; the decomposition must surface the two "
    "underlying country facts.)"
)

PLANNING_USER_TEMPLATE = (
    "Decompose the following question. Detect its type (bridge / comparison / yes-no), reason in "
    "one THINKING line, then output the sub-queries under QUERIES.\n\n"
    "Question: {question}{type_hint}"
)

# Regex to strip accidental leading numbering or bullets from sub-queries
_BULLET_RE = re.compile(r"^\s*(\d+[.)]\s*|[-*]\s*)")

# Regex to detect yes/no question starters
_YESNO_RE = re.compile(
    r"^\s*(is|was|did|does|can|are|were|have|has|do|will|would|could|should)\b",
    re.IGNORECASE,
)

# Regex to detect question type from THINKING line
_TYPE_BRIDGE_RE = re.compile(r"\bbridge\b", re.IGNORECASE)
_TYPE_COMPARISON_RE = re.compile(r"\bcomparison\b", re.IGNORECASE)
_TYPE_YESNO_RE = re.compile(r"\byes.?no\b", re.IGNORECASE)


def _detect_question_type_hint(question: str) -> str:
    """Return a short hint string appended to the user prompt, or empty string."""
    if _YESNO_RE.match(question.strip()):
        return "\n\n[Hint: this appears to be a YES/NO question — decompose into the underlying facts, not a yes/no sub-query.]"
    return ""


def _detect_type_from_thinking(thinking_line: str) -> str | None:
    """Extract question_type label from the THINKING line, if detectable."""
    if _TYPE_BRIDGE_RE.search(thinking_line):
        return "bridge"
    if _TYPE_COMPARISON_RE.search(thinking_line):
        return "comparison"
    if _TYPE_YESNO_RE.search(thinking_line):
        return "yes-no"
    return None


def _strip_bullet(line: str) -> str:
    """Remove leading numbering (1. 2. 3.) or bullets (- *) from a sub-query line."""
    return _BULLET_RE.sub("", line).strip()


def _dedup_queries(queries: list[str]) -> list[str]:
    """Remove near-duplicate sub-queries (case-insensitive, ignoring punctuation).

    Two sub-queries are considered near-duplicates when their normalised forms
    share more than 85 % of their tokens (Jaccard similarity).
    """
    def _normalise(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return set(tokens)

    kept: list[str] = []
    kept_sets: list[set[str]] = []
    for q in queries:
        q_set = _normalise(q)
        is_dup = False
        for existing_set in kept_sets:
            if not q_set or not existing_set:
                continue
            intersection = len(q_set & existing_set)
            union = len(q_set | existing_set)
            if union > 0 and intersection / union > 0.85:
                is_dup = True
                break
        if not is_dup:
            kept.append(q)
            kept_sets.append(q_set)
    return kept


def _parse_response(raw_text: str, question: str) -> tuple[list[str], str | None]:
    """Parse the LLM response into (sub_queries, question_type).

    Strategy:
    1. Look for a "QUERIES:" marker (case-insensitive).  Take everything after it.
    2. Scan for a "THINKING:" line to extract question_type.
    3. Fall back to treating every non-THINKING line as a sub-query if no marker found.
    4. Strip bullets/numbering, drop empties, cap at 3 sub-queries.
    5. Final fallback: [question] if still empty.
    """
    lines = raw_text.splitlines()

    # Extract THINKING line for type detection
    thinking_line = ""
    queries_start_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("THINKING:"):
            thinking_line = stripped
        if stripped.upper() == "QUERIES:":
            queries_start_idx = i + 1
            break

    question_type = _detect_type_from_thinking(thinking_line) if thinking_line else None

    # Extract candidate lines
    if queries_start_idx is not None:
        candidate_lines = lines[queries_start_idx:]
    else:
        # Fallback: treat every non-THINKING, non-QUERIES line as a sub-query
        candidate_lines = [
            l for l in lines
            if not l.strip().upper().startswith("THINKING:")
            and l.strip().upper() != "QUERIES:"
        ]

    # Clean up each candidate
    sub_queries: list[str] = []
    for line in candidate_lines:
        cleaned = _strip_bullet(line.strip())
        # Skip residual THINKING lines that may have slipped through
        if cleaned.upper().startswith("THINKING:"):
            continue
        if cleaned:
            sub_queries.append(cleaned)

    # Cap at 3
    sub_queries = sub_queries[:3]

    # Dedup
    sub_queries = _dedup_queries(sub_queries)

    # Final fallback
    if not sub_queries:
        sub_queries = [question]

    return sub_queries, question_type


def run_planning_agent(state: dict) -> dict:
    """Decompose the original question into sub-queries using the Groq LLM.

    Reads ``state["question"]`` and writes:
      - ``state["sub_queries"]``
      - ``state["question_type"]`` (when detectable: "bridge", "comparison", or "yes-no")
      - a reasoning-trace entry appended to ``state["reasoning_trace"]``

    Returns the updated state slice.
    """
    question: str = state["question"]
    client = get_client()

    type_hint = _detect_question_type_hint(question)
    user_message = PLANNING_USER_TEMPLATE.format(question=question, type_hint=type_hint)

    response = call_groq(
        client=client,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.2,
        max_tokens=512,
        caller="PlanningAgent",
    )

    raw_text: str = response.choices[0].message.content.strip()

    sub_queries, question_type = _parse_response(raw_text, question)

    trace_entry = (
        f"[PlanningAgent] Decomposed into {len(sub_queries)} sub-queries"
        + (f" (type={question_type})" if question_type else "")
        + ": "
        + " | ".join(sub_queries)
    )

    result: dict = {
        "sub_queries": sub_queries,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }
    if question_type is not None:
        result["question_type"] = question_type

    return result


