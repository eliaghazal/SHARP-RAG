"""LangGraph pipeline definition for SHARP-RAG."""

import os
import re
import time
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.vector_store import VectorStore
from core.groq_client import call_groq, get_client
from agents.planning_agent import run_planning_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.critique_agent import run_critique_agent

load_dotenv()

GENERATOR_SYSTEM_PROMPT = (
    "You are a precise, evidence-grounded multi-hop question-answering assistant. "
    "You must reason over the evidence to chain facts together, then commit to a "
    "SHORT final answer span suitable for exact-match scoring.\n\n"
    "RULES:\n"
    "1. Answer ONLY from the provided evidence. Never fabricate. If the evidence is "
    "genuinely insufficient, your FINAL ANSWER must be the single word: unknown.\n"
    "2. For BRIDGE questions, chain the hops explicitly in your reasoning: first "
    "identify the bridge entity from the evidence, then read off its target property. "
    "The FINAL ANSWER is the target property, not the bridge entity.\n"
    "3. For COMPARISON questions, state both compared values from the evidence, then "
    "give the entity or value the question asks for as the FINAL ANSWER.\n"
    "4. For YES/NO questions, the FINAL ANSWER must be exactly \"yes\" or \"no\" "
    "(lowercase, nothing else).\n"
    "5. If evidence is contradictory, give the best-supported value as the FINAL ANSWER "
    "and note the discrepancy in one clause of the explanation only.\n\n"
    "OUTPUT FORMAT (STRICT):\n"
    "Write 1-3 sentences of grounded explanation that show how the evidence yields the "
    "answer. Then, on its OWN final line, write exactly:\n"
    "FINAL ANSWER: <answer>\n"
    "where <answer> is the MINIMAL answer span — 1 to 5 words: an entity name, a "
    "number, a date, or yes/no. Do NOT include articles (\"the\", \"a\"), explanations, "
    "trailing punctuation, or a full sentence on the FINAL ANSWER line. Output the "
    "FINAL ANSWER line exactly once, as the last line.\n\n"
    "EXAMPLE (bridge):\n"
    "Evidence [2] says the 1994 film León was directed by Luc Besson; evidence [5] "
    "states Luc Besson is French. Chaining these, the director's nationality is French.\n"
    "FINAL ANSWER: French\n\n"
    "EXAMPLE (yes/no):\n"
    "Evidence [1] places Kurram Garhi in Pakistan and evidence [3] places Trogwood in "
    "the United States, so they are not in the same country.\n"
    "FINAL ANSWER: no\n\n"
    "EXAMPLE (comparison):\n"
    "Evidence [4] dates Arthur's Magazine to 1844 and evidence [6] dates First for Women "
    "to 1989, so Arthur's Magazine came first.\n"
    "FINAL ANSWER: Arthur's Magazine"
)

GENERATOR_USER_TEMPLATE = (
    "Question: {question}\n"
    "Question type (if known): {question_type}\n\n"
    "Evidence ({n_docs} retrieved chunks):\n"
    "{evidence_block}\n\n"
    "Reason over the evidence, then end with the FINAL ANSWER line.\n"
    "Answer:"
)


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------

def _extract_final_answer(text: str) -> str:
    """Extract the minimal answer span from the model's output.

    Strategy:
    1. Find the LAST occurrence of "FINAL ANSWER:" (case-insensitive).
    2. Take everything after it on the same line.
    3. Strip whitespace, trailing period, and leading articles (a / an / the).
    4. If no marker is found, fall back to the last non-empty line of the text.
    """
    # Look for the last FINAL ANSWER: marker
    pattern = re.compile(r"(?i)final\s+answer\s*:\s*(.*)")
    matches = list(pattern.finditer(text))

    if matches:
        raw_span = matches[-1].group(1).strip()
    else:
        # Fallback: last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        raw_span = lines[-1] if lines else text.strip()

    # Strip trailing period and leading articles
    raw_span = raw_span.rstrip(".")
    raw_span = re.sub(r"^(the|a|an)\s+", "", raw_span, flags=re.IGNORECASE)
    return raw_span.strip()


# --------------------------------------------------------------------------
# Graph builder
# --------------------------------------------------------------------------

def build_graph(vector_store: VectorStore) -> Any:
    """Compile and return the SHARP-RAG StateGraph.

    Nodes
    -----
    planner   → decomposes the question into sub-queries
    retriever → retrieves relevant docs from ChromaDB
    critic    → evaluates evidence quality
    generator → synthesises the final answer
    end       → terminal node (identity pass-through)

    The critic uses a conditional edge: if evidence is sufficient (or retries
    are exhausted), flow goes to the generator; otherwise flow loops back to
    the retriever for an additional retrieval pass.
    """
    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------ nodes

    def planner_node(state: AgentState) -> dict:
        """Planning agent node — decomposes question into sub-queries."""
        t0 = time.time()
        result = run_planning_agent(state)
        latency_ms = (time.time() - t0) * 1000
        trace = result.get("reasoning_trace", state.get("reasoning_trace", []))
        trace = trace + [f"[Latency] planner={latency_ms:.1f}ms"]
        result["planner_latency_ms"] = latency_ms
        result["reasoning_trace"] = trace
        return result

    def retriever_node(state: AgentState) -> dict:
        """Retrieval agent node — fetches relevant docs from the vector store."""
        t0 = time.time()
        result = run_retrieval_agent(state, vector_store)
        latency_ms = (time.time() - t0) * 1000
        trace = result.get("reasoning_trace", state.get("reasoning_trace", []))
        trace = trace + [f"[Latency] retriever={latency_ms:.1f}ms"]
        result["retriever_latency_ms"] = latency_ms
        result["reasoning_trace"] = trace
        return result

    def critic_node(state: AgentState) -> dict:
        """Critique agent node — judges evidence sufficiency."""
        t0 = time.time()
        result = run_critique_agent(state)
        latency_ms = (time.time() - t0) * 1000
        trace = result.get("reasoning_trace", state.get("reasoning_trace", []))
        trace = trace + [f"[Latency] critic={latency_ms:.1f}ms"]
        result["critic_latency_ms"] = latency_ms
        result["reasoning_trace"] = trace
        return result

    def generator_node(state: AgentState) -> dict:
        """Generator node — produces the final answer using Groq LLM."""
        t0 = time.time()
        result = _run_generator(state)
        latency_ms = (time.time() - t0) * 1000
        trace = result.get("reasoning_trace", state.get("reasoning_trace", []))
        trace = trace + [f"[Latency] generator={latency_ms:.1f}ms"]
        result["generator_latency_ms"] = latency_ms
        result["reasoning_trace"] = trace
        return result

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic", critic_node)
    graph.add_node("generator", generator_node)

    # ------------------------------------------------------------------ edges

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")

    def critic_router(state: AgentState) -> Literal["generator", "retriever"]:
        """Route from critic: loop back to retriever if insufficient and retries remain."""
        result = state.get("critique_result", "insufficient")
        retry_count = state.get("retry_count", 0)

        if result == "insufficient" and retry_count < 2:
            return "retriever"
        return "generator"

    graph.add_conditional_edges("critic", critic_router, ["generator", "retriever"])
    graph.add_edge("generator", END)

    return graph.compile()


# --------------------------------------------------------------------------
# Generator implementation
# --------------------------------------------------------------------------

def _run_generator(state: AgentState) -> dict:
    """Synthesise a final answer from retrieved evidence using the Groq LLM."""
    question: str = state["question"]
    question_type: str = state.get("question_type", "unknown")
    retrieved_docs: list[dict] = state.get("retrieved_docs", [])
    critique_result: str = state.get("critique_result", "sufficient")

    evidence_block = _format_evidence(retrieved_docs)
    user_message = GENERATOR_USER_TEMPLATE.format(
        question=question,
        question_type=question_type,
        n_docs=len(retrieved_docs),
        evidence_block=evidence_block,
    )

    client = get_client()
    response = call_groq(
        client=client,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.3,
        max_tokens=512,
        caller="Generator",
    )

    full_text: str = response.choices[0].message.content.strip()
    answer_span: str = _extract_final_answer(full_text)

    trace_entry = (
        f"[Generator] Produced answer (critique_result='{critique_result}', "
        f"docs_used={len(retrieved_docs)}): {answer_span[:120]}…"
        if len(answer_span) > 120
        else f"[Generator] Produced answer: {answer_span}"
    )

    return {
        "final_answer": answer_span,
        "final_answer_full": full_text,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def _format_evidence(docs: list[dict]) -> str:
    """Format docs for the generator prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "unknown")
        text = doc.get("text", "").strip()
        parts.append(f"[{i}] {source}\n{text}")
    return "\n\n".join(parts) if parts else "No evidence retrieved."


