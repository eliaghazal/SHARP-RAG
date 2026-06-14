"""Shared state schema for the SHARP-RAG agent pipeline."""

from typing import TypedDict


class AgentState(TypedDict):
    """Shared mutable state passed between all agents in the LangGraph pipeline."""

    question: str
    question_type: str                  # "bridge" | "comparison" | "yes/no" | "unknown"
    sub_queries: list[str]
    retrieved_docs: list[dict]          # each: {id, text, source, relevance_score}
    critique_result: str                # "sufficient" | "insufficient" | "contradictory"
    critique_feedback: str              # natural-language explanation of critique (from reasoning)
    critique_confidence: float          # 0.0-1.0 confidence in the verdict
    critique_missing_information: list[str]  # specific missing facts identified by critique
    critique_suggested_queries: list[str]    # concrete retrieval queries for missing facts
    seen_doc_ids: list[str]             # cross-pass dedup: IDs already returned to avoid repetition
    retry_count: int
    final_answer: str                   # extracted minimal answer span
    final_answer_full: str              # full model output (trace / demo use)
    reasoning_trace: list[str]          # ordered log of every agent decision
    # Per-node latency tracking (milliseconds)
    planner_latency_ms: float
    retriever_latency_ms: float
    critic_latency_ms: float
    generator_latency_ms: float
