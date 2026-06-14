"""Evaluation metrics: exact match, token-level F1, ROUGE-L, dataset-level evaluation."""

import re
import string
import time
from collections import Counter
from typing import Any

from rouge_score import rouge_scorer as _rouge_scorer_module


# --------------------------------------------------------------------------
# Per-prediction metrics
# --------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lower-case, strip punctuation and articles (SQuAD-style normalisation)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if normalised prediction equals normalised ground truth."""
    return normalize(prediction) == normalize(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth.

    Uses the same normalisation and bag-of-words overlap as SQuAD / HotpotQA.
    """
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F-measure between *prediction* and *ground_truth*.

    Uses the ``rouge_score`` library with stemming enabled to be consistent
    with standard NLP evaluation conventions.
    """
    scorer = _rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(ground_truth, prediction)
    return result["rougeL"].fmeasure


# --------------------------------------------------------------------------
# Dataset-level evaluation
# --------------------------------------------------------------------------

def evaluate_dataset(
    graph: Any,
    questions: list[dict],
    baseline: bool = False,
) -> dict:
    """Run *graph* on each question dict and aggregate metrics.

    Parameters
    ----------
    graph:
        A compiled LangGraph pipeline (or a simple callable for baseline).
    questions:
        List of dicts with at least ``question`` and ``answer`` keys
        (HotpotQA format from ``data.loader.load_hotpotqa``).
    baseline:
        If True, skip the critic loop by marking critique_result as "sufficient"
        before the first pass (simulates naive RAG).

    Returns
    -------
    dict with keys:
        exact_match_score, f1_score, rouge_l_score, avg_retry_count,
        avg_latency_ms, critique_distribution, n_evaluated
    """
    em_scores: list[float] = []
    f1_scores: list[float] = []
    rouge_scores: list[float] = []
    retry_counts: list[int] = []
    latency_ms_list: list[float] = []
    critique_counts: dict[str, int] = {
        "sufficient": 0,
        "insufficient": 0,
        "contradictory": 0,
    }

    for i, ex in enumerate(questions):
        question: str = ex["question"]
        ground_truth: str = ex["answer"]
        question_type: str = ex.get("type", "unknown")

        initial_state: dict = {
            "question": question,
            "question_type": question_type,
            "sub_queries": [],
            "retrieved_docs": [],
            "critique_result": "sufficient" if baseline else "",
            "critique_feedback": "",
            "critique_suggested_queries": [],
            "critique_missing_information": [],
            "seen_doc_ids": [],
            "retry_count": 0,
            "final_answer": "",
            "final_answer_full": "",
            "reasoning_trace": [],
            "planner_latency_ms": 0.0,
            "retriever_latency_ms": 0.0,
            "critic_latency_ms": 0.0,
            "generator_latency_ms": 0.0,
        }

        t0 = time.time()
        try:
            result = graph.invoke(initial_state)
            elapsed_ms = (time.time() - t0) * 1000

            prediction: str = result.get("final_answer", "")
            retry: int = result.get("retry_count", 0)
            verdict: str = result.get("critique_result", "sufficient")

            em_scores.append(float(exact_match(prediction, ground_truth)))
            f1_scores.append(f1_score(prediction, ground_truth))
            rouge_scores.append(rouge_l_score(prediction, ground_truth))
            retry_counts.append(retry)
            latency_ms_list.append(elapsed_ms)
            if verdict in critique_counts:
                critique_counts[verdict] += 1

        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000
            print(f"[Eval] Error on example {i}: {exc}")
            em_scores.append(0.0)
            f1_scores.append(0.0)
            rouge_scores.append(0.0)
            retry_counts.append(0)
            latency_ms_list.append(elapsed_ms)

        time.sleep(2)  # pace Groq calls to stay within rate limits
        if (i + 1) % 5 == 0:
            print(f"[Eval] Processed {i + 1}/{len(questions)} …")

    n = len(questions)
    return {
        "exact_match_score": sum(em_scores) / n if n else 0.0,
        "f1_score": sum(f1_scores) / n if n else 0.0,
        "rouge_l_score": sum(rouge_scores) / n if n else 0.0,
        "avg_retry_count": sum(retry_counts) / n if n else 0.0,
        "avg_latency_ms": sum(latency_ms_list) / n if n else 0.0,
        "critique_distribution": critique_counts,
        "n_evaluated": n,
    }


# --------------------------------------------------------------------------
# Baseline graphs
# --------------------------------------------------------------------------

def build_baseline_graph(vector_store: Any) -> Any:
    """Build a planning+retrieval graph with NO critique loop (planning baseline).

    The critic node is bypassed by always routing directly to the generator.
    This isolates the value of the critique-retry loop from planning gains.
    """
    from langgraph.graph import StateGraph, START, END
    from core.state import AgentState
    from agents.planning_agent import run_planning_agent
    from agents.retrieval_agent import run_retrieval_agent
    from core.graph import _run_generator

    bg = StateGraph(AgentState)

    def planner_node(state: AgentState) -> dict:
        return run_planning_agent(state)

    def retriever_node(state: AgentState) -> dict:
        return run_retrieval_agent(state, vector_store)

    def generator_node(state: AgentState) -> dict:
        patched = dict(state)
        patched["critique_result"] = "sufficient"
        return _run_generator(patched)

    bg.add_node("planner", planner_node)
    bg.add_node("retriever", retriever_node)
    bg.add_node("generator", generator_node)

    bg.add_edge(START, "planner")
    bg.add_edge("planner", "retriever")
    bg.add_edge("retriever", "generator")
    bg.add_edge("generator", END)

    return bg.compile()


def build_naive_rag_graph(vector_store: Any) -> Any:
    """Build a TRUE naive-RAG graph: no planning, no critique, single retrieval pass.

    The original question is used directly as the retrieval query — there is no
    planning decomposition and no critique-retry loop. This is the weakest
    baseline and establishes the floor for comparison.
    """
    import os
    from groq import Groq
    from dotenv import load_dotenv
    from langgraph.graph import StateGraph, START, END
    from core.state import AgentState
    from core.graph import (
        GENERATOR_SYSTEM_PROMPT,
        GENERATOR_USER_TEMPLATE,
        _format_evidence,
        _extract_final_answer,
        _call_groq_with_retry,
    )

    load_dotenv()

    ng = StateGraph(AgentState)

    def naive_retriever_node(state: AgentState) -> dict:
        """Retrieve directly using the original question (no sub-queries)."""
        question: str = state["question"]
        docs = vector_store.search(question, n_results=5)
        docs.sort(key=lambda d: d.get("relevance_score", 0.0), reverse=True)
        return {
            "retrieved_docs": docs,
            "sub_queries": [question],
            "reasoning_trace": state.get("reasoning_trace", [])
            + [f"[NaiveRetriever] Retrieved {len(docs)} docs for original question."],
        }

    def naive_generator_node(state: AgentState) -> dict:
        """Generate directly from retrieved docs, no critique."""
        question: str = state["question"]
        question_type: str = state.get("question_type", "unknown")
        retrieved_docs: list[dict] = state.get("retrieved_docs", [])

        evidence_block = _format_evidence(retrieved_docs)
        user_message = GENERATOR_USER_TEMPLATE.format(
            question=question,
            question_type=question_type,
            n_docs=len(retrieved_docs),
            evidence_block=evidence_block,
        )

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = _call_groq_with_retry(
            client=client,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_message=user_message,
        )

        full_text: str = response.choices[0].message.content.strip()
        answer_span: str = _extract_final_answer(full_text)

        return {
            "final_answer": answer_span,
            "final_answer_full": full_text,
            "critique_result": "sufficient",
            "reasoning_trace": state.get("reasoning_trace", [])
            + [f"[NaiveGenerator] Answer: {answer_span}"],
        }

    ng.add_node("retriever", naive_retriever_node)
    ng.add_node("generator", naive_generator_node)

    ng.add_edge(START, "retriever")
    ng.add_edge("retriever", "generator")
    ng.add_edge("generator", END)

    return ng.compile()
