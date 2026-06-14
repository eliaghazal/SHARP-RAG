"""
SHARP-RAG Evaluation Script
Compares: SHARP-RAG v2 vs Planning Baseline vs Naive RAG on 50 HotpotQA examples.
"""
import os
import json
import time
import sys

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Set GROQ_API_KEY before running evaluation.")
os.environ.setdefault("SHARP_RAG_EVAL_MODEL", "llama-3.1-8b-instant")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.vector_store import initialize_store
from core.graph import build_graph
from data.loader import load_hotpotqa, index_hotpotqa_contexts
from evaluation.metrics import evaluate_dataset, build_baseline_graph, build_naive_rag_graph

N_EVAL = 20

print("Loading HotpotQA...")
examples = load_hotpotqa(max_examples=N_EVAL + 200)[:N_EVAL]
print(f"Loaded {len(examples)} examples.")

print("Initializing vector store...")
_HERE = os.path.dirname(os.path.abspath(__file__))
vs = initialize_store(os.path.join(_HERE, "chroma_db"))

if vs.count() < 100:
    print("Indexing contexts...")
    index_hotpotqa_contexts(vs, examples)

print(f"Vector store has {vs.count()} docs")

print("\n=== Evaluating SHARP-RAG v2 ===")
sharp_rag_graph = build_graph(vs)
t0 = time.time()
sharp_results = evaluate_dataset(sharp_rag_graph, examples)
sharp_latency = (time.time() - t0) * 1000 / N_EVAL
sharp_results["avg_latency_ms"] = sharp_latency

print(f"SHARP-RAG: EM={sharp_results['exact_match_score']:.4f}, F1={sharp_results['f1_score']:.4f}, "
      f"retries={sharp_results['avg_retry_count']:.2f}, latency={sharp_latency:.0f}ms")

print("\n=== Evaluating Planning Baseline (no critique) ===")
planning_graph = build_baseline_graph(vs)
t0 = time.time()
planning_results = evaluate_dataset(planning_graph, examples, baseline=True)
planning_latency = (time.time() - t0) * 1000 / N_EVAL
planning_results["avg_latency_ms"] = planning_latency

print(f"Planning Baseline: EM={planning_results['exact_match_score']:.4f}, F1={planning_results['f1_score']:.4f}, "
      f"latency={planning_latency:.0f}ms")

print("\n=== Evaluating Naive RAG (no planning, no critique) ===")
naive_graph = build_naive_rag_graph(vs)
t0 = time.time()
naive_results = evaluate_dataset(naive_graph, examples)
naive_latency = (time.time() - t0) * 1000 / N_EVAL
naive_results["avg_latency_ms"] = naive_latency

print(f"Naive RAG: EM={naive_results['exact_match_score']:.4f}, F1={naive_results['f1_score']:.4f}, "
      f"latency={naive_latency:.0f}ms")

# Collect 5 sample outputs for paper
sample_outputs = []
print("\n=== Collecting 5 sample outputs ===")
for ex in examples[:5]:
    q = ex["question"]
    gt = ex["answer"]
    init = {
        "question": q,
        "question_type": ex.get("type", "unknown"),
        "sub_queries": [],
        "retrieved_docs": [],
        "critique_result": "",
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
    try:
        sr = sharp_rag_graph.invoke(dict(init))
        init_baseline = dict(init)
        init_baseline["critique_result"] = "sufficient"
        br = planning_graph.invoke(init_baseline)
        sample_outputs.append({
            "question": q,
            "ground_truth": gt,
            "sharp_rag_answer": sr.get("final_answer", ""),
            "baseline_answer": br.get("final_answer", ""),
            "retry_count": sr.get("retry_count", 0)
        })
        print(f"  Sample: Q={q[:60]}... -> SHARP={sr.get('final_answer','')}, GT={gt}")
    except Exception as e:
        print(f"Sample error: {e}")
        sample_outputs.append({
            "question": q,
            "ground_truth": gt,
            "sharp_rag_answer": "",
            "baseline_answer": "",
            "retry_count": 0
        })

final = {
    "sharp_rag_v2": {
        "exact_match": sharp_results["exact_match_score"],
        "f1_score": sharp_results["f1_score"],
        "avg_retry_count": sharp_results["avg_retry_count"],
        "avg_latency_ms": sharp_results["avg_latency_ms"],
        "critique_distribution": sharp_results["critique_distribution"]
    },
    "planning_baseline": {
        "exact_match": planning_results["exact_match_score"],
        "f1_score": planning_results["f1_score"],
        "avg_latency_ms": planning_results["avg_latency_ms"]
    },
    "naive_rag": {
        "exact_match": naive_results["exact_match_score"],
        "f1_score": naive_results["f1_score"],
        "avg_latency_ms": naive_results["avg_latency_ms"]
    },
    "n_evaluated": N_EVAL,
    "sample_outputs": sample_outputs
}

print("\n=== FINAL RESULTS ===")
print(json.dumps(final, indent=2))

with open(os.path.join(_HERE, "eval_results.json"), "w") as f:
    json.dump(final, f, indent=2)

print("\nResults saved to eval_results.json")
