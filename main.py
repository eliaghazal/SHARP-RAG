"""SHARP-RAG — main entry point.

Usage
-----
  python main.py --demo         # interactive demo
  python main.py --evaluate     # run evaluation on HotpotQA subset
  python main.py --sanity       # quick sanity check (default)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()


def _check_env() -> None:
    """Abort early if GROQ_API_KEY is not set."""
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[red]Error:[/red] GROQ_API_KEY not set. "
                      "Create a .env file with GROQ_API_KEY=<your-key>.")
        sys.exit(1)


def run_sanity_check() -> None:
    """Test Groq connectivity, ChromaDB init, and a full pipeline pass."""
    from core.vector_store import initialize_store
    from core.graph import build_graph
    from data.loader import load_hotpotqa, index_hotpotqa_contexts

    console.rule("[bold cyan]SHARP-RAG Sanity Check")

    # --- Groq connectivity ---
    console.print("\n[1/3] Testing Groq API connection…")
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content.strip()
        console.print(f"   [green]Groq OK[/green] — model replied: '{reply}'")
    except Exception as exc:
        console.print(f"   [red]Groq FAILED:[/red] {exc}")
        sys.exit(1)

    # --- ChromaDB init ---
    console.print("\n[2/3] Initialising ChromaDB vector store…")
    try:
        store = initialize_store()
        console.print(f"   [green]ChromaDB OK[/green] — {store.count()} docs in collection")
    except Exception as exc:
        console.print(f"   [red]ChromaDB FAILED:[/red] {exc}")
        sys.exit(1)

    # --- Index data if needed ---
    if store.count() == 0:
        console.print("   [yellow]Store empty — indexing first 20 HotpotQA examples…[/yellow]")
        examples = load_hotpotqa(max_examples=20)
        index_hotpotqa_contexts(store, examples=examples)

    # --- Full pipeline on 1 question ---
    console.print("\n[3/3] Running full pipeline on 1 HotpotQA question…")
    try:
        examples = load_hotpotqa(max_examples=1)
        sample = examples[0]
        graph = build_graph(store)

        initial_state = {
            "question": sample["question"],
            "question_type": sample.get("type", "unknown"),
            "sub_queries": [],
            "retrieved_docs": [],
            "critique_result": "",
            "critique_feedback": "",
            "critique_confidence": 0.0,
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

        with console.status("[bold green]Invoking graph…", spinner="dots"):
            result = graph.invoke(initial_state)

        console.print(f"\n   [bold]Question:[/bold]  {sample['question']}")
        console.print(f"   [bold]Ground truth:[/bold] {sample['answer']}")
        console.print(f"   [bold]SHARP-RAG:[/bold]    {result['final_answer']}")
        console.print(f"   [bold]Critique:[/bold]     {result['critique_result']}")
        console.print(f"   [bold]Retries:[/bold]      {result['retry_count']}")
        console.print("\n   [bold]Reasoning trace:[/bold]")
        for i, step in enumerate(result["reasoning_trace"], start=1):
            console.print(f"     {i}. {step}")

        console.print("\n[green bold]Sanity check passed.[/green bold]")

    except Exception as exc:
        console.print(f"   [red]Pipeline FAILED:[/red] {exc}")
        raise


def run_evaluation(n_eval: int = 20) -> None:
    """Evaluate SHARP-RAG and a naive-RAG baseline, then print a comparison table."""
    from core.vector_store import initialize_store
    from core.graph import build_graph
    from data.loader import load_hotpotqa, index_hotpotqa_contexts
    from evaluation.metrics import evaluate_dataset, build_baseline_graph

    console.rule("[bold cyan]SHARP-RAG Evaluation")

    store = initialize_store()
    if store.count() == 0:
        console.print("[yellow]Indexing HotpotQA contexts…[/yellow]")
        examples_all = load_hotpotqa()
        index_hotpotqa_contexts(store, examples=examples_all)

    examples = load_hotpotqa(max_examples=n_eval)
    console.print(f"[dim]Evaluating on {len(examples)} questions…[/dim]\n")

    # Build graphs
    sharp_graph = build_graph(store)
    baseline_graph = build_baseline_graph(store)

    console.print("[bold]Running SHARP-RAG…[/bold]")
    sharp_results = evaluate_dataset(sharp_graph, examples)

    console.print("[bold]Running Naive-RAG baseline…[/bold]")
    baseline_results = evaluate_dataset(baseline_graph, examples, baseline=True)

    # Print comparison table
    table = Table(title="SHARP-RAG vs Naive-RAG Baseline", border_style="cyan")
    table.add_column("Metric", style="bold white", width=28)
    table.add_column("SHARP-RAG", style="bold green", justify="right")
    table.add_column("Naive RAG", style="bold yellow", justify="right")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    metrics = [
        ("Exact Match", "exact_match_score"),
        ("F1 Score", "f1_score"),
        ("Avg Retry Count", "avg_retry_count"),
        ("N Evaluated", "n_evaluated"),
    ]
    for label, key in metrics:
        table.add_row(label, fmt(sharp_results[key]), fmt(baseline_results[key]))

    console.print(table)

    # Critique distribution
    console.print("\n[bold]Critique distribution (SHARP-RAG):[/bold]")
    dist = sharp_results["critique_distribution"]
    for verdict, count in dist.items():
        pct = 100 * count / len(examples) if examples else 0
        console.print(f"  {verdict:15s}  {count:3d}  ({pct:.1f}%)")


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="SHARP-RAG — Self-correcting Hierarchical Agentic RAG"
    )
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on HotpotQA")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check (default)")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=20,
        help="Number of examples for evaluation (default: 20)",
    )
    args = parser.parse_args()

    _check_env()

    if args.demo:
        from demo.demo import run_demo
        run_demo()
    elif args.evaluate:
        run_evaluation(n_eval=args.n_eval)
    else:
        # Default: sanity check
        run_sanity_check()


if __name__ == "__main__":
    main()
