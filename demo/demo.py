"""Interactive command-line demo with rich terminal output."""

import sys
import os

# Ensure the project root is on the path when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from core.vector_store import initialize_store
from core.graph import build_graph
from data.loader import load_hotpotqa, index_hotpotqa_contexts

console = Console()

DEMO_QUESTIONS = [
    "What nationality is the director of the film Ran?",
    "The band that performed 'Hotel California' was formed in which city?",
    "What is the birth city of the author who wrote '1984'?",
    "Which country did the creator of Wikipedia grow up in?",
    "What language is spoken in the country where the Eiffel Tower is located?",
]


def print_banner() -> None:
    """Print the SHARP-RAG banner."""
    banner = Text()
    banner.append("  SHARP-RAG", style="bold cyan")
    banner.append("  —  Self-correcting Hierarchical Agentic RAG\n", style="dim")
    banner.append("  Independent Research · Elia Ghazal · 2026", style="dim italic")
    console.print(Panel(banner, border_style="cyan"))


def print_reasoning_trace(trace: list[str]) -> None:
    """Render the reasoning trace as a numbered table."""
    table = Table(title="Reasoning Trace", show_lines=True, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent Step", style="white")

    for i, step in enumerate(trace, start=1):
        # Colour-code by agent prefix
        if step.startswith("[PlanningAgent]"):
            style = "bold yellow"
        elif step.startswith("[RetrievalAgent]"):
            style = "bold blue"
        elif step.startswith("[CritiqueAgent]"):
            style = "bold magenta"
        elif step.startswith("[Generator]"):
            style = "bold green"
        else:
            style = "white"
        table.add_row(str(i), Text(step, style=style))

    console.print(table)


def print_docs(docs: list[dict]) -> None:
    """Display retrieved documents in a collapsible summary."""
    if not docs:
        console.print("[dim]  No documents retrieved.[/dim]")
        return

    table = Table(title=f"Retrieved Documents ({len(docs)})", show_lines=True, border_style="dim blue")
    table.add_column("Score", width=7)
    table.add_column("Source", width=25)
    table.add_column("Excerpt", width=80)

    for doc in docs:
        score = f"{doc.get('relevance_score', 0.0):.3f}"
        source = doc.get("source", "?")
        excerpt = doc.get("text", "")[:200].replace("\n", " ") + "…"
        table.add_row(score, source, excerpt)

    console.print(table)


def run_question(graph, question: str) -> dict:
    """Run the SHARP-RAG pipeline on a single question and return the result."""
    initial_state = {
        "question": question,
        "sub_queries": [],
        "retrieved_docs": [],
        "critique_result": "",
        "critique_feedback": "",
        "critique_suggested_queries": [],
        "critique_missing_information": [],
        "seen_doc_ids": [],
        "retry_count": 0,
        "final_answer": "",
        "reasoning_trace": [],
    }

    console.rule(f"[bold cyan]Running SHARP-RAG")
    console.print(f"[bold white]Question:[/bold white] {question}\n")

    with console.status("[bold green]Thinking…", spinner="dots"):
        result = graph.invoke(initial_state)

    return result


def display_result(result: dict) -> None:
    """Pretty-print the full result including trace and answer."""
    console.print()
    console.print(Panel(
        Text(f"Sub-queries: {result.get('sub_queries', [])}", style="yellow"),
        title="Planning",
        border_style="yellow",
    ))

    console.print()
    print_docs(result.get("retrieved_docs", []))

    console.print()
    verdict = result.get("critique_result", "—")
    feedback = result.get("critique_feedback", "—")
    verdict_color = {
        "sufficient": "green",
        "insufficient": "red",
        "contradictory": "orange3",
    }.get(verdict, "white")
    console.print(Panel(
        Text(f"Verdict: [{verdict_color}]{verdict}[/{verdict_color}]\n{feedback}"),
        title="Critique",
        border_style="magenta",
    ))

    console.print()
    print_reasoning_trace(result.get("reasoning_trace", []))

    console.print()
    answer = result.get("final_answer", "No answer produced.")
    console.print(Panel(
        Text(answer, style="bold white"),
        title="[bold green]Final Answer[/bold green]",
        border_style="green",
    ))

    retries = result.get("retry_count", 0)
    console.print(f"[dim]  Retrieval retries: {retries}[/dim]\n")


def interactive_loop(graph) -> None:
    """Run the interactive demo loop."""
    console.print("\n[bold]Pre-loaded example questions:[/bold]")
    for i, q in enumerate(DEMO_QUESTIONS, start=1):
        console.print(f"  [cyan]{i}.[/cyan] {q}")

    console.print("\n[dim]Type a number to use a pre-loaded question, "
                  "or type your own question. Press Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]> [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(DEMO_QUESTIONS):
                question = DEMO_QUESTIONS[idx]
            else:
                console.print("[red]Invalid number.[/red]")
                continue
        else:
            question = user_input

        try:
            result = run_question(graph, question)
            display_result(result)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")


def run_demo(vector_store=None, graph=None) -> None:
    """Entry point called by main.py --demo flag."""
    print_banner()

    if vector_store is None:
        console.print("[dim]Initialising vector store…[/dim]")
        vector_store = initialize_store()

    if vector_store.count() == 0:
        console.print("[yellow]Vector store is empty. Indexing HotpotQA (first 100)…[/yellow]")
        index_hotpotqa_contexts(vector_store)

    if graph is None:
        console.print("[dim]Building SHARP-RAG graph…[/dim]")
        graph = build_graph(vector_store)

    interactive_loop(graph)


if __name__ == "__main__":
    run_demo()
