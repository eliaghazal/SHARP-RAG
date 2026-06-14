"""Visualisation utilities for SHARP-RAG evaluation results."""

from __future__ import annotations

import os
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(results: dict, save_path: str) -> None:
    """Bar chart comparing EM and F1 across evaluation systems.

    Parameters
    ----------
    results:
        Dict mapping system name -> metrics dict, e.g.::

            {
                "naive_rag":         {"exact_match_score": 0.21, "f1_score": 0.34},
                "planning_baseline": {"exact_match_score": 0.29, "f1_score": 0.41},
                "sharp_rag_v2":      {"exact_match_score": 0.37, "f1_score": 0.52},
            }

    save_path:
        Absolute or relative path where the PNG will be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    systems = list(results.keys())
    em_scores = [results[s].get("exact_match_score", 0.0) for s in systems]
    f1_scores = [results[s].get("f1_score", 0.0) for s in systems]

    x = np.arange(len(systems))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_em = ax.bar(x - bar_width / 2, em_scores, bar_width,
                     label="Exact Match", color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + bar_width / 2, f1_scores, bar_width,
                     label="F1 Score", color="#DD8452", edgecolor="white", linewidth=0.8)

    # Annotate bar values
    for bar in bars_em:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("System", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Exact Match & F1 by System", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in systems], fontsize=10)
    ax.set_ylim(0, min(1.05, max(em_scores + f1_scores) * 1.25 + 0.05))
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved metrics comparison chart → {save_path}")


# ---------------------------------------------------------------------------
# 2. Retry / critique verdict distribution pie chart
# ---------------------------------------------------------------------------

def plot_retry_distribution(critique_distribution: dict, save_path: str) -> None:
    """Pie chart of critique verdicts (sufficient / insufficient / contradictory).

    Parameters
    ----------
    critique_distribution:
        Dict with string keys and integer counts, e.g.::

            {"sufficient": 62, "insufficient": 25, "contradictory": 13}

    save_path:
        Path where the PNG will be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    sizes = []
    for key in ("sufficient", "insufficient", "contradictory"):
        count = critique_distribution.get(key, 0)
        if count > 0:
            labels.append(key.capitalize())
            sizes.append(count)

    if not sizes:
        print("[Plots] critique_distribution is empty — skipping pie chart.")
        return

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    explode = [0.05] * len(sizes)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors[: len(sizes)],
        explode=explode[: len(sizes)],
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title("Critique Verdict Distribution", fontsize=14, fontweight="bold", pad=14)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved critique distribution chart → {save_path}")


# ---------------------------------------------------------------------------
# 3. Latency breakdown stacked bar chart
# ---------------------------------------------------------------------------

def plot_latency_breakdown(latency_data: dict, save_path: str) -> None:
    """Stacked bar chart of per-node latency across systems.

    Parameters
    ----------
    latency_data:
        Dict mapping system name -> per-node latency dict, e.g.::

            {
                "naive_rag": {
                    "planner_latency_ms": 0,
                    "retriever_latency_ms": 310,
                    "critic_latency_ms": 0,
                    "generator_latency_ms": 480,
                },
                "sharp_rag_v2": {
                    "planner_latency_ms": 520,
                    "retriever_latency_ms": 340,
                    "critic_latency_ms": 430,
                    "generator_latency_ms": 510,
                },
            }

    save_path:
        Path where the PNG will be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nodes = ["planner_latency_ms", "retriever_latency_ms",
             "critic_latency_ms", "generator_latency_ms"]
    node_labels = ["Planner", "Retriever", "Critic", "Generator"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    systems = list(latency_data.keys())
    x = np.arange(len(systems))
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(systems))

    for node_key, node_label, color in zip(nodes, node_labels, colors):
        values = np.array(
            [latency_data[s].get(node_key, 0.0) for s in systems], dtype=float
        )
        bars = ax.bar(x, values, bar_width, bottom=bottoms,
                      label=node_label, color=color, edgecolor="white", linewidth=0.6)

        # Annotate segments that are tall enough to read
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 30:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + val / 2,
                    f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += values

    ax.set_xlabel("System", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Breakdown by Node and System", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in systems], fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved latency breakdown chart → {save_path}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they don't exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
