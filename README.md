# SHARP-RAG

**Self-correcting Hierarchical Agentic RAG**

> Independent research by Elia Ghazal · June 2026  
> Paper: [Link to paper](https://zenodo.org/records/20690440)

---

## Overview

Naive RAG retrieves documents once and generates an answer regardless of evidence quality. **SHARP-RAG** fixes this with a self-correcting loop: a _Critique Agent_ evaluates the retrieved evidence and, if it is insufficient, triggers targeted re-retrieval before generating a final answer.

Three specialised agents collaborate in a LangGraph pipeline:

| Agent | Role |
|---|---|
| **Planning Agent** | Decomposes a complex multi-hop question into 2–3 focused sub-queries |
| **Retrieval Agent** | Searches ChromaDB for the most relevant document chunks per sub-query |
| **Critique Agent** | Judges whether the evidence is *sufficient*, *insufficient*, or *contradictory* |
| **Generator** | Synthesises the final answer strictly from retrieved evidence |

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │               SHARP-RAG Pipeline             │
                     └─────────────────────────────────────────────┘

  Question
     │
     ▼
┌─────────────┐
│  Planning   │  Groq LLM → decomposes question into 2–3 sub-queries
│   Agent     │
└──────┬──────┘
       │ sub_queries
       ▼
┌─────────────┐
│  Retrieval  │  sentence-transformers + ChromaDB
│   Agent     │  → top-3 docs per sub-query, deduplicated
└──────┬──────┘
       │ retrieved_docs
       ▼
┌─────────────┐
│  Critique   │  Groq LLM → evaluates evidence quality
│   Agent     │
└──────┬──────┘
       │
       ├─── "sufficient" ──────────────────────────────┐
       │                                               │
       ├─── "contradictory" ───────────────────────────┤
       │                                               │
       └─── "insufficient" + retry_count < 2 ──────────┼──► Retrieval Agent
                                                       │    (reformulated queries)
                                                       ▼
                                               ┌─────────────┐
                                               │  Generator  │  Groq LLM → final answer
                                               └──────┬──────┘
                                                      │
                                                      ▼
                                               final_answer + reasoning_trace
```

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Agent orchestration | LangGraph |
| LLM inference | Groq API (`llama-3.3-70b-versatile`) |
| Vector store | ChromaDB (persistent) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Dataset | HotpotQA (HuggingFace `datasets`) |
| Terminal UI | Rich |

---

## Installation

```bash
# 1. Clone / navigate to the project
cd sharp-rag

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

Get a free Groq API key at <https://console.groq.com>.

---

## Usage

### Sanity check (default)
Verifies Groq connectivity, ChromaDB init, and runs the full pipeline on one sample question.

```bash
python main.py
# or explicitly:
python main.py --sanity
```

### Interactive demo
```bash
python main.py --demo
```

Presents a menu of pre-loaded example questions or accepts free-text input.  
Shows the full reasoning trace (sub-queries, retrieved docs, critique verdict, final answer) in a colourful terminal UI.

### Evaluation
```bash
python main.py --evaluate --n-eval 20
```

Runs SHARP-RAG and a naive-RAG baseline on `n-eval` HotpotQA questions and prints a comparison table:

```
╔══════════════════════════════╦═════════════╦═══════════╗
║ Metric                       ║  SHARP-RAG  ║ Naive RAG ║
╠══════════════════════════════╬═════════════╬═══════════╣
║ Exact Match                  ║  0.2500     ║ 0.1500    ║
║ F1 Score                     ║  0.4231     ║ 0.3102    ║
║ Avg Retry Count              ║  0.6000     ║ 0.0000    ║
║ N Evaluated                  ║  20         ║ 20        ║
╚══════════════════════════════╩═════════════╩═══════════╝
```

---

## Example Output (demo)

```
Question: What nationality is the director of the film Ran?

[PlanningAgent] Decomposed into 2 sub-queries:
  • Who directed the film Ran?
  • What nationality is [director's name]?

[RetrievalAgent] Retrieved 5 unique docs across 2 queries (retry=0).

[CritiqueAgent] Verdict='sufficient' | Evidence covers both facts needed.

[Generator] Answer: Akira Kurosawa, the director of Ran (1985), is Japanese.
```

---

## Project Structure

```
sharp-rag/
├── .env                     # GROQ_API_KEY
├── requirements.txt
├── README.md
├── main.py                  # Entry point (--demo | --evaluate | --sanity)
├── agents/
│   ├── planning_agent.py    # Decomposes questions into sub-queries
│   ├── retrieval_agent.py   # Retrieves docs from ChromaDB
│   └── critique_agent.py    # Evaluates evidence quality
├── core/
│   ├── state.py             # AgentState TypedDict
│   ├── vector_store.py      # ChromaDB + sentence-transformers
│   └── graph.py             # LangGraph pipeline
├── data/
│   └── loader.py            # HotpotQA loader & indexer
├── evaluation/
│   └── metrics.py           # Exact match, F1, dataset eval, baseline
└── demo/
    └── demo.py              # Rich terminal demo
```

---

## Research Context

SHARP-RAG is a foundation for the research paper:

> **SHARP-RAG: Self-correcting Hierarchical Agentic Retrieval-Augmented Generation for Multi-hop Question Answering**

Key research questions:
1. Does a critique loop improve factual accuracy on multi-hop benchmarks (HotpotQA, MuSiQue)?
2. How does the number of retrieval retries trade off with latency and accuracy?
3. Can a lightweight critique prompt reliably distinguish sufficient from insufficient evidence?

---

## License

MIT — built for academic purposes.
