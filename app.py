"""SHARP-RAG · Cinematic Web UI — Live Multi-Agent Pipeline"""

import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

DEMO_QUESTIONS = [
    "What nationality is the director of the film Ran?",
    "The band that performed 'Hotel California' was formed in which city?",
    "What is the birth city of the author who wrote '1984'?",
    "Which country did the creator of Wikipedia grow up in?",
    "What language is spoken in the country where the Eiffel Tower is located?",
    "Who was the US president when the Berlin Wall fell?",
    "What is the capital of the country where the Amazon River originates?",
]

_store = None
_graph = None


def _init():
    global _store, _graph
    if _store is None:
        from core.vector_store import initialize_store
        from core.graph import build_graph
        from data.loader import load_hotpotqa, index_hotpotqa_contexts

        _store = initialize_store()
        if _store.count() == 0:
            examples = load_hotpotqa(max_examples=150)
            index_hotpotqa_contexts(_store, examples=examples)
        _graph = build_graph(_store)


# ── HTML building blocks ────────────────────────────────────────────────────────


def _pipeline_html(active: str = "", completed: set = None, retry: bool = False) -> str:
    completed = completed or set()
    nodes = [
        ("planner",   "🧠", "Planning",   "#fbbf24"),
        ("retriever", "🔍", "Retrieval",  "#60a5fa"),
        ("critic",    "🧐", "Critique",   "#a78bfa"),
        ("generator", "✍️",  "Generator",  "#34d399"),
    ]

    retry_banner = ""
    if retry:
        retry_banner = '<div class="retry-banner">⟳ Insufficient evidence — retrying retrieval with refined queries</div>'

    node_parts = []
    for i, (nid, icon, label, color) in enumerate(nodes):
        if nid in completed:
            cls, status_txt = "node-done", "✓ Done"
        elif nid == active:
            cls, status_txt = "node-active", "● Running…"
        else:
            cls, status_txt = "node-idle", "◌ Waiting"

        node_parts.append(f"""
<div class="pnode {cls}" style="--nc:{color}">
  <div class="pnode-icon">{icon}</div>
  <div class="pnode-label">{label}</div>
  <div class="pnode-status">{status_txt}</div>
</div>""")

        if i < len(nodes) - 1:
            arrow_cls = "arrow-lit" if nid in completed else ""
            node_parts.append(f'<div class="parrow {arrow_cls}">→</div>')

    return f"""
<div class="pipeline-wrap">
  {retry_banner}
  <div class="pipeline-nodes">{''.join(node_parts)}</div>
</div>"""


def _planning_html(sub_queries: list) -> str:
    if not sub_queries:
        return ""
    items = "".join(
        f'<div class="sub-query"><span class="sq-num">{i+1}</span>'
        f'<span class="sq-text">{q}</span></div>'
        for i, q in enumerate(sub_queries)
    )
    return f"""
<div class="result-card" style="--cc:#fbbf24">
  <div class="card-header">
    <span class="card-icon">🧠</span>
    <span class="card-title">Planning Agent — Question Decomposition</span>
  </div>
  <div class="card-content">
    <div class="sq-label">Question broken into sub-queries:</div>
    {items}
  </div>
</div>"""


def _doc_card(doc: dict) -> str:
    score = doc.get("relevance_score", 0.0)
    source = doc.get("source", "Unknown")[:60]
    text = doc.get("text", "")[:300].replace("<", "&lt;").replace(">", "&gt;")
    bar_w = int(min(max(score, 0.0), 1.0) * 100)
    sc = "#34d399" if score > 0.7 else "#fbbf24" if score > 0.5 else "#f87171"
    return f"""
<div class="doc-card">
  <div class="doc-header">
    <span class="doc-source">📄 {source}</span>
    <span class="doc-score" style="color:{sc}">{score:.3f}</span>
  </div>
  <div class="doc-bar-wrap"><div class="doc-bar" style="width:{bar_w}%;background:{sc}"></div></div>
  <div class="doc-text">{text}…</div>
</div>"""


def _retrieval_html(docs: list, retry_count: int = 0) -> str:
    if not docs:
        return ""
    retry_badge = f' <span class="retry-badge">Attempt #{retry_count + 1}</span>' if retry_count > 0 else ""
    cards = "".join(_doc_card(d) for d in docs[:8])
    return f"""
<div class="result-card" style="--cc:#60a5fa">
  <div class="card-header">
    <span class="card-icon">🔍</span>
    <span class="card-title">Retrieval Agent — {len(docs)} Documents Found{retry_badge}</span>
  </div>
  <div class="card-content">{cards}</div>
</div>"""


def _critique_html(verdict: str, feedback: str, retries: int) -> str:
    icons  = {"sufficient": "✅", "insufficient": "❌", "contradictory": "⚠️"}
    colors = {"sufficient": "#34d399", "insufficient": "#f87171", "contradictory": "#fbbf24"}
    icon  = icons.get(verdict, "❓")
    color = colors.get(verdict, "#94a3b8")
    safe_fb = feedback.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div class="result-card" style="--cc:#a78bfa">
  <div class="card-header">
    <span class="card-icon">🧐</span>
    <span class="card-title">Critique Agent — Evidence Evaluation</span>
  </div>
  <div class="card-content">
    <div class="verdict" style="color:{color}">{icon} {verdict.capitalize()}</div>
    <div class="feedback">{safe_fb}</div>
    <div class="retries">Total retrieval attempts: {retries}</div>
  </div>
</div>"""


def _answer_html(answer: str) -> str:
    safe = answer.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div class="answer-box">
  <div class="answer-header">
    <span>✍️ Final Answer</span>
    <span class="answer-badge">SHARP-RAG</span>
  </div>
  <div class="answer-text">{safe}</div>
</div>"""


def _trace_html(trace: list) -> str:
    if not trace:
        return ""
    styles = {
        "[PlanningAgent]":  ("🧠", "#fbbf24"),
        "[RetrievalAgent]": ("🔍", "#60a5fa"),
        "[CritiqueAgent]":  ("🧐", "#a78bfa"),
        "[Generator]":      ("✍️",  "#34d399"),
    }
    steps = []
    for step in trace:
        icon, color = "📝", "#64748b"
        text = step
        for prefix, (ico, col) in styles.items():
            if step.startswith(prefix):
                icon, color = ico, col
                text = step[len(prefix):].strip()
                break
        safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
        steps.append(
            f'<div class="trace-step">'
            f'<span style="color:{color};flex-shrink:0;font-size:14px">{icon}</span>'
            f'<span class="trace-text">{safe_text}</span></div>'
        )
    return (
        '<div class="trace-wrap">'
        '<div class="trace-title">🗂 Reasoning Trace</div>'
        + "".join(steps)
        + "</div>"
    )


# ── Main streaming function ─────────────────────────────────────────────────────


def run_query(question: str):
    """Generator: yields (pipeline, planning, retrieval, critique, answer, trace) tuples."""
    if not question or not question.strip():
        yield _pipeline_html(), "", "", "", "", ""
        return

    if not os.environ.get("GROQ_API_KEY"):
        err = '<div class="error">⚠️ GROQ_API_KEY not set — add it to your .env file.</div>'
        yield _pipeline_html(), "", "", "", err, ""
        return

    completed: set = set()
    planning_h = retrieval_h = critique_h = answer_h = trace_h = ""
    all_trace: list = []
    in_retry = False

    # Yield immediately so the UI unfreezes — _init() may take time on first run
    loading_trace = (
        '<div class="trace-wrap">'
        '<div class="trace-title">🗂 Reasoning Trace</div>'
        '<div class="trace-step">'
        '<span style="color:#60a5fa;flex-shrink:0;font-size:14px">🔧</span>'
        '<span class="trace-text">Loading knowledge base… '
        '(first run indexes documents — may take ~30s)</span>'
        '</div></div>'
    )
    yield _pipeline_html("planner"), "", "", "", "", loading_trace

    _init()

    initial_state = {
        "question":                      question.strip(),
        "sub_queries":                   [],
        "retrieved_docs":                [],
        "critique_result":               "",
        "critique_feedback":             "",
        "critique_suggested_queries":    [],
        "critique_missing_information":  [],
        "seen_doc_ids":                  [],
        "retry_count":                   0,
        "final_answer":                  "",
        "reasoning_trace":               [],
    }

    try:
        for event in _graph.stream(initial_state):
            for node_name, output in event.items():

                # Accumulate trace (each agent returns the full list so far)
                if "reasoning_trace" in output:
                    all_trace = output["reasoning_trace"]
                trace_h = _trace_html(all_trace)

                if node_name == "planner":
                    completed.add("planner")
                    sub_queries = output.get("sub_queries", [])
                    planning_h = _planning_html(sub_queries)
                    yield (
                        _pipeline_html("retriever", completed),
                        planning_h, retrieval_h, critique_h, answer_h, trace_h,
                    )

                elif node_name == "retriever":
                    completed.add("retriever")
                    docs = output.get("retrieved_docs", [])
                    retry_count = output.get("retry_count", 0)
                    retrieval_h = _retrieval_html(docs, retry_count)
                    yield (
                        _pipeline_html("critic", completed, retry=in_retry),
                        planning_h, retrieval_h, critique_h, answer_h, trace_h,
                    )
                    in_retry = False

                elif node_name == "critic":
                    completed.add("critic")
                    verdict  = output.get("critique_result", "")
                    feedback = output.get("critique_feedback", "")
                    retries  = output.get("retry_count", 0)
                    critique_h = _critique_html(verdict, feedback, retries)

                    if verdict == "insufficient" and retries < 2:
                        # Loop back to retriever
                        in_retry = True
                        completed.discard("retriever")
                        completed.discard("critic")
                        yield (
                            _pipeline_html("retriever", completed, retry=True),
                            planning_h, retrieval_h, critique_h, answer_h, trace_h,
                        )
                    else:
                        yield (
                            _pipeline_html("generator", completed),
                            planning_h, retrieval_h, critique_h, answer_h, trace_h,
                        )

                elif node_name == "generator":
                    completed.add("generator")
                    answer  = output.get("final_answer", "No answer produced.")
                    answer_h = _answer_html(answer)
                    yield (
                        _pipeline_html("", completed),
                        planning_h, retrieval_h, critique_h, answer_h, trace_h,
                    )

    except Exception as exc:
        err = f'<div class="error">⚠️ Pipeline error: {exc}</div>'
        yield _pipeline_html(), planning_h, retrieval_h, critique_h, err, trace_h


def fill_question(q: str):
    return q


# ── CSS ─────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ──────────────────────────────────────────────────────────────── */
.gradio-container {
    background: #070b14 !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

.main { background: #070b14 !important; }

/* ── Sharp header ─────────────────────────────────────────────────────────── */
.sharp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 8px 12px;
    border-bottom: 1px solid #1e2d47;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 12px;
}
.header-logo {
    display: flex;
    align-items: center;
    gap: 16px;
}
.logo-icon {
    font-size: 36px;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.header-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4ff 30%, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-sub {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 2px;
}
.header-meta { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.meta-badge {
    background: #0d1424;
    border: 1px solid #1e2d47;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.04em;
}

/* ── Input area ───────────────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: #0d1424 !important;
    border: 1px solid #1e2d47 !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}
label { color: #64748b !important; font-size: 13px !important; }

button.primary {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

button.secondary {
    background: #0d1424 !important;
    border: 1px solid #1e2d47 !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
    background: rgba(0,212,255,0.05) !important;
}

/* ── Pipeline visualization ───────────────────────────────────────────────── */
.pipeline-wrap {
    background: #0d1424;
    border: 1px solid #1e2d47;
    border-radius: 16px;
    padding: 24px 20px 20px;
    margin: 4px 0 8px;
}

.retry-banner {
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    color: #f87171;
    letter-spacing: 0.05em;
    padding: 6px 12px;
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: 6px;
    margin-bottom: 16px;
    animation: slideIn 0.3s ease-out;
}

.pipeline-nodes {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    flex-wrap: wrap;
}

.pnode {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 18px 26px;
    border-radius: 14px;
    border: 2px solid #1e2d47;
    background: #080e1c;
    min-width: 120px;
    transition: all 0.45s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.pnode-icon { font-size: 28px; transition: transform 0.3s; }
.pnode-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #334155;
    transition: color 0.3s;
}
.pnode-status {
    font-size: 10px;
    color: #1e293b;
    transition: color 0.3s;
}

/* Idle */
.node-idle { opacity: 0.32; }

/* Active */
.node-active {
    border-color: var(--nc) !important;
    background: rgba(255,255,255,0.02) !important;
    box-shadow:
        0 0 0 1px var(--nc),
        0 0 18px rgba(255,255,255,0.05),
        0 0 30px rgba(0,0,0,0.3);
    animation: nodeGlow 2s ease-in-out infinite;
}
@keyframes nodeGlow {
    0%, 100% { box-shadow: 0 0 0 1px var(--nc), 0 0 20px rgba(255,255,255,0.04), 0 0 35px rgba(0,0,0,0.2); }
    50%       { box-shadow: 0 0 0 1px var(--nc), 0 0 28px rgba(255,255,255,0.07), 0 0 50px rgba(0,0,0,0.3); }
}

.node-active .pnode-icon { animation: iconFloat 1.8s ease-in-out infinite; }
@keyframes iconFloat {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-5px); }
}
.node-active .pnode-label { color: var(--nc) !important; }
.node-active .pnode-status { color: var(--nc); animation: blink 1.1s ease-in-out infinite; }
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

/* Done */
.node-done {
    border-color: #22c55e !important;
    opacity: 0.88;
}
.node-done .pnode-label { color: #22c55e !important; }
.node-done .pnode-status { color: #22c55e; }

/* Arrows */
.parrow {
    font-size: 22px;
    color: #1e2d47;
    padding: 0 2px;
    flex-shrink: 0;
    transition: color 0.5s, text-shadow 0.5s;
    margin-top: -2px;
}
.arrow-lit {
    color: #22c55e;
    text-shadow: 0 0 10px rgba(34,197,94,0.5);
}

/* ── Result cards ─────────────────────────────────────────────────────────── */
.result-card {
    background: #0d1424;
    border: 1px solid #1e2d47;
    border-left: 3px solid var(--cc, #475569);
    border-radius: 14px;
    padding: 20px 22px;
    margin: 10px 0;
    animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #0f1a2e;
}
.card-icon { font-size: 18px; }
.card-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cc, #94a3b8);
}

/* ── Planning ─────────────────────────────────────────────────────────────── */
.sq-label {
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 10px;
}
.sub-query {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 11px 14px;
    background: #080e1c;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    margin: 6px 0;
    animation: slideIn 0.35s ease-out;
}
.sq-num {
    background: #fbbf24;
    color: #000;
    font-size: 10px;
    font-weight: 900;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.sq-text { font-size: 13px; color: #cbd5e1; line-height: 1.5; }

/* ── Retrieval docs ───────────────────────────────────────────────────────── */
.doc-card {
    background: #080e1c;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 13px 15px;
    margin: 8px 0;
    animation: slideIn 0.3s ease-out;
}
.doc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
}
.doc-source {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 260px;
}
.doc-score { font-size: 12px; font-weight: 700; font-family: monospace; flex-shrink: 0; }
.doc-bar-wrap {
    background: #1e2d47;
    border-radius: 3px;
    height: 4px;
    margin-bottom: 9px;
    overflow: hidden;
}
.doc-bar { height: 100%; border-radius: 3px; transition: width 0.8s ease; }
.doc-text { font-size: 12px; color: #475569; line-height: 1.55; }

.retry-badge {
    background: rgba(248,113,113,0.15);
    color: #f87171;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid rgba(248,113,113,0.25);
    margin-left: 6px;
    vertical-align: middle;
}

/* ── Critique ─────────────────────────────────────────────────────────────── */
.verdict { font-size: 17px; font-weight: 800; margin-bottom: 10px; }
.feedback { font-size: 13px; color: #94a3b8; line-height: 1.65; margin-bottom: 8px; }
.retries { font-size: 11px; color: #334155; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }

/* ── Answer ───────────────────────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, #081a0e 0%, #070b14 60%);
    border: 1px solid #22c55e;
    border-radius: 16px;
    padding: 28px 30px;
    margin: 10px 0;
    box-shadow: 0 0 0 1px rgba(34,197,94,0.1), 0 0 40px rgba(34,197,94,0.06);
    animation: answerReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes answerReveal {
    from { opacity: 0; transform: scale(0.97); }
    to   { opacity: 1; transform: scale(1); }
}
.answer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #22c55e;
}
.answer-badge {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 10px;
    color: #22c55e;
}
.answer-text { font-size: 16px; color: #e2e8f0; line-height: 1.75; font-weight: 400; }

/* ── Reasoning trace ──────────────────────────────────────────────────────── */
.trace-wrap {
    background: #0a0f1e;
    border: 1px solid #1e2d47;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 10px 0;
    animation: slideIn 0.4s ease-out;
}
.trace-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #0f172a;
}
.trace-step {
    display: flex;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #0a0f1e;
    font-size: 12px;
    color: #475569;
    line-height: 1.55;
    align-items: flex-start;
}
.trace-text { flex: 1; }

/* ── Error ────────────────────────────────────────────────────────────────── */
.error {
    color: #f87171;
    padding: 18px 22px;
    background: #150a0a;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
    font-size: 14px;
    animation: slideIn 0.3s ease-out;
}

/* ── Gradio container cleanup ─────────────────────────────────────────────── */
.block.svelte-90oupt { background: transparent !important; border: none !important; }
.block { background: transparent !important; }
footer { display: none !important; }
"""

# ── Gradio theme ────────────────────────────────────────────────────────────────

dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#070b14",
    body_background_fill_dark="#070b14",
    block_background_fill="#0d1424",
    block_background_fill_dark="#0d1424",
    block_border_color="#1e2d47",
    block_border_color_dark="#1e2d47",
    block_title_text_color="#64748b",
    block_title_text_color_dark="#64748b",
    block_label_text_color="#64748b",
    block_label_text_color_dark="#64748b",
    button_primary_background_fill="linear-gradient(135deg, #0ea5e9, #7c3aed)",
    button_primary_background_fill_dark="linear-gradient(135deg, #0ea5e9, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #38bdf8, #9f5de5)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #38bdf8, #9f5de5)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="#0d1424",
    button_secondary_background_fill_dark="#0d1424",
    button_secondary_border_color="#1e2d47",
    button_secondary_border_color_dark="#1e2d47",
    button_secondary_text_color="#94a3b8",
    button_secondary_text_color_dark="#94a3b8",
    input_background_fill="#0d1424",
    input_background_fill_dark="#0d1424",
    input_border_color="#1e2d47",
    input_border_color_dark="#1e2d47",
    input_placeholder_color="#334155",
    input_placeholder_color_dark="#334155",
    checkbox_background_color="#0d1424",
    checkbox_background_color_dark="#0d1424",
)

HEADER_HTML = """
<div class="sharp-header">
  <div class="header-logo">
    <div class="logo-icon">◈</div>
    <div>
      <div class="header-title">SHARP-RAG</div>
      <div class="header-sub">Self-correcting Hierarchical Agentic Retrieval-Augmented Generation</div>
    </div>
  </div>
  <div class="header-meta">
    <span class="meta-badge">Independent Research</span>
    <span class="meta-badge">Elia Ghazal · 2026</span>
    <span class="meta-badge">HotpotQA · LangGraph · Groq</span>
  </div>
</div>
"""

# ── Layout ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SHARP-RAG") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        question_box = gr.Textbox(
            label="",
            placeholder=(
                "Ask a multi-hop question…  "
                "e.g. 'What city was the founder of Apple born in?'"
            ),
            lines=2,
            scale=5,
        )
        submit_btn = gr.Button("⚡ Ask SHARP-RAG", variant="primary", scale=1, min_width=180)

    with gr.Row():
        gr.HTML("<div style='font-size:11px;color:#475569;font-weight:700;"
                "letter-spacing:0.06em;text-transform:uppercase;padding:4px 0;"
                "'>Example questions (click to load):</div>")

    with gr.Row():
        for q in DEMO_QUESTIONS:
            btn = gr.Button(q, size="sm", variant="secondary")
            btn.click(fn=lambda x=q: x, outputs=question_box)

    # ── Pipeline visualization
    pipeline_comp = gr.HTML(value=_pipeline_html())

    # ── Results
    with gr.Row():
        with gr.Column(scale=3, min_width=400):
            planning_comp  = gr.HTML()
            critique_comp  = gr.HTML()
            answer_comp    = gr.HTML()

        with gr.Column(scale=2, min_width=320):
            retrieval_comp = gr.HTML()
            trace_comp     = gr.HTML()

    # Wire up
    outputs = [pipeline_comp, planning_comp, retrieval_comp, critique_comp, answer_comp, trace_comp]
    submit_btn.click(fn=run_query, inputs=question_box, outputs=outputs)
    question_box.submit(fn=run_query, inputs=question_box, outputs=outputs)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. Set it in .env before querying.")
    demo.launch(inbrowser=True, share=False, theme=dark_theme, css=CSS)
