"""HotpotQA data loader and ChromaDB indexer."""

from datasets import load_dataset

from core.vector_store import VectorStore

HOTPOTQA_SPLIT = "validation"
MAX_EXAMPLES = 100


def load_hotpotqa(max_examples: int = MAX_EXAMPLES) -> list[dict]:
    """Load the first *max_examples* from the HotpotQA validation split.

    Returns a list of dicts with keys:
        question, answer, supporting_facts, context
    """
    print(f"[Loader] Downloading HotpotQA ({HOTPOTQA_SPLIT}, first {max_examples})…")
    dataset = load_dataset("hotpot_qa", "fullwiki", split=HOTPOTQA_SPLIT)

    examples: list[dict] = []
    for row in dataset.select(range(min(max_examples, len(dataset)))):
        examples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type", "unknown"),
                "supporting_facts": row["supporting_facts"],
                "context": row["context"],
            }
        )

    print(f"[Loader] Loaded {len(examples)} HotpotQA examples.")
    return examples


def index_hotpotqa_contexts(
    vector_store: VectorStore,
    examples: list[dict] | None = None,
    max_examples: int = MAX_EXAMPLES,
) -> int:
    """Index all context paragraphs from HotpotQA into *vector_store*.

    Each paragraph becomes a separate document with ``source`` metadata set
    to the Wikipedia title it came from.

    Returns the total number of documents indexed.
    """
    if examples is None:
        examples = load_hotpotqa(max_examples)

    docs: list[dict] = []
    for ex in examples:
        # context is a dict with keys "title" (list) and "sentences" (list of lists)
        context = ex["context"]
        titles: list[str] = context["title"]
        sentences_per_title: list[list[str]] = context["sentences"]

        for title, sentences in zip(titles, sentences_per_title):
            paragraph_text = " ".join(sentences).strip()
            if paragraph_text:
                docs.append(
                    {
                        "text": paragraph_text,
                        "source": title,
                    }
                )

    if not docs:
        print("[Loader] No context paragraphs found — nothing indexed.")
        return 0

    print(f"[Loader] Indexing {len(docs)} paragraphs into ChromaDB…")
    vector_store.add_documents(docs)
    print(f"[Loader] Indexing complete. Total docs in store: {vector_store.count()}")
    return len(docs)
