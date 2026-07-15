from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_by_fixed_size(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_recursive(text: str, chunk_size: int = 512) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return splitter.split_text(text)


def chunk_by_sentence(text: str) -> list[str]:
    chunks = []
    for para in text.split("\n\n"):
        for sentence in para.split(". "):
            stripped = sentence.strip()
            if stripped:
                chunks.append(stripped)
    return chunks


def enrich_metadata(chunks: list[str], source: str, section: str = "") -> list[dict]:
    return [
        {"text": chunk, "source": source, "section": section, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]


CHUNKING_STRATEGIES: dict[str, object] = {
    "fixed_size": chunk_by_fixed_size,
    "recursive": chunk_recursive,
    "sentence": chunk_by_sentence,
}


if __name__ == "__main__":
    sample = (
        "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. "
        "The retriever fetches relevant documents from a vector store. "
        "The generator then conditions on those documents to produce an answer. "
        "This approach grounds the model in external knowledge. "
        "It reduces hallucination and keeps responses up to date.\n\n"
        "Chunking is a critical preprocessing step. "
        "Poor chunking leads to lost context or noisy retrieval. "
        "Fixed-size chunking is simple but ignores semantic boundaries. "
        "Recursive chunking respects natural text structure. "
        "Sentence-level chunking preserves complete thoughts."
    )

    for name, fn in CHUNKING_STRATEGIES.items():
        chunks = fn(sample)  # type: ignore[operator]
        print(f"\n--- {name} ({len(chunks)} chunks) ---")
        for c in chunks:
            print(f"  [{len(c):3d} chars] {c[:60]!r}")
