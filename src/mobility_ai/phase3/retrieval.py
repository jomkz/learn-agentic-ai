from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class VectorStoreConfig(BaseModel):
    store_type: Literal["pgvector", "qdrant", "chroma"]
    connection_string: str | None = None
    collection_name: str = "documents"


class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: dict


def build_chroma_retriever(docs: list[str], embeddings: object) -> object:
    from langchain_chroma import Chroma

    store = Chroma.from_texts(docs, embeddings)
    return store.as_retriever()


def hybrid_rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    scores: dict[str, float] = {}

    for rank, item in enumerate(dense_results):
        key = str(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(sparse_results):
        key = str(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    all_items = {str(item): item for item in dense_results + sparse_results}
    return sorted(all_items.values(), key=lambda item: scores[str(item)], reverse=True)


if __name__ == "__main__":
    dense = ["doc_A", "doc_B", "doc_C", "doc_D"]
    sparse = ["doc_A", "doc_C", "doc_E", "doc_B"]

    fused = hybrid_rrf_fusion(dense, sparse)
    print("RRF fusion result:")
    for rank, doc in enumerate(fused, 1):
        print(f"  {rank}. {doc}")
