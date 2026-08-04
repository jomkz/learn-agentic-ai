"""Phase 3 capstone: Technical Documentation Search using RAG + FastAPI."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from ingestion import chunk_recursive, enrich_metadata
from retrieval import VectorStoreConfig, hybrid_rrf_fusion  # noqa: F401

SAMPLE_DOCS: list[dict] = [
    {
        "title": "Kubernetes Pods",
        "content": (
            "A Pod is the smallest deployable unit in Kubernetes, representing a single instance of a "
            "running process. Pods encapsulate one or more containers that share network and storage "
            "resources. The Pod lifecycle moves through Pending, Running, Succeeded, Failed, and Unknown "
            "phases. Resource limits on CPU and memory are defined per container to ensure fair scheduling."
        ),
    },
    {
        "title": "Kubernetes Services",
        "content": (
            "A Service provides stable network access to a set of Pods selected by label selectors. "
            "ClusterIP exposes the service on a cluster-internal IP, while NodePort opens a port on every "
            "node and LoadBalancer provisions an external load balancer in supported cloud environments. "
            "Services decouple consumers from Pod IP churn caused by restarts and rescheduling."
        ),
    },
    {
        "title": "Kubernetes Deployments",
        "content": (
            "A Deployment manages a ReplicaSet to maintain a desired number of identical Pod replicas. "
            "Rolling updates replace Pods incrementally so the application stays available throughout the "
            "update process. If an update introduces a problem, kubectl rollout undo restores the previous "
            "ReplicaSet. Replica counts can be scaled manually or via a HorizontalPodAutoscaler."
        ),
    },
    {
        "title": "ConfigMaps and Secrets",
        "content": (
            "ConfigMaps store non-sensitive configuration data as key-value pairs that Pods can consume as "
            "environment variables or mounted files. Secrets hold sensitive data such as passwords and TLS "
            "certificates and are base64-encoded at rest by default. Both resources decouple configuration "
            "from container images, making applications portable across environments. Secrets can be "
            "mounted as volumes or injected as environment variables into containers."
        ),
    },
    {
        "title": "Kubernetes Ingress",
        "content": (
            "An Ingress resource defines HTTP and HTTPS routing rules that direct external traffic to "
            "Services inside the cluster. TLS termination is configured by referencing a Secret containing "
            "the certificate and private key. Annotations customize the behaviour of the underlying Ingress "
            "controller, such as rewrite rules and rate limits. Path-based and host-based routing allow a "
            "single IP to serve multiple applications."
        ),
    },
]


class DocSearchService:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = docs or SAMPLE_DOCS
        self.chunks: list[dict] = []
        self.ready = False

    def setup(self) -> None:
        for doc in self.docs:
            raw_chunks = chunk_recursive(doc["content"], chunk_size=200)
            enriched = enrich_metadata(raw_chunks, source=doc["title"])
            self.chunks.extend(enriched)
        self.ready = True

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.ready or not self.chunks:
            return []
        q_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            text = chunk["text"].lower()
            score = sum(1 for w in q_words if w in text) / max(len(q_words), 1)
            scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [c for c in scored[:k] if c["score"] > 0]

    def answer(self, query: str, llm: object | None = None) -> str:
        results = self.search(query)
        if not results:
            return "No relevant documents found."
        context = "\n\n".join(r["text"] for r in results[:3])
        if llm is None:
            return f"Based on {len(results)} document(s): {context[:200]}..."
        try:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            chain = prompt | llm
            return chain.invoke({"context": context, "query": query}).content
        except Exception:
            return f"Based on {len(results)} document(s): {context[:200]}..."


service = DocSearchService()
service.setup()

app = FastAPI(title="Technical Documentation Search")


@app.get("/search")
def search(q: str) -> dict:
    results = service.search(q)
    return {"query": q, "results": results, "count": len(results)}


@app.get("/ask")
def ask(q: str) -> dict:
    answer = service.answer(q)
    sources = [r["source"] for r in service.search(q)]
    return {"query": q, "answer": answer, "sources": sources}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "docs": len(SAMPLE_DOCS),
        "chunks": len(service.chunks),
        "ready": service.ready,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
