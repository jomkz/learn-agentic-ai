"""Reusable RAGAS evaluation harness. Run against any (retriever, dataset) pair."""

from __future__ import annotations

import json

from pydantic import BaseModel


class EvalSample(BaseModel):
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str


class RAGASReport(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    overall: float
    sample_count: int


def _heuristic_report(samples: list[EvalSample]) -> RAGASReport:
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    for s in samples:
        gt_words = set(s.ground_truth.lower().split())
        a_words = set(s.answer.lower().split())
        q_words = set(s.question.lower().split())

        faithfulness_scores.append(1.0 if gt_words & a_words else 0.0)

        if q_words:
            relevancy_scores.append(len(q_words & a_words) / len(q_words))
        else:
            relevancy_scores.append(0.0)

        all_context_words = set(" ".join(s.contexts).lower().split())
        precision_scores.append(1.0 if gt_words & all_context_words else 0.0)

    faith = sum(faithfulness_scores) / len(faithfulness_scores)
    relev = sum(relevancy_scores) / len(relevancy_scores)
    prec = sum(precision_scores) / len(precision_scores)
    overall = (faith + relev + prec) / 3.0

    return RAGASReport(
        faithfulness=faith,
        answer_relevancy=relev,
        context_precision=prec,
        overall=overall,
        sample_count=len(samples),
    )


def compute_report(samples: list[EvalSample]) -> RAGASReport:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        data = [
            {
                "question": s.question,
                "ground_truth": s.ground_truth,
                "contexts": s.contexts,
                "answer": s.answer,
            }
            for s in samples
        ]
        dataset = Dataset.from_list(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        df = result.to_pandas()
        faith = float(df["faithfulness"].mean())
        relev = float(df["answer_relevancy"].mean())
        prec = float(df["context_precision"].mean())
        return RAGASReport(
            faithfulness=faith,
            answer_relevancy=relev,
            context_precision=prec,
            overall=(faith + relev + prec) / 3.0,
            sample_count=len(samples),
        )
    except ImportError:
        return _heuristic_report(samples)


def save_report(report: RAGASReport, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)


if __name__ == "__main__":
    synthetic_samples = [
        EvalSample(
            question="What is RAG?",
            ground_truth="RAG stands for Retrieval-Augmented Generation.",
            contexts=[
                "Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
                "RAG retrieves documents and uses them to generate answers.",
            ],
            answer="RAG stands for Retrieval-Augmented Generation: retrieval + generation.",
        ),
        EvalSample(
            question="What is pgvector?",
            ground_truth="pgvector is a PostgreSQL extension for vector similarity search.",
            contexts=[
                "pgvector adds vector similarity search to PostgreSQL.",
                "It supports cosine and L2 distance metrics.",
            ],
            answer="pgvector is a PostgreSQL extension that enables vector similarity search.",
        ),
        EvalSample(
            question="What is chunking in RAG?",
            ground_truth="Chunking splits documents into smaller pieces for retrieval.",
            contexts=[
                "Documents are split into chunks before being embedded.",
                "Chunk size affects retrieval quality significantly.",
            ],
            answer="Chunking splits documents into smaller pieces before embedding.",
        ),
    ]

    report = compute_report(synthetic_samples)
    print("RAGAS Report:")
    print(f"  Faithfulness:      {report.faithfulness:.3f}")
    print(f"  Answer Relevancy:  {report.answer_relevancy:.3f}")
    print(f"  Context Precision: {report.context_precision:.3f}")
    print(f"  Overall:           {report.overall:.3f}")
    print(f"  Samples:           {report.sample_count}")
