"""RAFT dataset generator. Constructs training examples with oracle + distractor documents."""

from __future__ import annotations

import random

from pydantic import BaseModel


class RAFTExample(BaseModel):
    question: str
    oracle_doc: str
    distractor_docs: list[str]
    chain_of_thought: str
    answer: str


def answer_hint(question: str) -> str:
    return "the answer is contained in the oracle document."


def generate_cot(question: str, oracle_doc: str) -> str:
    return (
        f"Looking at the provided documents, I found relevant information in:"
        f" '{oracle_doc[:100]}...'. Based on this, {answer_hint(question)}"
    )


def build_raft_example(
    question: str,
    oracle_doc: str,
    all_docs: list[str],
    k_distractors: int = 3,
    include_oracle_prob: float = 0.8,
) -> RAFTExample:
    distractors = random.sample(
        [d for d in all_docs if d != oracle_doc],
        min(k_distractors, len(all_docs) - 1),
    )
    if random.random() < include_oracle_prob:
        cot = generate_cot(question, oracle_doc)
    else:
        cot = "The answer is not found in the provided documents."
    return RAFTExample(
        question=question,
        oracle_doc=oracle_doc,
        distractor_docs=distractors,
        chain_of_thought=cot,
        answer=oracle_doc,
    )


def generate_raft_dataset(
    qa_pairs: list[tuple[str, str, str]],
    all_docs: list[str],
    k_distractors: int = 3,
) -> list[RAFTExample]:
    examples: list[RAFTExample] = []
    for question, oracle_doc, answer in qa_pairs:
        ex = build_raft_example(question, oracle_doc, all_docs, k_distractors)
        ex = ex.model_copy(update={"answer": answer})
        examples.append(ex)
    return examples


def to_sft_format(examples: list[RAFTExample]) -> list[dict]:
    result: list[dict] = []
    for ex in examples:
        all_docs = [ex.oracle_doc] + ex.distractor_docs
        combined_docs = "\n\n".join(f"[Doc {i + 1}] {doc}" for i, doc in enumerate(all_docs))
        result.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Documents:\n{combined_docs}\n\nQuestion: {ex.question}",
                    },
                    {
                        "role": "assistant",
                        "content": ex.chain_of_thought,
                    },
                ]
            }
        )
    return result


if __name__ == "__main__":
    import json

    all_docs = [
        "OpenShift AI provides a managed MLOps platform built on Open Data Hub.",
        "vLLM uses PagedAttention for efficient large language model serving.",
        "KubeFlow Pipelines orchestrates ML workflows on Kubernetes.",
        "Ray is a distributed computing framework used for large-scale ML training.",
        "InstructLab enables community-driven fine-tuning of language models.",
    ]
    qa_pairs = [
        ("What is OpenShift AI?", all_docs[0], "A managed MLOps platform."),
        ("What does vLLM use for efficiency?", all_docs[1], "PagedAttention."),
        ("What is KFP used for?", all_docs[2], "Orchestrating ML workflows."),
    ]
    dataset = generate_raft_dataset(qa_pairs, all_docs)
    sft = to_sft_format(dataset)
    print("First example:")
    print(json.dumps(sft[0], indent=2))
