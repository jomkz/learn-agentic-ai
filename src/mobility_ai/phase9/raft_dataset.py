"""Seeded RAFT examples with explicit oracle inclusion and complete answer targets."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field, model_validator


class RAFTExample(BaseModel):
    question: str = Field(min_length=1)
    oracle_doc: str = Field(min_length=1)
    context_docs: list[str]
    oracle_included: bool
    answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_oracle(self) -> RAFTExample:
        if (self.oracle_doc in self.context_docs) != self.oracle_included:
            raise ValueError("oracle_included must match the documents actually supplied")
        if len(self.context_docs) != len(set(self.context_docs)):
            raise ValueError("Context documents must be unique")
        return self

    @property
    def distractor_docs(self) -> list[str]:
        return [d for d in self.context_docs if d != self.oracle_doc]


def build_raft_example(
    question: str,
    oracle_doc: str,
    all_docs: list[str],
    k_distractors: int = 3,
    include_oracle_prob: float = 0.8,
    *,
    answer: str,
    rng: random.Random | None = None,
) -> RAFTExample:
    if k_distractors < 0 or not 0 <= include_oracle_prob <= 1:
        raise ValueError("Require k_distractors >= 0 and 0 <= include_oracle_prob <= 1")
    randomizer = rng if rng is not None else random.Random(0)
    candidates = list(dict.fromkeys(d for d in all_docs if d != oracle_doc))
    docs = randomizer.sample(candidates, min(k_distractors, len(candidates)))
    include_oracle = randomizer.random() < include_oracle_prob
    if include_oracle:
        docs.append(oracle_doc)
    randomizer.shuffle(docs)
    return RAFTExample(
        question=question,
        oracle_doc=oracle_doc,
        context_docs=docs,
        oracle_included=include_oracle,
        answer=answer,
    )


def generate_raft_dataset(
    qa_pairs: list[tuple[str, str, str]],
    all_docs: list[str],
    k_distractors: int = 3,
    *,
    include_oracle_prob: float = 0.8,
    seed: int = 0,
) -> list[RAFTExample]:
    rng = random.Random(seed)
    return [
        build_raft_example(
            question, oracle, all_docs, k_distractors, include_oracle_prob, answer=answer, rng=rng
        )
        for question, oracle, answer in qa_pairs
    ]


def to_sft_format(examples: list[RAFTExample]) -> list[dict]:
    rows = []
    for example in examples:
        documents = "\n\n".join(
            f"[Doc {index}] {doc}" for index, doc in enumerate(example.context_docs, 1)
        )
        target = example.answer
        if example.oracle_included:
            index = example.context_docs.index(example.oracle_doc) + 1
            target = f"{target} [Doc {index}]"
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Documents:\n{documents}\n\nQuestion: {example.question}",
                    },
                    {"role": "assistant", "content": target},
                ]
            }
        )
    return rows


if __name__ == "__main__":
    import json

    docs = ["vLLM uses PagedAttention.", "KFP orchestrates ML workflows.", "Ray distributes work."]
    examples = generate_raft_dataset([("What does vLLM use?", docs[0], "PagedAttention.")], docs)
    print(json.dumps(to_sft_format(examples), indent=2))
