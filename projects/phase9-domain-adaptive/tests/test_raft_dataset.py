from __future__ import annotations

from monitoring import compute_text_drift
from raft_dataset import (
    RAFTExample,
    build_raft_example,
    generate_raft_dataset,
    to_sft_format,
)

ALL_DOCS = [
    "OpenShift AI provides a managed MLOps platform built on Open Data Hub.",
    "vLLM uses PagedAttention for efficient large language model serving.",
    "KubeFlow Pipelines orchestrates ML workflows on Kubernetes.",
    "Ray is a distributed computing framework used for large-scale ML training.",
    "InstructLab enables community-driven fine-tuning of language models.",
]


def test_build_raft_example_has_distractors() -> None:
    ex = build_raft_example("What is OpenShift AI?", ALL_DOCS[0], ALL_DOCS)
    assert isinstance(ex, RAFTExample)
    assert len(ex.distractor_docs) > 0


def test_raft_example_fields_populated() -> None:
    ex = build_raft_example("What is vLLM?", ALL_DOCS[1], ALL_DOCS)
    assert ex.chain_of_thought
    assert ex.question
    assert ex.oracle_doc


def test_generate_dataset_count() -> None:
    qa_pairs = [
        ("What is OpenShift AI?", ALL_DOCS[0], "A managed MLOps platform."),
        ("What does vLLM use?", ALL_DOCS[1], "PagedAttention."),
        ("What is KFP for?", ALL_DOCS[2], "Orchestrating ML workflows."),
    ]
    dataset = generate_raft_dataset(qa_pairs, ALL_DOCS)
    assert len(dataset) == 3


def test_sft_format_has_messages() -> None:
    qa_pairs = [
        ("What is Ray?", ALL_DOCS[3], "A distributed computing framework."),
    ]
    dataset = generate_raft_dataset(qa_pairs, ALL_DOCS)
    sft = to_sft_format(dataset)
    assert "messages" in sft[0]
    assert isinstance(sft[0]["messages"], list)


def test_drift_result_no_drift() -> None:
    texts = ["The model predicts customer churn accurately."] * 10
    result = compute_text_drift(texts, texts)
    assert result.has_drift is False


def test_drift_result_detects_drift() -> None:
    reference = ["short text"] * 10
    production = ["very very very long detailed document with many many words and concepts"] * 10
    result = compute_text_drift(reference, production)
    assert result.has_drift is True
