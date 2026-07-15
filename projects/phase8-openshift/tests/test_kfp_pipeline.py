"""Tests for kfp_pipeline module."""

from __future__ import annotations

import tempfile

from kfp_pipeline import (
    _KFP_AVAILABLE,
    fetch_documents,
    generate_embeddings,
    parse_documents,
    rag_ingestion_pipeline,
)


def test_kfp_available_is_bool():
    assert isinstance(_KFP_AVAILABLE, bool)


def test_fetch_documents_is_callable():
    assert callable(fetch_documents)


def test_parse_documents_is_callable():
    assert callable(parse_documents)


def test_generate_embeddings_is_callable():
    assert callable(generate_embeddings)


def test_pipeline_is_callable():
    assert callable(rag_ingestion_pipeline)


def test_fetch_documents_runs_without_kfp():
    if _KFP_AVAILABLE:
        # When kfp is present, fetch_documents is a kfp component; skip runtime call.
        assert callable(fetch_documents)
        return
    with tempfile.TemporaryDirectory() as d:
        fetch_documents("s3://test", d)


def test_pipeline_function_name():
    if not _KFP_AVAILABLE:
        assert rag_ingestion_pipeline.__name__ == "rag_ingestion_pipeline"
    else:
        assert callable(rag_ingestion_pipeline)


def test_kfp_component_decorator_passthrough():
    assert fetch_documents is not None and callable(fetch_documents)
