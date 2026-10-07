import json
from pathlib import Path

import pytest

from mobility_ai.capstone.app import VectorStore, ask, ingest
from mobility_ai.capstone.retrieval import Chunk, RetrievedChunk, build_context, retrieve


def chunks_for(text, *, source="facts.txt", size=8, overlap=2):
    return [
        Chunk(source=source, index=i, text=text[start : start + size], vector=[1.0])
        for i, start in enumerate(range(0, len(text), size - overlap))
    ]


def hits_for(chunks, *indices):
    return [
        RetrievedChunk(**chunks[index].model_dump(exclude={"vector"}), score=1 - rank / 10)
        for rank, index in enumerate(indices)
    ]


def expand(chunks, hits, **kwargs):
    return build_context(hits, chunks, chunk_size=8, overlap=2, adjacent_chunks=1, **kwargs)


@pytest.mark.parametrize("overlap", [0, 2, 7])
@pytest.mark.parametrize("text", ["ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "é猫🙂ab" * 8])
def test_overlapping_windows_reconstruct_exact_source_and_preserve_seed_rank(overlap, text):
    chunks = chunks_for(text, overlap=overlap)
    hits = hits_for(chunks, 3, 1)
    contexts = build_context(hits, chunks, chunk_size=8, overlap=overlap, adjacent_chunks=1)
    assert len(contexts) == 1
    span = contexts[0]
    assert span.text == text[: 4 * (8 - overlap) + 8]
    assert span.chunk_indices == [0, 1, 2, 3, 4]
    assert span.matched_indices == [1, 3]
    assert span.score == hits[0].score
    assert span.citation == "facts.txt#chunks-0-4"


def test_neighbors_are_bounded_and_never_cross_sources_or_reorder_best_matches():
    a = chunks_for("A" * 80, source="a.txt")
    b = chunks_for("B" * 30, source="b.txt")
    hits = hits_for(b, 0) + hits_for(a, 5)
    contexts = expand(a + b, hits)
    assert [c.source for c in contexts] == ["b.txt", "a.txt"]
    assert contexts[0].chunk_indices == [0, 1]
    assert contexts[0].text == "B" * 14
    assert contexts[1].chunk_indices == [4, 5, 6]
    assert contexts[1].text == "A" * 20


def test_gaps_in_chunk_indices_are_not_bridged():
    chunks = chunks_for("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    del chunks[1]
    hits = hits_for(chunks, 0, 1)
    contexts = expand(chunks, hits)
    assert [c.chunk_indices for c in contexts] == [[0], [2, 3]]
    assert contexts[0].citation == "facts.txt#chunk-0"


def test_budget_skips_whole_neighbors_without_dropping_seeds():
    chunks = chunks_for("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    hits = hits_for(chunks, 1)
    contexts = expand(chunks, hits, max_context_chars=14)
    assert contexts[0].text == "GHIJKLMNOPQRST"
    assert contexts[0].chunk_indices == [1, 2]
    assert contexts[0].matched_indices == [1]
    assert sum(len(c.text) for c in contexts) == 14
    assert expand(chunks, hits, max_context_chars=8)[0].text == chunks[1].text
    with pytest.raises(ValueError, match="Seed context exceeds"):
        expand(chunks, hits, max_context_chars=7)


def test_budget_reserves_lower_ranked_seeds_before_expanding():
    a = chunks_for("A" * 30, source="a.txt")
    b = chunks_for("B" * 30, source="b.txt")
    hits = hits_for(a, 0) + hits_for(b, 0)
    contexts = expand(a + b, hits, max_context_chars=22)
    assert [(c.source, c.chunk_indices) for c in contexts] == [("a.txt", [0, 1]), ("b.txt", [0])]


def test_disabled_expansion_keeps_original_passages_and_overlap():
    chunks = chunks_for("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    hits = hits_for(chunks, 1, 0)
    assert build_context(hits, chunks, chunk_size=8, overlap=2) == hits
    assert hits[0].citation == "facts.txt#chunk-1"


@pytest.mark.parametrize("text", ["WRONG!!!", "X"])
def test_inconsistent_overlap_or_missing_text_fails_explicitly(text):
    chunks = chunks_for("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunks[0].text = text
    with pytest.raises(ValueError, match="chunking configuration"):
        expand(chunks, hits_for(chunks, 0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adjacent_chunks": 2},
        {"adjacent_chunks": -1},
        {"max_context_chars": 0},
        {"chunk_size": 0},
        {"overlap": 8},
    ],
)
def test_invalid_context_settings_fail_even_without_hits(kwargs):
    with pytest.raises(ValueError):
        build_context([], [], **({"chunk_size": 8, "overlap": 2} | kwargs))


def test_empty_retrieval_does_not_add_unrelated_neighbors():
    chunks = chunks_for("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert expand(chunks, []) == []
    assert retrieve([0.0], chunks, 1) == []


def test_frozen_cache_namespace_failure_recovers_complete_evidence(tmp_path):
    suite = Path("evals/benchmarks/project-docs-v1")
    rows = [
        json.loads(line)
        for line in (suite / "results/2026-09-23/top3.jsonl").read_text().splitlines()
    ]
    case = next(row for row in rows if row["id"] == "q18")
    db = tmp_path / "frozen.db"
    ingest(suite / "corpus", db, provider="lexical")
    config, chunks = VectorStore(db).load()
    hits = [RetrievedChunk.model_validate(hit) for hit in case["result"]["retrieved"]]
    evidence = case["evidence"][0]["text"]
    assert not any(evidence in hit.text for hit in hits)
    contexts = build_context(
        hits, chunks, chunk_size=config.chunk_size, overlap=config.overlap, adjacent_chunks=1
    )
    recovered = next(c for c in contexts if evidence in c.text)
    assert recovered.source == "overview.md"
    assert recovered.chunk_indices == [9, 10, 11]
    assert recovered.matched_indices == [10]
    assert recovered.citation == "overview.md#chunks-9-11"
    source = (suite / "corpus/overview.md").read_text().strip()
    assert recovered.text == source[9 * 448 : 11 * 448 + 512]


def test_expansion_reopens_existing_database_and_exports_source_span(tmp_path):
    doc = tmp_path / "facts.txt"
    doc.write_text("HEADER namespace identifies model, prompt, corpus, tenant. TAIL")
    db = tmp_path / "db"
    ingest(doc, db, provider="lexical", chunk_size=32, overlap=8)
    before = db.read_bytes()
    baseline = ask(db, "namespace", top_k=1)
    expanded = ask(db, "namespace", top_k=1, adjacent_chunks=1)
    assert "corpus" not in baseline.answer
    assert "corpus" in expanded.answer
    assert expanded.citations == {"1": "facts.txt#chunks-0-1"}
    assert expanded.retrieved[0].chunk_indices == [0, 1]
    assert expanded.adjacent_chunks == 1
    assert expanded.top_k == 1
    assert expanded.max_context_chars == 6000
    assert db.read_bytes() == before
