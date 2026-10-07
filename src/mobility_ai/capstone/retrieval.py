"""Exact vector search and bounded context expansion for persisted character chunks."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    source: str
    index: int
    text: str
    vector: list[float]


class RetrievedChunk(BaseModel):
    source: str
    index: int
    text: str
    score: float
    chunk_indices: list[int] = Field(default_factory=list)
    matched_indices: list[int] = Field(default_factory=list)

    @property
    def citation(self) -> str:
        end = self.chunk_indices[-1] if self.chunk_indices else self.index
        fragment = f"chunk-{self.index}" if end == self.index else f"chunks-{self.index}-{end}"
        return f"{self.source}#{fragment}"


def validate_vectors(vectors: list[list[float]]) -> None:
    if not vectors or not vectors[0]:
        raise ValueError("Vectors must be non-empty")
    dimension = len(vectors[0])
    if any(len(v) != dimension or any(not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("Vectors must have consistent dimensions and finite values")


def retrieve(vector: list[float], chunks: list[Chunk], top_k: int) -> list[RetrievedChunk]:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    validate_vectors([vector] + [c.vector for c in chunks])
    norm = math.sqrt(sum(x * x for x in vector))
    ranked = []
    for chunk in chunks:
        other_norm = math.sqrt(sum(x * x for x in chunk.vector))
        score = (
            sum(a * b for a, b in zip(vector, chunk.vector, strict=True)) / (norm * other_norm)
            if norm and other_norm
            else 0.0
        )
        if score > 0:
            ranked.append(
                RetrievedChunk(
                    source=chunk.source,
                    index=chunk.index,
                    text=chunk.text,
                    score=score,
                    chunk_indices=[chunk.index],
                    matched_indices=[chunk.index],
                )
            )
    return sorted(ranked, key=lambda c: (-c.score, c.source, c.index))[:top_k]


def _join_context(
    selected: dict[tuple[str, int], Chunk], hits: list[RetrievedChunk], stride: int
) -> list[RetrievedChunk]:
    """Join contiguous source spans using offsets, never a guessed text overlap."""
    ranks = {(hit.source, hit.index): rank for rank, hit in enumerate(hits)}
    groups: list[list[Chunk]] = []
    for key in sorted(selected):
        chunk = selected[key]
        if (
            groups
            and chunk.source == groups[-1][-1].source
            and chunk.index == groups[-1][-1].index + 1
        ):
            groups[-1].append(chunk)
        else:
            groups.append([chunk])
    ranked = []
    for group in groups:
        first = group[0]
        text = first.text
        for chunk in group[1:]:
            offset = (chunk.index - first.index) * stride
            shared = min(len(chunk.text), len(text) - offset)
            if shared < 0 or text[offset : offset + shared] != chunk.text[:shared]:
                raise ValueError("Adjacent chunks do not match the stored chunking configuration")
            text += chunk.text[shared:]
        matches = [c.index for c in group if (c.source, c.index) in ranks]
        rank = min(ranks[(first.source, index)] for index in matches)
        ranked.append(
            (
                rank,
                RetrievedChunk(
                    source=first.source,
                    index=first.index,
                    text=text,
                    score=hits[rank].score,
                    chunk_indices=[c.index for c in group],
                    matched_indices=matches,
                ),
            )
        )
    return [context for _, context in sorted(ranked, key=lambda item: item[0])]


def build_context(
    hits: list[RetrievedChunk],
    chunks: list[Chunk],
    *,
    chunk_size: int,
    overlap: int,
    adjacent_chunks: int = 0,
    max_context_chars: int = 6000,
) -> list[RetrievedChunk]:
    """Keep every seed hit; add at most one neighbor on either side within the budget.

    Candidates follow seed rank, then next/previous order. Whole neighbors that exceed
    the budget are skipped. Expanded spans inherit their best seed's score and rank.
    """
    if adjacent_chunks not in (0, 1):
        raise ValueError("adjacent_chunks must be 0 or 1")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be positive")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    lookup = {(c.source, c.index): c for c in chunks}
    selected = {(hit.source, hit.index): lookup[(hit.source, hit.index)] for hit in hits}
    stride = chunk_size - overlap
    contexts = _join_context(selected, hits, stride) if adjacent_chunks else hits
    if sum(len(c.text) for c in contexts) > max_context_chars:
        raise ValueError("Seed context exceeds max_context_chars; lower top_k or raise the budget")
    if not adjacent_chunks:
        return contexts
    for hit in hits:
        for index in (hit.index + 1, hit.index - 1):
            key = (hit.source, index)
            if key not in lookup or key in selected:
                continue
            candidate = selected | {key: lookup[key]}
            expanded = _join_context(candidate, hits, stride)
            if sum(len(c.text) for c in expanded) <= max_context_chars:
                selected, contexts = candidate, expanded
    return contexts
