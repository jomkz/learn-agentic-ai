"""Explicit lexical diagnostics or RAGAS judging of recorded RAG outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Backend = Literal["lexical", "ragas"]


class EvalSample(BaseModel):
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    contexts: list[str]
    answer: str
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ScoredSample(BaseModel):
    sample: EvalSample
    metrics: dict[str, float]


class EvaluationReport(BaseModel):
    backend: Backend
    metrics: dict[str, float]
    sample_count: int
    dataset_sha256: str
    generated_at: str
    configuration: dict[str, str]
    per_sample: list[ScoredSample]
    limitations: str


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _lexical_scores(sample: EvalSample) -> dict[str, float]:
    reference = Counter(_tokens(sample.ground_truth))
    answer = Counter(_tokens(sample.answer))
    overlap = sum((reference & answer).values())
    total = sum(reference.values()) + sum(answer.values())
    reference_words = set(reference)
    context_words = set(_tokens(" ".join(sample.contexts)))
    return {
        "answer_exact_match": float(_tokens(sample.answer) == _tokens(sample.ground_truth)),
        "answer_token_f1": 2 * overlap / total if total else 0.0,
        "reference_context_token_recall": (
            len(reference_words & context_words) / len(reference_words) if reference_words else 0.0
        ),
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def compute_report(
    samples: list[EvalSample],
    *,
    backend: Backend = "ragas",
    llm: Any = None,
    embeddings: Any = None,
    judge_id: str | None = None,
    embedding_id: str | None = None,
    configuration: dict[str, str] | None = None,
    run_config: Any = None,
) -> EvaluationReport:
    """Score supplied outputs. Missing dependencies never change the scoring method."""
    if not samples:
        raise ValueError("An evaluation requires at least one sample")
    config = dict(configuration or {})
    config.update(python=platform.python_version(), evaluator_version="2")
    if backend == "lexical":
        rows = [_lexical_scores(s) for s in samples]
        limitations = "Lexical diagnostics only; these do not measure factual faithfulness."
    elif backend == "ragas":
        if llm is None or embeddings is None or not judge_id or not embedding_id:
            raise ValueError("RAGAS requires explicit llm, embeddings, judge_id and embedding_id")
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
        except ImportError as exc:
            raise RuntimeError("Install the evaluation extra to use RAGAS") from exc
        dataset = Dataset.from_list(
            [s.model_dump(exclude={"latency_ms", "cost_usd"}) for s in samples]
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=True,
            **({"run_config": run_config} if run_config is not None else {}),
        )
        names = ("faithfulness", "answer_relevancy", "context_precision")
        rows = [{name: float(row[name]) for name in names} for row in result.scores]
        config.update(judge=judge_id, embeddings=embedding_id, ragas=_package_version("ragas"))
        limitations = "LLM judge scores depend on the recorded model and evaluation configuration."
    else:
        raise ValueError(f"Unknown evaluation backend: {backend}")
    if len(rows) != len(samples):
        raise ValueError("Evaluation returned missing or invalid scores")
    # Answer relevancy is a cosine mean, so its mathematical range includes negatives.
    # Dot/norm roundoff can produce e.g. 1.0000000000000002 for identical vectors.
    tolerance = 1e-12
    for row in rows:
        for name, value in row.items():
            lower = -1.0 if name == "answer_relevancy" else 0.0
            if not math.isfinite(value) or not lower - tolerance <= value <= 1 + tolerance:
                raise ValueError(f"Evaluation returned missing or invalid scores: {name}={value!r}")
            row[name] = min(1.0, max(lower, value))
    config["score_validation"] = (
        "finite; answer_relevancy [-1,1], other metrics [0,1]; boundary roundoff tolerance 1e-12"
    )
    serialized = json.dumps([s.model_dump() for s in samples], sort_keys=True).encode()
    return EvaluationReport(
        backend=backend,
        metrics={name: sum(row[name] for row in rows) / len(rows) for name in rows[0]},
        sample_count=len(samples),
        dataset_sha256=hashlib.sha256(serialized).hexdigest(),
        generated_at=datetime.now(UTC).isoformat(),
        configuration=config,
        per_sample=[
            ScoredSample(sample=s, metrics=row) for s, row in zip(samples, rows, strict=True)
        ],
        limitations=limitations,
    )


def load_samples(path: str | Path) -> list[EvalSample]:
    samples = [
        EvalSample.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("Evaluation file is empty")
    questions = [s.question for s in samples]
    if len(set(questions)) != len(questions):
        raise ValueError("Evaluation questions must be unique")
    return samples


def save_report(report: BaseModel, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["lexical", "ragas"], required=True)
    parser.add_argument("--judge-model", default="llama3.2")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--revision", default="unspecified", help="Code revision for provenance")


def evaluation_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "backend": args.backend,
        "configuration": {"revision": args.revision},
    }
    if args.backend == "ragas":
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        options.update(
            llm=ChatOllama(model=args.judge_model, base_url=args.ollama_url, temperature=0),
            embeddings=OllamaEmbeddings(model=args.embedding_model, base_url=args.ollama_url),
            judge_id=args.judge_model,
            embedding_id=args.embedding_model,
        )
        options["configuration"].update(
            ollama_url=args.ollama_url,
            judge_temperature_initial="0",
            judge_temperature_policy=(
                "RAGAS controls temperature; defaults follow the recorded installed version"
            ),
        )
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    report = compute_report(load_samples(args.eval_set), **evaluation_options(args))
    save_report(report, args.output)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
