"""Compare recorded strategy outputs on the same questions and reference answers."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mobility_ai.evals.ragas_harness import (
    EvalSample,
    EvaluationReport,
    add_evaluation_arguments,
    compute_report,
    evaluation_options,
    load_samples,
    save_report,
)


class ApproachResult(BaseModel):
    approach: str
    evaluation: EvaluationReport
    mean_latency_ms: float | None
    mean_cost_usd: float | None


class ComparisonReport(BaseModel):
    results: list[ApproachResult]
    generated_at: str
    note: str = "Compare measured results; no strategy is assigned an assumed quality advantage."

    def to_markdown_table(self) -> str:
        header = "| Approach | Backend | Metrics | Mean latency (ms) |"
        rows = [header, "|---|---|---|---|"]
        for result in self.results:
            metrics = ", ".join(f"{k}: {v:.3f}" for k, v in result.evaluation.metrics.items())
            latency = (
                f"{result.mean_latency_ms:.1f}"
                if result.mean_latency_ms is not None
                else "not recorded"
            )
            rows.append(
                f"| {result.approach} | {result.evaluation.backend} | {metrics} | {latency} |"
            )
        return "\n".join(rows)


def _mean_if_complete(values: list[float | None]) -> float | None:
    return (
        sum(v for v in values if v is not None) / len(values)
        if all(v is not None for v in values)
        else None
    )


def run_comparison(runs: dict[str, list[EvalSample]], **evaluation_kwargs: Any) -> ComparisonReport:
    if len(runs) < 2:
        raise ValueError("Supply recorded outputs for at least two strategies")
    reference: list[tuple[str, str]] | None = None
    results = []
    for name, samples in runs.items():
        if not samples or len({s.question for s in samples}) != len(samples):
            raise ValueError("Each strategy needs non-empty, unique questions")
        samples = sorted(samples, key=lambda s: s.question)
        signature = [(s.question, s.ground_truth) for s in samples]
        if reference is not None and signature != reference:
            raise ValueError("Strategies must use the same questions and reference answers")
        reference = signature
        results.append(
            ApproachResult(
                approach=name,
                evaluation=compute_report(samples, **evaluation_kwargs),
                mean_latency_ms=_mean_if_complete([s.latency_ms for s in samples]),
                mean_cost_usd=_mean_if_complete([s.cost_usd for s in samples]),
            )
        )
    return ComparisonReport(results=results, generated_at=datetime.now(UTC).isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=JSONL_PATH")
    parser.add_argument("--output", required=True, type=Path)
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not name or name in runs:
            parser.error("Each --run must have a unique NAME=JSONL_PATH")
        runs[name] = load_samples(path)
    report = run_comparison(runs, **evaluation_options(args))
    save_report(report, args.output)
    print(report.to_markdown_table())


if __name__ == "__main__":
    main()
