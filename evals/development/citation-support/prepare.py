"""Prepare inspected citation failures and controls without changing saved generations."""

import argparse
import json
from pathlib import Path

from mobility_ai.evals.benchmark import digest, read_jsonl, write_json
from mobility_ai.evals.citation_support import AuditInput, parse_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline = root / "benchmarks/project-docs-v1/results/2026-09-23/top3.jsonl"
    structured = root / "development/structured-responses/results/2026-10-07"
    expanded = root / "development/chunk-boundaries/results/2026-10-07"
    records, origins, expected = [], {}, {}

    def add(case_id, sample, verdict, citation_verdicts, origin):
        records.append({"id": case_id, **sample.model_dump()})
        origins[case_id] = origin
        expected[case_id] = {"answer_support": verdict, "citations": citation_verdicts}

    def saved(case_id, path, record_id, verdict, citation_verdicts):
        row = next(r for r in read_jsonl(path) if r["id"] == record_id)
        sample = parse_record(row)
        add(
            case_id,
            sample,
            verdict,
            citation_verdicts,
            {
                "source": str(path.relative_to(root)),
                "sha256": digest(path),
                "record_id": record_id,
                "transformation": (
                    "Flatten recorded question, answer, contexts, citations, and abstention state."
                ),
            },
        )
        return sample

    saved("wrong-passage-ingest", baseline, "q08", "unsupported", {"1": "does_not_support"})
    saved("unsupported-reasoning-latency", baseline, "q13", "unsupported", {})
    saved(
        "extra-passage-chunk-size",
        structured / "top3.jsonl",
        "chunk-defaults",
        "supported",
        {"1": "supports_part", "2": "does_not_support"},
    )
    namespace = saved(
        "wrong-passage-namespace",
        expanded / "on/top3.jsonl",
        "q18",
        "unsupported",
        {"3": "does_not_support"},
    )
    saved(
        "unsupported-namespace",
        expanded / "on/top1.jsonl",
        "q18",
        "unsupported",
        {"1": "does_not_support"},
    )
    saved(
        "supported-vector-search",
        structured / "top1.jsonl",
        "vector-search",
        "supported",
        {"1": "supports_part"},
    )
    saved(
        "supported-missing-database",
        expanded / "on/top1.jsonl",
        "q08",
        "supported",
        {"1": "supports_part"},
    )
    saved("non-answer-throughput", expanded / "off/top3.jsonl", "q21", "non_answer", {})
    add(
        "corrected-namespace-citation",
        namespace.model_copy(
            update={
                "answer": namespace.answer.replace("[3]", "[2]"),
                "citations": {"2": "overview.md#chunks-9-11"},
            }
        ),
        "supported",
        {"2": "supports_part"},
        {
            "derived_from": "wrong-passage-namespace",
            "transformation": (
                "Controlled citation correction from [3] to [2]; answer words unchanged. "
                "Not a new model generation."
            ),
        },
    )
    add(
        "negation-overlap",
        AuditInput(
            question="Is SQLite intended for scalable production vector search?",
            answer="SQLite is intended for scalable production vector search. [1]",
            contexts=["SQLite is not intended for scalable production vector search."],
            citations={"1": "synthetic-negation.txt#chunk-0"},
            abstained=False,
        ),
        "unsupported",
        {"1": "does_not_support"},
        {"transformation": "Synthetic negation counterexample."},
    )
    add(
        "split-support",
        AuditInput(
            question="What are the chunk size and overlap?",
            answer="Chunks contain 512 characters and overlap by 64 characters. [1] [2]",
            contexts=[
                "Chunks contain 512 characters.",
                "Adjacent chunks overlap by 64 characters.",
            ],
            citations={"1": "size.txt#chunk-0", "2": "overlap.txt#chunk-0"},
            abstained=False,
        ),
        "supported",
        {"1": "supports_part", "2": "supports_part"},
        {"transformation": "Synthetic distributed-evidence control."},
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    write_json(args.output / "origins.json", origins)
    write_json(
        args.output / "expectations.json",
        {
            "reviewer": (
                "Provisional Codex labels; not independent human review or held-out calibration."
            ),
            "cases": expected,
        },
    )
    print(f"Prepared {len(records)} inspected development cases at {args.output}")


if __name__ == "__main__":
    main()
