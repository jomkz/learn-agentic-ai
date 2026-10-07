"""Build a reproducible 32-case development calibration set without suggested labels."""

import argparse
import json
import random
from pathlib import Path

from mobility_ai.evals.benchmark import digest, normalize, write_json
from mobility_ai.evals.calibration import load_records, prepare_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evals = Path(__file__).resolve().parents[2]
    previous = evals / "development/citation-support/results/2026-10-07/records.jsonl"
    corpus = evals / "benchmarks/project-docs-v1/corpus"
    cases = []
    for line in previous.read_text().splitlines():
        record = json.loads(line)
        previous_id = record.pop("id")
        cases.append(
            (
                record,
                {
                    "kind": "previously_inspected",
                    "source": str(previous.relative_to(evals)),
                    "sha256": digest(previous),
                    "original_id": previous_id,
                    "note": "Existing development case, not a new model generation.",
                },
            )
        )

    def add(question, answer, texts, topic, *, file=None, cited=None):
        if file:
            source = corpus / file
            assert all(normalize(text) in normalize(source.read_text()) for text in texts)
            provenance = {"source": str(source.relative_to(evals)), "sha256": digest(source)}
        else:
            provenance = {"source": "Synthetic evidence; no real system measurements implied."}
        ids = cited or list(range(1, len(texts) + 1))
        cases.append(
            (
                {
                    "question": question,
                    "answer": answer + " " + " ".join(f"[{i}]" for i in ids),
                    "contexts": texts,
                    "citations": {str(i): f"{file or 'synthetic.txt'}#passage-{i}" for i in ids},
                    "abstained": False,
                },
                {
                    "kind": "authored_control",
                    "topic": topic,
                    "note": "Codex-authored answer/evidence pairing; not a new model generation.",
                    **provenance,
                },
            )
        )

    chunk = "Defaults are 512 **characters** per\nchunk and 64 characters overlap."
    for answer in (
        "512 characters with 64 characters overlap.",
        "512 tokens with 64 tokens overlap.",
    ):
        add(
            "What are the default chunk size and overlap?",
            answer,
            [chunk],
            "units",
            file="runbook.md",
        )
    add(
        "What is the default chunk overlap?",
        "128 characters.",
        [chunk],
        "quantity",
        file="runbook.md",
    )
    database = "- Missing database: run `ingest`; `ask` never silently creates a new corpus."
    for answer in ("Run ingest.", "Run ask; it silently creates a new corpus."):
        add(
            "What should I run when the database is missing?",
            answer,
            [database],
            "recovery",
            file="runbook.md",
        )
    model = "Re-ingest when switching embedding\nmodels."
    add(
        "What must I do after changing embedding models?",
        "Re-ingest the corpus.",
        [model],
        "paraphrase",
        file="overview.md",
    )
    cost = (
        "Local inference does not establish a\n"
        "zero infrastructure cost, so unmeasured costs remain null."
    )
    for answer in (
        "Local inference establishes zero infrastructure cost.",
        "Unmeasured infrastructure costs remain null.",
    ):
        add(
            "What can we say about local inference infrastructure cost?",
            answer,
            [cost],
            "negation",
            file="evaluation.md",
        )
    latency = "mean total query latency, not p95 or time to first token."
    add(
        "Does the comparison report time to first token?",
        "No, it does not report time to first token.",
        [latency],
        "latency_scope",
        file="evaluation.md",
    )
    add(
        "What is the measured time to first token?",
        "The cited text does not provide that measurement.",
        [latency],
        "non_answer",
        file="evaluation.md",
    )
    transaction = "Failed\nembedding calls leave the previous database intact."
    add(
        "What happens after failed embedding calls?",
        "The previous database stays intact, so every query is guaranteed to succeed.",
        [transaction],
        "unsupported_reasoning",
        file="architecture.md",
    )
    namespace = "queries and a namespace identifying the model, prompt, corpus version, and tenant."
    add(
        "Which identifiers does the namespace include?",
        "Model, prompt, corpus version, and tenant.",
        [namespace],
        "list_scope",
        file="overview.md",
    )
    add(
        "Which identifiers does the namespace include?",
        "Only the model, and no other identifiers.",
        [namespace],
        "list_scope",
        file="overview.md",
    )
    for answer in ("The median is 12 ms.", "The p95 is 12 ms."):
        add(
            "What latency statistic was measured?",
            answer,
            ["The measured median query latency was 12 ms. The p95 was not measured."],
            "statistic_scope",
        )
    add(
        "How many requests per second were measured?",
        "40 requests per second.",
        ["The benchmark processed 40 requests in 20 seconds."],
        "unsupported_arithmetic",
    )
    add(
        "Is feature X enabled?",
        "Yes, for every tenant.",
        ["Feature X is enabled for tenant alpha. No other tenants were tested."],
        "qualification",
    )
    add(
        "Is feature X enabled?",
        "Yes.",
        [
            "The release note says feature X is enabled.",
            "The same release note says feature X is disabled. "
            "Neither statement has a date or precedence.",
        ],
        "conflicting_evidence",
    )
    add(
        "What are the timeout and retry count?",
        "The timeout is 5 seconds and the retry count is 2.",
        ["The timeout is 5 seconds.", "The retry count is 2."],
        "distributed_evidence",
    )
    add(
        "What are the timeout and retry count?",
        "The timeout is 5 seconds and the retry count is 2.",
        ["The timeout is 5 seconds.", "The retry count is 2."],
        "uncited_evidence",
        cited=[1],
    )
    add(
        "What is the timeout?",
        "The timeout is 5 seconds.",
        ["The timeout is 5 seconds.", "Logs are stored as JSON."],
        "extra_citation",
    )
    assert len(cases) == 32
    random.Random(20261007).shuffle(cases)
    records, origins = [], {}
    for i, (record, origin) in enumerate(cases, 1):
        case_id = f"case-{i:03d}"
        records.append({"id": case_id, **record})
        origins[case_id] = origin
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    load_records(path)
    write_json(args.output / "origins.json", origins)
    prepare_packet(path, args.output / "packet")
    write_json(
        args.output / "artifacts.sha256.json",
        {
            str(p.relative_to(args.output)): digest(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    print(f"Prepared {len(cases)} unlabelled cases at {args.output}")


if __name__ == "__main__":
    main()
