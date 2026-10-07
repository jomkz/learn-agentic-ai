"""Prepare blind human review packets and compare completed reviews with citation audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mobility_ai.capstone.app import ABSTENTION
from mobility_ai.evals.benchmark import digest, write_json
from mobility_ai.evals.citation_support import cited_passages, parse_record, validate_judgment

AnswerLabel = Literal["supported", "unsupported", "uncertain", "non_answer"]
CitationLabel = Literal["supports_part", "does_not_support", "uncertain"]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
LIMITATIONS = (
    "Agreement with one independent reviewer is not established accuracy or a correctness "
    "gate. Judge errors remain in overall denominators. Inspect rationales even when labels "
    "agree; resolve ambiguity and obtain further review before adoption. This is development "
    "calibration, not held-out answer-quality evaluation."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HumanCitation(StrictModel):
    citation_id: Annotated[int, Field(gt=0)]
    verdict: CitationLabel


class HumanCase(StrictModel):
    id: Text
    answer_support: AnswerLabel
    reason: Text
    citations: list[HumanCitation]


class HumanReview(StrictModel):
    format_version: Annotated[int, Field(ge=1, le=1)]
    records_sha256: str
    reviewer: Text
    reviewed_without_judge: Literal[True]
    cases: list[HumanCase]

    @field_validator("reviewed_without_judge", mode="before")
    @classmethod
    def explicit_attestation(cls, value: object) -> bool:
        if value is not True:
            raise ValueError("Independent review must be explicitly confirmed")
        return True


def load_records(path: Path) -> tuple[str, list[dict]]:
    raw = path.read_bytes()
    records = []
    seen = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Each calibration record must be an object")
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("Calibration IDs must be nonempty, unique strings")
        sample = parse_record(row)
        passages = cited_passages(sample)
        abstained = (
            sample.abstained if sample.abstained is not None else sample.answer == ABSTENTION
        )
        if (
            row.get("status") == "error"
            or abstained
            or not passages
            or not sample.question.strip()
            or not sample.answer.strip()
        ):
            raise ValueError("Calibration requires non-abstaining answers with cited passages")
        seen.add(case_id)
        records.append(
            {
                "id": case_id,
                "input_line": line_number,
                "question": sample.question,
                "answer": sample.answer,
                "cited_passages": passages,
            }
        )
    if not records:
        raise ValueError("Calibration records are empty")
    return hashlib.sha256(raw).hexdigest(), records


def prepare_packet(records: Path, output: Path) -> dict:
    input_hash, samples = load_records(records)
    template = {
        "format_version": 1,
        "records_sha256": input_hash,
        "reviewer": "",
        "reviewed_without_judge": False,
        "cases": [
            {
                "id": sample["id"],
                "answer_support": None,
                "reason": "",
                "citations": [
                    {"citation_id": p["citation_id"], "verdict": None}
                    for p in sample["cited_passages"]
                ],
            }
            for sample in samples
        ],
    }
    # Match the judge's evidence view; filenames and uncited contexts can bias the review.
    packet = {
        "template": template,
        "samples": [
            {
                "id": s["id"],
                "question": s["question"],
                "answer": s["answer"],
                "cited_passages": [
                    {"citation_id": p["citation_id"], "text": p["text"]}
                    for p in s["cited_passages"]
                ],
            }
            for s in samples
        ],
    }
    # JSON is inside an inert script element; escape HTML delimiters before insertion.
    data = (
        json.dumps(packet).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    html = Path(__file__).with_name("calibration_review.html").read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.html").write_text(html.replace("__PACKET_JSON__", data), encoding="utf-8")
    write_json(output / "labels.template.json", template)
    manifest = {
        "records_sha256": input_hash,
        "case_count": len(samples),
        "citation_count": sum(len(s["cited_passages"]) for s in samples),
        "labels_complete": False,
        "limitations": LIMITATIONS,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_review(path: Path, input_hash: str, samples: list[dict]) -> HumanReview:
    review = HumanReview.model_validate_json(path.read_text(encoding="utf-8"))
    if review.records_sha256 != input_hash:
        raise ValueError("Human review input hash does not match the records")
    ids = [c.id for c in review.cases]
    if len(ids) != len(set(ids)) or set(ids) != {s["id"] for s in samples}:
        raise ValueError("Human review must label every case exactly once and no others")
    by_id = {c.id: c for c in review.cases}
    for sample in samples:
        case = by_id[sample["id"]]
        citation_ids = [c.citation_id for c in case.citations]
        if len(citation_ids) != len(set(citation_ids)) or set(citation_ids) != {
            p["citation_id"] for p in sample["cited_passages"]
        }:
            raise ValueError(f"Human review must label every citation once: {case.id}")
        if case.answer_support == "supported" and not any(
            c.verdict == "supports_part" for c in case.citations
        ):
            raise ValueError(f"Supported human label needs a supporting citation: {case.id}")
    return review


def load_audit(directory: Path, input_hash: str, samples: list[dict]) -> tuple[dict, dict]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("input_sha256") != input_hash or not metadata.get("completed_at"):
        raise ValueError("Audit must be complete and use the exact calibration records")
    rows = [
        json.loads(line)
        for line in (directory / "judgments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows):
        raise ValueError("Audit rows require string case IDs")
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != {s["id"] for s in samples}:
        raise ValueError("Audit must include every case exactly once and no others")
    for sample in samples:
        row = by_id[sample["id"]]
        if type(row.get("input_line")) is not int or row["input_line"] != sample["input_line"]:
            raise ValueError(f"Audit input line mismatch: {sample['id']}")
        # Eligible inputs always record this evidence, even on HTTP/validator/budget errors.
        for key in ("question", "answer", "cited_passages"):
            if row.get(key) != sample[key]:
                raise ValueError(f"Audit {key} does not match input: {sample['id']}")
        status = row.get("status")
        if status not in {"assessed", "judge_error", "input_too_large"}:
            raise ValueError(f"Unexpected audit status for an eligible case: {status}")
        if row.get("identifiers_valid") is not True:
            raise ValueError("Audit must validate the recorded citation identifiers")
        if status == "assessed":
            judgment = validate_judgment(
                json.dumps(row.get("semantic_support")),
                sample["cited_passages"],
                sample["answer"],
            )
            provider = row.get("provider_response")
            if not isinstance(provider, dict):
                raise ValueError("Assessed audit rows require the raw judge response")
            message = provider.get("message")
            raw = message.get("content") if isinstance(message, dict) else None
            if (
                not isinstance(raw, str)
                or provider.get("done") is False
                or provider.get("done_reason") == "length"
            ):
                raise ValueError("Assessed audit rows require a complete raw judgment")
            original = validate_judgment(raw, sample["cited_passages"], sample["answer"])
            if original != judgment:
                raise ValueError("Saved judgment differs from the raw judge response")
            if row.get("evidence_quotes_valid") is not True:
                raise ValueError("Assessed audit rows require validated excerpts")
        elif row.get("semantic_support") is not None:
            raise ValueError("Failed audit rows cannot contain a support verdict")
    return metadata, by_id


def agreement(matching: int, total: int) -> dict:
    return {"matching": matching, "total": total, "fraction": matching / total if total else None}


def compare(records: Path, labels: Path, audit: Path, output: Path) -> dict:
    input_hash, samples = load_records(records)
    review = load_review(labels, input_hash, samples)
    metadata, judgments = load_audit(audit, input_hash, samples)
    labels_by_id = {c.id: c for c in review.cases}
    statuses: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    citation_confusion: dict[str, Counter] = defaultdict(Counter)
    matching = citation_matching = citation_total = citation_assessed = 0
    cases = []
    for sample in samples:
        case_id = sample["id"]
        human = labels_by_id[case_id]
        row = judgments[case_id]
        status = row["status"]
        statuses[status] += 1
        prediction = row["semantic_support"]
        observed = prediction["answer_support"] if prediction else status
        confusion[human.answer_support][observed] += 1
        agrees = observed == human.answer_support
        matching += agrees
        citations = []
        predicted_citations = (
            {c["citation_id"]: c for c in prediction["citations"]} if prediction else {}
        )
        for citation in human.citations:
            predicted = predicted_citations.get(citation.citation_id)
            verdict = predicted["verdict"] if predicted else status
            citation_confusion[citation.verdict][verdict] += 1
            citation_total += 1
            citation_assessed += predicted is not None
            citation_matching += citation.verdict == verdict
            citations.append(
                {
                    "citation_id": citation.citation_id,
                    "human": citation.verdict,
                    "judge": verdict,
                    "agrees": citation.verdict == verdict,
                    "judge_reason": predicted["reason"] if predicted else None,
                }
            )
        cases.append(
            {
                "id": case_id,
                "status": status,
                "human": human.answer_support,
                "judge": observed,
                "agrees": agrees,
                "human_reason": human.reason,
                "judge_reason": prediction["reason"] if prediction else None,
                "citations": citations,
                "error": row.get("error"),
                "rationale_review": None,
                "rationale_notes": "",
            }
        )
    report = {
        "total": len(samples),
        "reviewer": review.reviewer,
        "independence": "Reviewer attestation, not independently verifiable by this tool.",
        "status_counts": dict(statuses),
        "answer_agreement_all_inputs": agreement(matching, len(samples)),
        "answer_agreement_assessed_only": agreement(matching, statuses["assessed"]),
        "citation_agreement_all_citations": agreement(citation_matching, citation_total),
        "citation_agreement_assessed_only": agreement(citation_matching, citation_assessed),
        "answer_confusion": dict(confusion),
        "citation_confusion": dict(citation_confusion),
        "label_disagreement_ids": [
            c["id"] for c in cases if c["status"] == "assessed" and not c["agrees"]
        ],
        "citation_disagreement_ids": [
            c["id"]
            for c in cases
            if c["status"] == "assessed" and any(not p["agrees"] for p in c["citations"])
        ],
        "unassessed_ids": [c["id"] for c in cases if c["status"] != "assessed"],
        "rationale_review_complete": False,
        "limitations": LIMITATIONS,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    write_json(output / "cases.json", cases)
    write_json(
        output / "metadata.json",
        {
            "records_sha256": input_hash,
            "labels_sha256": digest(labels),
            "audit_metadata_sha256": digest(audit / "metadata.json"),
            "audit_judgments_sha256": digest(audit / "judgments.jsonl"),
            "comparison_source_sha256": digest(Path(__file__)),
            "judge": metadata,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build an unlabelled local browser review packet")
    prepare.add_argument("--records", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser(
        "compare", help="Compare complete independent labels with an audit"
    )
    comparison.add_argument("--records", type=Path, required=True)
    comparison.add_argument("--labels", type=Path, required=True)
    comparison.add_argument("--audit", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_packet(args.records, args.output)
    else:
        result = compare(args.records, args.labels, args.audit, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
