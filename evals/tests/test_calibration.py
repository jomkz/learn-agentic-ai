import copy
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

from mobility_ai.capstone.app import OllamaClient
from mobility_ai.evals import calibration
from mobility_ai.evals.benchmark import digest, write_json
from mobility_ai.evals.citation_support import audit_record


@pytest.fixture
def fixture(tmp_path):
    samples = [
        {
            "id": "case-001",
            "question": "What is the timeout?",
            "answer": "5 seconds. [1] [2]",
            "contexts": ["The timeout is 5 seconds.", "Logs use JSON.", "UNCITED SECRET"],
            "citations": {"1": "HIDDEN_NAME_1", "2": "HIDDEN_NAME_2"},
            "abstained": False,
            "ground_truth": "REFERENCE SECRET",
        },
        {
            "id": "case-002",
            "question": "How many retries?",
            "answer": "2 retries. [1]",
            "contexts": ["The retry count is 2."],
            "citations": {"1": "retries.txt"},
            "abstained": False,
        },
    ]
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    prediction = {
        "answer_support": "supported",
        "reason": "The first passage supplies the timeout.",
        "citations": [
            {
                "citation_id": 1,
                "verdict": "supports_part",
                "claim": "5 seconds",
                "quote": "The timeout is 5 seconds.",
                "reason": "It states the timeout.",
            },
            {
                "citation_id": 2,
                "verdict": "does_not_support",
                "claim": "",
                "quote": "",
                "reason": "It is about logging.",
            },
        ],
    }
    client = OllamaClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"message": {"content": json.dumps(prediction)}}
            )
        )
    )
    first = audit_record(samples[0], client, "test-judge")
    error_client = OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    second = audit_record(samples[1], error_client, "test-judge")
    rows = [{**first, "input_line": 1}, {**second, "input_line": 2}]
    audit = tmp_path / "audit"
    audit.mkdir()
    write_json(audit / "metadata.json", {"input_sha256": digest(records), "completed_at": "test"})
    save_rows(audit, rows)
    labels = {
        "format_version": 1,
        "records_sha256": digest(records),
        "reviewer": "Unit-test reviewer; synthetic fixture only",
        "reviewed_without_judge": True,
        "cases": [
            {
                "id": "case-001",
                "answer_support": "supported",
                "reason": "Timeout is explicit; the logging citation is irrelevant.",
                "citations": [
                    {"citation_id": 1, "verdict": "supports_part"},
                    {"citation_id": 2, "verdict": "does_not_support"},
                ],
            },
            {
                "id": "case-002",
                "answer_support": "supported",
                "reason": "Retry count is explicit.",
                "citations": [{"citation_id": 1, "verdict": "supports_part"}],
            },
        ],
    }
    labels_path = tmp_path / "labels.json"
    write_json(labels_path, labels)
    return {
        "records": records,
        "samples": samples,
        "audit": audit,
        "rows": rows,
        "labels": labels,
        "labels_path": labels_path,
        "output": tmp_path / "comparison",
    }


def save_rows(directory, rows):
    (directory / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def compare(fixture):
    return calibration.compare(
        fixture["records"], fixture["labels_path"], fixture["audit"], fixture["output"]
    )


def test_errors_remain_in_answer_and_citation_denominators(fixture):
    report = compare(fixture)
    assert report["status_counts"] == {"assessed": 1, "judge_error": 1}
    assert report["answer_agreement_all_inputs"] == {"matching": 1, "total": 2, "fraction": 0.5}
    assert report["answer_agreement_assessed_only"]["fraction"] == 1
    assert report["citation_agreement_all_citations"]["total"] == 3
    assert report["citation_agreement_all_citations"]["matching"] == 2
    assert report["citation_agreement_assessed_only"]["fraction"] == 1
    assert report["answer_confusion"] == {"supported": {"supported": 1, "judge_error": 1}}
    assert report["unassessed_ids"] == ["case-002"]
    assert report["rationale_review_complete"] is False
    cases = json.loads((fixture["output"] / "cases.json").read_text())
    assert cases[0]["human_reason"] and cases[0]["judge_reason"]
    assert cases[0]["rationale_review"] is None  # Matching labels do not approve reasoning.
    provenance = json.loads((fixture["output"] / "metadata.json").read_text())
    assert provenance["labels_sha256"] == digest(fixture["labels_path"])
    assert provenance["audit_judgments_sha256"] == digest(fixture["audit"] / "judgments.jsonl")
    with pytest.raises(FileExistsError):
        compare(fixture)


def test_extra_citation_disagreement_is_separate_from_answer_agreement(fixture):
    fixture["labels"]["cases"][0]["citations"][1]["verdict"] = "uncertain"
    write_json(fixture["labels_path"], fixture["labels"])
    report = compare(fixture)
    assert report["label_disagreement_ids"] == []
    assert report["citation_disagreement_ids"] == ["case-001"]
    assert report["citation_confusion"]["uncertain"] == {"does_not_support": 1}


@pytest.mark.parametrize("label", ["unsupported", "uncertain", "non_answer"])
def test_all_human_labels_and_confusion_are_preserved(fixture, label):
    fixture["labels"]["cases"][0]["answer_support"] = label
    write_json(fixture["labels_path"], fixture["labels"])
    report = compare(fixture)
    assert report["label_disagreement_ids"] == ["case-001"]
    assert report["answer_confusion"][label] == {"supported": 1}


def test_no_assessed_judgments_has_null_conditional_rates(fixture):
    row = fixture["rows"][0]
    row.update(status="input_too_large", semantic_support=None, evidence_quotes_valid=None)
    save_rows(fixture["audit"], fixture["rows"])
    report = compare(fixture)
    assert report["answer_agreement_all_inputs"]["fraction"] == 0
    assert report["answer_agreement_assessed_only"]["fraction"] is None
    assert report["citation_agreement_assessed_only"]["fraction"] is None


@pytest.mark.parametrize(
    "variant",
    [
        "wrong_hash",
        "no_reviewer",
        "not_independent",
        "numeric_attestation",
        "boolean_version",
        "unlabelled",
        "blank_reason",
        "missing_case",
        "duplicate_case",
        "extra_case",
        "missing_citation",
        "duplicate_citation",
        "extra_citation",
        "unsupported_citations",
        "wrong_type",
        "extra_field",
    ],
)
def test_incomplete_or_mismatched_human_reviews_cannot_produce_reports(fixture, variant):
    value = fixture["labels"]
    case = value["cases"][0]
    if variant == "wrong_hash":
        value["records_sha256"] = "wrong"
    elif variant == "no_reviewer":
        value["reviewer"] = " "
    elif variant == "not_independent":
        value["reviewed_without_judge"] = False
    elif variant == "numeric_attestation":
        value["reviewed_without_judge"] = 1
    elif variant == "boolean_version":
        value["format_version"] = True
    elif variant == "unlabelled":
        case["answer_support"] = None
    elif variant == "blank_reason":
        case["reason"] = " \n "
    elif variant == "missing_case":
        value["cases"].pop()
    elif variant == "duplicate_case":
        value["cases"].append(case)
    elif variant == "extra_case":
        value["cases"].append({**case, "id": "other"})
    elif variant == "missing_citation":
        case["citations"].pop()
    elif variant == "duplicate_citation":
        case["citations"].append(case["citations"][0])
    elif variant == "extra_citation":
        case["citations"].append({"citation_id": 3, "verdict": "uncertain"})
    elif variant == "unsupported_citations":
        case["citations"][0]["verdict"] = "does_not_support"
    elif variant == "wrong_type":
        case["citations"][0]["citation_id"] = True
    elif variant == "extra_field":
        value["suggested_labels"] = True
    write_json(fixture["labels_path"], value)
    with pytest.raises(ValueError):
        compare(fixture)
    assert not fixture["output"].exists()


@pytest.mark.parametrize("variant", ["incomplete", "different_input"])
def test_audit_metadata_must_match(fixture, variant):
    metadata = {"input_sha256": digest(fixture["records"]), "completed_at": "test"}
    if variant == "incomplete":
        metadata.pop("completed_at")
    else:
        metadata["input_sha256"] = "different"
    write_json(fixture["audit"] / "metadata.json", metadata)
    with pytest.raises(ValueError, match="complete"):
        compare(fixture)


@pytest.mark.parametrize(
    "variant",
    [
        "missing",
        "duplicate",
        "unknown",
        "not_object",
        "id_type",
        "line",
        "answer",
        "passage",
        "status",
        "ids_invalid",
        "quotes_invalid",
        "bad_judgment",
        "error_verdict",
        "raw_missing",
        "raw_incomplete",
        "raw_missing_message",
        "raw_disagrees",
    ],
)
def test_mismatched_or_invalid_audit_rows_are_rejected(fixture, variant):
    rows = fixture["rows"]
    row = rows[0]
    if variant == "missing":
        rows.pop()
    elif variant == "duplicate":
        rows.append(row)
    elif variant == "unknown":
        row["id"] = "other"
    elif variant == "not_object":
        rows[0] = []
    elif variant == "id_type":
        row["id"] = 123
    elif variant == "line":
        row["input_line"] = 2
    elif variant == "answer":
        row["answer"] = "different"
    elif variant == "passage":
        row["cited_passages"][0]["text"] = "different"
    elif variant == "status":
        row["status"] = "not_applicable"
    elif variant == "ids_invalid":
        row["identifiers_valid"] = False
    elif variant == "quotes_invalid":
        row["evidence_quotes_valid"] = None
    elif variant == "bad_judgment":
        row["semantic_support"]["citations"][0]["quote"] = "invented"
    elif variant == "error_verdict":
        rows[1]["semantic_support"] = row["semantic_support"]
    elif variant == "raw_missing":
        row["provider_response"] = None
    elif variant == "raw_incomplete":
        row["provider_response"]["done"] = False
    elif variant == "raw_missing_message":
        row["provider_response"]["message"] = {}
    elif variant == "raw_disagrees":
        row["semantic_support"]["reason"] = "Manually changed after the judge ran."
    save_rows(fixture["audit"], rows)
    with pytest.raises(ValueError):
        compare(fixture)
    assert not fixture["output"].exists()


def test_packet_has_no_predictions_references_filenames_or_uncited_text(fixture, tmp_path):
    out = tmp_path / "packet"
    summary = calibration.prepare_packet(fixture["records"], out)
    assert summary["case_count"] == 2
    assert summary["citation_count"] == 3
    html = (out / "review.html").read_text()
    assert all(
        secret not in html for secret in ("UNCITED SECRET", "REFERENCE SECRET", "HIDDEN_NAME")
    )
    value = json.loads((out / "labels.template.json").read_text())
    assert all(c["answer_support"] is None for c in value["cases"])
    assert value["reviewed_without_judge"] is False
    assert value["reviewer"] == ""
    with pytest.raises(FileExistsError):
        calibration.prepare_packet(fixture["records"], out)


def test_packet_treats_markup_and_script_delimiters_as_data(fixture, tmp_path):
    malicious = '</script><script>alert("not executable")</script><img src=x onerror=alert(1)>'
    samples = copy.deepcopy(fixture["samples"])
    samples[0]["question"] = malicious
    fixture["records"].write_text("\n".join(json.dumps(s) for s in samples))
    out = tmp_path / "packet"
    calibration.prepare_packet(fixture["records"], out)
    html = (out / "review.html").read_text()
    assert malicious not in html
    embedded = re.search(r'<script id="packet" type="application/json">(.*?)</script>', html, re.S)
    assert embedded
    assert json.loads(embedded[1])["samples"][0]["question"] == malicious


@pytest.mark.parametrize(
    "variant",
    [
        "empty",
        "not_object",
        "duplicate",
        "missing_id",
        "abstained",
        "rejected",
        "no_citations",
        "blank_question",
    ],
)
def test_packet_rejects_ineligible_inputs(fixture, tmp_path, variant):
    samples = fixture["samples"]
    if variant == "empty":
        samples = []
    elif variant == "not_object":
        samples[0] = []
    elif variant == "duplicate":
        samples.append(samples[0])
    elif variant == "missing_id":
        samples[0].pop("id")
    elif variant == "abstained":
        samples[0]["abstained"] = True
    elif variant == "rejected":
        samples[0]["status"] = "error"
    elif variant == "no_citations":
        samples[0]["citations"] = {}
    elif variant == "blank_question":
        samples[0]["question"] = " "
    fixture["records"].write_text("\n".join(json.dumps(s) for s in samples))
    with pytest.raises(ValueError):
        calibration.prepare_packet(fixture["records"], tmp_path / "packet")


def test_blank_lines_preserve_audit_line_numbers(fixture):
    fixture["records"].write_text("\n" + fixture["records"].read_text())
    _, rows = calibration.load_records(fixture["records"])
    assert [r["input_line"] for r in rows] == [2, 3]


@pytest.mark.parametrize("command", ["prepare", "compare"])
def test_cli(fixture, monkeypatch, capsys, command):
    args = [
        "calibration",
        command,
        "--records",
        str(fixture["records"]),
        "--output",
        str(fixture["output"]),
    ]
    if command == "compare":
        args += ["--labels", str(fixture["labels_path"]), "--audit", str(fixture["audit"])]
    monkeypatch.setattr(sys, "argv", args)
    calibration.main()
    assert json.loads(capsys.readouterr().out)


def test_committed_review_suite_is_complete_unlabelled_and_hashed():
    root = Path(__file__).resolve().parents[1] / "calibration/citation-support/suite"
    hashes = json.loads((root / "artifacts.sha256.json").read_text())
    for name, expected in hashes.items():
        assert digest(root / name) == expected
    input_hash, records = calibration.load_records(root / "records.jsonl")
    assert len(records) == 32
    labels = json.loads((root / "packet/labels.template.json").read_text())
    assert labels["records_sha256"] == input_hash
    assert labels["cases"] and all(c["answer_support"] is None for c in labels["cases"])
