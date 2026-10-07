import copy
import json
import sys

import httpx
import pytest

from mobility_ai.capstone.app import ABSTENTION, OllamaClient
from mobility_ai.evals import citation_support as audit


@pytest.fixture
def record():
    return {
        "id": "test",
        "question": "What does pgvector add?",
        "answer": "pgvector adds vector search. [2]",
        "contexts": ["UNCITED DISTRACTOR", "pgvector adds vector search."],
        "citations": {"2": "facts.txt#chunks-1-2"},
        "abstained": False,
        "ground_truth": "REFERENCE MUST NOT REACH JUDGE",
    }


def judgment(verdict="supported", citation_verdict="supports_part", quote="vector search"):
    return {
        "answer_support": verdict,
        "reason": "Test judgment.",
        "citations": [
            {
                "citation_id": 2,
                "verdict": citation_verdict,
                "reason": "Test reason.",
                "claim": "pgvector adds vector search.",
                "quote": quote,
            }
        ],
    }


def client_for(raw, requests=None, status=200):
    def handle(request):
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx.Response(status, json={"message": {"content": raw}, "eval_count": 10})

    return OllamaClient(transport=httpx.MockTransport(handle))


def test_only_cited_passages_reach_judge_and_ids_are_not_renumbered(record):
    requests = []
    result = audit.audit_record(record, client_for(json.dumps(judgment()), requests), "judge")
    assert result["status"] == "assessed"
    assert result["identifiers_valid"] is True
    assert result["evidence_quotes_valid"] is True
    assert result["semantic_support"]["answer_support"] == "supported"
    payload = requests[0]
    assert payload["model"] == "judge"
    assert payload["stream"] is False
    assert payload["format"]["additionalProperties"] is False
    assert payload["options"]["temperature"] == 0
    assert "UNCITED DISTRACTOR" not in json.dumps(payload)
    assert "REFERENCE MUST NOT REACH JUDGE" not in json.dumps(payload)
    assert "facts.txt" not in json.dumps(payload)
    prompt = json.loads(payload["messages"][1]["content"])
    assert prompt["cited_passages"] == [{"citation_id": 2, "text": record["contexts"][1]}]


def test_extra_citation_can_be_irrelevant_even_when_answer_is_supported(record):
    record["citations"]["1"] = "unrelated.txt#chunk-0"
    value = judgment()
    value["citations"].append(
        {
            "citation_id": 1,
            "verdict": "does_not_support",
            "reason": "Unrelated.",
            "claim": "",
            "quote": "",
        }
    )
    result = audit.audit_record(record, client_for(json.dumps(value)), "judge")
    assert result["status"] == "assessed"
    assert result["semantic_support"]["answer_support"] == "supported"
    assert result["semantic_support"]["citations"][1]["verdict"] == "does_not_support"


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain", "non_answer"])
def test_advisory_verdicts_are_preserved_without_overriding_them(record, verdict):
    value = judgment(verdict, "does_not_support", "")
    result = audit.audit_record(record, client_for(json.dumps(value)), "judge")
    assert result["status"] == "assessed"
    assert result["semantic_support"]["answer_support"] == verdict


@pytest.mark.parametrize(
    "change",
    [
        {"citation_id": 1},
        {"citation_id": True},
        {"quote": "Invented evidence"},
        {"quote": " "},
        {"claim": "Invented claim"},
        {"claim": " "},
    ],
)
def test_invalid_judge_ids_or_quotes_preserve_raw_output_without_a_verdict(record, change):
    value = judgment()
    value["citations"][0].update(change)
    raw = json.dumps(value)
    result = audit.audit_record(record, client_for(raw), "judge")
    assert result["status"] == "judge_error"
    assert result["identifiers_valid"] is True
    assert result["semantic_support"] is None
    assert result["provider_response"]["message"]["content"] == raw


@pytest.mark.parametrize(
    "variant", ["duplicate", "missing", "extra", "unsupported_only", "malformed"]
)
def test_incomplete_or_contradictory_judgments_are_not_support_scores(record, variant):
    value = judgment()
    if variant == "duplicate":
        value["citations"] *= 2
    elif variant == "missing":
        value["citations"] = []
    elif variant == "extra":
        value["extra"] = "value"
    elif variant == "unsupported_only":
        value["citations"][0]["verdict"] = "does_not_support"
    raw = "not JSON" if variant == "malformed" else json.dumps(value)
    assert audit.audit_record(record, client_for(raw), "judge")["status"] == "judge_error"


def test_every_citation_must_be_assessed_and_quotes_must_come_from_its_own_passage(record):
    record["citations"]["1"] = "unrelated.txt#chunk-0"
    assert (
        audit.audit_record(record, client_for(json.dumps(judgment())), "judge")["status"]
        == "judge_error"
    )
    value = judgment()
    value["citations"].append(
        {
            "citation_id": 1,
            "verdict": "does_not_support",
            "reason": "Unrelated",
            "claim": "",
            "quote": "vector search",
        }
    )
    assert (
        audit.audit_record(record, client_for(json.dumps(value)), "judge")["status"]
        == "judge_error"
    )


def test_quote_matching_normalizes_only_whitespace(record):
    record["contexts"][1] = "pgvector adds vector\nsearch."
    assert (
        audit.audit_record(record, client_for(json.dumps(judgment())), "judge")["status"]
        == "assessed"
    )
    value = judgment(quote="Vector Search")
    assert (
        audit.audit_record(record, client_for(json.dumps(value)), "judge")["status"]
        == "judge_error"
    )


@pytest.mark.parametrize("key", ["0", "-1", "01", "3", "source"])
def test_invalid_citations_do_not_call_judge(record, key):
    record["citations"] = {key: "facts#chunk-0"}
    requests = []
    result = audit.audit_record(record, client_for("unused", requests), "judge")
    assert result["status"] == "invalid_record"
    assert result["identifiers_valid"] is False
    assert requests == []


@pytest.mark.parametrize(
    "change",
    [
        {"citations": {}},
        {"citations": {"2": " "}},
        {"contexts": ["text", " "]},
        {"citations": None},
        {"abstained": True},
        {"answer": " "},
        {"result": "bad"},
        {"result": {"answer": "Different answer"}},
        {"result": {"retrieved": [None]}},
        {"result": {"retrieved": [{"text": "Different context"}]}},
    ],
)
def test_missing_or_conflicting_record_metadata_is_explicit(record, change):
    record.update(change)
    requests = []
    result = audit.audit_record(record, client_for("unused", requests), "judge")
    assert result["status"] == "invalid_record"
    assert requests == []


def test_legacy_benchmark_metadata_is_read_without_inferring_citations(record):
    old = copy.deepcopy(record)
    old.pop("citations")
    old.pop("abstained")
    old["result"] = {
        "citations": record["citations"],
        "retrieved": [{"text": c} for c in record["contexts"]],
    }
    assert (
        audit.audit_record(old, client_for(json.dumps(judgment())), "judge")["status"] == "assessed"
    )
    old.pop("contexts")
    assert audit.parse_record(old).contexts == record["contexts"]


@pytest.mark.parametrize(
    "change",
    [
        {"status": "error"},
        {"abstained": True, "citations": {}, "answer": "Alternate abstention"},
        {"abstained": None, "citations": {}, "answer": ABSTENTION},
    ],
)
def test_rejected_generation_and_recorded_abstentions_are_counted_as_exclusions(record, change):
    requests = []
    result = audit.audit_record(record | change, client_for("unused", requests), "judge")
    assert result["status"] == "not_applicable"
    assert result["semantic_support"] is None
    assert result["latency_ms"] >= 0
    assert requests == []


def test_input_limits_never_silently_truncate_evidence(record):
    requests = []
    result = audit.audit_record(record, client_for("unused", requests), "judge", max_input_chars=1)
    assert result["status"] == "input_too_large"
    assert result["cited_passages"][0]["text"] == record["contexts"][1]
    assert requests == []
    assert (
        "positive"
        in audit.audit_record(record, client_for("unused"), "judge", max_input_chars=0)["error"]
    )


def test_service_errors_and_malformed_envelopes_are_not_judgments(record):
    result = audit.audit_record(record, client_for("unavailable", status=503), "judge")
    assert result["status"] == "judge_error"
    assert result["semantic_support"] is None
    for response in [[], "bad", {"message": None}, {"message": "bad"}, {"message": {"content": 1}}]:
        client = OllamaClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
        )
        result = audit.audit_record(record, client, "judge")
        assert result["status"] == "judge_error"
        assert result["provider_response"] == response


def test_too_many_citations_fail_before_judging(record):
    record.update(contexts=["text"] * 21, citations={str(i): "source" for i in range(1, 22)})
    assert "At most 20" in audit.audit_record(record, client_for("unused"), "judge")["error"]


@pytest.mark.parametrize("end", [{"done": False}, {"done_reason": "length"}])
def test_even_valid_json_is_rejected_after_incomplete_generation(record, end):
    response = {"message": {"content": json.dumps(judgment())}, **end}
    client = OllamaClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    )
    result = audit.audit_record(record, client, "judge")
    assert result["status"] == "judge_error"
    assert result["semantic_support"] is None
    assert result["provider_response"] == response


def api_client(*, changed_digest=False):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            digest = "changed" if changed_digest and calls.count("/api/tags") > 1 else "original"
            return httpx.Response(200, json={"models": [{"name": "judge", "digest": digest}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        return httpx.Response(200, json={"message": {"content": json.dumps(judgment())}})

    return OllamaClient(transport=httpx.MockTransport(handle))


def test_audit_retains_every_line_and_cli_uses_explicit_model(record, tmp_path, monkeypatch):
    records, output = tmp_path / "records.jsonl", tmp_path / "audit"
    records.write_text(
        json.dumps(record) + "\nnot JSON\n[]\n" + json.dumps(record | {"status": "error"}) + "\n"
    )
    client = api_client()
    monkeypatch.setattr(audit, "OllamaClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--records", str(records), "--output", str(output), "--judge-model", "judge"],
    )
    audit.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["total"] == 4
    assert summary["status_counts"] == {"assessed": 1, "invalid_record": 2, "not_applicable": 1}
    assert summary["advisory_verdict_counts"] == {"supported": 1}
    assert summary["citation_verdict_counts"] == {"supports_part": 1}
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["model"]["digest"] == "original"
    assert metadata["completed_at"] and metadata["input_sha256"] and metadata["source_sha256"]
    assert len((output / "judgments.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        audit.audit_file(records, output, "judge", client)


def test_empty_inputs_invalid_limits_and_missing_models_fail_explicitly(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        audit.audit_file(records, tmp_path / "out", "judge", api_client())
    with pytest.raises(ValueError, match="positive"):
        audit.audit_file(records, tmp_path / "out", "judge", api_client(), max_input_chars=0)
    with pytest.raises(ValueError, match="not installed"):
        audit.model_metadata(api_client(), "absent")


def test_digest_change_keeps_partial_results_without_completed_report(record, tmp_path):
    records, output = tmp_path / "records.jsonl", tmp_path / "out"
    records.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="digest changed"):
        audit.audit_file(records, output, "judge", api_client(changed_digest=True))
    assert (output / "judgments.jsonl").exists()
    assert "completed_at" not in json.loads((output / "metadata.json").read_text())
    assert not (output / "summary.json").exists()
