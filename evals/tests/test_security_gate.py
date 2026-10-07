import json
import subprocess
import sys
from pathlib import Path

import pytest

from mobility_ai.evals import security_gate
from mobility_ai.evals.security_gate import PROBES, check_report


def records():
    return (
        [{"entry_type": "init", "run": "test"}]
        + [
            {
                "entry_type": "eval",
                "probe": probe,
                "passed": 3,
                "total_evaluated": 3,
                "nones": 0,
                "fails": 0,
                "total_processed": 3,
            }
            for probe in PROBES
        ]
        + [{"entry_type": "completion", "run": "test"}]
    )


def write(tmp_path, rows):
    path = tmp_path / "scan.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_complete_passing_scan(tmp_path):
    assert check_report(write(tmp_path, records()))["passed"] == 6


@pytest.mark.parametrize(
    "change", ["failure", "no_results", "incomplete", "missing_probe", "nulls"]
)
def test_bad_scan_cannot_pass(tmp_path, change):
    rows = records()
    if change == "failure":
        rows[1].update(passed=2, fails=1)
    elif change == "no_results":
        rows = [rows[0], rows[-1]]
    elif change == "incomplete":
        rows.pop()
    elif change == "missing_probe":
        rows.pop(1)
    else:
        rows[1].update(nones=1, total_processed=4)
    with pytest.raises(ValueError):
        check_report(write(tmp_path, rows))


def test_missing_report_does_not_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_report(tmp_path / "missing.jsonl")


def test_scan_invokes_scanner_and_checks_actual_output(tmp_path, monkeypatch, capsys):
    def fake_scan(command, *, check, timeout):
        assert check is True and timeout == 1800
        assert command[command.index("--target_name") + 1] == "model; literal"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert (
            config["plugins"]["generators"]["ollama"]["OllamaGeneratorChat"]["host"]
            == "http://localhost:11434"
        )
        prefix = command[command.index("--report_prefix") + 1]
        Path(prefix + ".report.jsonl").write_text("\n".join(json.dumps(r) for r in records()))

    monkeypatch.setattr(security_gate.subprocess, "run", fake_scan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate",
            "--model",
            "model; literal",
            "--endpoint",
            "http://localhost:11434",
            "--output-dir",
            str(tmp_path),
        ],
    )
    security_gate.main()
    assert "Security gate passed" in capsys.readouterr().out
    report = next(tmp_path.glob("*.report.jsonl"))
    monkeypatch.setattr(sys, "argv", ["gate", "--report", str(report)])
    security_gate.main()
    assert json.loads(capsys.readouterr().out)["passed"] == 6


def test_scanner_failure_propagates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "garak")

    monkeypatch.setattr(security_gate.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        security_gate.run_scan("http://localhost:11434", "model", tmp_path)


def test_missing_endpoint_fails_before_scan(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        security_gate.run_scan("file:///tmp", "model", tmp_path)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["gate", "--model", "model"])
    with pytest.raises(SystemExit) as error:
        security_gate.main()
    assert error.value.code == 2
