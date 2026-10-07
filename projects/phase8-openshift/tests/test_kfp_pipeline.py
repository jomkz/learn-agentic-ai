import sys
from unittest.mock import patch

import pytest

from mobility_ai.phase8.kfp_pipeline import build_pipeline, main


def test_missing_kfp_fails_explicitly():
    with patch.dict(sys.modules, {"kfp": None}):
        with pytest.raises(RuntimeError, match="pipelines extra"):
            build_pipeline("example.test/mobility-ai:test")


def test_real_pipeline_compiles(tmp_path, monkeypatch):
    pytest.importorskip("kfp")
    import yaml

    output = tmp_path / "pipeline.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile", "--output", str(output), "--image", "example.test/mobility-ai:test"],
    )
    main()
    spec = yaml.safe_load(output.read_text())
    executors = spec["deploymentSpec"]["executors"]
    containers = [value["container"] for value in executors.values() if "container" in value]
    assert len(containers) == 2
    assert all(c["image"] == "example.test/mobility-ai:test" for c in containers)
    assert {c["command"][-1] for c in containers} == {"ingest", "evaluate"}
