from __future__ import annotations

from integration import SystemComponent, SystemHealthReport, check_component, run_health_check


def test_system_component_ok():
    assert SystemComponent(name="x", phase=1, module="y").status == "ok"


def test_system_component_error():
    assert SystemComponent(name="x", phase=1, module="y", status="error").status == "error"


def _make_degraded_report():
    return SystemHealthReport(
        components=[
            SystemComponent(name="a", phase=1, module="m"),
            SystemComponent(name="b", phase=2, module="n", status="error", note="x"),
        ],
        phases_covered=[1, 2],
        overall_status="degraded",
    )


def test_health_report_healthy_count():
    assert _make_degraded_report().healthy_count == 1


def test_health_report_total_count():
    assert _make_degraded_report().total_count == 2


def test_check_component_success():
    assert check_component("test", 1, "m", lambda: None).status == "ok"


def test_check_component_error():
    assert check_component("test", 1, "m", lambda: 1 / 0).status == "error"


def test_check_component_error_note():
    assert check_component("test", 1, "m", lambda: 1 / 0).note != ""


def test_run_health_check_returns_report():
    assert isinstance(run_health_check(), SystemHealthReport)


def test_run_health_check_phases_covered():
    assert set(run_health_check().phases_covered) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_run_health_check_component_count():
    assert len(run_health_check().components) >= 20


def test_health_report_overall_status_valid():
    assert run_health_check().overall_status in ("healthy", "degraded")
