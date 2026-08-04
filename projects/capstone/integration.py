"""Enterprise Knowledge Assistant — system health check verifying all phase components are importable."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

for phase_dir in sorted(Path(__file__).parent.parent.glob("phase*/")):
    if str(phase_dir) not in sys.path:
        sys.path.insert(0, str(phase_dir))
_evals = str(Path(__file__).parent.parent / "evals")
if _evals not in sys.path:
    sys.path.insert(0, _evals)


class SystemComponent(BaseModel):
    name: str
    phase: int
    module: str
    status: str = "ok"
    note: str = ""


class SystemHealthReport(BaseModel):
    components: list[SystemComponent]
    phases_covered: list[int]
    overall_status: str

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.components if c.status == "ok")

    @property
    def total_count(self) -> int:
        return len(self.components)


def check_component(name: str, phase: int, module_path: str, import_fn) -> SystemComponent:
    try:
        import_fn()
        return SystemComponent(name=name, phase=phase, module=module_path)
    except Exception as e:
        return SystemComponent(
            name=name, phase=phase, module=module_path, status="error", note=str(e)[:80]
        )


def run_health_check() -> SystemHealthReport:
    checks = [
        ("Pydantic AgentConfig", 1, "config.AgentConfig", lambda: __import__("config")),
        ("Async LLM Client", 1, "async_client.CompletionResult", lambda: __import__("async_client")),
        (
            "Prompt Engineering",
            1,
            "prompt_engineering.TRANSCRIPT",
            lambda: __import__("prompt_engineering"),
        ),
        ("LCEL Chains", 2, "chains.build_qa_chain", lambda: __import__("chains")),
        ("Tool Agent", 2, "agent.search_web", lambda: __import__("agent")),
        ("FastAPI Streaming", 2, "streaming.app", lambda: __import__("streaming")),
        ("Research Capstone", 2, "capstone.ResearchReport", lambda: __import__("capstone")),
        ("RAG Ingestion", 3, "ingestion.chunk_recursive", lambda: __import__("ingestion")),
        ("Hybrid Retrieval", 3, "retrieval.hybrid_rrf_fusion", lambda: __import__("retrieval")),
        ("RAGAS Harness", 3, "ragas_harness.compute_report", lambda: __import__("ragas_harness")),
        ("Semantic Cache", 4, "cache.SemanticCache", lambda: __import__("cache")),
        (
            "Advanced Techniques",
            4,
            "techniques.rerank_with_scores",
            lambda: __import__("techniques"),
        ),
        ("Token Budget", 4, "cost.TokenBudget", lambda: __import__("cost")),
        (
            "LangGraph StateGraph",
            5,
            "graphs.build_research_graph",
            lambda: __import__("graphs"),
        ),
        (
            "Supervisor Multi-Agent",
            5,
            "multi_agent.build_supervisor_graph",
            lambda: __import__("multi_agent"),
        ),
        ("MCP Server Tools", 6, "mcp_server.run_query", lambda: __import__("mcp_server")),
        (
            "Context Budget",
            6,
            "context_budget.ContextBudget",
            lambda: __import__("context_budget"),
        ),
        (
            "LlamaStack Client",
            7,
            "llamastack_client.LlamaStackConfig",
            lambda: __import__("llamastack_client"),
        ),
        (
            "QLoRA Config",
            8,
            "qlora_finetune.FinetuneConfig",
            lambda: __import__("qlora_finetune"),
        ),
        (
            "MLflow Tracking",
            8,
            "mlflow_tracking.ExperimentConfig",
            lambda: __import__("mlflow_tracking"),
        ),
        (
            "KFP Pipeline",
            8,
            "kfp_pipeline.rag_ingestion_pipeline",
            lambda: __import__("kfp_pipeline"),
        ),
        (
            "Neo4j Graph Model",
            9,
            "neo4j_basics.ServiceGraph",
            lambda: __import__("neo4j_basics"),
        ),
        ("RAFT Dataset", 9, "raft_dataset.build_raft_example", lambda: __import__("raft_dataset")),
        (
            "Drift Monitoring",
            9,
            "monitoring.compute_text_drift",
            lambda: __import__("monitoring"),
        ),
    ]
    components = [check_component(n, p, m, fn) for n, p, m, fn in checks]
    phases = sorted(set(c.phase for c in components))
    ok = all(c.status == "ok" for c in components)
    return SystemHealthReport(
        components=components,
        phases_covered=phases,
        overall_status="healthy" if ok else "degraded",
    )


if __name__ == "__main__":
    report = run_health_check()
    print(f"System Health: {report.overall_status}")
    print(f"Components: {report.healthy_count}/{report.total_count} healthy")
    print(f"Phases covered: {report.phases_covered}")
    for c in report.components:
        icon = "v" if c.status == "ok" else "x"
        note = f" ({c.note})" if c.note else ""
        print(f"  [{icon}] Phase {c.phase}: {c.name}{note}")
