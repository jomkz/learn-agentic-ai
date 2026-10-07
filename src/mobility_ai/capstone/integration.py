"""Component import diagnostic; does not check service health."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from pydantic import BaseModel


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
        imported = import_fn()
        if isinstance(imported, ModuleType):
            getattr(imported, module_path.rsplit(".", 1)[-1])
        return SystemComponent(name=name, phase=phase, module=module_path)
    except Exception as e:
        return SystemComponent(
            name=name, phase=phase, module=module_path, status="error", note=str(e)[:80]
        )


def run_health_check() -> SystemHealthReport:
    checks = [
        (
            "Pydantic AgentConfig",
            1,
            "mobility_ai.phase1.config.AgentConfig",
            lambda: import_module("mobility_ai.phase1.config"),
        ),
        (
            "Async LLM Client",
            1,
            "mobility_ai.phase1.async_client.CompletionResult",
            lambda: import_module("mobility_ai.phase1.async_client"),
        ),
        (
            "Prompt Engineering",
            1,
            "mobility_ai.phase1.prompt_engineering.TRANSCRIPT",
            lambda: import_module("mobility_ai.phase1.prompt_engineering"),
        ),
        (
            "LCEL Chains",
            2,
            "mobility_ai.phase2.chains.build_qa_chain",
            lambda: import_module("mobility_ai.phase2.chains"),
        ),
        (
            "Tool Agent",
            2,
            "mobility_ai.phase2.agent.search_web",
            lambda: import_module("mobility_ai.phase2.agent"),
        ),
        (
            "FastAPI Streaming",
            2,
            "mobility_ai.phase2.streaming.app",
            lambda: import_module("mobility_ai.phase2.streaming"),
        ),
        (
            "Research Capstone",
            2,
            "mobility_ai.phase2.capstone.ResearchReport",
            lambda: import_module("mobility_ai.phase2.capstone"),
        ),
        (
            "RAG Ingestion",
            3,
            "mobility_ai.phase3.ingestion.chunk_recursive",
            lambda: import_module("mobility_ai.phase3.ingestion"),
        ),
        (
            "Hybrid Retrieval",
            3,
            "mobility_ai.phase3.retrieval.hybrid_rrf_fusion",
            lambda: import_module("mobility_ai.phase3.retrieval"),
        ),
        (
            "RAGAS Harness",
            3,
            "mobility_ai.evals.ragas_harness.compute_report",
            lambda: import_module("mobility_ai.evals.ragas_harness"),
        ),
        (
            "Exact Response Cache",
            4,
            "mobility_ai.phase4.cache.ExactMatchCache",
            lambda: import_module("mobility_ai.phase4.cache"),
        ),
        (
            "Advanced Techniques",
            4,
            "mobility_ai.phase4.techniques.rerank_with_scores",
            lambda: import_module("mobility_ai.phase4.techniques"),
        ),
        (
            "Token Budget",
            4,
            "mobility_ai.phase4.cost.TokenBudget",
            lambda: import_module("mobility_ai.phase4.cost"),
        ),
        (
            "LangGraph StateGraph",
            5,
            "mobility_ai.phase5.graphs.build_research_graph",
            lambda: import_module("mobility_ai.phase5.graphs"),
        ),
        (
            "Supervisor Multi-Agent",
            5,
            "mobility_ai.phase5.multi_agent.build_supervisor_graph",
            lambda: import_module("mobility_ai.phase5.multi_agent"),
        ),
        (
            "MCP Server Tools",
            6,
            "mobility_ai.phase6.mcp_server.run_query",
            lambda: import_module("mobility_ai.phase6.mcp_server"),
        ),
        (
            "Context Budget",
            6,
            "mobility_ai.phase6.context_budget.ContextBudget",
            lambda: import_module("mobility_ai.phase6.context_budget"),
        ),
        (
            "LlamaStack Client",
            7,
            "mobility_ai.phase7.llamastack_client.LlamaStackConfig",
            lambda: import_module("mobility_ai.phase7.llamastack_client"),
        ),
        (
            "QLoRA Config",
            8,
            "mobility_ai.phase8.qlora_finetune.FinetuneConfig",
            lambda: import_module("mobility_ai.phase8.qlora_finetune"),
        ),
        (
            "MLflow Tracking",
            8,
            "mobility_ai.phase8.mlflow_tracking.ExperimentConfig",
            lambda: import_module("mobility_ai.phase8.mlflow_tracking"),
        ),
        (
            "KFP Pipeline",
            8,
            "mobility_ai.phase8.kfp_pipeline.build_pipeline",
            lambda: import_module("mobility_ai.phase8.kfp_pipeline"),
        ),
        (
            "Neo4j Graph Model",
            9,
            "mobility_ai.phase9.neo4j_basics.ServiceGraph",
            lambda: import_module("mobility_ai.phase9.neo4j_basics"),
        ),
        (
            "RAFT Dataset",
            9,
            "mobility_ai.phase9.raft_dataset.build_raft_example",
            lambda: import_module("mobility_ai.phase9.raft_dataset"),
        ),
        (
            "Drift Monitoring",
            9,
            "mobility_ai.phase9.monitoring.compute_text_drift",
            lambda: import_module("mobility_ai.phase9.monitoring"),
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
    print(f"Component import check: {report.overall_status}")
    print(f"Components: {report.healthy_count}/{report.total_count} healthy")
    print(f"Phases covered: {report.phases_covered}")
    for c in report.components:
        icon = "v" if c.status == "ok" else "x"
        note = f" ({c.note})" if c.note else ""
        print(f"  [{icon}] Phase {c.phase}: {c.name}{note}")
