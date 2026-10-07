"""Record a paired live Ollama benchmark and optionally judge its frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx

from mobility_ai.capstone.app import (
    ABSTENTION,
    GENERATION_OPTIONS,
    PROMPT_VERSION,
    OllamaClient,
    VectorStore,
    ask,
    ingest,
)
from mobility_ai.evals.ragas_harness import EvalSample, compute_report, save_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.split())


def load_suite(suite: Path) -> list[dict]:
    manifest = json.loads((suite / "corpus-manifest.json").read_text())
    documents = {}
    for name, entry in manifest.items():
        path = suite / "corpus" / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Frozen corpus changed: {name}")
        documents[name] = normalize(path.read_text())
    questions = read_jsonl(suite / "questions.jsonl")
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be non-empty and unique")
    for question in questions:
        if question["answerable"] != bool(question["evidence"]):
            raise ValueError("Answerable questions require evidence; abstention cases have none")
        for evidence in question["evidence"]:
            if normalize(evidence["text"]) not in documents[evidence["source"]]:
                raise ValueError(f"Reference evidence is absent: {question['id']}")
    return questions


class RecordingClient(OllamaClient):
    """Keep the actual response even when application citation validation rejects it."""

    def reset(self) -> None:
        self.contexts: list[str] = []
        self.raw_answer = ""
        self.calls: list[dict] = []

    def request(self, path: str, payload: dict) -> dict:
        response = super().request(path, payload)
        self.calls.append(
            {
                "path": path,
                "model": payload["model"],
                "statistics": {
                    key: value
                    for key, value in response.items()
                    if key.endswith("duration") or key.endswith("count") or key == "done_reason"
                },
            }
        )
        return response

    def generate(self, question: str, contexts: list[str], model: str) -> str:
        self.contexts = contexts
        self.raw_answer = super().generate(question, contexts, model)
        return self.raw_answer


def capture_one(db: Path, question: dict, top_k: int, client: RecordingClient) -> dict:
    client.reset()
    started = time.perf_counter()
    result, error = None, None
    try:
        result = ask(db, question["question"], top_k=top_k, client=client).model_dump()
    except (ValueError, httpx.HTTPError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": question["id"],
        "top_k": top_k,
        "question": question["question"],
        "ground_truth": question["ground_truth"],
        "answerable": question["answerable"],
        "category": question["category"],
        "evidence": question["evidence"],
        "status": "ok" if error is None else "error",
        "error": error,
        # Rejected payloads remain auditable but are not scored as answer prose.
        "answer": result["answer"] if result else "",
        "abstained": result["abstained"] if result else None,
        "citations": result["citations"] if result else None,
        "raw_response": client.raw_answer,
        "contexts": [c["text"] for c in result["retrieved"]] if result else client.contexts,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "result": result,
        "provider_calls": client.calls,
    }


def is_abstention(row: dict) -> bool:
    # Frozen v1 records predate explicit state; never infer new-record state from prose.
    if "abstained" in row:
        return row["abstained"] is True
    return row["answer"] == ABSTENTION


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unknown = [r for r in rows if not r["answerable"]]
    latencies = sorted(r["latency_ms"] for r in rows)
    supported = [r for r in rows if r["status"] == "ok" and not is_abstention(r)]
    hits = sum(
        all(normalize(e["text"]) in normalize(" ".join(r["contexts"])) for e in r["evidence"])
        for r in answerable
    )
    return {
        "total": len(rows),
        "application_errors": sum(r["status"] != "ok" for r in rows),
        "answerable_total": len(answerable),
        "unanswerable_total": len(unknown),
        "answerable_with_all_reference_evidence": hits,
        "answerable_abstentions": sum(is_abstention(r) for r in answerable),
        "unanswerable_exact_abstentions": sum(
            r["status"] == "ok" and is_abstention(r) for r in unknown
        ),
        "accepted_nonabstaining_answers": len(supported),
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "latency_definition": (
            "sequential warm requests; total client wall time; nearest-rank p95; errors included"
        ),
    }


def sample(row: dict) -> EvalSample:
    return EvalSample(
        **{
            key: row[key]
            for key in ("question", "ground_truth", "contexts", "answer", "latency_ms")
        },
        abstained=row.get("abstained"),
        citations=row.get("citations"),
    )


def capture(suite: Path, output: Path, endpoint: str, generation: str, embedding: str) -> None:
    questions = load_suite(suite)
    output.mkdir(parents=True, exist_ok=False)
    with httpx.Client(base_url=endpoint, timeout=30) as api:
        tags = api.get("/api/tags").raise_for_status().json()["models"]
        models = {
            name: next(m for m in tags if m["name"] == name) for name in (generation, embedding)
        }
        server = api.get("/api/version").raise_for_status().json()
    source = Path(__file__).parents[1]
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "server": server,
        "models": models,
        "python": platform.python_version(),
        "cpu_threads": os.cpu_count(),
        "generation_options": GENERATION_OPTIONS,
        "prompt_version": PROMPT_VERSION,
        "top_k": [1, 3],
        "questions_sha256": digest(suite / "questions.jsonl"),
        "manifest_sha256": digest(suite / "corpus-manifest.json"),
        "lockfile_sha256": digest(Path("uv.lock")),
        "source_sha256": {
            str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": (
            "Source hashes identify the working tree; HEAD alone is not the run revision."
        ),
        "packages": {name: version(name) for name in ("mobility-ai", "httpx", "pydantic")},
    }
    write_json(output / "metadata.json", metadata)
    client = RecordingClient(endpoint, timeout=600)
    client.reset()
    db = output / "corpus.db"
    started = time.perf_counter()
    chunks = ingest(
        suite / "corpus",
        db,
        provider="ollama",
        generation_model=generation,
        embedding_model=embedding,
        client=client,
    )
    metadata.update(
        chunks=chunks,
        ingestion_seconds=time.perf_counter() - started,
        corpus_config=VectorStore(db).load()[0].model_dump(),
    )
    # Use a smoke question, not a held-out question, to load the generation model.
    client.reset()
    client.generate(
        "What information is available?", ["This is a project documentation corpus."], generation
    )
    runs: dict[int, list[dict]] = {1: [], 3: []}
    for index, question in enumerate(questions):
        for top_k in [1, 3] if index % 2 == 0 else [3, 1]:
            row = capture_one(db, question, top_k, client)
            runs[top_k].append(row)
            with (output / f"top{top_k}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"{question['id']} top{top_k}: {row['status']}, {row['latency_ms'] / 1000:.1f}s",
                flush=True,
            )
    with httpx.Client(base_url=endpoint, timeout=30) as api:
        after = {
            m["name"]: m["digest"] for m in api.get("/api/tags").raise_for_status().json()["models"]
        }
        metadata["loaded_models"] = api.get("/api/ps").raise_for_status().json()
    if any(after.get(name) != model["digest"] for name, model in models.items()):
        raise ValueError("Model digests changed during capture")
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", {f"top{k}": summarize(rows) for k, rows in runs.items()})
    for top_k, rows in runs.items():
        save_report(
            compute_report([sample(r) for r in rows], backend="lexical"),
            output / f"top{top_k}-lexical.json",
        )


def judge(output: Path, endpoint: str, model: str, embedding: str) -> None:
    # This benchmark sends prompts only to its explicitly configured local endpoint.
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.run_config import RunConfig

    destination = output / "judgments.jsonl"
    if destination.exists():
        raise ValueError("Judgments already exist; preserve the previous attempt before rerunning")
    metadata = json.loads((output / "metadata.json").read_text())
    if not metadata.get("completed_at"):
        raise ValueError("Capture is incomplete; finish it before judging")
    runs = {k: read_jsonl(output / f"top{k}.jsonl") for k in (1, 3)}
    signatures = [[(r["id"], r["question"], r["ground_truth"]) for r in runs[k]] for k in (1, 3)]
    if signatures[0] != signatures[1] or not signatures[0]:
        raise ValueError("Recorded configurations must contain the same questions and references")
    with httpx.Client(base_url=endpoint, timeout=30) as api:
        tags = api.get("/api/tags").raise_for_status().json()["models"]
    models = {name: next(m for m in tags if m["name"] == name) for name in (model, embedding)}
    provenance = {
        "started_at": datetime.now(UTC).isoformat(),
        "models": models,
        "endpoint": endpoint,
        "input_sha256": {f"top{k}.jsonl": digest(output / f"top{k}.jsonl") for k in (1, 3)},
        "packages": {
            name: version(name) for name in ("ragas", "langchain-ollama", "langchain-community")
        },
        "seed": 0,
        "num_ctx": 8192,
        "num_predict": 1024,
        "format": "JSON requested by RAGAS prompts; native output unconstrained",
        "temperature_policy": (
            "RAGAS controls temperature; defaults follow the recorded installed version"
        ),
        "max_workers": 1,
        "timeout_seconds": 600,
        "max_retries": 1,
        "generator_judge_same_digest": models[model]["digest"]
        in {m["digest"] for m in metadata.get("models", {}).values()},
    }
    write_json(output / "judge-metadata.json", provenance)
    llm = ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=0,
        seed=0,
        num_ctx=8192,
        num_predict=1024,
        client_kwargs={"timeout": 600},
    )
    embeddings = OllamaEmbeddings(
        model=embedding, base_url=endpoint, client_kwargs={"timeout": 600}
    )
    for top_k in (1, 3):
        for row in runs[top_k]:
            result = {"id": row["id"], "top_k": top_k, "status": "not_applicable"}
            # Judge answerable, non-abstaining outputs. Report exclusions explicitly.
            if row["answerable"] and row["answer"] and not is_abstention(row):
                try:
                    report = compute_report(
                        [sample(row)],
                        llm=llm,
                        embeddings=embeddings,
                        judge_id=model,
                        embedding_id=embedding,
                        run_config=RunConfig(timeout=600, max_retries=1, max_workers=1, seed=0),
                        configuration={
                            "seed": "0",
                            "num_ctx": "8192",
                            "num_predict": "1024",
                            "format": (
                                "JSON requested by RAGAS prompts; native output unconstrained"
                            ),
                            "temperature_policy": str(provenance["temperature_policy"]),
                        },
                    )
                    result.update(status="ok", report=report.model_dump())
                except Exception as exc:
                    result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            print(f"Judge {row['id']} top{top_k}: {result['status']}", flush=True)
    with httpx.Client(base_url=endpoint, timeout=30) as api:
        after = {
            m["name"]: m["digest"] for m in api.get("/api/tags").raise_for_status().json()["models"]
        }
    if any(after.get(name) != model["digest"] for name, model in models.items()):
        raise ValueError("Judge model digests changed during scoring")
    provenance["completed_at"] = datetime.now(UTC).isoformat()
    write_json(output / "judge-metadata.json", provenance)
    judgments = read_jsonl(destination)
    summary = {}
    for top_k in (1, 3):
        group = [r for r in judgments if r["top_k"] == top_k]
        valid = [r for r in group if r["status"] == "ok"]
        summary[f"top{top_k}"] = {
            "scored": len(valid),
            "errors": sum(r["status"] == "error" for r in group),
            "not_applicable": sum(r["status"] == "not_applicable" for r in group),
            "conditional_mean_metrics": {
                name: statistics.mean(r["report"]["metrics"][name] for r in valid)
                for name in valid[0]["report"]["metrics"]
            }
            if valid
            else None,
        }
    write_json(output / "judge-summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["capture", "judge"])
    parser.add_argument("--suite", type=Path, default=Path("evals/benchmarks/project-docs-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--generation-model", default="llama3.2:3b")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:v1.5")
    args = parser.parse_args()
    if args.stage == "capture":
        capture(
            args.suite, args.output, args.ollama_url, args.generation_model, args.embedding_model
        )
    else:
        judge(args.output, args.ollama_url, args.judge_model, args.embedding_model)


if __name__ == "__main__":
    main()
