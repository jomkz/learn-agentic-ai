"""Compile a KFP pipeline that runs the same ingestion/evaluation CLI as the local lab."""

import argparse
from pathlib import Path


def build_pipeline(image: str):
    """Require a built application image; no decorator stubs or fake successful runs."""
    try:
        from kfp import dsl
    except ImportError as exc:
        raise RuntimeError("Install the pipelines extra to compile a KFP pipeline") from exc

    @dsl.container_component
    def ingest_documents(
        corpus: dsl.Input[dsl.Dataset],
        database: dsl.Output[dsl.Artifact],
        provider: str,
        ollama_url: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "mobility_ai.capstone.app", "ingest"],
            args=[
                "--corpus",
                corpus.path,
                "--db",
                database.path,
                "--provider",
                provider,
                "--ollama-url",
                ollama_url,
            ],
        )

    @dsl.container_component
    def evaluate_documents(
        database: dsl.Input[dsl.Artifact],
        questions: dsl.Input[dsl.Dataset],
        records: dsl.Output[dsl.Dataset],
        report: dsl.Output[dsl.Artifact],
        ollama_url: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "mobility_ai.capstone.app", "evaluate"],
            args=[
                "--db",
                database.path,
                "--eval-set",
                questions.path,
                "--records",
                records.path,
                "--output",
                report.path,
                "--backend",
                "lexical",
                "--ollama-url",
                ollama_url,
            ],
        )

    @dsl.pipeline(name="rag-ingestion-evaluation")
    def rag_ingestion_pipeline(
        corpus_uri: str,
        questions_uri: str,
        provider: str = "lexical",
        ollama_url: str = "http://ollama:11434",
    ):
        corpus = dsl.importer(artifact_uri=corpus_uri, artifact_class=dsl.Dataset, reimport=True)
        questions = dsl.importer(
            artifact_uri=questions_uri, artifact_class=dsl.Dataset, reimport=True
        )
        ingest_task = ingest_documents(
            corpus=corpus.output, provider=provider, ollama_url=ollama_url
        )
        ingest_task.set_caching_options(False)
        evaluate_task = evaluate_documents(
            database=ingest_task.outputs["database"],
            questions=questions.output,
            ollama_url=ollama_url,
        )
        evaluate_task.set_caching_options(False)

    return rag_ingestion_pipeline


def compile_pipeline(output: Path, image: str) -> None:
    pipeline = build_pipeline(image)
    from kfp import compiler

    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline, str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Built and pushed application image reference"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/kfp-pipeline.yaml"))
    args = parser.parse_args()
    compile_pipeline(args.output, args.image)
    print(f"Compiled {args.output}; cluster execution is a separate integration check")


if __name__ == "__main__":
    main()
