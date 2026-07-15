"""KFP v2 pipeline for the RAG ingestion workflow. Compile and submit to OpenShift AI."""

from __future__ import annotations

import os

try:
    from kfp import compiler
    from kfp.dsl import InputPath, OutputPath, component, pipeline

    _KFP_AVAILABLE = True
except ImportError:
    _KFP_AVAILABLE = False

    def component(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator if args and callable(args[0]) else decorator

    def pipeline(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator if args and callable(args[0]) else decorator

    class InputPath:
        def __new__(cls, *args, **kwargs):
            return str

    class OutputPath:
        def __new__(cls, *args, **kwargs):
            return str


@component(base_image="python:3.11")
def fetch_documents(source_url: str, output_dir: OutputPath(str)):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Fetching from {source_url}")


@component(base_image="python:3.11", packages_to_install=["docling"])
def parse_documents(input_dir: InputPath(str), output_dir: OutputPath(str)):
    print(f"Parsing documents from {input_dir}")


@component(
    base_image="python:3.11",
    packages_to_install=["sentence-transformers", "pgvector"],
)
def generate_embeddings(input_dir: InputPath(str), collection_name: str):
    print(f"Generating embeddings into {collection_name}")


@pipeline(name="rag-ingestion-pipeline")
def rag_ingestion_pipeline(
    source_url: str = "s3://my-bucket/docs",
    collection_name: str = "documents",
):
    fetch_task = fetch_documents(source_url=source_url)
    parse_task = parse_documents(input_dir=fetch_task.output)
    generate_embeddings(input_dir=parse_task.output, collection_name=collection_name)


if __name__ == "__main__":
    if _KFP_AVAILABLE:
        try:
            compiler.Compiler().compile(rag_ingestion_pipeline, "rag_ingestion_pipeline.yaml")
            print("Pipeline compiled")
        except Exception as exc:
            print(f"Compile error: {exc}")
            print(repr(rag_ingestion_pipeline))
    else:
        print(repr(rag_ingestion_pipeline))
