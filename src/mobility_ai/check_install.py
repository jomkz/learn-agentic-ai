"""Import the selected optional integration dependencies without contacting providers."""

import os
from importlib import import_module

MODULES = {
    "rag": ["docling", "pgvector", "qdrant_client", "mobility_ai.phase3.ingestion"],
    "advanced-rag": ["dspy", "redis", "flashrank", "mobility_ai.phase4.dspy_rag"],
    "llamastack": ["llama_stack_client", "mobility_ai.phase7.llamastack_client"],
    "ml": ["peft", "trl", "mlflow", "kfp", "mobility_ai.phase8.qlora_finetune"],
    "advanced": ["graphrag", "neo4j", "mobility_ai.phase9.raft_dataset"],
    "evaluation": ["ragas", "datasets", "langchain_ollama"],
}


def main() -> None:
    extra = os.environ["PHASE_EXTRA"]
    for name in MODULES[extra]:
        import_module(name)
        print(f"Imported {name}")


if __name__ == "__main__":
    main()
