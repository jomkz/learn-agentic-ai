"""DSPy RAG module definition. Run optimizer when dspy and a labeled eval set are available."""

from __future__ import annotations

try:
    import dspy  # pragma: no cover

    _DSPY_AVAILABLE = True  # pragma: no cover
except ImportError:
    dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False


if _DSPY_AVAILABLE:  # pragma: no cover

    class RAGSignature(dspy.Signature):  # pragma: no cover
        """Answer a question given retrieved context passages."""

        context: list[str] = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class RAGModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generate = dspy.ChainOfThought(RAGSignature)

        def forward(self, question: str, context: list[str]) -> object:
            return self.generate(question=question, context=context)

else:

    class RAGSignature:  # type: ignore[no-redef]
        """Answer a question given retrieved context passages."""

        context: list[str]
        question: str
        answer: str

    class RAGModule:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.generate = None

        def forward(self, question: str, context: list[str]) -> object:
            return {"question": question, "context": context, "answer": "[dspy not installed]"}


def compile_with_bootstrap(module: object, trainset: list, metric: object) -> object:
    if _DSPY_AVAILABLE:
        optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
        return optimizer.compile(module, trainset=trainset)
    return module


if __name__ == "__main__":
    print(f"DSPy available: {_DSPY_AVAILABLE}")
    print("\nRAGSignature fields:")
    if _DSPY_AVAILABLE:
        for name, field in RAGSignature.model_fields.items():
            print(f"  {name}: {field}")
    else:
        print("  context: list[str]  (input)")
        print("  question: str       (input)")
        print("  answer: str         (output)")

    module = RAGModule()
    result = module.forward(
        question="What is retrieval augmented generation?",
        context=["RAG combines retrieval with LLM generation."],
    )
    print(f"\nForward pass result: {result}")
