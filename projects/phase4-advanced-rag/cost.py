"""Cost optimization patterns: token budgeting, tier routing, and cost estimation."""

from __future__ import annotations

_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "claude-haiku": (0.25, 1.25),
}

_DEFAULT_COMPLEX_KEYWORDS = ["analyze", "synthesize", "compare", "evaluate", "architecture"]


class TokenBudget:
    def __init__(self, max_tokens: int, model: str = "gpt-4o-mini") -> None:
        self._max = max_tokens
        self._model = model
        self._used = 0

    def count(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.remaining()

    def add(self, text: str) -> None:
        self._used += self.count(text)

    def remaining(self) -> int:
        return max(0, self._max - self._used)

    def reset(self) -> None:
        self._used = 0


def tier_route(query: str, complex_keywords: list[str] | None = None) -> str:
    keywords = complex_keywords if complex_keywords is not None else _DEFAULT_COMPLEX_KEYWORDS
    words = query.lower().split()
    if len(words) < 20 and not any(kw in words for kw in keywords):
        return "cheap"
    return "expensive"


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    if model not in _PRICES:
        raise ValueError(f"Unknown model '{model}'. Known models: {list(_PRICES)}")
    in_price, out_price = _PRICES[model]
    return (input_tokens / 1e6) * in_price + (output_tokens / 1e6) * out_price


if __name__ == "__main__":
    queries = [
        "what is RAG",
        "analyze and compare the architecture of two retrieval systems",
        "how do embeddings work",
        "evaluate and synthesize approaches to fine-tuning",
    ]

    print("=== Tier Routing ===")
    for q in queries:
        route = tier_route(q)
        print(f"  [{route:9s}]  {q}")

    print("\n=== Cost Comparison (1k input / 500 output tokens) ===")
    for model in _PRICES:
        cost = estimate_cost_usd(1000, 500, model)
        print(f"  {model:20s}  ${cost:.6f}")

    print("\n=== Token Budget Demo ===")
    budget = TokenBudget(max_tokens=50)
    sample = "This is a short test sentence for the token budget."
    print(f"  text tokens (est): {budget.count(sample)}")
    print(f"  fits in 50:        {budget.fits(sample)}")
    budget.add(sample)
    print(f"  remaining:         {budget.remaining()}")
