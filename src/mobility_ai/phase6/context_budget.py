"""Token-aware context window management for LLM conversations."""

from __future__ import annotations


class ContextBudget:
    def __init__(self, max_tokens: int, model: str = "gpt-4o-mini") -> None:
        self.max_tokens = max_tokens
        self.model = model
        self._messages: list[dict] = []
        self._used: int = 0

    def _count(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return round(len(text.split()) * 1.3)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._used += self._count(content)

    def remaining(self) -> int:
        return max(0, self.max_tokens - self._used)

    def is_full(self) -> bool:
        return self.remaining() < 100

    def trim_oldest(self) -> bool:
        for i, msg in enumerate(self._messages):
            if msg["role"] != "system":
                removed = self._messages.pop(i)
                self._used -= self._count(removed["content"])
                self._used = max(0, self._used)
                return True
        return False

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages = []
        self._used = 0


def trim_to_budget(
    messages: list[dict],
    max_tokens: int,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Keep system messages and the most-recent messages that fit in max_tokens."""
    budget = ContextBudget(max_tokens, model)
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    for msg in system_msgs:
        budget.add_message(msg["role"], msg["content"])

    fitting: list[dict] = []
    for msg in reversed(other_msgs):
        tokens = budget._count(msg["content"])
        if budget.remaining() >= tokens:
            fitting.insert(0, msg)
            budget._used += tokens
        else:
            break

    return system_msgs + fitting


if __name__ == "__main__":
    cb = ContextBudget(max_tokens=500)
    cb.add_message("system", "You are a helpful assistant.")
    cb.add_message("user", "Hello, how are you?")
    cb.add_message("assistant", "I am doing well, thank you for asking!")
    cb.add_message("user", "Tell me about MCP.")

    print(f"Messages: {len(cb.get_messages())}")
    print(f"Remaining tokens: {cb.remaining()}")
    print(f"Is full: {cb.is_full()}")

    removed = cb.trim_oldest()
    print(f"Trimmed oldest non-system message: {removed}")
    print(f"Messages after trim: {len(cb.get_messages())}")
    print(f"Remaining tokens after trim: {cb.remaining()}")
