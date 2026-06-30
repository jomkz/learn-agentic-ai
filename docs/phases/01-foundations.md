# Phase 1: AI/ML Foundations, Python Modernization, and Local Dev Environment

**Duration: 3 weeks** | [← Index](../index.md) | [Phase 2 →](02-langchain.md)

**Project directory:** [`projects/phase1-foundations/`](../../projects/phase1-foundations/)

---

## Objectives

- Understand how LLMs work: tokens, attention, context windows, temperature, sampling
- Master modern Python patterns used across all AI frameworks: async/await, type hints, Pydantic v2
- Set up a productive local development environment with offline model serving via Ollama
- Understand model formats and what drives the choice between them

---

## Key Concepts

### LLM Fundamentals
- Tokenization, embeddings, attention mechanism (conceptual, not mathematical)
- Context windows, temperature, top-p, top-k sampling
- Prompt engineering: system prompts, few-shot examples, chain-of-thought prompting
- LLM API patterns: completion vs. chat, streaming vs. blocking, token limits

### Modern Python for AI
- Pydantic v2: `BaseModel`, field validators, `model_config`, discriminated unions, `model_json_schema()`
- Python `async`/`await`: event loops, `asyncio.gather`, `TaskGroup`, async generators, `async for`
- Python typing: generics, `TypeVar`, `Protocol`, `Annotated`, `Literal`, `TypedDict`
- `uv`: fast Python package manager — workspace management, lockfiles, tool installs

### Local Model Development with Ollama
- Why local models matter: no API costs during development, data privacy, offline capability
- Ollama architecture: model library, REST API, OpenAI-compatible endpoint
- Running models locally: `ollama pull`, `ollama run`, `ollama serve`
- Interacting via Python: `ollama` Python library and OpenAI SDK pointed at `http://localhost:11434/v1`
- Choosing local models for dev: Llama 3.2 (3B for fast iteration), Llama 3.1 8B (quality)
- Model file formats overview:
  - **GGUF**: CPU/GPU quantized format used by Ollama/llama.cpp; most accessible
  - **safetensors**: HuggingFace's secure, fast-loading model format; used in training
  - **GPTQ / AWQ**: GPU-quantized formats for vLLM deployment
  - Quantization tradeoffs: 4-bit vs 8-bit vs full precision — quality vs memory vs speed

---

## Resources

- Andrej Karpathy — "Intro to Large Language Models" (YouTube, 1hr): https://youtu.be/zjkBMFhNj_g
- Andrej Karpathy — "Let's build GPT from scratch" (YouTube): https://youtu.be/kCc8FmEb1nY
- "The Illustrated Transformer" by Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- Anthropic Prompt Engineering Guide: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
- FastAPI docs (excellent async + Pydantic patterns): https://fastapi.tiangolo.com/
- uv package manager: https://docs.astral.sh/uv/
- Ollama docs: https://ollama.com/docs
- Ollama GitHub: https://github.com/ollama/ollama
- Ollama Python library: https://github.com/ollama/ollama-python

**Key paper:** "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762

---

## Hands-on Projects

1. **Dev environment setup** — Install `uv` and Ollama; pull `llama3.2`, `llama3.1:8b`, and `nomic-embed-text`; verify local inference from Python; set up Jupyter
2. **Pydantic v2 model** — Build an "AI Agent configuration" model with nested models, discriminated unions for provider selection (`openai` | `ollama` | `anthropic`), and field validators
3. **Async LLM client** — Async script calling both Ollama (local) and a cloud API concurrently with multiple prompts; measure and compare latency
4. **Prompt engineering iteration** — Take a vague task and iterate on system prompt + few-shot examples; test against both local Llama and a cloud model to understand quality gaps

---

## Completion Checklist

- [ ] `ollama run llama3.2` produces a response in the terminal
- [ ] `uv sync` succeeds and `uv run python -c "import pydantic; print(pydantic.VERSION)"` shows v2.x
- [ ] Async script successfully calls Ollama and a cloud LLM concurrently
- [ ] Pydantic model rejects invalid provider names and invalid nested fields with clear errors
- [ ] Prompt produces consistent, high-quality output after ≥3 iterations
- [ ] Jupyter Lab launches with `uv run jupyter lab`
