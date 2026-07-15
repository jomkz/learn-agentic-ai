"""
Environment smoke-test for Phase 1.

Runs sequential checks and prints PASS/FAIL for each prerequisite.
Exits 0 if all pass, 1 if any fail.

Usage:
    uv run python projects/phase1-foundations/verify_setup.py
"""

from __future__ import annotations

import sys


def _check(label: str, fn: object) -> bool:
    try:
        fn()  # type: ignore[operator]
        print(f"  PASS  {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {label}")
        print(f"        {exc}")
        return False


def check_pydantic_v2() -> None:
    import pydantic

    major = int(pydantic.VERSION.split(".")[0])
    if major < 2:
        raise RuntimeError(f"pydantic {pydantic.VERSION} — need v2.x")


def check_openai_sdk() -> None:
    import openai  # noqa: F401
    import openai as _openai

    _ = _openai.__version__


def check_httpx() -> None:
    import httpx  # noqa: F401


def check_dotenv() -> None:
    import dotenv  # noqa: F401


def check_ollama_running() -> None:
    import httpx

    r = httpx.get("http://localhost:11434/api/tags", timeout=3)
    r.raise_for_status()


def check_ollama_models() -> None:
    import httpx

    r = httpx.get("http://localhost:11434/api/tags", timeout=3)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    required = ["llama3.2", "nomic-embed-text"]
    missing = [m for m in required if not any(m in name for name in models)]
    if missing:
        available = ", ".join(models) if models else "(none)"
        raise RuntimeError(
            f"Missing models: {missing}\n"
            f"        Available: {available}\n"
            f"        Run: ollama pull {' '.join(missing)}"
        )


def check_jupyterlab() -> None:
    import jupyterlab  # noqa: F401
    import jupyterlab as _jl

    _ = _jl.__version__


def check_langchain_available() -> None:
    """Optional — only present after `uv sync --extra langchain`."""
    import langchain  # noqa: F401


def main() -> None:
    print("Phase 1 — Environment Verification\n")

    results: list[bool] = []

    print("── Python packages ──────────────────────────────────────")
    results.append(_check("pydantic >= 2.x", check_pydantic_v2))
    results.append(_check("openai SDK", check_openai_sdk))
    results.append(_check("httpx", check_httpx))
    results.append(_check("python-dotenv", check_dotenv))
    results.append(_check("jupyterlab", check_jupyterlab))

    print("\n── Ollama (local model server) ───────────────────────────")
    ollama_up = _check("Ollama API responding at localhost:11434", check_ollama_running)
    results.append(ollama_up)
    if ollama_up:
        results.append(
            _check("Required models pulled (llama3.2, nomic-embed-text)", check_ollama_models)
        )
    else:
        print("  SKIP  Model check (Ollama not running — start with `ollama serve`)")

    print("\n── Optional (Phase 2+) ───────────────────────────────────")
    _check("langchain (install with: uv sync --extra langchain)", check_langchain_available)

    passed = sum(results)
    total = len(results)
    print(f"\n{'─' * 56}")
    print(f"  {passed}/{total} checks passed")

    if passed < total:
        print("  Some checks failed — see messages above.")
        sys.exit(1)
    else:
        print("  All checks passed. Ready to start Phase 1 projects.")
        sys.exit(0)


if __name__ == "__main__":
    main()
