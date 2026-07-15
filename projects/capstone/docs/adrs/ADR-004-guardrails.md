# ADR-004 — Guardrails Strategy

**Status:** Accepted
**Date:** 2026-07-15
**Deciders:** Platform AI team

## Context

The production agent handles user-submitted queries that may contain prompt injections, jailbreak attempts,
PII, or requests that fall outside the permitted topic scope. The agent's outputs must also be validated for
schema correctness and sanitized for PII before being returned to callers. A single-layer defense is
insufficient; a multi-layer strategy is required to avoid single-point-of-failure risks.

Requirements:
- Block off-topic and jailbreak inputs before they reach the LLM.
- Validate and sanitize structured outputs before returning to the caller.
- Classify high-stakes interactions (medical, legal, financial, safety-critical) before responding.
- Keep added latency under 200ms p95 for the guardrail stack.

## Options Evaluated

### Option A — NeMo Guardrails (Colang rules)

NVIDIA NeMo Guardrails uses a domain-specific language (Colang) to define topic rails, jailbreak detection
flows, and dialog management rules. Runs as a wrapper around the LLM call; can short-circuit before the LLM
is invoked.

**Pros:** Declarative and auditable; strong community rail libraries; handles multi-turn dialog context;
low-latency rule evaluation.
**Cons:** Colang rules require ongoing maintenance; limited to rule-expressible patterns; false positive risk
on edge cases.

### Option B — Guardrails AI (Python validators)

Guardrails AI provides a Python SDK for defining input/output validators, including PII detection, JSON
schema validation, and custom regex-based checks. Integrates as a post-processing step on LLM output.

**Pros:** Pythonic and easy to extend; strong out-of-the-box validators (PII, profanity, JSON schema);
output re-asking loop for schema failures.
**Cons:** Output-only by default; adds a re-ask round-trip when validation fails; PII detection accuracy
depends on the underlying NER model.

### Option C — LlamaGuard (LLM classifier)

Meta's LlamaGuard is a fine-tuned Llama model that classifies both inputs and outputs against a taxonomy of
unsafe content categories (violence, self-harm, illicit content, etc.). Can be served locally via Ollama or
vLLM.

**Pros:** Strong classification accuracy on safety-critical categories; model-level understanding of
nuanced harm; handles input and output classification.
**Cons:** Adds one full LLM inference per guarded call (50–150ms on GPU); requires a GPU endpoint; model
itself can make classification errors.

### Option D — Custom regex/rules

Hand-crafted regular expressions and keyword blocklists for known-bad patterns (credit card numbers, SSNs,
known jailbreak phrases).

**Pros:** Zero latency; fully auditable; no external dependencies.
**Cons:** Poor generalization; high maintenance burden; easily bypassed with paraphrasing; not suitable as a
primary defense.

## Decision

**Defense-in-depth: all three principal layers (A, B, C) are deployed; custom regex (D) is used as a
supplementary fast-path only.**

| Layer | Tool | Position | Purpose |
|---|---|---|---|
| 1 — Input rails | NeMo Guardrails (Colang) | Before LLM call | Topic filtering, jailbreak detection, off-topic rejection |
| 2 — Output validation | Guardrails AI | After LLM call | PII redaction, JSON schema validation, profanity filter |
| 3 — Safety classification | LlamaGuard | Conditional (high-stakes paths) | Content safety classification for flagged interaction types |
| Fast-path | Custom regex | Parallel with Layer 1 | Block obvious PII patterns (SSN, CC) with zero latency |

LlamaGuard is invoked selectively: a lightweight query classifier flags interactions involving medical,
legal, financial, or safety-critical topics, and only those interactions are sent through LlamaGuard. This
bounds the p95 latency impact to high-stakes paths rather than all requests.

## Consequences

**What becomes easier:**
- No single guardrail layer is a single point of failure; bypass of one layer is caught by another.
- Colang rules provide an auditable, declarative record of what the system will and will not discuss.
- Guardrails AI output validation prevents malformed or PII-leaking responses from reaching callers.
- LlamaGuard classification provides semantic-level safety checks that regex cannot replicate.

**What becomes harder:**
- Total guardrail overhead is 50–150ms per request on guarded paths; LlamaGuard adds a full inference
  call on high-stakes paths (targeting <150ms p95 with a batched LlamaGuard endpoint).
- Colang rules must be versioned and tested as use cases evolve; stale rules cause false positives.
- Three separate systems increase operational surface: NeMo, Guardrails AI, and a LlamaGuard endpoint
  each require monitoring and SLO tracking.
- The re-ask loop in Guardrails AI can cause 1–2x additional LLM calls when output schema validation
  fails; this must be capped (max 2 re-asks) to bound latency.
