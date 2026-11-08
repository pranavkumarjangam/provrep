"""LLM backends for the adjudication agents.

AnthropicClient         real agents (Claude); needs ANTHROPIC_API_KEY in the environment
OfflineHeuristicClient  deterministic stand-in so the whole pipeline, its tests and
                        its evaluation run with no API key and no network. It builds
                        claims directly from the retrieved evidence and scores with a
                        transparent rule set. Results produced with it measure the
                        *pipeline*, not LLM reasoning - reports label them as such.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from ..filters.metapaths import MECHANISTIC, SIMILARITY
from ..utils import get_logger

log = get_logger(__name__)


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: type[BaseModel], context: dict | None = None) -> BaseModel: ...


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1500, temperature: float = 0.0,
                 api_key: str | None = None):
        import anthropic  # optional dependency

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("agents.llm.provider=anthropic needs ANTHROPIC_API_KEY set in the environment")
        self.client = anthropic.Anthropic(api_key=key)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.name = f"anthropic:{model}"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def complete_json(self, system: str, user: str, schema: type[BaseModel], context: dict | None = None) -> BaseModel:
        prompt = user
        last_err: Exception | None = None
        for _ in range(3):
            msg = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                system=system + "\nRespond with one JSON object and nothing else.",
                messages=[{"role": "user", "content": prompt}],
            )
            self.usage["calls"] += 1
            self.usage["input_tokens"] += getattr(msg.usage, "input_tokens", 0)
            self.usage["output_tokens"] += getattr(msg.usage, "output_tokens", 0)
            text = "".join(getattr(b, "text", "") for b in msg.content)
            try:
                return schema.model_validate_json(_extract_json(text))
            except (ValidationError, ValueError) as e:
                last_err = e
                prompt = f"{user}\n\nYour previous reply was not valid JSON for the required schema ({e}). Reply again with valid JSON only."
        raise RuntimeError(f"LLM did not return valid {schema.__name__}: {last_err}")


class OfflineHeuristicClient:
    name = "offline-heuristic"

    def complete_json(self, system: str, user: str, schema: type[BaseModel], context: dict | None = None) -> BaseModel:
        ctx = context or {}
        task = ctx.get("task")
        if task in ("proponent", "falsifier"):
            return schema.model_validate(self._claims(ctx["evidence"]))
        if task == "referee":
            return schema.model_validate(self._referee(ctx))
        if task == "nli":
            from .verifier import LexicalVerifier

            r = LexicalVerifier().verify(ctx["claim"], ctx["evidence"])
            label = "neutral" if r.label == "missing" else r.label
            return schema.model_validate({"label": label, "confidence": r.score})
        raise ValueError(f"offline client has no handler for task {task!r}")

    @staticmethod
    def _claims(evidence: list) -> dict:
        ranked = sorted(evidence, key=lambda e: (e.kind in SIMILARITY, -e.weight))[:6]
        return {"claims": [{"text": e.text, "evidence_ids": [e.id]} for e in ranked]}

    @staticmethod
    def _referee(ctx: dict[str, Any]) -> dict:
        passed = ctx["passed_claims"]
        ev = ctx["evidence"]
        n_total = max(1, ctx.get("n_claims_total", len(passed)))
        support = [g for g in passed if g.claim.stance == "supports"]
        refute = [g for g in passed if g.claim.stance == "refutes"]
        kinds_refute = [ev[i].kind for g in refute for i in g.claim.evidence_ids if i in ev]
        hub_genes = {n for g in refute for i in g.claim.evidence_ids if i in ev and ev[i].kind == "hub_target" for n in ev[i].node_ids}
        specific_targets, mech_claims = set(), 0
        for g in support:
            for i in g.claim.evidence_ids:
                e = ev.get(i)
                if e and e.kind in MECHANISTIC and len(e.node_ids) > 2:
                    mech_claims += 1
                    inner = set(e.node_ids[1:-1])
                    if not inner & hub_genes:
                        specific_targets.add(e.node_ids[1])
        mech = 1 - math.exp(-len(specific_targets) / 1.5)
        quality = (len(passed) / n_total) * (1.0 if mech_claims else 0.5)
        safety = 0.6 if "side_effect_burden" in kinds_refute else 0.1
        weights = {"no_mechanistic_path": 0.9, "similarity_driven": 0.6, "hub_target": 0.3, "off_target": 0.3,
                   "pubmed_abstract": 0.3, "side_effect_burden": 0.1}
        keep = 1.0
        for k in kinds_refute:
            keep *= 1 - weights.get(k, 0.2)
        counter = 1 - keep
        z = 3 * mech - 3 * counter - 1.0 * safety + 1.0 * quality - 0.5
        overall = 1 / (1 + math.exp(-z))
        tier = "strong" if overall >= 0.7 else "moderate" if overall >= 0.5 else "weak" if overall >= 0.3 else "reject"
        cited = [g.claim.id for g in support + refute]
        rationale = (f"{len(specific_targets)} specific (non-hub) target(s) with verified mechanistic paths; "
                     f"{len(refute)} verified counter-claim(s) ({', '.join(sorted(set(kinds_refute))) or 'none'}).")
        return {"scores": {"mechanistic_plausibility": mech, "evidence_quality": min(1.0, quality), "safety_concern": safety,
                           "counter_evidence_strength": counter},
                "overall": overall, "tier": tier, "rationale": rationale, "cited_claim_ids": cited, "rejected_claim_ids": []}


def make_llm(cfg: dict) -> LLMClient:
    lc = cfg["agents"]["llm"]
    if lc["provider"] == "anthropic":
        return AnthropicClient(lc["model"], lc["max_tokens"], lc["temperature"])
    if lc["provider"] == "offline":
        return OfflineHeuristicClient()
    raise ValueError(f"unknown LLM provider {lc['provider']}")


def dumps_evidence(evidence: list) -> str:
    return json.dumps([{"id": e.id, "kind": e.kind, "text": e.text} for e in evidence], indent=1)
