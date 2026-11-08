"""NLI claim-verification gate (M4).

Every claim must be entailed by the evidence it cites. A claim fails if it
cites an evidence id that was never retrieved, if no cited evidence entails
it, or if any cited evidence contradicts it. Provenance is therefore
enforced by the pipeline, not requested in a prompt.

Verifiers
---------
HFNLIVerifier     cross-encoder NLI model (default cross-encoder/nli-deberta-v3-base);
                  the intended production verifier. Needs ``transformers`` and
                  a one-time model download from huggingface.co.
LLMVerifier       asks the configured LLM for an entailment judgement.
LexicalVerifier   token-coverage heuristic with negation checks. Offline stand-in
                  only: it catches hallucinated entities and misquoted evidence
                  but is not a real NLI model.
"""
from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel

from ..utils import get_logger
from .schemas import Claim, Evidence, GatedClaim, GateResult

log = get_logger(__name__)

_STOP = set("""a an the of to in on for and or is are was were be been being by with as at from that this these those
it its into via which who whom than then there their its also may might can could would should has have had do does
did so such both either each other more most less least any all some very""".split())
_NEG = re.compile(r"\b(no|not|none|never|without|lacks?|fails?|failed|cannot|unable|absence|ineffective)\b", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-:%.]*[A-Za-z0-9%]|[A-Za-z0-9]")


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _tokens(text: str) -> list[str]:
    """Content tokens, lightly stemmed; negation words are excluded (handled separately)."""
    out = []
    for t in _TOKEN.findall(text):
        t = t.lower().rstrip(".")
        if t in _STOP or _NEG.fullmatch(t):
            continue
        out.append(_stem(t))
    return out


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, evidence: Evidence) -> GateResult: ...


class LexicalVerifier:
    name = "lexical-heuristic"

    def __init__(self, min_coverage: float = 0.7):
        self.min_coverage = min_coverage

    def verify(self, claim: str, evidence: Evidence) -> GateResult:
        ev_tokens = set(_tokens(evidence.text)) | {t for e in evidence.entities for t in _tokens(e)}
        cl = _tokens(claim)
        coverage = sum(t in ev_tokens for t in cl) / max(1, len(cl))
        mentioned = [e for e in evidence.entities if e.lower() in claim.lower()]
        need = min(2, len(evidence.entities))
        neg_claim, neg_ev = bool(_NEG.search(claim)), bool(_NEG.search(evidence.text))
        if neg_claim != neg_ev and coverage >= self.min_coverage and len(mentioned) >= need:
            label, score = "contradiction", coverage
        elif coverage >= self.min_coverage and len(mentioned) >= need:
            label, score = "entailment", coverage
        else:
            label, score = "neutral", coverage
        return GateResult(evidence_id=evidence.id, label=label, score=float(score), verifier=self.name)


class HFNLIVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: int = -1):
        from transformers import pipeline  # optional dependency

        self.name = f"hf:{model_name}"
        self.pipe = pipeline("text-classification", model=model_name, top_k=None, device=device)

    def verify(self, claim: str, evidence: Evidence) -> GateResult:
        scores = self.pipe({"text": evidence.text, "text_pair": claim})
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        best = {s["label"].lower(): float(s["score"]) for s in scores}
        get = lambda key: max((v for k, v in best.items() if key in k), default=0.0)  # noqa: E731
        ent, con, neu = get("entail"), get("contra"), get("neutral")
        label = max((("entailment", ent), ("contradiction", con), ("neutral", neu)), key=lambda x: x[1])[0]
        return GateResult(evidence_id=evidence.id, label=label, score=max(ent, con, neu), verifier=self.name)


class NLIJudgement(BaseModel):
    label: Literal["entailment", "neutral", "contradiction"]
    confidence: float


class LLMVerifier:
    def __init__(self, llm):
        self.llm = llm
        self.name = f"llm:{llm.name}"

    def verify(self, claim: str, evidence: Evidence) -> GateResult:
        from .prompts import NLI_SYSTEM

        user = f"PREMISE (evidence {evidence.id}):\n{evidence.text}\n\nHYPOTHESIS (claim):\n{claim}"
        out = self.llm.complete_json(NLI_SYSTEM, user, NLIJudgement, context={"task": "nli", "claim": claim, "evidence": evidence})
        return GateResult(evidence_id=evidence.id, label=out.label, score=float(out.confidence), verifier=self.name)


def make_verifier(kind: str, llm=None, hf_model: str = "cross-encoder/nli-deberta-v3-base"):
    if kind in ("auto", "hf"):
        try:
            return HFNLIVerifier(hf_model)
        except Exception as e:  # ImportError, or the model cannot be downloaded
            if kind == "hf":
                raise
            log.warning("HF NLI verifier unavailable (%s); falling back", type(e).__name__)
    if kind in ("auto", "llm") and llm is not None and getattr(llm, "name", "").startswith("anthropic"):
        return LLMVerifier(llm)
    if kind == "llm":
        raise ValueError("verifier=llm needs a real LLM provider")
    log.warning("using the lexical heuristic verifier (offline stand-in, not a real NLI model)")
    return LexicalVerifier()


def gate_claim(claim: Claim, evidence: dict[str, Evidence], verifier: Verifier, threshold: float = 0.5) -> GatedClaim:
    if not claim.evidence_ids:
        return GatedClaim(claim=claim, passed=False, results=[], reason="cites no evidence")
    results: list[GateResult] = []
    for eid in claim.evidence_ids:
        ev = evidence.get(eid)
        if ev is None:
            results.append(GateResult(evidence_id=eid, label="missing", score=0.0, verifier=verifier.name))
            continue
        results.append(verifier.verify(claim.text, ev))
    if any(r.label == "missing" for r in results):
        return GatedClaim(claim=claim, passed=False, results=results, reason="cites evidence that was never retrieved")
    if any(r.label == "contradiction" and r.score >= threshold for r in results):
        return GatedClaim(claim=claim, passed=False, results=results, reason="contradicted by cited evidence")
    if not any(r.label == "entailment" and r.score >= threshold for r in results):
        return GatedClaim(claim=claim, passed=False, results=results, reason="not entailed by cited evidence")
    return GatedClaim(claim=claim, passed=True, results=results)
