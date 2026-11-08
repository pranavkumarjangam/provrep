"""Typed objects passed between the adjudication agents (M4).

Provenance is structural: a Claim can only point at Evidence by id, Evidence
is only ever created by retrieval tools (never by an LLM), and the Referee's
output schema has no field in which it could introduce a claim or evidence.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..filters.rules import Candidate

Role = Literal["proponent", "falsifier"]


class Evidence(BaseModel):
    id: str                               # stable, content-derived (kg:... / pmid:...)
    source: Literal["kg", "pubmed"]
    kind: str                             # metapath name, "pubmed_abstract", "hub_target", ...
    text: str                             # what the evidence states, in plain words
    entities: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)   # KG nodes the evidence routes through
    weight: float = 1.0                   # retrieval-time specificity (e.g. path DWPC)
    retrieved_by: Role
    url: str | None = None


class Claim(BaseModel):
    id: str
    role: Role
    stance: Literal["supports", "refutes"]
    text: str
    evidence_ids: list[str]


class ClaimDraft(BaseModel):
    """What an LLM is allowed to produce: text plus citations to existing evidence."""
    text: str
    evidence_ids: list[str]


class ClaimDrafts(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)


class GateResult(BaseModel):
    evidence_id: str
    label: Literal["entailment", "neutral", "contradiction", "missing"]
    score: float
    verifier: str


class GatedClaim(BaseModel):
    claim: Claim
    passed: bool
    results: list[GateResult]
    reason: str = ""


class RubricScores(BaseModel):
    mechanistic_plausibility: float = Field(ge=0, le=1)
    evidence_quality: float = Field(ge=0, le=1)
    safety_concern: float = Field(ge=0, le=1)          # higher = more concerning
    counter_evidence_strength: float = Field(ge=0, le=1)


class RefereeOutput(BaseModel):
    """The Referee's entire allowed output: scores and references to claim ids."""
    scores: RubricScores
    overall: float = Field(ge=0, le=1)
    tier: Literal["strong", "moderate", "weak", "reject"]
    rationale: str
    cited_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    candidate: Candidate
    evidence: list[Evidence]
    claims: list[GatedClaim]
    verdict: RefereeOutput
    llm: str
    verifier: str
    notes: list[str] = Field(default_factory=list)

    def passed(self, stance: str | None = None) -> list[GatedClaim]:
        return [g for g in self.claims if g.passed and (stance is None or g.claim.stance == stance)]

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}
