"""Stage 4a: cross-candidate arbitration over a shared evidence ledger (M5).

Independent per-candidate briefs over-count correlated support: biomedical
evidence is heavily shared (the same target gene, the same hub, the same
paper), so twenty "independent" briefs can all rest on one weak fact.

The ledger records which verified evidence supports which candidate and
applies invalidations *across the whole shortlist*:

* node invalidation - a verified Falsifier claim that a gene is a promiscuous
  hub down-weights every support path through that gene, for every candidate
* evidence invalidation - evidence behind a claim the Referee rejected is
  removed for every candidate that cites it

Candidates are then rescored by *effective independent support*: paths are
grouped by the target they act through (correlated within a group -> count
the best one), and a group shared by k shortlisted candidates contributes 1/k
to each.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..agents.schemas import Brief, Evidence
from ..filters.metapaths import MECHANISTIC, SIMILARITY

KIND_WEIGHT = {**{m: 1.0 for m in MECHANISTIC}, **{m: 0.3 for m in SIMILARITY}, "pubmed_abstract": 0.6}
REFUTE_WEIGHT = {"no_mechanistic_path": 1.0, "similarity_driven": 0.6, "hub_target": 0.3, "off_target": 0.3,
                 "side_effect_burden": 0.3, "pubmed_abstract": 0.5}


@dataclass
class Invalidation:
    target: str            # node id or evidence id
    scope: str             # "node" | "evidence"
    reason: str
    factor: float          # multiplier applied to affected evidence (0 = removed)
    raised_by: str         # "compound_id:claim_id"
    affected: list[str] = field(default_factory=list)   # compound ids whose support changed


class EvidenceLedger:
    def __init__(self, hub_weight: float = 0.5, sharing: str = "inverse"):
        self.hub_weight, self.sharing = hub_weight, sharing
        self.briefs: dict[str, Brief] = {}
        self.node_factor: dict[str, float] = {}
        self.evidence_factor: dict[str, float] = {}
        self.events: list[Invalidation] = []

    def add(self, brief: Brief) -> None:
        self.briefs[brief.candidate.compound_id] = brief

    # ------------------------------------------------------ invalidation
    def invalidate_node(self, node_id: str, reason: str, factor: float, raised_by: str) -> Invalidation:
        self.node_factor[node_id] = min(self.node_factor.get(node_id, 1.0), factor)
        ev = Invalidation(node_id, "node", reason, factor, raised_by)
        ev.affected = [cid for cid, b in self.briefs.items() if any(node_id in e.node_ids[1:-1] for e in self._support_evidence(b))]
        self.events.append(ev)
        return ev

    def invalidate_evidence(self, evidence_id: str, reason: str, raised_by: str) -> Invalidation:
        self.evidence_factor[evidence_id] = 0.0
        ev = Invalidation(evidence_id, "evidence", reason, 0.0, raised_by)
        ev.affected = [cid for cid, b in self.briefs.items() if any(e.id == evidence_id for e in self._support_evidence(b))]
        self.events.append(ev)
        return ev

    def arbitrate(self) -> None:
        """Apply the automatic cross-candidate invalidations."""
        for cid, b in self.briefs.items():
            ev = b.evidence_by_id()
            for g in b.passed("refutes"):
                for eid in g.claim.evidence_ids:
                    e = ev.get(eid)
                    if e and e.kind == "hub_target":
                        for node in e.node_ids:
                            if self.node_factor.get(node, 1.0) > self.hub_weight:
                                self.invalidate_node(node, "verified hub-gene claim", self.hub_weight, f"{cid}:{g.claim.id}")
            rejected = set(b.verdict.rejected_claim_ids)
            for g in b.claims:
                if g.claim.id in rejected and g.claim.stance == "supports":
                    for eid in g.claim.evidence_ids:
                        if self.evidence_factor.get(eid, 1.0) > 0:
                            self.invalidate_evidence(eid, "referee rejected the claim citing it", f"{cid}:{g.claim.id}")

    # ---------------------------------------------------------- scoring
    @staticmethod
    def _support_evidence(b: Brief) -> list[Evidence]:
        ev = b.evidence_by_id()
        return [ev[i] for g in b.passed("supports") for i in g.claim.evidence_ids if i in ev]

    def _weight(self, e: Evidence) -> float:
        w = KIND_WEIGHT.get(e.kind, 0.5) * self.evidence_factor.get(e.id, 1.0)
        for n in e.node_ids[1:-1]:
            w *= self.node_factor.get(n, 1.0)
        return w

    @staticmethod
    def _group(e: Evidence) -> str:
        # correlated support = same first hop (the target a KG path acts through), or same paper
        if e.source == "kg" and len(e.node_ids) > 2:
            return f"{e.kind in SIMILARITY and 'sim' or 'mech'}:{e.node_ids[1]}"
        return e.id

    def groups(self, b: Brief) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self._support_evidence(b):
            k = self._group(e)
            out[k] = max(out.get(k, 0.0), self._weight(e))
        return out

    def usage(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for b in self.briefs.values():
            for k, w in self.groups(b).items():
                if w > 0:
                    counts[k] += 1
        return counts

    def effective_support(self, compound_id: str) -> tuple[float, float]:
        """(effective independent support, raw support) for one candidate."""
        b = self.briefs[compound_id]
        use = self.usage()
        groups = self.groups(b)
        raw = sum(groups.values())
        eff = 0.0
        for k, w in groups.items():
            n = max(1, use.get(k, 1))
            eff += w / (n if self.sharing == "inverse" else n ** 0.5)
        return eff, raw

    def refutation(self, compound_id: str) -> float:
        b = self.briefs[compound_id]
        ev = b.evidence_by_id()
        return float(sum(REFUTE_WEIGHT.get(ev[i].kind, 0.3) for g in b.passed("refutes") for i in g.claim.evidence_ids if i in ev))

    def features(self, compound_id: str) -> dict[str, float]:
        b = self.briefs[compound_id]
        eff, raw = self.effective_support(compound_id)
        v = b.verdict
        return {
            "eff_support": eff, "raw_support": raw, "refutation": self.refutation(compound_id),
            "referee_overall": v.overall,
            "mechanistic_plausibility": v.scores.mechanistic_plausibility,
            "evidence_quality": v.scores.evidence_quality,
            "safety_concern": v.scores.safety_concern,
            "counter_evidence_strength": v.scores.counter_evidence_strength,
            "n_supports_passed": float(len(b.passed("supports"))),
            "n_refutes_passed": float(len(b.passed("refutes"))),
            "n_claims_failed": float(sum(1 for g in b.claims if not g.passed)),
        }

    def report(self) -> list[dict]:
        return [e.__dict__ for e in self.events]
