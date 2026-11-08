"""Stage 3: adversarial adjudication as a LangGraph state machine (M4).

    retrieve_support -> proponent -> gate_support
        -> retrieve_counter -> falsifier -> gate_counter -> referee -> END

* the Proponent and Falsifier have separate retrieval budgets
* the Falsifier's retrieval excludes everything the Proponent retrieved
* every claim is gated by the NLI verifier against its cited evidence
* the Referee sees only gated claims and can only score and reference them;
  any claim id it cites that does not exist is dropped and noted
"""
from __future__ import annotations

import itertools
import json
from typing import Any, TypedDict

from ..filters.rules import Candidate
from ..utils import get_logger
from . import prompts
from .llm import LLMClient, dumps_evidence
from .schemas import Brief, Claim, ClaimDrafts, Evidence, GatedClaim, RefereeOutput
from .tools import Budget, KnowledgeGraphTool, PubMedTool
from .verifier import Verifier, gate_claim

log = get_logger(__name__)


class AdjudicationState(TypedDict, total=False):
    candidate: Candidate
    evidence: dict[str, Evidence]
    support_ids: list[str]
    counter_ids: list[str]
    claims: list[GatedClaim]
    verdict: RefereeOutput
    notes: list[str]


class Adjudicator:
    def __init__(self, kg_tool: KnowledgeGraphTool, llm: LLMClient, verifier: Verifier, cfg: dict,
                 pubmed: PubMedTool | None = None):
        ac = cfg["agents"]
        self.kg_tool, self.llm, self.verifier, self.pubmed = kg_tool, llm, verifier, pubmed
        self.pro_budget, self.fal_budget = ac["proponent_budget"], ac["falsifier_budget"]
        self.graph = self._build_graph()

    # ------------------------------------------------------------- nodes
    def _retrieve_support(self, s: AdjudicationState) -> dict:
        cand, budget = s["candidate"], Budget(self.pro_budget)
        ev = self.kg_tool.supporting(cand, budget)
        if self.pubmed:
            ev += self.pubmed.search(PubMedTool.support_query(cand.compound_name, cand.disease_name), "proponent", budget)
        evidence = {e.id: e for e in ev}
        return {"evidence": evidence, "support_ids": list(evidence), "notes": s.get("notes", []) + [f"proponent retrieved {len(ev)}/{self.pro_budget}"]}

    def _proponent(self, s: AdjudicationState) -> dict:
        cand, ev = s["candidate"], [s["evidence"][i] for i in s["support_ids"]]
        if not ev:
            return {"claims": [], "notes": s["notes"] + ["proponent had no evidence"]}
        user = (f"Candidate: {cand.compound_name} for {cand.disease_name}.\n"
                f"Stage-1 rank {cand.stage1_rank}; mechanistic share of path support {cand.mechanistic_share:.0%}.\n"
                f"EVIDENCE (cite by id):\n{dumps_evidence(ev)}")
        drafts = self.llm.complete_json(prompts.PROPONENT_SYSTEM, user, ClaimDrafts, context={"task": "proponent", "evidence": ev})
        return {"claims": self._to_claims(drafts, "proponent", "supports", 0)}

    def _gate(self, s: AdjudicationState, role: str) -> dict:
        claims = [self._gate_one(g, s["evidence"]) if g.claim.role == role and not g.results and g.reason == "" else g for g in s["claims"]]
        n = sum(1 for g in claims if g.claim.role == role)
        ok = sum(1 for g in claims if g.claim.role == role and g.passed)
        return {"claims": claims, "notes": s["notes"] + [f"{role}: {ok}/{n} claims passed the NLI gate"]}

    def _gate_one(self, g: GatedClaim, evidence: dict[str, Evidence]) -> GatedClaim:
        # The gate only accepts citations to evidence this claim's own agent retrieved.
        own = {k: v for k, v in evidence.items() if v.retrieved_by == g.claim.role}
        return gate_claim(g.claim, own, self.verifier)

    def _retrieve_counter(self, s: AdjudicationState) -> dict:
        cand, budget = s["candidate"], Budget(self.fal_budget)
        exclude = set(s["support_ids"])
        ev = self.kg_tool.counter(cand, budget, exclude)
        if self.pubmed:
            ev += self.pubmed.search(PubMedTool.counter_query(cand.compound_name, cand.disease_name), "falsifier", budget, exclude)
        evidence = dict(s["evidence"])
        evidence.update({e.id: e for e in ev})
        return {"evidence": evidence, "counter_ids": [e.id for e in ev], "notes": s["notes"] + [f"falsifier retrieved {len(ev)}/{self.fal_budget}"]}

    def _falsifier(self, s: AdjudicationState) -> dict:
        cand, ev = s["candidate"], [s["evidence"][i] for i in s["counter_ids"]]
        if not ev:
            return {"notes": s["notes"] + ["falsifier found no counter-evidence"]}
        pro = [{"id": g.claim.id, "text": g.claim.text} for g in s["claims"] if g.passed]
        user = (f"Candidate: {cand.compound_name} for {cand.disease_name}.\n"
                f"PROPONENT CLAIMS (verified):\n{json.dumps(pro, indent=1)}\n"
                f"YOUR EVIDENCE (cite by id):\n{dumps_evidence(ev)}")
        drafts = self.llm.complete_json(prompts.FALSIFIER_SYSTEM, user, ClaimDrafts, context={"task": "falsifier", "evidence": ev})
        start = len(s["claims"])
        return {"claims": s["claims"] + self._to_claims(drafts, "falsifier", "refutes", start)}

    def _referee(self, s: AdjudicationState) -> dict:
        cand = s["candidate"]
        passed = [g for g in s["claims"] if g.passed]
        payload = [{"id": g.claim.id, "role": g.claim.role, "stance": g.claim.stance, "text": g.claim.text} for g in passed]
        user = (f"Candidate: {cand.compound_name} for {cand.disease_name}. Stage-2 audit: "
                f"mechanistic share {cand.mechanistic_share:.0%}, degenerate={cand.degenerate}, "
                f"{cand.n_side_effects} side effects.\nVERIFIED CLAIMS:\n{json.dumps(payload, indent=1)}")
        out: RefereeOutput = self.llm.complete_json(
            prompts.REFEREE_SYSTEM, user, RefereeOutput,
            context={"task": "referee", "passed_claims": passed, "evidence": s["evidence"], "n_claims_total": len(s["claims"]), "candidate": cand})
        valid = {g.claim.id for g in passed}
        bad = [i for i in out.cited_claim_ids + out.rejected_claim_ids if i not in valid]
        notes = list(s["notes"])
        if bad:
            notes.append(f"referee referenced unknown/unverified claim ids {bad}; dropped")
            out = out.model_copy(update={
                "cited_claim_ids": [i for i in out.cited_claim_ids if i in valid],
                "rejected_claim_ids": [i for i in out.rejected_claim_ids if i in valid]})
        return {"verdict": out, "notes": notes}

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _to_claims(drafts: ClaimDrafts, role: str, stance: str, start: int) -> list[GatedClaim]:
        prefix = "P" if role == "proponent" else "F"
        out = []
        for i, d in zip(itertools.count(start + 1), drafts.claims):
            c = Claim(id=f"{prefix}{i}", role=role, stance=stance, text=d.text, evidence_ids=d.evidence_ids)
            out.append(GatedClaim(claim=c, passed=False, results=[]))
        return out

    def _build_graph(self):
        steps = [
            ("retrieve_support", self._retrieve_support),
            ("proponent", self._proponent),
            ("gate_support", lambda s: self._gate(s, "proponent")),
            ("retrieve_counter", self._retrieve_counter),
            ("falsifier", self._falsifier),
            ("gate_counter", lambda s: self._gate(s, "falsifier")),
            ("referee", self._referee),
        ]
        self._steps = steps
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:  # pragma: no cover
            log.warning("langgraph not installed; running the adjudication steps sequentially")
            return None
        g = StateGraph(AdjudicationState)
        for name, fn in steps:
            g.add_node(name, fn)
        g.set_entry_point(steps[0][0])
        for (a, _), (b, _) in zip(steps, steps[1:]):
            g.add_edge(a, b)
        g.add_edge(steps[-1][0], END)
        return g.compile()

    def adjudicate(self, cand: Candidate) -> Brief:
        state: dict[str, Any] = {"candidate": cand, "evidence": {}, "support_ids": [], "counter_ids": [], "claims": [], "notes": []}
        if self.graph is not None:
            state = self.graph.invoke(state)
        else:  # pragma: no cover
            for _, fn in self._steps:
                state.update(fn(state))
        return Brief(candidate=cand, evidence=list(state["evidence"].values()), claims=state["claims"],
                     verdict=state["verdict"], llm=self.llm.name, verifier=self.verifier.name, notes=state["notes"])
