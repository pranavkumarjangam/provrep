"""Stage 2: deterministic structured filters (M3).

Takes the Stage-1 ranking, prunes on rules that need no model, and annotates
each survivor with an auditable metapath profile. Uses *only* knowledge that
is visible at training time (training indications, palliative edges, training
non-indications) - never the held-out labels.

Rules
-----
already_indicated   compound already has a known (training) indication or
                    palliative use for this disease
known_failure       compound-disease pair is a known non-indication in training
contraindicated     pair appears in an optional contraindications file
side_effect_burden  compound causes more side effects than ``max_side_effects``
delivery            hook for delivery-feasibility rules (e.g. blood-brain
                    barrier permeability for CNS diseases); no public source
                    is wired in yet, so it is a documented no-op.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..data.kg import COMPOUND, KnowledgeGraph
from ..data.splits import Split
from .metapaths import MECHANISTIC, SIMILARITY, MetapathProfiler


class Candidate(BaseModel):
    compound: int
    compound_id: str
    compound_name: str
    disease: int
    disease_id: str
    disease_name: str
    stage1_logit: float
    stage1_rank: int
    dwpc: dict[str, float] = Field(default_factory=dict)
    mechanistic_share: float = 0.0
    degenerate: bool = False            # support is mostly similarity, not mechanism
    n_targets: int = 0
    n_side_effects: int = 0
    targets: list[str] = Field(default_factory=list)
    audit_score: float | None = None    # Stage-1/2 calibrated probability, when a selector is fitted


def stage12_features(c: Candidate) -> dict[str, float]:
    """Model-free audit features available for every Stage-2 survivor."""
    mech = sum(c.dwpc.get(m, 0.0) for m in MECHANISTIC)
    sim = sum(c.dwpc.get(m, 0.0) for m in SIMILARITY)
    return {
        "stage1_logit": c.stage1_logit, "log_stage1_rank": math.log(c.stage1_rank),
        "mechanistic_share": c.mechanistic_share, "log_mech_dwpc": math.log1p(1000 * mech),
        "log_sim_dwpc": math.log1p(1000 * sim), "degenerate": float(c.degenerate),
        "log_side_effects": math.log1p(c.n_side_effects),
    }


class PrunedCandidate(BaseModel):
    compound_id: str
    compound_name: str
    stage1_rank: int
    reason: str


class FilterResult(BaseModel):
    disease_id: str
    disease_name: str
    shortlist: list[Candidate]
    survivors: list[Candidate] = Field(default_factory=list)   # all unpruned top-k, in Stage-2 order
    pruned: list[PrunedCandidate]
    n_scored: int
    n_considered: int


class StructuredFilter:
    def __init__(self, kg: KnowledgeGraph, split: Split, cfg: dict):
        fc = cfg["filters"]
        self.kg, self.cfg = kg, fc
        self.top_k, self.shortlist_k = fc["stage1_top_k"], fc["shortlist_k"]
        self.min_share = fc["degeneracy_min_mechanistic_share"]
        self.demote_degenerate = fc.get("demote_degenerate", False)
        # "audit": choose the shortlist by the Stage-1/2 calibrated probability (needs a fitted
        # selector, see pipeline.fit_calibrators); "stage1": keep the raw Stage-1 order.
        self.shortlist_by = fc.get("shortlist_by", "audit")
        self.selector = None
        self.compounds = kg.members(COMPOUND)
        self.profiler = MetapathProfiler(kg, split.train_pos, fc["dwpc_damping"])

        self.known: dict[int, set[int]] = {}
        for c, d in split.train_pos:
            self.known.setdefault(int(d), set()).add(int(c))
        for c, d in kg.edges.get("CpD", np.empty((0, 2), dtype=np.int64)):
            self.known.setdefault(int(d), set()).add(int(c))
        self.failures: dict[int, set[int]] = {}
        for c, d in split.train_neg:
            self.failures.setdefault(int(d), set()).add(int(c))
        self.contra: dict[int, set[int]] = {}
        if fc.get("contraindications_file"):
            df = pd.read_csv(fc["contraindications_file"], sep="\t")
            for c, d in df[["compound_id", "disease_id"]].itertuples(index=False):
                if c in kg.id_to_idx and d in kg.id_to_idx:
                    self.contra.setdefault(kg.idx(d), set()).add(kg.idx(c))
        cse = kg.edges.get("CcSE")
        self.se_count = np.bincount(kg.local_idx[cse[:, 0]], minlength=len(self.compounds)) if cse is not None and len(cse) else np.zeros(len(self.compounds), int)
        cbg = kg.edges.get("CbG")
        self.targets: dict[int, list[int]] = {}
        if cbg is not None:
            for c, g in cbg:
                self.targets.setdefault(int(c), []).append(int(g))
        self.max_se = fc.get("max_side_effects")

    def _prune_reason(self, compound: int, disease: int, local: int) -> str | None:
        if compound in self.known.get(disease, ()):
            return "already_indicated"
        if compound in self.failures.get(disease, ()):
            return "known_failure"
        if compound in self.contra.get(disease, ()):
            return "contraindicated"
        if self.max_se is not None and self.se_count[local] > self.max_se:
            return "side_effect_burden"
        return self.delivery_rule(compound, disease)

    def delivery_rule(self, compound: int, disease: int) -> str | None:  # noqa: ARG002
        """Override to add delivery-feasibility pruning (returns a reason or None)."""
        return None

    def run(self, disease: int, stage1_logits: np.ndarray) -> FilterResult:
        kg = self.kg
        order = np.argsort(-stage1_logits, kind="stable")
        profile = self.profiler.profile(disease)
        share = self.profiler.mechanistic_share(profile)
        survivors, pruned = [], []
        for rank, local in enumerate(order[: self.top_k].tolist(), start=1):
            c = int(self.compounds[local])
            reason = self._prune_reason(c, disease, local)
            if reason:
                pruned.append(PrunedCandidate(compound_id=kg.node_ids[c], compound_name=kg.name(c), stage1_rank=rank, reason=reason))
                continue
            mech_total = sum(profile[m][local] for m in self.profiler.mechanistic)
            degenerate = bool(share[local] < self.min_share or mech_total == 0)
            tg = self.targets.get(c, [])
            survivors.append(Candidate(
                compound=c, compound_id=kg.node_ids[c], compound_name=kg.name(c),
                disease=int(disease), disease_id=kg.node_ids[disease], disease_name=kg.name(disease),
                stage1_logit=float(stage1_logits[local]), stage1_rank=rank,
                dwpc={m: float(v[local]) for m, v in profile.items()},
                mechanistic_share=float(share[local]), degenerate=degenerate,
                n_targets=len(tg), n_side_effects=int(self.se_count[local]),
                targets=[kg.name(g) for g in tg[:10]],
            ))
        if survivors and self.selector is not None:
            probs = self.selector.predict(pd.DataFrame([stage12_features(c) for c in survivors]))
            for c, p in zip(survivors, probs):
                c.audit_score = float(p)
            if self.shortlist_by == "audit":
                survivors.sort(key=lambda x: (-x.audit_score, x.stage1_rank))
        if self.demote_degenerate:
            survivors.sort(key=lambda x: x.degenerate)          # stable: keeps the order above within groups
        return FilterResult(disease_id=kg.node_ids[disease], disease_name=kg.name(disease),
                            shortlist=survivors[: self.shortlist_k], survivors=survivors, pruned=pruned,
                            n_scored=len(stage1_logits), n_considered=min(self.top_k, len(order)))

    def rerank_scores(self, disease: int, stage1_logits: np.ndarray) -> np.ndarray:
        """Full-length scores reflecting Stage 2 - pruned -> -inf, survivors of the top-k
        placed first in Stage-2 order, everything else after in Stage-1 order.
        Used by the evaluation harness for the ablation ladder."""
        scores = stage1_logits.astype(float).copy()
        res = self.run(disease, stage1_logits)
        pos = {int(c): i for i, c in enumerate(self.compounds)}
        for p in res.pruned:
            scores[pos[self.kg.idx(p.compound_id)]] = -np.inf
        finite = scores[np.isfinite(scores)]
        top = (finite.max() if len(finite) else 0.0) + 1.0
        n = len(res.survivors)
        for i, cand in enumerate(res.survivors):
            scores[pos[cand.compound]] = top + (n - i)
        return scores


def save_filter_result(res: FilterResult, path: str | Path) -> None:
    Path(path).write_text(res.model_dump_json(indent=2))
