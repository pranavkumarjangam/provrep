"""The full cost-tiered cascade: Stage 1 -> 2 -> 3 -> 4 for one disease query."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .agents import Adjudicator, KnowledgeGraphTool, PubMedTool, make_llm, make_verifier
from .agents.schemas import Brief
from .arbitration import FULL_FEATURES, STAGE12_FEATURES, Calibrator, EvidenceLedger
from .data.kg import DISEASE, KnowledgeGraph
from .data.splits import Split
from .filters import Candidate, StructuredFilter, stage12_features
from .filters.rules import PrunedCandidate
from .scorer import WideScorer
from .utils import get_logger

log = get_logger(__name__)


class RankedCandidate(BaseModel):
    rank: int
    probability: float | None
    candidate: Candidate
    features: dict[str, float]
    brief: Brief | None = None


class TriageResult(BaseModel):
    id: str
    created_at: str
    disease_id: str
    disease_name: str
    recommendations: list[RankedCandidate]      # calibrated probability >= threshold (may be empty)
    shortlist: list[RankedCandidate]
    pruned: list[PrunedCandidate]
    invalidations: list[dict] = Field(default_factory=list)
    abstained: bool
    threshold: float
    settings: dict
    timings: dict[str, float]


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


class TriagePipeline:
    def __init__(self, kg: KnowledgeGraph, split: Split, scorer, cfg: dict,
                 calibrator_full: Calibrator | None = None, calibrator_stage12: Calibrator | None = None):
        self.kg, self.split, self.scorer, self.cfg = kg, split, scorer, cfg
        self.filter = StructuredFilter(kg, split, cfg)
        ac = cfg["agents"]
        self.adjudicator = None
        if ac.get("enabled", True):
            llm = make_llm(cfg)
            verifier = make_verifier(ac["verifier"], llm, ac.get("hf_nli_model", "cross-encoder/nli-deberta-v3-base"))
            pm = ac.get("pubmed", {})
            pubmed = PubMedTool(pm.get("max_results", 5), pm.get("email"), pm.get("api_key")) if pm.get("enabled") else None
            tool = KnowledgeGraphTool(kg, self.filter.profiler, ac["max_paths_per_metapath"], cfg["arbitration"]["hub_degree_quantile"])
            self.adjudicator = Adjudicator(tool, llm, verifier, cfg, pubmed)
        self.cal_full, self.cal_s12 = calibrator_full, calibrator_stage12
        self.filter.selector = calibrator_stage12          # audit-based shortlist selection
        self.threshold = cfg["arbitration"]["confidence_threshold"]

    # --------------------------------------------------------------- load
    @classmethod
    def from_artifacts(cls, cfg: dict) -> "TriagePipeline":
        work = Path(cfg["paths"]["work_dir"])
        kg = KnowledgeGraph.load(work / "kg")
        split = Split.load(work / "split")
        import torch

        device = "cuda" if cfg["scorer"].get("device") == "cuda" and torch.cuda.is_available() else "cpu"
        scorer = WideScorer.load(work / "scorer", kg, split, device)
        if cfg["scorer"].get("ensemble", {}).get("enabled", True):
            from .scorer.ensemble import EnsembleScorer

            scorer = EnsembleScorer.maybe_load(work / "scorer", scorer, kg, split) or scorer
        full = Calibrator.load(work / "calibrator_full.joblib") if (work / "calibrator_full.joblib").exists() else None
        s12 = Calibrator.load(work / "calibrator_stage12.joblib") if (work / "calibrator_stage12.joblib").exists() else None
        return cls(kg, split, scorer, cfg, full, s12)

    def resolve_disease(self, query: str) -> int:
        if query in self.kg.id_to_idx:
            return self.kg.idx(query)
        q = query.lower().strip()
        for i in self.kg.members(DISEASE):
            if self.kg.name(int(i)).lower() == q or self.kg.node_ids[int(i)].lower().endswith(q):
                return int(i)
        raise KeyError(f"unknown disease {query!r}")

    # --------------------------------------------------------------- run
    def shortlist(self, disease: int, use_agents: bool = True):
        t = {}
        t0 = time.perf_counter()
        scores = self.scorer.scores(disease)
        t["stage1_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fres = self.filter.run(disease, scores)
        t["stage2_s"] = time.perf_counter() - t0
        briefs: dict[str, Brief] = {}
        ledger = None
        if use_agents and self.adjudicator is not None:
            t0 = time.perf_counter()
            for c in fres.shortlist:
                briefs[c.compound_id] = self.adjudicator.adjudicate(c)
            t["stage3_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            ledger = EvidenceLedger(self.cfg["arbitration"]["hub_evidence_weight"])
            for b in briefs.values():
                ledger.add(b)
            ledger.arbitrate()
            t["stage4_s"] = time.perf_counter() - t0
        rows = []
        for c in fres.shortlist:
            f = stage12_features(c)
            f["audit_logit"] = _logit(c.audit_score) if c.audit_score is not None else c.stage1_logit
            if ledger is not None:
                f.update(ledger.features(c.compound_id))
            rows.append(f)
        return fres, briefs, ledger, pd.DataFrame(rows), t

    def triage(self, query: str | int, use_agents: bool = True) -> TriageResult:
        disease = query if isinstance(query, int) else self.resolve_disease(query)
        fres, briefs, ledger, X, t = self.shortlist(disease, use_agents)
        cal = self.cal_full if (ledger is not None and self.cal_full is not None) else self.cal_s12
        probs = cal.predict(X) if (cal is not None and len(X)) else np.full(len(X), np.nan)
        order = np.argsort(-np.nan_to_num(probs, nan=-1.0), kind="stable") if cal is not None else np.arange(len(X))
        ranked = []
        for r, i in enumerate(order, start=1):
            c = fres.shortlist[int(i)]
            p = None if np.isnan(probs[i]) else float(probs[i])
            ranked.append(RankedCandidate(rank=r, probability=p, candidate=c, features=X.iloc[int(i)].to_dict(), brief=briefs.get(c.compound_id)))
        recs = [rc for rc in ranked if rc.probability is not None and rc.probability >= self.threshold]
        ac = self.cfg["agents"]
        settings = {
            "agents": bool(ledger is not None),
            "llm": self.adjudicator.llm.name if self.adjudicator and ledger is not None else None,
            "verifier": self.adjudicator.verifier.name if self.adjudicator and ledger is not None else None,
            "calibrator": "full" if cal is self.cal_full and cal is not None else ("stage1-2" if cal is not None else None),
            "shortlist_k": self.filter.shortlist_k, "stage1_top_k": self.filter.top_k,
            "pubmed": bool(ac.get("pubmed", {}).get("enabled")),
        }
        return TriageResult(
            id=uuid.uuid4().hex[:12], created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            disease_id=fres.disease_id, disease_name=fres.disease_name, recommendations=recs, shortlist=ranked,
            pruned=fres.pruned, invalidations=ledger.report() if ledger else [], abstained=(cal is not None and not recs),
            threshold=self.threshold, settings=settings, timings={k: round(v, 4) for k, v in t.items()},
        )


def labelled_shortlists(pipe: TriagePipeline, diseases: np.ndarray, positives: np.ndarray, use_agents: bool,
                        max_diseases: int | None = None) -> pd.DataFrame:
    """Features + labels for every shortlisted candidate of the given diseases."""
    pos = {(int(c), int(d)) for c, d in positives}
    frames = []
    for d in (diseases[:max_diseases] if max_diseases else diseases):
        fres, briefs, ledger, X, _ = pipe.shortlist(int(d), use_agents)
        if not len(X):
            continue
        X = X.copy()
        X["disease"] = int(d)
        X["compound"] = [c.compound for c in fres.shortlist]
        X["label"] = [int((c.compound, int(d)) in pos) for c in fres.shortlist]
        frames.append(X)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def labelled_survivors(pipe: TriagePipeline, diseases: np.ndarray, positives: np.ndarray,
                       max_diseases: int | None = None) -> pd.DataFrame:
    """Stage-1/2 features + labels for *every* unpruned top-k candidate (no agents)."""
    pos = {(int(c), int(d)) for c, d in positives}
    frames = []
    for d in (diseases[:max_diseases] if max_diseases else diseases):
        fres = pipe.filter.run(int(d), pipe.scorer.scores(int(d)))
        if not fres.survivors:
            continue
        X = pd.DataFrame([stage12_features(c) for c in fres.survivors])
        X["disease"] = int(d)
        X["compound"] = [c.compound for c in fres.survivors]
        X["label"] = [int((c.compound, int(d)) in pos) for c in fres.survivors]
        frames.append(X)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_calibrators(pipe: TriagePipeline, out_dir: str | Path, max_diseases: int | None = None) -> dict:
    """Fit calibrators on validation diseases.

    1. Stage-1/2 calibrator on every unpruned top-k candidate. It doubles as the
       shortlist selector: Stage 2 then picks the shortlist by audit score
       (Stage-1 score + mechanistic path evidence) instead of raw Stage-1 rank.
    2. Full calibrator on the (audit-selected) shortlist, with Stage-3/4 features.
    """
    out = Path(out_dir)
    report = {}
    val_d, val_pos = pipe.split.diseases("val"), pipe.split.val_pos
    pipe.filter.selector = None
    s12 = labelled_survivors(pipe, val_d, val_pos, max_diseases)
    cal_s12 = Calibrator(STAGE12_FEATURES).fit(s12, s12["label"].to_numpy(), s12["disease"].to_numpy())
    cal_s12.save(out / "calibrator_stage12.joblib")
    pipe.cal_s12 = cal_s12
    pipe.filter.selector = cal_s12
    report["stage12"] = {"n": len(s12), "positives": int(s12["label"].sum()), "C": cal_s12.C,
                         "cv_logloss": cal_s12.cv_scores, "coef": cal_s12.coefficients()}
    if pipe.adjudicator is not None:
        full = labelled_shortlists(pipe, val_d, val_pos, use_agents=True, max_diseases=max_diseases)
        cal_full = Calibrator(FULL_FEATURES).fit(full, full["label"].to_numpy(), full["disease"].to_numpy())
        cal_full.save(out / "calibrator_full.joblib")
        pipe.cal_full = cal_full
        report["full"] = {"n": len(full), "positives": int(full["label"].sum()), "C": cal_full.C,
                          "cv_logloss": cal_full.cv_scores, "coef": cal_full.coefficients()}
    log.info("calibrators fitted: %s", {k: (v["n"], v["positives"]) for k, v in report.items()})
    return report
