"""Evaluation harness and ablation ladder (M6).

Levels, all evaluated on *test* diseases (never seen by the scorer as
indications, never used for calibration):

  B1  popularity baseline                (no learning)
  B2  gene-overlap baseline              (CbG-DaG path count, no learning)
  L1a Stage 1  R-GCN alone
  L1  Stage 1  ensemble (R-GCN + popularity + gene overlap, weights fitted on validation)
  L2  + Stage 2 structured filters       (pruning; then shortlist chosen by audit score)
  L3  + Stage 3 adversarial adjudication (shortlist re-ranked by Referee verdict)
  L4  + Stage 4 arbitration & calibration (shortlist re-ranked by calibrated probability)

L3/L4 only reorder the top-``shortlist_k`` block, so their gains show up in
top-heavy metrics (MAP, recall@5/10, nDCG@10), not in recall@20+.

Calibration is measured on test-disease shortlist candidates for naive
sigmoid(Stage-1 logit), the raw Referee score, the Stage-1/2 calibrator and
the full calibrator. If an oracle truth file is supplied (synthetic data
only) precision against the *complete* ground truth is reported too, which
also credits correct "undiscovered" repurposing hits.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.kg import COMPOUND
from ..pipeline import TriagePipeline, labelled_shortlists
from ..scorer.baselines import GeneOverlapBaseline, PopularityBaseline
from ..utils import ensure_dir, get_logger, write_json
from .metrics import auroc, precision_at_k, rank_order, ranking_report
from ..arbitration.calibrate import calibration_report, plot_reliability

log = get_logger(__name__)
KS = (5, 10, 20, 50, 100)


def _reorder_block(base: np.ndarray, block_local: list[int], keys: list[float]) -> np.ndarray:
    """Put ``block_local`` at the top ordered by ``keys`` (desc); keep the rest in base order."""
    s = base.astype(float).copy()
    top = np.nanmax(s[np.isfinite(s)]) + 1.0 if np.isfinite(s).any() else 1.0
    order = sorted(range(len(block_local)), key=lambda i: -keys[i])
    for rank, i in enumerate(order):
        s[block_local[i]] = top + (len(order) - rank)
    return s


def run_evaluation(pipe: TriagePipeline, cfg: dict, out_dir: str | Path, truth_file: str | Path | None = None) -> dict:
    out = ensure_dir(out_dir)
    kg, split = pipe.kg, pipe.split
    ks = tuple(cfg.get("eval", {}).get("ks") or KS)
    ks = tuple(sorted(set(ks) | {5, 10}))
    diseases = split.diseases("test")
    cap = cfg.get("eval", {}).get("max_eval_diseases")
    if cap:
        diseases = diseases[:cap]
    compounds = kg.members(COMPOUND)
    pos_local = {int(c): i for i, c in enumerate(compounds)}
    test = split.test_pos
    positives = {int(d): {pos_local[int(c)] for c in test[test[:, 1] == d, 0]} for d in diseases}
    exclude = {int(d): {pos_local[int(c)] for c in split.train_pos[split.train_pos[:, 1] == d, 0]} for d in diseases}
    truth = None
    if truth_file and Path(truth_file).exists():
        t = pd.read_csv(truth_file, sep="\t")
        truth = {(kg.idx(c), kg.idx(d)) for c, d in t.itertuples(index=False) if c in kg.id_to_idx and d in kg.id_to_idx}

    prune_only = copy.copy(pipe.filter)
    prune_only.selector, prune_only.shortlist_by = None, "stage1"
    pop, overlap = PopularityBaseline(kg, split), GeneOverlapBaseline(kg)
    levels: dict[str, dict[int, np.ndarray]] = {k: {} for k in
        ["B1 popularity", "B2 gene overlap", "L1a stage1 R-GCN only", "L1 stage1 (ensemble)", "L2 +filters (prune)", "L2 +filters (prune+audit)",
         "L3 +adjudication", "L4 +arbitration & calibration"]}
    use_agents = pipe.adjudicator is not None
    cal_rows = []
    for d in diseases:
        d = int(d)
        s1 = pipe.scorer.scores(d)
        levels["B1 popularity"][d] = pop.scores(d)
        levels["B2 gene overlap"][d] = overlap.scores(d)
        if hasattr(pipe.scorer, "rgcn"):
            levels["L1a stage1 R-GCN only"][d] = pipe.scorer.rgcn.scores(d)
            levels["L1 stage1 (ensemble)"][d] = s1
        else:
            levels["L1a stage1 R-GCN only"][d] = s1
        levels["L2 +filters (prune)"][d] = prune_only.rerank_scores(d, s1)
        l2 = pipe.filter.rerank_scores(d, s1)
        levels["L2 +filters (prune+audit)"][d] = l2
        fres, briefs, ledger, X, _ = pipe.shortlist(d, use_agents)
        block = [pos_local[c.compound] for c in fres.shortlist]
        if use_agents and len(X):
            levels["L3 +adjudication"][d] = _reorder_block(l2, block, list(X["referee_overall"] + 1e-6 * -X["log_stage1_rank"]))
            if pipe.cal_full is not None:
                p_full = pipe.cal_full.predict(X)
                levels["L4 +arbitration & calibration"][d] = _reorder_block(l2, block, list(p_full))
        X = X.copy()
        X["label"] = [int(pos_local[c.compound] in positives[d]) for c in fres.shortlist]
        X["oracle"] = [int((c.compound, d) in truth) for c in fres.shortlist] if truth is not None else np.nan
        X["disease"] = d
        cal_rows.append(X)

    report: dict = {"n_test_diseases": int(len(diseases)), "split": split.summary(), "llm": None, "verifier": None, "levels": {}}
    if use_agents:
        report["llm"], report["verifier"] = pipe.adjudicator.llm.name, pipe.adjudicator.verifier.name
    for name, scores in levels.items():
        if not scores:
            continue
        rep = ranking_report(scores, positives, ks, exclude)
        if truth is not None:
            p10 = []
            for d, s in scores.items():
                sc = s.astype(float).copy()
                sc[list(exclude[d])] = -np.inf
                order = rank_order(sc)
                oracle_pos = {pos_local[c] for c in compounds if (int(c), d) in truth} - exclude[d]
                p10.append(precision_at_k(order, oracle_pos, 10))
            rep["oracle_precision@10"] = float(np.mean(p10))
        report["levels"][name] = rep

    # ---------------------------------------------------- calibration
    C = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    cal: dict[str, dict] = {}
    if len(C):
        y = C["label"].to_numpy()
        raw = C["stage1_logit"].to_numpy()
        naive = raw if (raw.min() >= 0 and raw.max() <= 1) else 1 / (1 + np.exp(-raw))
        cands = {"raw stage-1 score": naive}
        if "referee_overall" in C:
            cands["raw referee score"] = C["referee_overall"].to_numpy()
        if pipe.cal_s12 is not None:
            cands["calibrator (stage 1-2)"] = pipe.cal_s12.predict(C)
        if pipe.cal_full is not None and "referee_overall" in C:
            cands["calibrator (full)"] = pipe.cal_full.predict(C)
        for name, p in cands.items():
            r = calibration_report(p, y)
            r["auroc"] = auroc(p, y)
            cal[name] = r
        plot_reliability(cal, out / "reliability.png", "Test-disease shortlist calibration")
        final = cands.get("calibrator (full)", cands.get("calibrator (stage 1-2)"))
        if final is not None:
            C["p_final"] = final
            per_d = C.groupby("disease")
            sweep = []
            for thr in sorted({0.2, 0.3, 0.4, 0.5, pipe.threshold}):
                returned = C[C["p_final"] >= thr]
                sweep.append({
                    "threshold": thr,
                    "empty_sets": int(sum((g["p_final"] < thr).all() for _, g in per_d)),
                    "n_recommended": int(len(returned)),
                    "precision_known": float(returned["label"].mean()) if len(returned) else float("nan"),
                    "precision_oracle": float(returned["oracle"].mean()) if (truth is not None and len(returned)) else float("nan"),
                })
            report["abstention"] = {
                "configured_threshold": pipe.threshold, "n_diseases": int(C["disease"].nunique()),
                "shortlist_base_rate_known": float(C["label"].mean()),
                "shortlist_base_rate_oracle": float(C["oracle"].mean()) if truth is not None else None,
                "sweep": sweep,
            }
        C.to_csv(out / "test_shortlist_features.tsv", sep="\t", index=False)
    report["calibration"] = {k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in cal.items()}
    write_json(out / "eval_report.json", report)
    (out / "eval_report.md").write_text(to_markdown(report))
    return report


def to_markdown(report: dict) -> str:
    lines = ["# Evaluation report", ""]
    lines.append(f"Test diseases: {report['n_test_diseases']}  |  split: {report['split']['strategy']}  |  "
                 f"LLM: {report.get('llm') or 'agents off'}  |  verifier: {report.get('verifier') or '-'}")
    if report.get("llm") == "offline-heuristic":
        lines += ["", "> Stage 3 used the **offline heuristic** stand-in, not a real LLM. L3/L4 numbers measure the",
                  "> pipeline's evidence handling, not LLM reasoning. Re-run with `agents.llm.provider=anthropic`."]
    lines += ["", "## Ranking (macro-averaged over test diseases)", ""]
    levels = report["levels"]
    if levels:
        cols = [c for c in next(iter(levels.values())).keys() if c != "n_queries"]
        lines.append("| level | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for name, rep in levels.items():
            lines.append(f"| {name} | " + " | ".join(f"{rep.get(c, float('nan')):.3f}" for c in cols) + " |")
    if report.get("calibration"):
        lines += ["", "## Calibration (test-disease shortlist candidates)", "", "| scorer | n | positives | ECE | Brier | AUROC |", "|---|---|---|---|---|---|"]
        for name, r in report["calibration"].items():
            lines.append(f"| {name} | {r['n']} | {r['positives']} | {r['ece']:.3f} | {r['brier']:.3f} | {r['auroc']:.3f} |")
    if report.get("abstention"):
        a = report["abstention"]
        lines += ["", "## Abstention: what is returned at each confidence threshold", "",
                  f"Shortlist base rate: {a['shortlist_base_rate_known']:.3f} (known labels)"
                  + (f", {a['shortlist_base_rate_oracle']:.3f} (oracle)" if a.get("shortlist_base_rate_oracle") is not None else "")
                  + f". Configured threshold: {a['configured_threshold']}. Diseases: {a['n_diseases']}.", "",
                  "| threshold | diseases with empty set | recommended | precision (known) | precision (oracle) |", "|---|---|---|---|---|"]
        for r in a["sweep"]:
            lines.append(f"| {r['threshold']} | {r['empty_sets']} | {r['n_recommended']} | {r['precision_known']:.3f} | {r['precision_oracle']:.3f} |")
    lines += ["", "Positives = held-out indications; everything else in the shortlist counts as negative, so known-label",
              "precision is a lower bound. `oracle_*` columns (synthetic data only) score against complete ground truth."]
    return "\n".join(lines) + "\n"
