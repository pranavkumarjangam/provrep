"""Stage 4b: confidence calibration (M5).

A logistic-regression calibrator maps candidate features (Stage-1 score,
Stage-2 audit, Stage-3 verdict, Stage-4 effective support) to a probability,
fitted on held-out labelled pairs from *validation* diseases and assessed on
*test* diseases with ECE, Brier score and a reliability diagram.

Labels: held-out indications are positives; known non-indications (NOT) and
unlabelled shortlist candidates are negatives. Treating unlabelled pairs as
negative is the standard assumption for Hetionet link prediction, but it
means some "negatives" are undiscovered positives - so calibrated
probabilities are conservative (a lower bound on true precision).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..eval.metrics import brier_score, expected_calibration_error, reliability_curve

STAGE12_FEATURES = ["stage1_logit", "log_stage1_rank", "mechanistic_share", "log_mech_dwpc", "log_sim_dwpc", "degenerate", "log_side_effects"]
STAGE34_FEATURES = ["referee_overall", "mechanistic_plausibility", "evidence_quality", "safety_concern",
                    "counter_evidence_strength", "eff_support", "raw_support", "refutation",
                    "n_supports_passed", "n_refutes_passed", "n_claims_failed"]
# The full calibrator is stacked: the Stage-1/2 calibrator's logit summarises Stages 1-2,
# so the small shortlist calibration set only has to learn how Stages 3-4 adjust it.
FULL_FEATURES = ["audit_logit"] + STAGE34_FEATURES
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


class Calibrator:
    def __init__(self, features: list[str], C: float = 1.0):
        self.features, self.C = features, C
        self.model = None
        self.constant: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None) -> "Calibrator":
        """Fit; if ``groups`` (disease ids) are given, choose the L2 strength C by
        grouped cross-validated log-loss so no disease is in both train and test folds."""
        y = np.asarray(y, int)
        if len(np.unique(y)) < 2:          # degenerate calibration set: predict the base rate
            self.constant = float(y.mean()) if len(y) else 0.0
            return self
        Xa = X[self.features].to_numpy(float)
        if groups is not None and len(np.unique(groups)) >= 3:
            self.C, self.cv_scores = _select_c(Xa, y, np.asarray(groups))
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.model.fit(Xa, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return self.model.predict_proba(X[self.features].to_numpy(float))[:, 1]

    def coefficients(self) -> dict[str, float]:
        if self.model is None:
            return {}
        lr = self.model[-1]
        return {f: float(c) for f, c in zip(self.features, lr.coef_[0])}

    def save(self, path: str | Path) -> None:
        joblib.dump({"features": self.features, "C": self.C, "model": self.model, "constant": self.constant}, path)

    cv_scores: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        d = joblib.load(path)
        c = cls(d["features"], d["C"])
        c.model, c.constant = d["model"], d["constant"]
        return c


def calibration_report(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "n": int(len(labels)), "positives": int(np.sum(labels)),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "reliability": reliability_curve(probs, labels, n_bins),
    }


def plot_reliability(curves: dict[str, dict], path: str | Path, title: str = "Reliability diagram") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="perfect calibration")
    colors = ["#4F81BD", "#E07B24", "#1A7A6E", "#7B5EA7"]
    for (name, rep), col in zip(curves.items(), colors):
        rc = rep["reliability"]
        pts = [(c, a) for c, a, n in zip(rc["confidence"], rc["accuracy"], rc["count"]) if n > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=col, lw=1.8, label=f"{name} (ECE {rep['ece']:.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _select_c(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold

    k = min(5, len(np.unique(groups)))
    scores = {}
    for C in C_GRID:
        losses = []
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
            losses.append(log_loss(y[te], m.predict_proba(X[te])[:, 1], labels=[0, 1]))
        scores[C] = float(np.mean(losses)) if losses else float("inf")
    best = min(scores, key=scores.get)
    return best, scores
