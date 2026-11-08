"""Ranking and calibration metrics (M6).

Ranking metrics are computed per query disease over the full compound list,
with training positives excluded from the ranking (the "filtered" setting
standard in link prediction), then macro-averaged over diseases.
"""
from __future__ import annotations

import numpy as np


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Indices sorted by descending score (stable, so ties keep input order)."""
    return np.argsort(-scores, kind="stable")


def recall_at_k(order: np.ndarray, positives: set[int], k: int) -> float:
    if not positives:
        return float("nan")
    return len(set(order[:k].tolist()) & positives) / len(positives)


def precision_at_k(order: np.ndarray, positives: set[int], k: int) -> float:
    return len(set(order[:k].tolist()) & positives) / k


def average_precision(order: np.ndarray, positives: set[int]) -> float:
    if not positives:
        return float("nan")
    hits, total = 0, 0.0
    for rank, item in enumerate(order.tolist(), start=1):
        if item in positives:
            hits += 1
            total += hits / rank
            if hits == len(positives):
                break
    return total / len(positives)


def ndcg_at_k(order: np.ndarray, positives: set[int], k: int) -> float:
    if not positives:
        return float("nan")
    gains = np.array([1.0 if i in positives else 0.0 for i in order[:k].tolist()])
    dcg = float((gains / np.log2(np.arange(2, len(gains) + 2))).sum())
    ideal = min(len(positives), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def ranking_report(scores_by_query: dict[int, np.ndarray], positives_by_query: dict[int, set[int]],
                   ks=(10, 20, 50, 100), exclude_by_query: dict[int, set[int]] | None = None) -> dict:
    """Macro-averaged ranking metrics. ``scores`` are indexed by candidate
    position; ``exclude`` candidate positions are pushed to the bottom."""
    rows = []
    for q, scores in scores_by_query.items():
        pos = positives_by_query.get(q, set())
        if not pos:
            continue
        s = scores.astype(float).copy()
        if exclude_by_query and exclude_by_query.get(q):
            s[list(exclude_by_query[q])] = -np.inf
        order = rank_order(s)
        row = {"ap": average_precision(order, pos)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(order, pos, k)
            row[f"ndcg@{k}"] = ndcg_at_k(order, pos, k)
        rows.append(row)
    if not rows:
        return {"n_queries": 0}
    out = {"n_queries": len(rows), "map": float(np.mean([r["ap"] for r in rows]))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([r[f"recall@{k}"] for r in rows]))
        out[f"ndcg@{k}"] = float(np.mean([r[f"ndcg@{k}"] for r in rows]))
    return out


# ------------------------------------------------------------ calibration
def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((probs - labels) ** 2)) if len(probs) else float("nan")


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        count.append(int(m.sum()))
        conf.append(float(probs[m].mean()) if m.any() else float("nan"))
        acc.append(float(labels[m].mean()) if m.any() else float("nan"))
    return {"bin_edges": edges.tolist(), "confidence": conf, "accuracy": acc, "count": count}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    rc = reliability_curve(probs, labels, n_bins)
    n = sum(rc["count"])
    if n == 0:
        return float("nan")
    return float(sum(c * abs(a - f) for c, a, f in zip(rc["count"], rc["accuracy"], rc["confidence"]) if c) / n)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie handling."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks
