"""Stage-1 ensemble: R-GCN + cheap exhaustive priors, combined by rank.

On real Hetionet the R-GCN alone under-recalls on held-out diseases, because
a large share of indications go to broad-use drugs (e.g. corticosteroids)
and a specific-target signal matters too. Rank-averaging the R-GCN with a
popularity prior and the gene-overlap count keeps Stage 1 exhaustive and
millisecond-cheap while lifting recall of the top-k funnel. Weights are
chosen on *validation* diseases only (``fit_weights``).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.kg import KnowledgeGraph
from ..data.splits import Split
from ..utils import read_json, write_json
from .baselines import GeneOverlapBaseline, PopularityBaseline
from .train import WideScorer, evaluate_scorer

WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)


class EnsembleScorer:
    names = ("rgcn", "popularity", "gene_overlap")

    def __init__(self, rgcn: WideScorer, kg: KnowledgeGraph, split: Split, weights: dict[str, float] | None = None):
        self.rgcn = rgcn
        self.parts = {"rgcn": rgcn, "popularity": PopularityBaseline(kg, split), "gene_overlap": GeneOverlapBaseline(kg)}
        self.weights = weights or {"rgcn": 1.0, "popularity": 1.0, "gene_overlap": 1.0}
        self.compounds = rgcn.compounds
        self.kg = kg

    def components(self, disease: int) -> dict[str, np.ndarray]:
        return {k: p.scores(disease) for k, p in self.parts.items()}

    def scores(self, disease: int) -> np.ndarray:
        """Weighted mean of per-component normalised ranks, in [0, 1] (higher = better)."""
        n = len(self.compounds)
        total = sum(self.weights.values()) or 1.0
        out = np.zeros(n)
        for k, s in self.components(disease).items():
            w = self.weights.get(k, 0.0)
            if w:
                out += w * rankdata(s) / n
        return out / total

    def fit_weights(self, split: Split, metric: str = "recall@200") -> dict:
        best, best_w, table = -1.0, self.weights, []
        k = int(metric.split("@")[1]) if "@" in metric else 100
        for combo in itertools.product(WEIGHT_GRID, repeat=len(self.names)):
            if not any(combo):
                continue
            self.weights = dict(zip(self.names, combo))
            r = evaluate_scorer(self, split, "val", ks=(20, k))
            table.append({**self.weights, metric: r[metric], "map": r["map"]})
            if r[metric] > best + 1e-9:
                best, best_w = r[metric], dict(self.weights)
        self.weights = best_w
        return {"weights": best_w, "val_" + metric: best, "grid": table}

    def save(self, directory: str | Path, extra: dict | None = None) -> None:
        write_json(Path(directory) / "ensemble.json", {"weights": self.weights, **(extra or {})})

    @classmethod
    def maybe_load(cls, directory: str | Path, rgcn: WideScorer, kg: KnowledgeGraph, split: Split):
        f = Path(directory) / "ensemble.json"
        if not f.exists():
            return None
        return cls(rgcn, kg, split, read_json(f)["weights"])
