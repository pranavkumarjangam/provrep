"""Non-learned reference scorers, so Stage-1 numbers have something honest to beat."""
from __future__ import annotations

import numpy as np

from ..data.kg import COMPOUND, KnowledgeGraph
from ..data.splits import Split


class PopularityBaseline:
    """Rank compounds by how many training indications they already have."""

    def __init__(self, kg: KnowledgeGraph, split: Split):
        self.compounds = kg.members(COMPOUND)
        counts = np.bincount(kg.local_idx[split.train_pos[:, 0]], minlength=len(self.compounds)) if len(split.train_pos) else np.zeros(len(self.compounds))
        self._scores = counts.astype(float)

    def scores(self, disease: int) -> np.ndarray:
        return self._scores


class GeneOverlapBaseline:
    """Rank compounds by the number of their target genes associated with the
    disease (CbG-DaG path count). The classic "shared targets" heuristic."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.compounds = kg.members(COMPOUND)
        self.cbg = kg.adjacency("CbG")
        self.dag = kg.adjacency("DaG")

    def scores(self, disease: int) -> np.ndarray:
        d = self.kg.local_idx[disease]
        return np.asarray(self.cbg @ self.dag[d].T.toarray()).ravel()
