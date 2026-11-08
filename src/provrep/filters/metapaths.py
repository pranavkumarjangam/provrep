"""Metapath profiles between a compound and a disease (M3).

Degree-weighted path counts (DWPC, Himmelstein et al. 2017 / Rephetio) down-
weight paths through high-degree nodes, so a link through a promiscuous hub
gene counts for much less than one through a specific target.

Metapaths are written in Hetionet abbreviations and split into two families:
* mechanistic - through the compound's targets and the disease's genes
* similarity  - "resembles a known treatment" / "treats a similar disease"
A candidate whose support is mostly similarity is flagged as degenerate:
its score says "looks like something that works", not "acts on the disease".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..data.kg import COMPOUND, DISEASE, KnowledgeGraph

# (metaedge, reversed?) steps; kinds are checked against the KG at build time.
METAPATHS: dict[str, list[tuple[str, bool]]] = {
    "CbGaD": [("CbG", False), ("DaG", True)],
    "CbGiGaD": [("CbG", False), ("GiG", False), ("DaG", True)],
    "CbGpPWpGaD": [("CbG", False), ("GpPW", False), ("GpPW", True), ("DaG", True)],
    "CuGdD": [("CuG", False), ("DdG", True)],     # real Hetionet: upregulates / downregulates
    "CdGuD": [("CdG", False), ("DuG", True)],
    "CrCtD": [("CrC", False), ("CtD", False)],
    "CtDrD": [("CtD", False), ("DrD", False)],
}
MECHANISTIC = ["CbGaD", "CbGiGaD", "CbGpPWpGaD", "CuGdD", "CdGuD"]
SIMILARITY = ["CrCtD", "CtDrD"]


def _dw_normalize(a: sp.csr_matrix, w: float) -> sp.csr_matrix:
    row_deg = np.asarray(a.sum(axis=1)).ravel()
    col_deg = np.asarray(a.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        rs = np.where(row_deg > 0, row_deg ** -w, 0.0)
        cs = np.where(col_deg > 0, col_deg ** -w, 0.0)
    return sp.diags(rs) @ a @ sp.diags(cs)


@dataclass
class PathInstance:
    metapath: str
    nodes: list[int]        # global indices, compound first, disease last
    weight: float           # product of per-step degree^-w terms


class MetapathProfiler:
    """DWPC profiles and concrete path instances. ``known_ctd`` must contain only
    training-visible indications, otherwise similarity paths would leak labels."""

    def __init__(self, kg: KnowledgeGraph, known_ctd: np.ndarray, damping: float = 0.4,
                 metapaths: dict[str, list[tuple[str, bool]]] | None = None):
        self.kg, self.w = kg, damping
        self.edges = dict(kg.edges)
        self.edges["CtD"] = known_ctd
        self.raw: dict[str, sp.csr_matrix] = {}
        self.norm: dict[str, sp.csr_matrix] = {}
        self.metapaths: dict[str, list[tuple[str, bool]]] = {}
        for name, steps in (metapaths or METAPATHS).items():
            if all(me in self.edges and len(self.edges[me]) for me, _ in steps):
                self.metapaths[name] = steps
        for me in {me for steps in self.metapaths.values() for me, _ in steps}:
            a = kg.adjacency(me, self.edges[me]) if me == "CtD" else kg.adjacency(me)
            self.raw[me] = a.tocsr()
            self.norm[me] = _dw_normalize(a, damping).tocsr()
        self.mechanistic = [m for m in MECHANISTIC if m in self.metapaths]
        self.similarity = [m for m in SIMILARITY if m in self.metapaths]

    def _mat(self, me: str, rev: bool, normed: bool) -> sp.csr_matrix:
        m = (self.norm if normed else self.raw)[me]
        return m.T.tocsr() if rev else m

    def dwpc_vector(self, metapath: str, disease: int) -> np.ndarray:
        """DWPC of every compound (local order) to ``disease`` along ``metapath``."""
        steps = self.metapaths[metapath]
        d_local = self.kg.local_idx[disease]
        last_me, last_rev = steps[-1]
        v = self._mat(last_me, last_rev, True)[:, d_local].toarray().ravel()
        for me, rev in reversed(steps[:-1]):
            v = self._mat(me, rev, True) @ v
        return np.asarray(v).ravel()

    def profile(self, disease: int) -> dict[str, np.ndarray]:
        return {m: self.dwpc_vector(m, disease) for m in self.metapaths}

    def mechanistic_share(self, profile: dict[str, np.ndarray]) -> np.ndarray:
        """Share of *scale-normalised* support that is mechanistic, per compound.
        Each metapath's DWPC is divided by its mean over compounds with a nonzero
        value, so metapaths on different scales are comparable."""
        def scaled(ms):
            total = np.zeros(len(self.kg.members(COMPOUND)))
            for m in ms:
                v = profile[m]
                nz = v[v > 0]
                if len(nz):
                    total += v / nz.mean()
            return total
        mech, sim = scaled(self.mechanistic), scaled(self.similarity)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(mech + sim > 0, mech / (mech + sim), 0.0)
        return share

    # ------------------------------------------------------------ instances
    def paths(self, compound: int, disease: int, metapath: str, limit: int = 5) -> list[PathInstance]:
        """Concrete node paths, highest DWPC weight first (most specific first)."""
        steps = self.metapaths[metapath]
        kg = self.kg
        # backward reachability sets (local indices) from the disease
        allowed: list[np.ndarray] = [None] * (len(steps) + 1)  # type: ignore[list-item]
        allowed[len(steps)] = np.array([kg.local_idx[disease]])
        for i in range(len(steps) - 1, -1, -1):
            me, rev = steps[i]
            m = self._mat(me, rev, False)
            reach = np.asarray(m[:, allowed[i + 1]].sum(axis=1)).ravel()
            allowed[i] = np.flatnonzero(reach > 0)
        c_local = kg.local_idx[compound]
        if c_local not in set(allowed[0].tolist()):
            return []
        kinds = [self._step_kinds(me, rev)[0] for me, rev in steps] + [DISEASE]
        out: list[PathInstance] = []
        allowed_sets = [set(a.tolist()) for a in allowed]
        mats = [self._mat(me, rev, False) for me, rev in steps]
        in_deg = [np.asarray(m.sum(axis=0)).ravel() for m in mats]

        def dfs(i: int, node_local: int, trail: list[int], weight: float):
            if len(out) >= limit * 20:
                return
            if i == len(steps):
                out.append(PathInstance(metapath, [int(kg.members(k)[j]) for k, j in zip(kinds, trail)], weight))
                return
            m = mats[i]
            nbrs = m.indices[m.indptr[node_local]:m.indptr[node_local + 1]]
            deg_out = len(nbrs)
            for nxt in nbrs:
                nxt = int(nxt)
                repeat = any(t == nxt and kinds[j] == kinds[i + 1] for j, t in enumerate(trail))
                if nxt in allowed_sets[i + 1] and not repeat:
                    step_w = (deg_out ** -self.w) * (in_deg[i][nxt] ** -self.w)
                    dfs(i + 1, nxt, trail + [nxt], weight * step_w)

        dfs(0, int(c_local), [int(c_local)], 1.0)
        out.sort(key=lambda p: -p.weight)
        return out[:limit]

    def _step_kinds(self, me: str, rev: bool) -> tuple[str, str]:
        s, d = self.kg.metaedge_kinds.get(me, ("Compound", "Disease"))
        return (d, s) if rev else (s, d)
