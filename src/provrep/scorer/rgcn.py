"""Relational GCN encoder + DistMult decoder in plain PyTorch (M2, Stage 1).

No PyTorch Geometric / DGL dependency: message passing is a per-relation
gather -> transform -> ``index_add_`` over edges pre-sorted by relation type,
with basis decomposition of the relation weights (Schlichtkrull et al., 2018).
Every metaedge gets an inverse relation so information flows both ways.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphTensors:
    src: torch.Tensor        # (E,) long, sorted by relation
    dst: torch.Tensor
    norm: torch.Tensor       # (E,) float, 1 / in-degree of dst under that relation
    offsets: list[int]       # relation r occupies [offsets[r], offsets[r+1])
    num_nodes: int
    src_unique: torch.Tensor | None = None   # distinct source nodes per relation, concatenated
    unique_offsets: list[int] | None = None  # relation r's distinct sources: [unique_offsets[r], unique_offsets[r+1])
    src_inv: torch.Tensor | None = None      # (E,) position of each edge's source within its relation's block

    def to(self, device) -> "GraphTensors":
        mv = lambda t: t.to(device) if t is not None else None  # noqa: E731
        return GraphTensors(self.src.to(device), self.dst.to(device), self.norm.to(device), self.offsets, self.num_nodes,
                            mv(self.src_unique), self.unique_offsets, mv(self.src_inv))


def relation_names(metaedges: list[str]) -> list[str]:
    base = sorted(metaedges)
    return base + [f"{m}__rev" for m in base]


def build_graph_tensors(edges: dict[str, np.ndarray], relations: list[str], num_nodes: int) -> GraphTensors:
    srcs, dsts, norms, offsets = [], [], [], [0]
    uniq, invs, u_offsets = [], [], [0]
    for rel in relations:
        base, rev = (rel[:-5], True) if rel.endswith("__rev") else (rel, False)
        e = edges.get(base)
        if e is None or len(e) == 0:
            offsets.append(offsets[-1])
            u_offsets.append(u_offsets[-1])
            continue
        s, d = (e[:, 1], e[:, 0]) if rev else (e[:, 0], e[:, 1])
        deg = np.bincount(d, minlength=num_nodes).astype(np.float32)
        srcs.append(s)
        dsts.append(d)
        norms.append(1.0 / deg[d])
        offsets.append(offsets[-1] + len(s))
        u, inv = np.unique(s, return_inverse=True)
        uniq.append(u)
        invs.append(inv)
        u_offsets.append(u_offsets[-1] + len(u))
    cat = (lambda xs, dt: torch.from_numpy(np.concatenate(xs).astype(dt)) if xs else torch.empty(0, dtype=torch.long))
    return GraphTensors(cat(srcs, np.int64), cat(dsts, np.int64),
                        torch.from_numpy(np.concatenate(norms).astype(np.float32)) if norms else torch.empty(0),
                        offsets, num_nodes, cat(uniq, np.int64), u_offsets, cat(invs, np.int64))


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_rels: int, num_bases: int, activation: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_rels = num_rels
        self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coeffs = nn.Parameter(torch.empty(num_rels, num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeffs)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        weights = torch.einsum("rb,bio->rio", self.coeffs, self.bases)
        out = self.self_loop(h)
        for r in range(self.num_rels):
            a, b = g.offsets[r], g.offsets[r + 1]
            if a == b:
                continue
            if g.src_unique is not None and h.is_cuda:
                # transform each distinct source node once, then gather per edge: identical
                # result, but autograd keeps only (distinct nodes x dim) tensors instead of
                # (edges x dim) - the difference between fitting in 4 GB of VRAM or not.
                # Used on GPU only: on CPU the plain per-edge path is ~2x faster.
                ua, ub = g.unique_offsets[r], g.unique_offsets[r + 1]
                transformed = h[g.src_unique[ua:ub]] @ weights[r]
                msg = transformed[g.src_inv[a:b]] * g.norm[a:b].unsqueeze(1)
            else:
                msg = (h[g.src[a:b]] @ weights[r]) * g.norm[a:b].unsqueeze(1)
            out = out.index_add(0, g.dst[a:b], msg)
        if self.activation:
            out = F.relu(out)
        return self.dropout(out)


class RGCNScorer(nn.Module):
    """Encoder over the whole KG; DistMult scores (compound, disease) pairs.

    Inductive by design: nodes of the *query* kinds (Compound, Disease) share one
    learned vector per kind, so their representation comes entirely from their
    biology (targets, associated genes, side effects, similarity edges) rather
    than a memorised per-node embedding. That is what lets the scorer rank
    compounds for a disease it has never seen an indication for. Context nodes
    (genes, pathways, side effects, ...) keep free per-node embeddings.
    """

    def __init__(self, num_nodes: int, num_rels: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_bases: int = 8, dropout: float = 0.1, node_kind: torch.Tensor | None = None,
                 shared_kinds: list[int] | None = None, num_kinds: int = 1):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.kind_embed = nn.Embedding(max(1, num_kinds), hidden_dim)
        nn.init.xavier_uniform_(self.kind_embed.weight)
        kind = node_kind if node_kind is not None else torch.zeros(num_nodes, dtype=torch.long)
        shared = torch.zeros(num_nodes, dtype=torch.bool)
        for k in shared_kinds or []:
            shared |= kind == k
        self.register_buffer("node_kind", kind)
        self.register_buffer("shared_mask", shared.unsqueeze(1))
        self.layers = nn.ModuleList(
            RGCNLayer(hidden_dim, hidden_dim, num_rels, min(num_bases, num_rels),
                      activation=(i < num_layers - 1), dropout=(dropout if i < num_layers - 1 else 0.0))
            for i in range(num_layers)
        )
        self.rel = nn.Parameter(torch.ones(hidden_dim))                      # target relation (CtD)
        self.rel_aux = nn.Parameter(torch.ones(max(1, num_rels // 2), hidden_dim))  # auxiliary KG relations

    def encode(self, g: GraphTensors) -> torch.Tensor:
        h = torch.where(self.shared_mask, self.kind_embed(self.node_kind), self.embed.weight)
        for layer in self.layers:
            h = layer(h, g)
        return h

    def decode(self, h: torch.Tensor, compounds: torch.Tensor, diseases: torch.Tensor) -> torch.Tensor:
        return (h[compounds] * self.rel * h[diseases]).sum(-1)

    def decode_aux(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, rel: int) -> torch.Tensor:
        """DistMult for an auxiliary (non-target) metaedge; used only as a training signal."""
        return (h[src] * self.rel_aux[rel] * h[dst]).sum(-1)

    @torch.no_grad()
    def score_all(self, h: torch.Tensor, compounds: torch.Tensor, disease: int) -> torch.Tensor:
        """Logits of every compound in ``compounds`` against one disease."""
        return (h[compounds] * self.rel * h[disease]).sum(-1)
