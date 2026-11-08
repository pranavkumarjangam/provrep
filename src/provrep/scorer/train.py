"""Training and inference for the Stage-1 wide scorer (M2)."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.kg import COMPOUND, DISEASE, KnowledgeGraph
from ..data.splits import Split
from ..eval.metrics import ranking_report
from ..utils import ensure_dir, get_logger, read_json, seed_everything, write_json
from .rgcn import RGCNScorer, build_graph_tensors, relation_names

log = get_logger(__name__)


class WideScorer:
    """Stage 1: score *every* compound against a disease (exhaustive, no retrieval)."""

    def __init__(self, kg: KnowledgeGraph, model: RGCNScorer, relations: list[str],
                 message_edges: dict[str, np.ndarray], target: str = "CtD", device: str = "cpu"):
        self.kg, self.model, self.relations, self.target = kg, model.to(device).eval(), relations, target
        self.device = device
        self.message_edges = message_edges
        self.compounds = kg.members(COMPOUND)
        self._compounds_t = torch.as_tensor(self.compounds, device=device)
        g = build_graph_tensors(message_edges, relations, kg.num_nodes).to(device)
        with torch.no_grad():
            self.h = self.model.encode(g)

    def scores(self, disease: int) -> np.ndarray:
        """Logits aligned with ``self.compounds``."""
        return self.model.score_all(self.h, self._compounds_t, int(disease)).cpu().numpy()

    def pair_logits(self, pairs: np.ndarray) -> np.ndarray:
        if len(pairs) == 0:
            return np.empty(0)
        c = torch.as_tensor(pairs[:, 0], device=self.device)
        d = torch.as_tensor(pairs[:, 1], device=self.device)
        with torch.no_grad():
            return self.model.decode(self.h, c, d).cpu().numpy()

    def save(self, directory: str | Path, hparams: dict) -> None:
        d = ensure_dir(directory)
        torch.save(self.model.state_dict(), d / "scorer.pt")
        write_json(d / "scorer_meta.json", {"relations": self.relations, "target": self.target, "hparams": hparams})

    @classmethod
    def load(cls, directory: str | Path, kg: KnowledgeGraph, split: Split, device: str = "cpu") -> "WideScorer":
        d = Path(directory)
        meta = read_json(d / "scorer_meta.json")
        hp = meta["hparams"]
        model = build_model(kg, len(meta["relations"]), hp)
        model.load_state_dict(torch.load(d / "scorer.pt", map_location=device))
        edges = message_edges(kg, split, meta["target"], hp.get("target_message_passing", "none"))
        return cls(kg, model, meta["relations"], edges, meta["target"], device)


def _mode(m) -> str:
    return {True: "both", False: "none", None: "none"}.get(m, m)


def message_edges(kg: KnowledgeGraph, split: Split, target: str, mode, msg_pos: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Edges visible to the encoder. Held-out target edges are never included.

    mode="none"           no target (CtD) edges at all
    mode="compound_side"  training CtD edges flow only disease -> compound (relation "DtC"):
                          a compound's representation knows which training diseases it
                          treats (capturing broad-use drugs), while no disease - train or
                          held-out - receives indication messages, so they look alike
    mode="both"           training CtD edges in both directions (transductive setting)
    """
    mode = _mode(mode)
    pos = split.train_pos if msg_pos is None else msg_pos
    edges = split.message_passing_edges(kg, target, include_target=(mode == "both"))
    if mode == "both":
        edges[target] = pos
    if mode == "compound_side":
        edges["DtC"] = pos[:, ::-1].copy()
    return edges


def relations_for(kg: KnowledgeGraph, mode) -> list[str]:
    rels = relation_names(list(kg.edges.keys()))
    return rels + (["DtC"] if _mode(mode) == "compound_side" else [])


def build_model(kg: KnowledgeGraph, num_rels: int, hp: dict) -> RGCNScorer:
    kind_index = {k: i for i, k in enumerate(kg.kinds)}
    node_kind = torch.as_tensor([kind_index[k] for k in kg.node_kinds], dtype=torch.long)
    shared = [kind_index[k] for k in (COMPOUND, DISEASE) if k in kind_index] if hp.get("inductive", True) else []
    return RGCNScorer(kg.num_nodes, num_rels, hp["hidden_dim"], hp["num_layers"], hp["num_bases"], hp["dropout"],
                      node_kind=node_kind, shared_kinds=shared, num_kinds=len(kg.kinds))


HPARAM_KEYS = ("hidden_dim", "num_layers", "num_bases", "dropout", "inductive", "target_message_passing", "select_metric")


def evaluate_scorer(scorer: WideScorer, split: Split, part: str = "val", ks=(10, 20, 50, 100),
                    max_diseases: int | None = None) -> dict:
    pos = getattr(split, f"{part}_pos")
    diseases = np.unique(pos[:, 1])
    if max_diseases:
        diseases = diseases[:max_diseases]
    pos_by, excl_by, scores_by = {}, {}, {}
    comp_pos = {int(c): i for i, c in enumerate(scorer.compounds)}
    train = split.train_pos
    for d in diseases:
        pos_by[int(d)] = {comp_pos[int(c)] for c in pos[pos[:, 1] == d, 0]}
        excl_by[int(d)] = {comp_pos[int(c)] for c in train[train[:, 1] == d, 0]}
        scores_by[int(d)] = scorer.scores(int(d))
    return ranking_report(scores_by, pos_by, ks, excl_by)


def train_scorer(kg: KnowledgeGraph, split: Split, cfg: dict, target: str = "CtD",
                 out_dir: str | Path | None = None) -> tuple[WideScorer, dict]:
    hp = cfg["scorer"]
    seed_everything(hp["seed"])
    device = hp.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; using CPU")
        device = "cpu"
    rng = np.random.default_rng(hp["seed"])
    hp = {"inductive": True, "target_message_passing": "compound_side", "select_metric": "recall@100", **hp}
    hp["target_message_passing"] = _mode(hp["target_message_passing"])
    relations = relations_for(kg, hp["target_message_passing"])
    model = build_model(kg, len(relations), hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    compounds, diseases = kg.members(COMPOUND), kg.members(DISEASE)
    mp_mode = hp["target_message_passing"]
    base_edges = message_edges(kg, split, target, mp_mode)
    select = hp["select_metric"]
    train_pos, hard_neg = split.train_pos, split.train_neg
    k = hp["neg_per_pos"]
    aux_weight, aux_per_rel = hp.get("aux_weight", 1.0), hp.get("aux_edges_per_relation", 2000)
    base_rels = sorted(kg.edges.keys())
    aux_rels = [(i, m) for i, m in enumerate(base_rels) if m != target and len(kg.edges[m])]
    min_epochs = hp.get("min_epochs", 50)
    eval_every = int(hp.get("eval_every", 5))
    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(train_pos))
        n_sup = max(1, int(len(train_pos) * hp["supervision_edge_frac"]))
        sup, msg = train_pos[perm[:n_sup]], train_pos[perm[n_sup:]]
        # supervision edges are hidden from message passing this epoch (no trivial leakage)
        edges = message_edges(kg, split, target, mp_mode, msg_pos=msg)
        g = build_graph_tensors(edges, relations, kg.num_nodes).to(device)
        h = model.encode(g)

        # negatives: corrupt the compound (k/2) and the disease (k/2) of each supervision edge
        half = max(1, k // 2)
        neg_c = np.stack([rng.choice(compounds, size=len(sup) * half), np.repeat(sup[:, 1], half)], 1)
        neg_d = np.stack([np.repeat(sup[:, 0], half), rng.choice(diseases, size=len(sup) * half)], 1)
        pairs = np.concatenate([sup, neg_c, neg_d] + ([hard_neg] if len(hard_neg) else []))
        labels = np.concatenate([np.ones(len(sup)), np.zeros(len(neg_c) + len(neg_d) + len(hard_neg))])
        weights = np.concatenate([np.full(len(sup), float(half * 2)), np.ones(len(neg_c) + len(neg_d)),
                                  np.full(len(hard_neg), hp["hard_negative_weight"])])
        pt = torch.as_tensor(pairs, device=device)
        logits = model.decode(h, pt[:, 0], pt[:, 1])
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(labels, dtype=torch.float32, device=device),
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device) / weights.mean())
        if aux_weight > 0 and aux_rels:
            loss = loss + aux_weight * _aux_loss(model, h, kg, aux_rels, aux_per_rel, rng, device)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % eval_every == 0 or epoch == 1:
            scorer = WideScorer(kg, copy.deepcopy(model), relations, base_edges, target, device)
            val = evaluate_scorer(scorer, split, "val", ks=(20, 100))
            history.append({"epoch": epoch, "loss": float(loss.item()), "val_map": val.get("map"),
                            "val_recall@20": val.get("recall@20"), "val_recall@100": val.get("recall@100")})
            log.info("epoch %3d loss %.4f val MAP %.4f R@20 %.4f R@100 %.4f (%.0fs)", epoch, loss.item(),
                     val.get("map", float("nan")), val.get("recall@20", float("nan")), val.get("recall@100", float("nan")), time.time() - t0)
            if val.get(select, -1) > best:
                best, best_state, bad = val[select], copy.deepcopy(model.state_dict()), 0
            elif epoch >= min_epochs:
                bad += eval_every
                if bad >= hp["patience"]:
                    log.info("early stopping at epoch %d (best val %s %.4f)", epoch, select, best)
                    break

    model.load_state_dict(best_state)
    scorer = WideScorer(kg, model, relations, base_edges, target, device)
    report = {"best_val": best, "select_metric": select, "epochs_run": epoch, "seconds": time.time() - t0, "history": history}
    if out_dir:
        scorer.save(out_dir, {k: hp[k] for k in HPARAM_KEYS})
        write_json(Path(out_dir) / "train_report.json", report)
    return scorer, report


def _aux_loss(model, h, kg: KnowledgeGraph, aux_rels, per_rel: int, rng, device) -> torch.Tensor:
    """Multi-relational link prediction on the non-target metaedges (standard
    R-GCN objective). Gives context-node embeddings structure (pathway
    co-membership, interactions) that the sparse CtD signal alone cannot."""
    losses = []
    for r, me in aux_rels:
        e = kg.edges[me]
        pick = e[rng.choice(len(e), size=min(per_rel, len(e)), replace=False)]
        dst_kind = kg.metaedge_kinds[me][1]
        corrupt = rng.choice(kg.members(dst_kind), size=len(pick))
        src = torch.as_tensor(np.concatenate([pick[:, 0], pick[:, 0]]), device=device)
        dst = torch.as_tensor(np.concatenate([pick[:, 1], corrupt]), device=device)
        y = torch.cat([torch.ones(len(pick)), torch.zeros(len(pick))]).to(device)
        losses.append(F.binary_cross_entropy_with_logits(model.decode_aux(h, src, dst, r), y))
    return torch.stack(losses).mean()
