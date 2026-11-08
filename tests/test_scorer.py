import numpy as np
import torch

from provrep.data.kg import COMPOUND
from provrep.scorer.rgcn import RGCNScorer, build_graph_tensors, relation_names
from provrep.scorer.train import build_model

from .conftest import hand_kg


def test_graph_tensors_add_inverse_relations():
    kg = hand_kg()
    rels = relation_names(list(kg.edges))
    g = build_graph_tensors(kg.edges, rels, kg.num_nodes)
    assert len(rels) == 2 * len(kg.edges)
    assert g.offsets[-1] == 2 * kg.num_edges
    r = rels.index("CbG__rev")
    a, b = g.offsets[r], g.offsets[r + 1]
    assert g.src[a:b].tolist() == [kg.idx("Gene::1")] and g.dst[a:b].tolist() == [kg.idx("Compound::C1")]


def test_forward_backward_shapes():
    kg = hand_kg()
    rels = relation_names(list(kg.edges))
    model = build_model(kg, len(rels), {"hidden_dim": 16, "num_layers": 2, "num_bases": 4, "dropout": 0.0})
    g = build_graph_tensors(kg.edges, rels, kg.num_nodes)
    h = model.encode(g)
    assert h.shape == (kg.num_nodes, 16)
    logit = model.decode(h, torch.tensor([kg.idx("Compound::C1")]), torch.tensor([kg.idx("Disease::DOID:1")]))
    logit.sum().backward()
    assert model.layers[0].bases.grad is not None


def test_inductive_query_nodes_share_input_vector():
    kg = hand_kg()
    model = build_model(kg, 2, {"hidden_dim": 8, "num_layers": 1, "num_bases": 1, "dropout": 0.0})
    x = torch.where(model.shared_mask, model.kind_embed(model.node_kind), model.embed.weight)
    c = kg.members(COMPOUND)
    assert torch.allclose(x[c[0]], x[c[1]])                        # compounds: identical input
    assert not torch.allclose(x[kg.idx("Gene::1")], x[kg.idx("Gene::2")])   # genes: free embeddings


def test_trained_scorer_beats_popularity(tiny_pipeline):
    from provrep.scorer import PopularityBaseline, evaluate_scorer

    s = tiny_pipeline.split
    rgcn = evaluate_scorer(tiny_pipeline.scorer, s, "test")
    pop = evaluate_scorer(PopularityBaseline(tiny_pipeline.kg, s), s, "test")
    assert rgcn["recall@100"] >= pop["recall@100"]
    assert np.isfinite(tiny_pipeline.scorer.scores(int(s.diseases("test")[0]))).all()


def test_ensemble_scores_and_weight_fit(tiny_pipeline):
    from provrep.scorer.ensemble import EnsembleScorer

    base = tiny_pipeline.scorer.rgcn if hasattr(tiny_pipeline.scorer, "rgcn") else tiny_pipeline.scorer
    ens = EnsembleScorer(base, tiny_pipeline.kg, tiny_pipeline.split)
    d = int(tiny_pipeline.split.diseases("val")[0])
    s = ens.scores(d)
    assert s.shape == base.scores(d).shape and 0 <= s.min() and s.max() <= 1
    fit = ens.fit_weights(tiny_pipeline.split, "recall@20")
    assert set(fit["weights"]) == {"rgcn", "popularity", "gene_overlap"} and any(fit["weights"].values())
    ens.weights = {"rgcn": 1.0, "popularity": 0.0, "gene_overlap": 0.0}
    assert (np.argsort(-ens.scores(d))[:5] == np.argsort(-base.scores(d), kind="stable")[:5]).mean() >= 0.6
