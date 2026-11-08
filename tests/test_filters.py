import math

import numpy as np

from provrep.data.splits import Split
from provrep.filters import MetapathProfiler, StructuredFilter

from .conftest import hand_kg

CFG = {"filters": {"stage1_top_k": 10, "shortlist_k": 5, "dwpc_damping": 0.4, "degeneracy_min_mechanistic_share": 0.25,
                   "max_side_effects": None, "contraindications_file": None, "shortlist_by": "stage1"}}


def _split(kg):
    c3, d1, c2 = kg.idx("Compound::C3"), kg.idx("Disease::DOID:1"), kg.idx("Compound::C2")
    e = np.empty((0, 2), dtype=np.int64)
    return Split(np.array([[c3, d1]]), e, e, np.array([[c2, d1]]), e, e, "manual")


def test_dwpc_matches_hand_computation():
    kg = hand_kg()
    prof = MetapathProfiler(kg, _split(kg).train_pos, damping=0.4)
    v = prof.dwpc_vector("CbGaD", kg.idx("Disease::DOID:1"))
    c1 = kg.local_idx[kg.idx("Compound::C1")]
    # C1 has 1 target, G1 has 1 binder; D1 has 1 gene, G1 has 1 disease -> every degree is 1
    assert math.isclose(v[c1], 1.0)
    assert v.sum() == v[c1]
    sim = prof.dwpc_vector("CrCtD", kg.idx("Disease::DOID:1"))
    assert sim[kg.local_idx[kg.idx("Compound::C2")]] > 0 and sim[c1] == 0


def test_path_instances_and_pathway_route():
    kg = hand_kg()
    prof = MetapathProfiler(kg, _split(kg).train_pos)
    paths = prof.paths(kg.idx("Compound::C1"), kg.idx("Disease::DOID:1"), "CbGaD")
    assert [[kg.name(n) for n in p.nodes] for p in paths] == [["c-one", "G1", "d-one"]]
    via_pw = prof.paths(kg.idx("Compound::C1"), kg.idx("Disease::DOID:2"), "CbGpPWpGaD")
    assert [kg.name(n) for n in via_pw[0].nodes] == ["c-one", "G1", "pw-one", "G2", "d-two"]
    assert prof.paths(kg.idx("Compound::C2"), kg.idx("Disease::DOID:1"), "CbGaD") == []


def test_filter_prunes_and_flags_similarity_only():
    kg = hand_kg()
    f = StructuredFilter(kg, _split(kg), CFG)
    d1 = kg.idx("Disease::DOID:1")
    logits = np.array([3.0, 2.0, 1.0])   # C1, C2, C3 in local order
    res = f.run(d1, logits)
    reasons = {p.compound_name: p.reason for p in res.pruned}
    assert reasons == {"c-two": "known_failure", "c-three": "already_indicated"}
    assert [c.compound_name for c in res.shortlist] == ["c-one"]
    assert res.shortlist[0].degenerate is False and res.shortlist[0].mechanistic_share == 1.0


def test_degenerate_flag_on_synthetic(tiny_pipeline):
    """Similarity-flagged candidates should be less often truly efficacious."""
    import pandas as pd

    kg, s = tiny_pipeline.kg, tiny_pipeline.split
    truth = pd.read_csv(tiny_pipeline.cfg["paths"]["raw_dir"] + "/truth.tsv", sep="\t")
    T = {(kg.idx(c), kg.idx(d)) for c, d in truth.itertuples(index=False)}
    rate = {True: [], False: []}
    for d in s.diseases("test"):
        for c in tiny_pipeline.filter.run(int(d), tiny_pipeline.scorer.scores(int(d))).survivors:
            rate[c.degenerate].append((c.compound, int(d)) in T)
    if rate[True] and rate[False]:
        assert np.mean(rate[True]) < np.mean(rate[False])
