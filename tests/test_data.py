import numpy as np
import pandas as pd

from provrep.data import formats
from provrep.data.kg import KnowledgeGraph
from provrep.data.splits import Split, make_split

from .conftest import hand_kg


def test_formats_roundtrip(tmp_path):
    nodes = pd.DataFrame([("Compound::DB1", "a", "Compound"), ("Disease::DOID:1", "b", "Disease")], columns=formats.NODE_COLUMNS)
    edges = pd.DataFrame([("Compound::DB1", "CtD", "Disease::DOID:1")], columns=formats.EDGE_COLUMNS)
    formats.write_nodes(nodes, tmp_path / "n.tsv")
    formats.write_edges(edges, tmp_path / "e.sif.gz")
    assert formats.read_nodes(tmp_path / "n.tsv").equals(nodes)
    assert formats.read_edges(tmp_path / "e.sif.gz").equals(edges)


def test_indications_get_hetionet_ids(tmp_path):
    df = pd.DataFrame({"doid_id": ["DOID:1"], "drugbank_id": ["DB1"], "disease": ["d"], "drug": ["c"],
                       "category": ["DM"], "n_curators": [1], "n_resources": [1]})
    formats.write_indications(df, tmp_path / "i.tsv")
    out = formats.read_indications(tmp_path / "i.tsv")
    assert out.loc[0, "compound_id"] == "Compound::DB1" and out.loc[0, "disease_id"] == "Disease::DOID:1"


def test_kg_basics_and_symmetric_adjacency():
    kg = hand_kg()
    assert kg.num_nodes == 8 and set(kg.kinds) == {"Compound", "Gene", "Pathway", "Disease"}
    crc = kg.adjacency("CrC").toarray()
    assert crc[kg.local_idx[kg.idx("Compound::C2")], kg.local_idx[kg.idx("Compound::C3")]] == 1
    assert crc[kg.local_idx[kg.idx("Compound::C3")], kg.local_idx[kg.idx("Compound::C2")]] == 1   # symmetrised
    cbg = kg.adjacency("CbG")
    assert cbg.shape == (3, 2) and cbg.nnz == 1
    assert set(kg.labels["category"]) == {"DM", "NOT"}


def test_kg_save_load(tmp_path):
    kg = hand_kg()
    kg.save(tmp_path)
    kg2 = KnowledgeGraph.load(tmp_path)
    assert kg2.node_ids == kg.node_ids and kg2.num_edges == kg.num_edges and len(kg2.labels) == 2


def test_drop_metaedges():
    nodes = pd.DataFrame([("Gene::1", "g", "Gene"), ("Anatomy::1", "a", "Anatomy")], columns=formats.NODE_COLUMNS)
    edges = pd.DataFrame([("Anatomy::1", "AeG", "Gene::1")], columns=formats.EDGE_COLUMNS)
    assert "AeG" not in KnowledgeGraph.from_frames(nodes, edges, drop_metaedges=["AeG"]).edges


def test_disease_holdout_split_has_no_leakage(tiny_kg, tiny_split):
    s = tiny_split
    train_d = set(s.train_pos[:, 1].tolist())
    assert not train_d & set(s.test_pos[:, 1].tolist())
    assert not train_d & set(s.val_pos[:, 1].tolist())
    assert len(s.train_pos) + len(s.val_pos) + len(s.test_pos) == len(tiny_kg.edges["CtD"])
    mp = s.message_passing_edges(tiny_kg, "CtD", include_target=True)
    held = {tuple(x) for x in np.concatenate([s.val_pos, s.test_pos]).tolist()}
    assert not held & {tuple(x) for x in mp["CtD"].tolist()}
    assert len(s.message_passing_edges(tiny_kg, "CtD", include_target=False)["CtD"]) == 0


def test_random_split_and_persistence(tiny_kg, tmp_path):
    s = make_split(tiny_kg, "CtD", "random", 0.1, 0.2, seed=1)
    n = len(tiny_kg.edges["CtD"])
    assert len(s.test_pos) == int(n * 0.2) and len(s.val_pos) == int(n * 0.1)
    s.save(tmp_path)
    s2 = Split.load(tmp_path)
    assert np.array_equal(s2.test_pos, s.test_pos) and s2.strategy == "random"


def test_temporal_split(tiny_kg, tmp_path):
    pos = tiny_kg.edges["CtD"]
    rows = [(tiny_kg.node_ids[c], tiny_kg.node_ids[d], 2000 + (i % 20)) for i, (c, d) in enumerate(pos)]
    f = tmp_path / "dates.tsv"
    pd.DataFrame(rows, columns=["compound_id", "disease_id", "year"]).to_csv(f, sep="\t", index=False)
    s = make_split(tiny_kg, "CtD", "temporal", dates_file=str(f), temporal_cutoff_year=2015)
    years = {(tiny_kg.node_ids[c], tiny_kg.node_ids[d]): y for (c, d), (_, _, y) in zip(pos, rows)}
    assert all(years[(tiny_kg.node_ids[c], tiny_kg.node_ids[d])] >= 2015 for c, d in s.test_pos)
    assert all(years[(tiny_kg.node_ids[c], tiny_kg.node_ids[d])] < 2015 for c, d in s.train_pos)
