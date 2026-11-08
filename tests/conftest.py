"""Shared fixtures: a small synthetic KG and a quickly trained pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from provrep.config import load_config
from provrep.data import formats
from provrep.data.kg import KnowledgeGraph
from provrep.data.splits import make_split
from provrep.data.synthetic import generate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    cfg = load_config(CONFIGS / "synthetic.yaml", {
        "paths": {"raw_dir": str(work / "raw"), "work_dir": str(work / "artifacts")},
        "data": {"synthetic": {"seed": 3, "n_genes": 200, "n_pathways": 16, "n_diseases": 30,
                               "n_compounds": 120, "n_side_effects": 40},
                 "split": {"val_frac": 0.2, "test_frac": 0.2}},
        "scorer": {"epochs": 60, "min_epochs": 10, "patience": 60, "hidden_dim": 32},
        "filters": {"stage1_top_k": 60, "shortlist_k": 8},
    })
    return cfg


@pytest.fixture(scope="session")
def tiny_kg(tiny_cfg):
    p = generate(tiny_cfg["paths"]["raw_dir"], **tiny_cfg["data"]["synthetic"])
    kg = KnowledgeGraph.from_hetionet_files(p["nodes"], p["edges"], p["indications"])
    return kg


@pytest.fixture(scope="session")
def tiny_split(tiny_kg, tiny_cfg):
    sc = tiny_cfg["data"]["split"]
    return make_split(tiny_kg, "CtD", "disease_holdout", sc["val_frac"], sc["test_frac"], sc["seed"])


@pytest.fixture(scope="session")
def tiny_pipeline(tiny_kg, tiny_split, tiny_cfg):
    from provrep.pipeline import TriagePipeline, fit_calibrators
    from provrep.scorer import train_scorer

    work = Path(tiny_cfg["paths"]["work_dir"])
    tiny_kg.save(work / "kg")
    tiny_split.save(work / "split")
    train_scorer(tiny_kg, tiny_split, tiny_cfg, out_dir=work / "scorer")
    pipe = TriagePipeline.from_artifacts(tiny_cfg)
    fit_calibrators(pipe, work)
    return pipe


def hand_kg() -> KnowledgeGraph:
    """Seven-node KG with known paths, for exact tests.

        C1 -binds-> G1 <-assoc- D1        mechanistic path C1..D1
        C2 -resembles- C3 -treats-> D1    similarity-only path C2..D1
        G1 -participates-> PW1 <- G2 <-assoc- D2
    """
    nodes = pd.DataFrame([
        ("Compound::C1", "c-one", "Compound"), ("Compound::C2", "c-two", "Compound"), ("Compound::C3", "c-three", "Compound"),
        ("Gene::1", "G1", "Gene"), ("Gene::2", "G2", "Gene"), ("Pathway::PW1", "pw-one", "Pathway"),
        ("Disease::DOID:1", "d-one", "Disease"), ("Disease::DOID:2", "d-two", "Disease"),
    ], columns=formats.NODE_COLUMNS)
    edges = pd.DataFrame([
        ("Compound::C1", "CbG", "Gene::1"), ("Disease::DOID:1", "DaG", "Gene::1"),
        ("Compound::C2", "CrC", "Compound::C3"), ("Compound::C3", "CtD", "Disease::DOID:1"),
        ("Gene::1", "GpPW", "Pathway::PW1"), ("Gene::2", "GpPW", "Pathway::PW1"), ("Disease::DOID:2", "DaG", "Gene::2"),
    ], columns=formats.EDGE_COLUMNS)
    ind = pd.DataFrame({"compound_id": ["Compound::C3", "Compound::C2"], "disease_id": ["Disease::DOID:1", "Disease::DOID:1"],
                        "category": ["DM", "NOT"]})
    return KnowledgeGraph.from_frames(nodes, edges, ind)
