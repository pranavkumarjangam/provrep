"""Small Stage-1 hyperparameter sweep on the synthetic KG (dev utility)."""
import itertools, json, sys
from provrep.config import load_config
from provrep.data.kg import KnowledgeGraph
from provrep.data.splits import make_split
from provrep.scorer import train_scorer, evaluate_scorer

cfg0 = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml")
kg = KnowledgeGraph.load(cfg0["paths"]["work_dir"] + "/kg")
s = make_split(kg, "CtD", "disease_holdout")
grid = [
    {"epochs": 600, "lr": 0.01, "dropout": 0.1, "aux_weight": 1.0},
    {"epochs": 600, "lr": 0.01, "dropout": 0.3, "aux_weight": 1.0},
    {"epochs": 600, "lr": 0.01, "dropout": 0.1, "aux_weight": 0.5, "weight_decay": 1e-5},
    {"epochs": 600, "lr": 0.005, "dropout": 0.2, "aux_weight": 1.0, "hidden_dim": 96},
]
for g in grid:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml", {"scorer": {**g, "patience": 150}})
    sc, rep = train_scorer(kg, s, cfg)
    t = evaluate_scorer(sc, s, "test")
    print(json.dumps({**g, "val": round(rep["best_val"], 4), "test_map": round(t["map"], 4),
                      "test_r20": round(t["recall@20"], 4), "epochs": rep["epochs_run"], "sec": round(rep["seconds"])}), flush=True)
