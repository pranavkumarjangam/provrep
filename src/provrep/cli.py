"""Command-line entry point: ``provrep <command> -c CONFIG [--set key=value ...]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, set_dotted
from .utils import get_logger, write_json

log = get_logger("provrep")


def _cfg(args) -> dict:
    cfg = load_config(args.config)
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        set_dotted(cfg, k, v)
    return cfg


def cmd_synth(cfg, args):
    from .data.synthetic import generate

    generate(cfg["paths"]["raw_dir"], **cfg["data"].get("synthetic", {}))


def cmd_download(cfg, args):
    from .data.sources import download_all

    download_all(cfg["paths"]["raw_dir"], force=args.force)


def cmd_build_kg(cfg, args):
    from .data.kg import KnowledgeGraph
    from .data.sources import source_paths
    from .data.splits import make_split

    p = source_paths(cfg["paths"]["raw_dir"])
    missing = [str(v) for k, v in p.items() if not v.exists()]
    if missing:
        sys.exit(f"missing source files: {missing}. Run `provrep download` (or `provrep synth` for synthetic data).")
    kg = KnowledgeGraph.from_hetionet_files(p["nodes"], p["edges"], p["indications"], cfg["data"]["drop_metaedges"])
    work = Path(cfg["paths"]["work_dir"])
    kg.save(work / "kg")
    sc = cfg["data"]["split"]
    split = make_split(kg, cfg["data"]["target_metaedge"], sc["strategy"], sc["val_frac"], sc["test_frac"], sc["seed"],
                       sc.get("dates_file"), sc.get("temporal_cutoff_year", 2010))
    split.save(work / "split")
    write_json(work / "kg_summary.json", {"kg": kg.summary(), "split": split.summary()})
    print(json.dumps({"kg": kg.summary(), "split": split.summary()}, indent=2))


def cmd_train(cfg, args):
    from .data.kg import KnowledgeGraph
    from .data.splits import Split
    from .scorer import GeneOverlapBaseline, PopularityBaseline, evaluate_scorer, train_scorer

    work = Path(cfg["paths"]["work_dir"])
    kg, split = KnowledgeGraph.load(work / "kg"), Split.load(work / "split")
    scorer, rep = train_scorer(kg, split, cfg, cfg["data"]["target_metaedge"], work / "scorer")
    ks = (10, 20, 50, 100, 200)
    res = {"val": evaluate_scorer(scorer, split, "val", ks), "test": evaluate_scorer(scorer, split, "test", ks)}
    ens_cfg = cfg["scorer"].get("ensemble", {})
    if ens_cfg.get("enabled", True):
        from .scorer.ensemble import EnsembleScorer

        ens = EnsembleScorer(scorer, kg, split)
        fit = ens.fit_weights(split, ens_cfg.get("select_metric", "recall@200"))
        ens.save(work / "scorer", {k: v for k, v in fit.items() if k != "grid"})
        res["ensemble_weights"] = fit["weights"]
        res["val_ensemble"] = evaluate_scorer(ens, split, "val", ks)
        res["test_ensemble"] = evaluate_scorer(ens, split, "test", ks)
    for name, b in (("popularity", PopularityBaseline(kg, split)), ("gene_overlap", GeneOverlapBaseline(kg))):
        res[f"test_{name}"] = evaluate_scorer(b, split, "test", ks)
    write_json(work / "scorer" / "stage1_eval.json", res)
    print(json.dumps({k: ({m: round(v, 4) for m, v in r.items() if m in ("map", "recall@20", "recall@100", "recall@200")}
                       if k != "ensemble_weights" else r) for k, r in res.items()}, indent=2))


def cmd_calibrate(cfg, args):
    from .pipeline import TriagePipeline, fit_calibrators

    pipe = TriagePipeline.from_artifacts(cfg)
    rep = fit_calibrators(pipe, cfg["paths"]["work_dir"], cfg.get("eval", {}).get("max_eval_diseases"))
    write_json(Path(cfg["paths"]["work_dir"]) / "calibration_fit.json", rep)
    print(json.dumps(rep, indent=2))


def cmd_triage(cfg, args):
    from .pipeline import TriagePipeline

    pipe = TriagePipeline.from_artifacts(cfg)
    res = pipe.triage(args.disease, use_agents=not args.no_agents)
    out = Path(args.out) if args.out else Path(cfg["paths"]["work_dir"]) / "triage" / f"{res.id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.model_dump_json(indent=2))
    print(f"{res.disease_name} ({res.disease_id}) - {len(res.recommendations)} recommendation(s) "
          f"at p >= {res.threshold}{' (abstained)' if res.abstained else ''}; timings {res.timings}")
    for rc in res.shortlist[: args.show]:
        v = rc.brief.verdict if rc.brief else None
        p = f"{rc.probability:.3f}" if rc.probability is not None else "  -  "
        flag = "similarity-driven" if rc.candidate.degenerate else ""
        print(f"  {rc.rank:>2}. p={p}  {rc.candidate.compound_name:<28} stage1 #{rc.candidate.stage1_rank:<4}"
              f"{(v.tier + ' ' + format(v.overall, '.2f')) if v else '':<16} {flag}")
    print(f"full brief: {out}")


def cmd_eval(cfg, args):
    from .eval.harness import run_evaluation
    from .pipeline import TriagePipeline

    pipe = TriagePipeline.from_artifacts(cfg)
    truth = Path(cfg["paths"]["raw_dir"]) / "truth.tsv"
    rep = run_evaluation(pipe, cfg, Path(cfg["paths"]["work_dir"]) / "eval", truth if truth.exists() else None)
    print((Path(cfg["paths"]["work_dir"]) / "eval" / "eval_report.md").read_text())
    return rep


def cmd_serve(cfg, args):
    import uvicorn

    from .api.app import create_app

    uvicorn.run(create_app(cfg), host=args.host, port=args.port)


def cmd_run_all(cfg, args):
    if cfg["data"]["source"] == "synthetic":
        cmd_synth(cfg, args)
    else:
        args.force = False
        cmd_download(cfg, args)
    cmd_build_kg(cfg, args)
    cmd_train(cfg, args)
    cmd_calibrate(cfg, args)
    cmd_eval(cfg, args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="provrep", description=__doc__)
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. agents.llm.provider=anthropic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth", help="generate the synthetic Hetionet-format KG")
    d = sub.add_parser("download", help="download Hetionet v1.0 + PharmacotherapyDB")
    d.add_argument("--force", action="store_true")
    sub.add_parser("build-kg", help="M1: build the typed KG and the train/val/test split")
    sub.add_parser("train", help="M2: train the Stage-1 R-GCN wide scorer")
    sub.add_parser("calibrate", help="M5: fit calibrators on validation diseases")
    t = sub.add_parser("triage", help="run the full cascade for one disease")
    t.add_argument("disease", help="disease id (Disease::DOID:...) or exact name")
    t.add_argument("--no-agents", action="store_true", help="stop after Stage 2 (no LLM calls)")
    t.add_argument("--show", type=int, default=20)
    t.add_argument("--out")
    sub.add_parser("eval", help="M6: ablation ladder + calibration report on test diseases")
    s = sub.add_parser("serve", help="M7: FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("run-all", help="data -> KG -> train -> calibrate -> eval")
    args = ap.parse_args(argv)
    cfg = _cfg(args)
    {"synth": cmd_synth, "download": cmd_download, "build-kg": cmd_build_kg, "train": cmd_train,
     "calibrate": cmd_calibrate, "triage": cmd_triage, "eval": cmd_eval, "serve": cmd_serve,
     "run-all": cmd_run_all}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
