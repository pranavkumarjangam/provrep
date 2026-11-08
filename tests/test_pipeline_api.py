import json

from fastapi.testclient import TestClient

from provrep.api.app import create_app
from provrep.cli import main
from provrep.eval.harness import run_evaluation


def test_triage_result_structure(tiny_pipeline):
    d = int(tiny_pipeline.split.diseases("test")[0])
    res = tiny_pipeline.triage(tiny_pipeline.kg.node_ids[d])
    assert res.shortlist and all(r.brief is not None for r in res.shortlist)
    probs = [r.probability for r in res.shortlist]
    assert probs == sorted(probs, reverse=True)
    assert all(r.probability >= res.threshold for r in res.recommendations)
    assert res.abstained == (len(res.recommendations) == 0)
    assert {"stage1_s", "stage2_s", "stage3_s", "stage4_s"} <= set(res.timings)
    no_agents = tiny_pipeline.triage(d, use_agents=False)
    assert all(r.brief is None for r in no_agents.shortlist) and no_agents.settings["calibrator"] == "stage1-2"


def test_resolve_disease_by_name(tiny_pipeline):
    kg = tiny_pipeline.kg
    d = int(tiny_pipeline.split.diseases("test")[0])
    assert tiny_pipeline.resolve_disease(kg.name(d)) == d


def test_api_endpoints(tiny_pipeline):
    client = TestClient(create_app(tiny_pipeline.cfg, tiny_pipeline))
    home = client.get("/")
    assert home.status_code == 200 and 'id="root"' in home.text
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm"] == "offline-heuristic" and h["calibrator_full"]
    ds = client.get("/diseases", params={"q": "synthetic", "limit": 5}).json()
    assert len(ds) == 5
    d = tiny_pipeline.kg.node_ids[int(tiny_pipeline.split.diseases("test")[0])]
    summary = client.post("/triage", json={"disease": d}).json()
    full = client.get(f"/triage/{summary['id']}").json()
    assert full["disease_id"] == d and len(full["shortlist"]) == summary["shortlist_size"]
    cid = full["shortlist"][0]["candidate"]["compound_id"]
    b = client.get(f"/triage/{summary['id']}/briefs/{cid}").json()
    assert b["brief"]["candidate"]["compound_id"] == cid and "verdict" in b["brief"]
    assert client.post("/triage", json={"disease": "no such disease"}).status_code == 404
    assert client.get("/triage/doesnotexist").status_code == 404


def test_evaluation_report(tiny_pipeline, tmp_path):
    rep = run_evaluation(tiny_pipeline, tiny_pipeline.cfg, tmp_path, tiny_pipeline.cfg["paths"]["raw_dir"] + "/truth.tsv")
    assert {"L1a stage1 R-GCN only", "L2 +filters (prune+audit)", "L3 +adjudication", "L4 +arbitration & calibration"} <= set(rep["levels"])
    assert "calibrator (full)" in rep["calibration"] and (tmp_path / "reliability.png").exists()
    assert (tmp_path / "eval_report.md").read_text().startswith("# Evaluation report")
    json.loads((tmp_path / "eval_report.json").read_text())


def test_cli_triage(tiny_pipeline, tmp_path, capsys):
    import yaml

    cfgf = tmp_path / "cfg.yaml"
    cfgf.write_text(yaml.safe_dump(tiny_pipeline.cfg))
    d = tiny_pipeline.kg.node_ids[int(tiny_pipeline.split.diseases("test")[0])]
    main(["-c", str(cfgf), "triage", d, "--show", "3", "--out", str(tmp_path / "t.json")])
    out = capsys.readouterr().out
    assert "recommendation" in out and (tmp_path / "t.json").exists()
