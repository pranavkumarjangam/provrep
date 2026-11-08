"""M7 - FastAPI service exposing the cascade and its audit trail.

GET  /health                          status, KG size, LLM/verifier, calibrators
GET  /diseases?q=...&limit=...        search diseases by name or id
POST /triage                          {"disease": "...", "use_agents": true}
GET  /triage/{id}                     full result (shortlist, briefs, pruned, invalidations)
GET  /triage/{id}/briefs/{compound}   one candidate's brief: evidence, gated claims, verdict
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

from ..data.kg import DISEASE
from ..pipeline import TriagePipeline, TriageResult


class TriageRequest(BaseModel):
    disease: str
    use_agents: bool = True


class TriageSummary(BaseModel):
    id: str
    disease_id: str
    disease_name: str
    abstained: bool
    threshold: float
    recommendations: list[dict]
    shortlist_size: int
    timings: dict[str, float]
    settings: dict


def create_app(cfg: dict, pipeline: TriagePipeline | None = None) -> FastAPI:
    state: dict = {"pipe": pipeline, "results": {}}
    lock = threading.Lock()
    store = Path(cfg["paths"]["work_dir"]) / "triage"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["pipe"] is None:
            state["pipe"] = TriagePipeline.from_artifacts(cfg)
        yield

    app = FastAPI(title="provrep - provenance-grounded drug repurposing", version="0.1.0", lifespan=lifespan)

    def pipe() -> TriagePipeline:
        if state["pipe"] is None:
            state["pipe"] = TriagePipeline.from_artifacts(cfg)
        return state["pipe"]

    def get_result(rid: str) -> TriageResult:
        if rid in state["results"]:
            return state["results"][rid]
        f = store / f"{rid}.json"
        if f.exists():
            res = TriageResult.model_validate_json(f.read_text())
            state["results"][rid] = res
            return res
        raise HTTPException(404, f"no triage result {rid}")

    @app.get("/", response_class=HTMLResponse)
    def ui():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<h1>provrep API</h1><p>UI file missing; see /docs for the API.</p>")

    @app.get("/health")
    def health():
        p = pipe()
        return {
            "status": "ok", "nodes": p.kg.num_nodes, "edges": p.kg.num_edges,
            "llm": p.adjudicator.llm.name if p.adjudicator else None,
            "verifier": p.adjudicator.verifier.name if p.adjudicator else None,
            "calibrator_full": p.cal_full is not None, "calibrator_stage12": p.cal_s12 is not None,
        }

    @app.get("/diseases")
    def diseases(q: str = "", limit: int = Query(20, le=500)):
        p = pipe()
        ql = q.lower()
        known = {}
        for c, d in p.split.train_pos:
            known[int(d)] = known.get(int(d), 0) + 1
        out = []
        for i in p.kg.members(DISEASE):
            i = int(i)
            name, nid = p.kg.name(i), p.kg.node_ids[i]
            if ql and ql not in name.lower() and ql not in nid.lower():
                continue
            out.append({"id": nid, "name": name, "known_indications": known.get(i, 0)})
            if len(out) >= limit:
                break
        return out

    @app.post("/triage", response_model=TriageSummary)
    def triage(req: TriageRequest):
        p = pipe()
        try:
            with lock:
                res = p.triage(req.disease, use_agents=req.use_agents)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        state["results"][res.id] = res
        store.mkdir(parents=True, exist_ok=True)
        (store / f"{res.id}.json").write_text(res.model_dump_json())
        return TriageSummary(
            id=res.id, disease_id=res.disease_id, disease_name=res.disease_name, abstained=res.abstained,
            threshold=res.threshold, shortlist_size=len(res.shortlist), timings=res.timings, settings=res.settings,
            recommendations=[{"rank": r.rank, "compound_id": r.candidate.compound_id, "compound_name": r.candidate.compound_name,
                              "probability": r.probability, "tier": r.brief.verdict.tier if r.brief else None}
                             for r in res.recommendations],
        )

    @app.get("/triage/{rid}", response_model=TriageResult)
    def triage_result(rid: str):
        return get_result(rid)

    @app.get("/triage/{rid}/briefs/{compound_id:path}")
    def brief(rid: str, compound_id: str):
        res = get_result(rid)
        for rc in res.shortlist:
            if rc.candidate.compound_id == compound_id or rc.candidate.compound_name == compound_id:
                if rc.brief is None:
                    raise HTTPException(404, "this result was produced without agents (no brief)")
                return {"rank": rc.rank, "probability": rc.probability, "features": rc.features, "brief": rc.brief,
                        "invalidations": [e for e in res.invalidations if compound_id in e.get("affected", [])]}
        raise HTTPException(404, f"{compound_id} not in shortlist")

    return app
