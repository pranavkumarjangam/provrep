import numpy as np
import pandas as pd

from provrep.agents.schemas import Brief, Claim, Evidence, GatedClaim, RefereeOutput, RubricScores
from provrep.arbitration import Calibrator, EvidenceLedger
from provrep.filters.rules import Candidate


def _brief(cid, support_paths, refute=()):
    cand = Candidate(compound=0, compound_id=cid, compound_name=cid, disease=1, disease_id="D", disease_name="D",
                     stage1_logit=0.0, stage1_rank=1)
    ev, claims = [], []
    for i, nodes in enumerate(support_paths):
        e = Evidence(id=f"kg:CbGaD:{cid}|{'|'.join(nodes)}", source="kg", kind="CbGaD", text="t",
                     node_ids=[cid, *nodes, "D"], retrieved_by="proponent")
        ev.append(e)
        claims.append(GatedClaim(claim=Claim(id=f"P{i}", role="proponent", stance="supports", text="t", evidence_ids=[e.id]), passed=True, results=[]))
    for j, (kind, node) in enumerate(refute):
        e = Evidence(id=f"kg:{kind}:{node}", source="kg", kind=kind, text="t", node_ids=[node], retrieved_by="falsifier")
        ev.append(e)
        claims.append(GatedClaim(claim=Claim(id=f"F{j}", role="falsifier", stance="refutes", text="t", evidence_ids=[e.id]), passed=True, results=[]))
    verdict = RefereeOutput(scores=RubricScores(mechanistic_plausibility=0.5, evidence_quality=0.5, safety_concern=0.1,
                                                counter_evidence_strength=0.2), overall=0.5, tier="moderate", rationale="r")
    return Brief(candidate=cand, evidence=ev, claims=claims, verdict=verdict, llm="t", verifier="t")


def test_shared_support_is_split_and_correlated_paths_count_once():
    led = EvidenceLedger()
    led.add(_brief("A", [["HUB"], ["G1"], ["G1"]]))        # two paths via G1 are correlated
    led.add(_brief("B", [["HUB"]]))
    eff_a, raw_a = led.effective_support("A")
    assert raw_a == 2.0                  # groups: HUB, G1
    assert eff_a == 0.5 + 1.0            # HUB shared by 2 candidates -> 1/2
    assert led.effective_support("B") == (0.5, 1.0)


def test_hub_invalidation_propagates_across_candidates():
    led = EvidenceLedger(hub_weight=0.5)
    led.add(_brief("A", [["HUB"], ["G1"]], refute=[("hub_target", "HUB")]))
    led.add(_brief("B", [["HUB"]]))
    before = led.effective_support("B")[0]
    led.arbitrate()
    assert len(led.events) == 1 and set(led.events[0].affected) == {"A", "B"}
    assert led.effective_support("B")[0] == before * 0.5      # B never raised the issue but is corrected
    assert led.features("A")["refutation"] > 0


def test_referee_rejection_invalidates_evidence_everywhere():
    a = _brief("A", [["G1"]])
    a = a.model_copy(update={"verdict": a.verdict.model_copy(update={"rejected_claim_ids": ["P0"]})})
    led = EvidenceLedger()
    led.add(a)
    led.arbitrate()
    assert led.effective_support("A") == (0.0, 0.0)


def test_calibrator_grouped_cv_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = (rng.random(400) < 1 / (1 + np.exp(-2 * x))).astype(int)
    X = pd.DataFrame({"f": x})
    cal = Calibrator(["f"]).fit(X, y, groups=np.repeat(np.arange(20), 20))
    assert cal.C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0) and cal.coefficients()["f"] > 0
    cal.save(tmp_path / "c.joblib")
    p = Calibrator.load(tmp_path / "c.joblib").predict(X)
    assert np.allclose(p, cal.predict(X)) and 0 < p.min() and p.max() < 1


def test_calibrator_single_class_falls_back_to_base_rate():
    cal = Calibrator(["f"]).fit(pd.DataFrame({"f": [1.0, 2.0]}), np.array([0, 0]))
    assert (cal.predict(pd.DataFrame({"f": [5.0]})) == 0).all()
