import httpx
import pytest

from provrep.agents import Adjudicator, KnowledgeGraphTool, LexicalVerifier, OfflineHeuristicClient, PubMedTool, gate_claim
from provrep.agents.llm import AnthropicClient, _extract_json
from provrep.agents.schemas import Claim, ClaimDrafts, Evidence, RefereeOutput
from provrep.agents.tools import Budget


def ev(eid="kg:x", text="Aspirin binds PTGS2; Pain is associated with PTGS2.", entities=("Aspirin", "PTGS2", "Pain"), role="proponent"):
    return Evidence(id=eid, source="kg", kind="CbGaD", text=text, entities=list(entities), retrieved_by=role)


def claim(text, ids, role="proponent"):
    return Claim(id="P1", role=role, stance="supports", text=text, evidence_ids=ids)


def test_gate_accepts_faithful_claim():
    e = ev()
    g = gate_claim(claim("Aspirin binds PTGS2, which is associated with Pain.", [e.id]), {e.id: e}, LexicalVerifier())
    assert g.passed


def test_gate_rejects_uncited_and_invented_citations():
    e = ev()
    v = LexicalVerifier()
    assert not gate_claim(claim("Aspirin binds PTGS2.", []), {e.id: e}, v).passed
    g = gate_claim(claim("Aspirin binds PTGS2.", ["pmid:999"]), {e.id: e}, v)
    assert not g.passed and "never retrieved" in g.reason


def test_gate_rejects_hallucinated_detail():
    e = ev()
    g = gate_claim(claim("Aspirin binds PTGS2 and EGFR in a phase 3 trial of 4000 patients with Pain.", [e.id]), {e.id: e}, LexicalVerifier())
    assert not g.passed


def test_gate_rejects_negated_claim():
    e = ev()
    g = gate_claim(claim("Aspirin does not bind PTGS2 associated with Pain.", [e.id]), {e.id: e}, LexicalVerifier())
    assert not g.passed and g.reason == "contradicted by cited evidence"


def test_budget():
    b = Budget(2)
    assert b.take() and b.take() and not b.take() and b.remaining == 0


def _pipe_parts(tiny_pipeline):
    p = tiny_pipeline
    d = int(p.split.diseases("test")[0])
    return p, d, p.filter.run(d, p.scorer.scores(d)).shortlist


def test_adjudication_end_to_end_offline(tiny_pipeline):
    p, d, shortlist = _pipe_parts(tiny_pipeline)
    b = p.adjudicator.adjudicate(shortlist[0])
    assert b.llm == "offline-heuristic"
    ids = {e.id for e in b.evidence}
    for g in b.claims:                      # every passed claim cites retrieved evidence of its own role
        if g.passed:
            assert set(g.claim.evidence_ids) <= ids
            assert all(b.evidence_by_id()[i].retrieved_by == g.claim.role for i in g.claim.evidence_ids)
    assert set(b.verdict.cited_claim_ids) <= {g.claim.id for g in b.claims if g.passed}


def test_falsifier_never_echoes_proponent_evidence(tiny_pipeline):
    p, d, shortlist = _pipe_parts(tiny_pipeline)
    for cand in shortlist:
        b = p.adjudicator.adjudicate(cand)
        pro = {e.id for e in b.evidence if e.retrieved_by == "proponent"}
        fal = {e.id for e in b.evidence if e.retrieved_by == "falsifier"}
        assert not pro & fal


class BadRefereeLLM(OfflineHeuristicClient):
    name = "bad-referee"

    def complete_json(self, system, user, schema, context=None):
        out = super().complete_json(system, user, schema, context)
        if schema is RefereeOutput:
            out = out.model_copy(update={"cited_claim_ids": out.cited_claim_ids + ["X99"], "rejected_claim_ids": ["P404"]})
        return out


class CheatingProponentLLM(OfflineHeuristicClient):
    name = "cheater"

    def complete_json(self, system, user, schema, context=None):
        if schema is ClaimDrafts and context["task"] == "proponent":
            return ClaimDrafts.model_validate({"claims": [{"text": "It cures everything.", "evidence_ids": ["pmid:1"]}]})
        return super().complete_json(system, user, schema, context)


def test_referee_cannot_reference_unknown_claims(tiny_pipeline):
    p, d, shortlist = _pipe_parts(tiny_pipeline)
    adj = Adjudicator(p.adjudicator.kg_tool, BadRefereeLLM(), LexicalVerifier(), p.cfg)
    b = adj.adjudicate(shortlist[0])
    assert "X99" not in b.verdict.cited_claim_ids and "P404" not in b.verdict.rejected_claim_ids
    assert any("unknown/unverified claim ids" in n for n in b.notes)


def test_fabricated_citation_is_blocked(tiny_pipeline):
    p, d, shortlist = _pipe_parts(tiny_pipeline)
    adj = Adjudicator(p.adjudicator.kg_tool, CheatingProponentLLM(), LexicalVerifier(), p.cfg)
    b = adj.adjudicate(shortlist[0])
    pro = [g for g in b.claims if g.claim.role == "proponent"]
    assert pro and not any(g.passed for g in pro)


def test_pubmed_tool_parses_and_respects_budget_and_exclusions():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
      <ArticleTitle>Drug X in disease Y</ArticleTitle><Abstract><AbstractText>X reduced Y activity.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

    def handler(req: httpx.Request):
        if "esearch" in req.url.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": ["111", "222"]}})
        assert req.url.params["id"] == "111"          # 222 excluded
        return httpx.Response(200, text=xml)

    tool = PubMedTool(client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = tool.search("q", "falsifier", Budget(5), exclude={"pmid:222"})
    assert [e.id for e in out] == ["pmid:111"] and "reduced" in out[0].text and out[0].url.endswith("/111/")
    assert tool.search("q", "falsifier", Budget(0)) == []


def test_pubmed_network_failure_is_soft():
    def handler(req):
        raise httpx.ConnectError("blocked")

    tool = PubMedTool(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert tool.search("q", "proponent", Budget(3)) == []


def test_extract_json_variants():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json('Sure! {"a": {"b": 2}} done') == '{"a": {"b": 2}}'


def test_anthropic_client_retries_invalid_json(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient("some-model")
    replies = iter(["not json at all", '{"claims": [{"text": "t", "evidence_ids": ["e"]}]}'])

    class Msg:
        def __init__(self, text):
            self.content = [type("B", (), {"text": text})()]
            self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()

    monkeypatch.setattr(client.client.messages, "create", lambda **kw: Msg(next(replies)))
    out = client.complete_json("sys", "user", ClaimDrafts)
    assert out.claims[0].text == "t" and client.usage["calls"] == 2
