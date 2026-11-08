"""Retrieval tools for the adjudication agents (M4).

KnowledgeGraphTool  concrete metapath instances and KG facts; always available
PubMedTool          NCBI E-utilities (esearch + efetch); needs network access
                    to eutils.ncbi.nlm.nih.gov

Each agent owns a separate retrieval budget, and the Falsifier's retrieval
skips anything the Proponent already retrieved, so it cannot simply echo the
Proponent's sources.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx
import numpy as np

from ..data.kg import COMPOUND, GENE, KnowledgeGraph
from ..filters.metapaths import MetapathProfiler, PathInstance
from ..filters.rules import Candidate
from ..utils import get_logger
from .schemas import Evidence, Role

log = get_logger(__name__)

VERBS = {
    "CbG": "binds", "DaG": "is associated with", "GpPW": "participates in", "GiG": "interacts with",
    "CrC": "resembles", "CtD": "treats", "DrD": "resembles", "CuG": "upregulates", "CdG": "downregulates",
    "DuG": "upregulates", "DdG": "downregulates", "CpD": "palliates", "CcSE": "causes",
}


@dataclass
class Budget:
    limit: int
    used: int = 0

    def take(self, n: int = 1) -> bool:
        if self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def remaining(self) -> int:
        return self.limit - self.used


class KnowledgeGraphTool:
    def __init__(self, kg: KnowledgeGraph, profiler: MetapathProfiler, max_paths: int = 4, hub_quantile: float = 0.99):
        self.kg, self.profiler, self.max_paths = kg, profiler, max_paths
        genes = kg.members(GENE)
        deg = np.zeros(kg.num_nodes)
        for me in ("CbG", "DaG"):          # the gene is the target end of both metaedges
            e = kg.edges.get(me)
            if e is not None and len(e):
                np.add.at(deg, e[:, 1], 1)
        self.gene_degree = deg
        gdeg = deg[genes] if len(genes) else np.zeros(1)
        self.hub_threshold = float(np.quantile(gdeg, hub_quantile)) if len(gdeg) else np.inf
        cse = kg.edges.get("CcSE")
        se = np.bincount(kg.local_idx[cse[:, 0]], minlength=len(kg.members(COMPOUND))) if cse is not None and len(cse) else np.zeros(len(kg.members(COMPOUND)))
        self.se_count = se
        self.se_q90 = float(np.quantile(se, 0.9)) if len(se) else np.inf

    def is_hub(self, node: int) -> bool:
        return self.kg.node_kinds[node] == GENE and self.gene_degree[node] > self.hub_threshold

    # ---------------------------------------------------------- evidence
    def path_evidence(self, p: PathInstance, role: Role) -> Evidence:
        kg = self.kg
        steps = self.profiler.metapaths[p.metapath]
        parts = []
        for (me, rev), a, b in zip(steps, p.nodes[:-1], p.nodes[1:]):
            s, o = (b, a) if rev else (a, b)
            parts.append(f"{kg.name(s)} {VERBS.get(me, me)} {kg.name(o)}")
        ids = [kg.node_ids[n] for n in p.nodes]
        return Evidence(
            id=f"kg:{p.metapath}:" + "|".join(ids), source="kg", kind=p.metapath,
            text="; ".join(parts) + ".", entities=[kg.name(n) for n in p.nodes], node_ids=ids,
            weight=float(p.weight), retrieved_by=role,
        )

    def supporting(self, cand: Candidate, budget: Budget, exclude: set[str] = frozenset()) -> list[Evidence]:
        out: list[Evidence] = []
        order = self.profiler.mechanistic + self.profiler.similarity
        for mp in order:
            for p in self.profiler.paths(cand.compound, cand.disease, mp, self.max_paths):
                ev = self.path_evidence(p, "proponent")
                if ev.id in exclude:
                    continue
                if not budget.take():
                    return out
                out.append(ev)
        return out

    def counter(self, cand: Candidate, budget: Budget, exclude: set[str] = frozenset()) -> list[Evidence]:
        """Facts that bear against the candidate. All are checkable KG facts."""
        kg = self.kg
        c, d = cand.compound, cand.disease
        cname, dname = kg.name(c), kg.name(d)
        out: list[Evidence] = []

        def add(ev: Evidence):
            if ev.id not in exclude and budget.take():
                out.append(ev)

        mech_paths = {mp: self.profiler.paths(c, d, mp, 50) for mp in self.profiler.mechanistic}
        n_mech = sum(len(v) for v in mech_paths.values())
        if n_mech == 0:
            add(Evidence(
                id=f"kg:no_mechanism:{kg.node_ids[c]}|{kg.node_ids[d]}", source="kg", kind="no_mechanistic_path",
                text=f"No mechanistic path connects {cname} to {dname}: none of its targets reach genes associated with {dname} "
                     f"(checked {', '.join(self.profiler.mechanistic)}).",
                entities=[cname, dname], node_ids=[kg.node_ids[c], kg.node_ids[d]], retrieved_by="falsifier"))
        # mechanistic links that route through promiscuous hub genes
        hubs = sorted({n for ps in mech_paths.values() for p in ps for n in p.nodes[1:-1] if self.is_hub(n)})
        via_hub_only = n_mech > 0 and all(any(self.is_hub(n) for n in p.nodes[1:-1]) for ps in mech_paths.values() for p in ps)
        for h in hubs[:3]:
            add(Evidence(
                id=f"kg:hub_target:{kg.node_ids[h]}", source="kg", kind="hub_target",
                text=f"{kg.name(h)} is a promiscuous hub gene with {int(self.gene_degree[h])} compound-binding and disease-association edges, "
                     f"so a path through {kg.name(h)} is weak evidence of a specific mechanism."
                     + (f" All mechanistic paths from {cname} to {dname} pass through hub genes." if via_hub_only else ""),
                entities=[kg.name(h), cname, dname] if via_hub_only else [kg.name(h)], node_ids=[kg.node_ids[h]],
                retrieved_by="falsifier"))
        if cand.degenerate:
            add(Evidence(
                id=f"kg:similarity_driven:{kg.node_ids[c]}|{kg.node_ids[d]}", source="kg", kind="similarity_driven",
                text=f"Only {cand.mechanistic_share:.0%} of the degree-weighted path support from {cname} to {dname} is mechanistic; "
                     f"the rest comes from resemblance to known treatments or similar diseases.",
                entities=[cname, dname], node_ids=[kg.node_ids[c], kg.node_ids[d]], retrieved_by="falsifier"))
        local = kg.local_idx[c]
        if self.se_count[local] > self.se_q90:
            add(Evidence(
                id=f"kg:side_effect_burden:{kg.node_ids[c]}", source="kg", kind="side_effect_burden",
                text=f"{cname} causes {int(self.se_count[local])} recorded side effects, more than 90% of compounds.",
                entities=[cname], node_ids=[kg.node_ids[c]], retrieved_by="falsifier"))
        # targets with no route to the disease (off-target share)
        cbg = self.profiler.raw.get("CbG")
        if cbg is not None and cand.n_targets:
            on = {p.nodes[1] for mp in ("CbGaD", "CbGpPWpGaD", "CbGiGaD") for p in mech_paths.get(mp, [])}
            off = cand.n_targets - len(on)
            if off > 0 and off / cand.n_targets >= 0.5:
                add(Evidence(
                    id=f"kg:off_target:{kg.node_ids[c]}|{kg.node_ids[d]}", source="kg", kind="off_target",
                    text=f"{off} of {cname}'s {cand.n_targets} known targets have no path to genes associated with {dname}.",
                    entities=[cname, dname], node_ids=[kg.node_ids[c], kg.node_ids[d]], retrieved_by="falsifier"))
        return out


class PubMedTool:
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, max_results: int = 5, email: str | None = None, api_key: str | None = None,
                 client: httpx.Client | None = None, timeout: float = 20.0):
        self.max_results, self.email, self.api_key = max_results, email, api_key
        self.client = client or httpx.Client(timeout=timeout)

    def _params(self, **kw) -> dict:
        p = {"db": "pubmed", **kw}
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def search(self, query: str, role: Role, budget: Budget, exclude: set[str] = frozenset()) -> list[Evidence]:
        if budget.remaining <= 0:
            return []
        try:
            r = self.client.get(self.ESEARCH, params=self._params(term=query, retmode="json", retmax=self.max_results))
            r.raise_for_status()
            pmids = [p for p in r.json().get("esearchresult", {}).get("idlist", []) if f"pmid:{p}" not in exclude]
            if not pmids:
                return []
            r = self.client.get(self.EFETCH, params=self._params(id=",".join(pmids), retmode="xml"))
            r.raise_for_status()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("PubMed query failed (%s): %s", query, e)
            return []
        out = []
        for art in ET.fromstring(r.text).iter("PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = (art.findtext(".//ArticleTitle") or "").strip()
            abstract = " ".join((t.text or "") for t in art.iter("AbstractText")).strip()
            if not pmid or not (title or abstract) or not budget.take():
                continue
            out.append(Evidence(
                id=f"pmid:{pmid}", source="pubmed", kind="pubmed_abstract",
                text=re.sub(r"\s+", " ", f"{title} {abstract}")[:3000], entities=[],
                retrieved_by=role, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))
        return out

    @staticmethod
    def support_query(drug: str, disease: str) -> str:
        return f'"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract]'

    @staticmethod
    def counter_query(drug: str, disease: str) -> str:
        return (f'"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract] AND '
                '(ineffective OR "no significant" OR failed OR adverse OR toxicity OR contraindicated)')
