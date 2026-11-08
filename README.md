# provrep — Provenance-Grounded Drug Repurposing via Agentic Evidence Adjudication

KMIT CSE (Data Science) major project, A.Y. 2026–27 · Team 14, CSD A
J Pranav Kumar (23BD1A6722) · V Sri Teja (23BD1A6758) · Shishir Gella (23BD1A6751) · C Girish (23BD1A6710)

Given a disease, `provrep` screens **every approved compound** and returns a small set of
**audited, cited, calibrated** repurposing candidates — or an empty list when nothing clears
the confidence threshold. Computational cost rises only as the candidate set narrows:

```
disease query
  │
  ▼  Stage 1  M2  R-GCN wide scorer ............ all ~1,500 compounds, milliseconds, no LLM
  ▼  Stage 2  M3  structured filters + audit ... prune -> metapath (DWPC) audit -> top 20
  ▼  Stage 3  M4  adversarial adjudication ..... Proponent / Falsifier / Referee, NLI-gated claims
  ▼  Stage 4  M5  arbitration + calibration .... shared evidence ledger, fitted calibrator
  │
  ▼  ranked, tiered, fully-sourced briefs  (or an empty list)
```

## Modules

| Module | What it does | Code |
|---|---|---|
| M1 KG construction | Hetionet v1.0 + PharmacotherapyDB loader, typed KG, leakage-free splits (`random`, `disease_holdout`, `temporal`) | `src/provrep/data/` |
| M2 Wide scorer | Relational GCN (basis decomposition) + DistMult, plain PyTorch, inductive | `src/provrep/scorer/` |
| M3 Structured filter | pruning rules, degree-weighted path counts, similarity-degeneracy flag, audit-based shortlist | `src/provrep/filters/` |
| M4 Adjudication | LangGraph state machine, separate retrieval budgets, KG + PubMed tools, NLI claim gate | `src/provrep/agents/` |
| M5 Arbitration & calibration | shared evidence ledger with cross-candidate invalidation, effective independent support, grouped-CV logistic calibrator | `src/provrep/arbitration/` |
| M6 Evaluation | ablation ladder, Recall@K / MAP / nDCG, ECE / Brier / AUROC, reliability diagram, abstention sweep | `src/provrep/eval/` |
| M7 Serving | FastAPI: `/health`, `/diseases`, `/triage`, `/triage/{id}`, `/triage/{id}/briefs/{compound}` | `src/provrep/api/` |

## Quick start

Python 3.10+. Works on Windows (PowerShell), macOS and Linux.

```bash
pip install -e ".[dev,llm]"          # add ",nli" for the HuggingFace NLI verifier
pytest                               # 44 tests, ~15 s, fully offline
```

### 1. Offline, synthetic data (no downloads, no API key)

```bash
provrep -c configs/synthetic.yaml run-all          # ~2 min on a laptop CPU
provrep -c configs/synthetic.yaml triage "Synthetic disease 22"
```

The synthetic KG uses Hetionet's exact file formats, so this exercises the same code path as real
data. It is a controlled world with known ground truth (see `data/synthetic.py`), not biology.

### 2. Real data (Hetionet v1.0 + PharmacotherapyDB)

```bash
provrep -c configs/default.yaml download          # or place the 3 files in data/raw/ yourself
provrep -c configs/default.yaml build-kg
provrep -c configs/default.yaml train             # CPU: tens of minutes; see configs/hetionet_gpu.yaml
provrep -c configs/default.yaml calibrate
provrep -c configs/default.yaml eval
provrep -c configs/default.yaml triage "multiple sclerosis"
```

Manual download (e.g. Windows `cmd`), into `data\raw\`:

```
curl -L -O https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv
curl -L -O https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz
curl -L -o indications.tsv https://raw.githubusercontent.com/dhimmel/indications/v1.0/catalog/indications.tsv
```

### 3. Real LLM agents, literature retrieval, real NLI

```bash
# PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."      bash: export ANTHROPIC_API_KEY=sk-ant-...
provrep -c configs/default.yaml --set agents.llm.provider=anthropic \
        --set agents.pubmed.enabled=true --set agents.pubmed.email=you@example.com \
        --set agents.verifier=hf  triage "multiple sclerosis"
```

Each adjudicated candidate costs about three LLM calls, so a triage run is about 60 calls.
Re-run `calibrate` and `eval` after switching providers; calibrators are specific to the agents
that produced their features. Use `--set eval.max_eval_diseases=5` to cap cost.

### 4. API

```bash
provrep -c configs/default.yaml serve --port 8000     # docs at http://127.0.0.1:8000/docs
curl -X POST localhost:8000/triage -H "content-type: application/json" -d '{"disease": "multiple sclerosis"}'
```

## Results so far

### Real Hetionet v1.0 (`configs/default.yaml`)

Split: `disease_holdout`, which is the zero-shot setting. The 15 test diseases have **no**
indication visible during training. There are 15 validation diseases for calibration, and 530
training indications. Stage 3 uses the **offline heuristic** agents, not a real LLM.
Full report: `artifacts/eval/eval_report.md`.

| level | MAP | recall@10 | nDCG@10 | recall@20 |
|---|---|---|---|---|
| B1 popularity baseline | 0.127 | 0.172 | 0.161 | 0.230 |
| B2 gene-overlap baseline | 0.054 | 0.103 | 0.068 | 0.125 |
| L1a Stage 1 R-GCN alone | 0.081 | 0.082 | 0.111 | 0.122 |
| L1 Stage 1 ensemble (R-GCN + priors) | 0.125 | 0.093 | 0.162 | 0.165 |
| L2 + filters (prune only) | 0.134 | 0.100 | 0.182 | 0.165 |
| L2 + filters (prune + audit shortlist) | 0.205 | 0.204 | 0.280 | 0.396 |
| L3 + adjudication (offline heuristic) | 0.157 | 0.203 | 0.189 | 0.396 |
| **L4 + arbitration & calibration** | **0.216** | **0.280** | **0.317** | **0.396** |

| calibration on test shortlists (n=300, 50 known positives) | ECE | Brier | AUROC |
|---|---|---|---|
| raw Stage-1 score | 0.664 | 0.569 | 0.718 |
| raw Referee score | 0.449 | 0.350 | 0.606 |
| Stage-1/2 calibrator | 0.097 | 0.145 | 0.655 |
| **full calibrator** | **0.035** | **0.131** | 0.705 |

At the default threshold of p ≥ 0.3, the system returns 26 recommendations across the 15 test
diseases and abstains on 3. Of those recommendations, 31% are known held-out indications,
against a shortlist base rate of 17%. The rest are unlabelled, not necessarily wrong.

Example, rheumatoid arthritis (a test disease; none of its 15 known drugs was visible in training):

```
 1. p=0.41 Dexamethasone          <- held-out known indication
 2. p=0.30 Prednisone             <- held-out known indication
 3. p=0.30 Cyclosporine           <- held-out known indication
 4. p=0.29 Betamethasone          <- held-out known indication
 5. p=0.27 Mycophenolic acid         (used off-label in RA)
 6. p=0.26 Methotrexate           <- held-out known indication
 7. p=0.21 Mycophenolate mofetil
 8. p=0.20 Leflunomide            <- held-out known indication
 9. p=0.20 Triamcinolone          <- held-out known indication
10. p=0.19 Hydrocortisone         <- held-out known indication
```

For migraine, it recovers topiramate and valproate (known) and also ranks zonisamide, pregabalin and
lamotrigine, all of which have been studied for migraine prophylaxis.

What the real-data run shows:

- **The R-GCN alone is weaker than a popularity prior on held-out diseases.** Many Hetionet
  indications go to broad-use drugs such as corticosteroids.
- **The Stage-1 ensemble closes that gap.** It combines the R-GCN with a popularity prior and gene
  overlap, with weights chosen on validation diseases only (R-GCN 1, popularity 0.5, gene overlap 1).
- **Most of the gain comes from Stages 2 and 4.** The audit-selected shortlist raises MAP from 0.13
  to 0.21. Arbitration and calibration give the best overall ranking (recall@10 0.28 vs 0.17 for the
  best baseline) and cut calibration error nineteen-fold (ECE 0.66 → 0.035).
- **The offline heuristic Referee, used alone to re-rank (L3), hurts.** Its hand-set rubric
  over-penalises hub-gene counter-claims on the real graph. The calibrator learns how much to
  trust it, which is why L4 recovers. A real LLM Referee is the next thing to test.
- **This is only 15 test diseases.** Differences of a few points are within noise.

### Synthetic benchmark (`configs/synthetic.yaml`)

A controlled world where the full ground truth is known, including undiscovered indications:
16 test diseases, offline agents.

| level | MAP | recall@10 | nDCG@5 | recall@20 | oracle precision@10 |
|---|---|---|---|---|---|
| B2 gene-overlap baseline | 0.196 | 0.317 | 0.247 | 0.396 | 0.637 |
| L1 Stage 1 ensemble | 0.214 | 0.362 | 0.216 | 0.479 | 0.656 |
| L2 + filters (prune + audit) | 0.343 | 0.503 | 0.265 | 0.801 | **0.969** |
| L3 + adjudication | **0.403** | 0.492 | **0.392** | 0.801 | 0.900 |
| L4 + arbitration & calibration | 0.337 | 0.446 | 0.286 | 0.801 | 0.962 |

In this synthetic world, 97% of the audited top-10 are truly efficacious. The full calibrator
reduces ECE from 0.55 to 0.07.

## Design decisions (and why)

- **Inductive scorer.** Compounds and diseases share one learned input vector per kind, so a
  node's representation comes from its biology (targets, genes, side effects) instead of a
  memorised embedding. With free per-node embeddings the model memorised the training
  indications and was near chance on held-out diseases (test MAP 0.03 vs 0.13).
- **Target edges hidden from the encoder, plus an auxiliary loss.** CtD edges never pass messages,
  so training and held-out diseases look alike to the encoder. A multi-relational link-prediction
  loss on all other metaedges gives gene and pathway embeddings structure that the ~750 CtD
  edges alone cannot.
- **Drugs see their own known uses.** In the default "compound_side" message-passing mode, training
  indications flow from disease to compound only. A compound therefore learns that it is a
  broad-use drug, while no disease, train or held-out, receives indication messages. The edge being
  scored is hidden each epoch. This raised real-data test MAP from 0.052 to 0.081.
- **Stage-1 ensemble.** The R-GCN is rank-averaged with a popularity prior and gene overlap, with
  weights chosen on validation diseases. On real data this raised recall@200 into the Stage-2
  funnel from 0.34 to 0.52.
- **Audit-based shortlist.** Stage 2 selects the top 20 by a Stage-1/2 calibrated score (Stage-1
  score plus mechanistic and similarity DWPC), not by raw Stage-1 rank. On real data this lifted
  shortlist recall@20 from 0.17 to 0.40.
- **Provenance is structural, not prompted.** Evidence objects come only from retrieval tools;
  LLM output can only cite evidence ids; the gate rejects any claim citing unretrieved
  evidence or not entailed by it; the Referee's schema has no field for new claims.
- **Stacked, cross-validated calibration.** The full calibrator uses the Stage-1/2 logit plus the
  Stage-3/4 features, with L2 strength chosen by disease-grouped cross-validation on validation
  diseases only.
- **Honest negatives.** PharmacotherapyDB `NOT` pairs are hard negatives in training and true
  negatives in evaluation (a proxy for repoDB's failed trials). Unlabelled pairs count as negative,
  so reported precision is a lower bound.

## Known limitations / next steps

- **Offline stand-ins.** Stage-3 results reported here use `OfflineHeuristicClient`, which
  builds claims straight from retrieved evidence, together with the lexical verifier. They
  measure the pipeline, not LLM reasoning. The real run needs `agents.llm.provider=anthropic`
  and `agents.verifier=hf`.
- **Stage 3 with offline agents is inconsistent.** L3 helps on synthetic data and hurts on real
  Hetionet. L4 is the best level on real data but not on synthetic. The heuristic agents only
  re-read KG facts that the audit already uses. Literature (PubMed) and a real LLM are where
  Stage 3 can add new information.
- **Default CPU config drops some edge types.** It removes GO-annotation and gene-regulation edges
  to make training feasible (about 15 minutes on 2 cores). `configs/hetionet_gpu.yaml` keeps
  all edges.
- **Time split.** Hetionet has no dates. `temporal` is implemented but needs a dates file
  (e.g. first-approval years from DrugCentral). The default is `disease_holdout`.
- **Not wired in yet:** repoDB (failed trials), delivery-feasibility rules (`StructuredFilter.delivery_rule`
  hook), OpenFDA FAERS and the React audit UI.
- Not a clinical tool: it narrows a search space for experts and makes no treatment claims.

## Layout

```
configs/          default.yaml (real data), synthetic.yaml, hetionet_gpu.yaml
src/provrep/      data/ scorer/ filters/ agents/ arbitration/ eval/ api/ pipeline.py cli.py
tests/            44 offline tests (formats, splits, R-GCN, DWPC, gate, ledger, calibration, API, CLI)
scripts/          sweep_scorer.py (Stage-1 hyper-parameter sweep)
```
