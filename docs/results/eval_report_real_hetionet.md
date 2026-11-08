# Evaluation report

Test diseases: 15  |  split: disease_holdout  |  LLM: offline-heuristic  |  verifier: lexical-heuristic

> Stage 3 used the **offline heuristic** stand-in, not a real LLM. L3/L4 numbers measure the
> pipeline's evidence handling, not LLM reasoning. Re-run with `agents.llm.provider=anthropic`.

## Ranking (macro-averaged over test diseases)

| level | map | recall@5 | ndcg@5 | recall@10 | ndcg@10 | recall@20 | ndcg@20 | recall@50 | ndcg@50 | recall@100 | ndcg@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 popularity | 0.127 | 0.088 | 0.161 | 0.172 | 0.161 | 0.230 | 0.178 | 0.467 | 0.246 | 0.524 | 0.268 |
| B2 gene overlap | 0.054 | 0.083 | 0.059 | 0.103 | 0.068 | 0.125 | 0.072 | 0.278 | 0.124 | 0.409 | 0.158 |
| L1a stage1 R-GCN only | 0.081 | 0.057 | 0.118 | 0.082 | 0.111 | 0.122 | 0.116 | 0.168 | 0.135 | 0.235 | 0.161 |
| L1 stage1 (ensemble) | 0.125 | 0.062 | 0.183 | 0.093 | 0.162 | 0.165 | 0.173 | 0.309 | 0.220 | 0.439 | 0.253 |
| L2 +filters (prune) | 0.134 | 0.062 | 0.199 | 0.100 | 0.182 | 0.165 | 0.182 | 0.309 | 0.228 | 0.439 | 0.261 |
| L2 +filters (prune+audit) | 0.205 | 0.103 | 0.271 | 0.204 | 0.280 | 0.396 | 0.316 | 0.442 | 0.327 | 0.487 | 0.343 |
| L3 +adjudication | 0.157 | 0.075 | 0.148 | 0.203 | 0.189 | 0.396 | 0.262 | 0.442 | 0.277 | 0.487 | 0.293 |
| L4 +arbitration & calibration | 0.216 | 0.103 | 0.279 | 0.280 | 0.317 | 0.396 | 0.330 | 0.442 | 0.341 | 0.487 | 0.358 |

## Calibration (test-disease shortlist candidates)

| scorer | n | positives | ECE | Brier | AUROC |
|---|---|---|---|---|---|
| raw stage-1 score | 300 | 50 | 0.664 | 0.569 | 0.718 |
| raw referee score | 300 | 50 | 0.449 | 0.350 | 0.606 |
| calibrator (stage 1-2) | 300 | 50 | 0.097 | 0.145 | 0.655 |
| calibrator (full) | 300 | 50 | 0.035 | 0.131 | 0.705 |

## Abstention: what is returned at each confidence threshold

Shortlist base rate: 0.167 (known labels). Configured threshold: 0.3. Diseases: 15.

| threshold | diseases with empty set | recommended | precision (known) | precision (oracle) |
|---|---|---|---|---|
| 0.2 | 2 | 92 | 0.293 | nan |
| 0.3 | 3 | 26 | 0.308 | nan |
| 0.4 | 10 | 5 | 0.400 | nan |
| 0.5 | 15 | 0 | nan | nan |

Positives = held-out indications; everything else in the shortlist counts as negative, so known-label
precision is a lower bound. `oracle_*` columns (synthetic data only) score against complete ground truth.
