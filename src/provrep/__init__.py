"""provrep: provenance-grounded drug repurposing via agentic evidence adjudication.

Pipeline (cost rises as the candidate set narrows):
  Stage 1  scorer/      R-GCN wide scorer over every approved compound
  Stage 2  filters/     deterministic pruning + metapath audit
  Stage 3  agents/      Proponent / Falsifier / Referee with NLI-gated claims
  Stage 4  arbitration/ shared evidence ledger + fitted calibrator
"""

__version__ = "0.1.0"
