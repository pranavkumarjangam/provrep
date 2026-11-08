"""M4 - Stage 3 adversarial adjudication (Proponent / Falsifier / Referee)."""
from .graph import Adjudicator
from .llm import AnthropicClient, OfflineHeuristicClient, make_llm
from .schemas import Brief, Claim, Evidence, GatedClaim, RefereeOutput
from .tools import KnowledgeGraphTool, PubMedTool
from .verifier import HFNLIVerifier, LexicalVerifier, LLMVerifier, gate_claim, make_verifier

__all__ = [
    "Adjudicator", "AnthropicClient", "OfflineHeuristicClient", "make_llm", "Brief", "Claim", "Evidence",
    "GatedClaim", "RefereeOutput", "KnowledgeGraphTool", "PubMedTool", "HFNLIVerifier", "LexicalVerifier",
    "LLMVerifier", "gate_claim", "make_verifier",
]
