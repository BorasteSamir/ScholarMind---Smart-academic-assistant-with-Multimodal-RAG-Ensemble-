"""
agents/agent_router.py
=======================
Intent-based agent router — now delegates generation to the LangGraph
graph in graph.py, while keeping the identical public interface that
app.py depends on.

Public API (unchanged):
  - detect_intent(text) → IntentResult
  - AgentRouter.route(query, doc_id, force_intent) → (GenerationResult, IntentResult)
  - AgentRouter.get_agent(intent) → BaseAgent
  - get_router() → singleton AgentRouter
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from agents.qa_agent import QAAgent
from agents.summarization_agent import SummarizationAgent
from agents.explanation_agent import ExplanationAgent
from agents.formula_agent import FormulaAgent
from agents.diagram_agent import DiagramAgent
from agents.base_agent import BaseAgent
from rag.generator import GenerationResult, get_generator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# INTENT DETECTION  (unchanged — zero LLM cost)
# ─────────────────────────────────────────────

INTENT_PATTERNS: dict[str, list[str]] = {
    "summarize": [
        r"\bsummar(ize|y|ise)\b",
        r"\boverview\b",
        r"\bbrief(ly)?\b",
        r"\bwhat (is|are) (this|the) (paper|document|article|study) about\b",
        r"\btldr\b",
        r"\bkey (points|takeaways|findings|contributions)\b",
        r"\bmain (idea|ideas|argument)\b",
    ],
    "explain": [
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bwhat (is|are|does)\b",
        r"\bmeaning of\b",
        r"\bin simple (terms|words|language)\b",
        r"\bsimplify\b",
        r"\bhelp me understand\b",
        r"\bdescribe\b",
        r"\belaborate\b",
    ],
    "formula": [
        r"\bformula\b",
        r"\bequation\b",
        r"\bmath(ematics|ematical)?\b",
        r"\blatex\b",
        r"[\$\\]",
        r"\b(sum|integral|derivative|gradient|matrix|vector|eigenvalue)\b",
        r"\b(alpha|beta|gamma|delta|theta|sigma|lambda|epsilon|mu|pi)\b",
        r"\b(log|exp|sin|cos|tan|norm|loss|function)\s*\(",
    ],
    "diagram": [
        r"\b(diagram|figure|figure|fig\.?|image|chart|graph|plot|illustration)\b",
        r"\bwhat (does|is) (this|the) (image|figure|diagram|chart)\b",
        r"\bexplain (the )?(image|figure|diagram)\b",
        r"\bdraw(ing)?\b",
        r"\bvisuali[sz]ation\b",
        r"\barchitecture (diagram)?\b",
    ],
    "qa": [
        r"\b(what|who|when|where|which|how|why|did|does|is|are|was|were|can|could|should)\b",
        r"\?",
    ],
}

_COMPILED_PATTERNS = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_PATTERNS.items()
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_patterns: list[str]


def detect_intent(text: str) -> IntentResult:
    priority_order = ["summarize", "formula", "diagram", "explain", "qa"]
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for intent in priority_order:
        hits = [p.pattern for p in _COMPILED_PATTERNS[intent] if p.search(text)]
        scores[intent] = len(hits)
        matched[intent] = hits

    for intent in priority_order:
        if scores[intent] > 0:
            conf = min(scores[intent] / 5.0, 1.0)
            return IntentResult(intent=intent, confidence=conf,
                                matched_patterns=matched[intent])

    return IntentResult(intent="qa", confidence=0.3, matched_patterns=[])


# ─────────────────────────────────────────────
# AGENT ROUTER
# ─────────────────────────────────────────────

class AgentRouter:
    """
    Routes queries to specialized agents.

    .route() now delegates to the LangGraph pipeline in graph.py via
    RAGGenerator, while still exposing the same (GenerationResult, IntentResult)
    return type that app.py expects.

    .get_agent() still returns the original BaseAgent subclasses so that
    app.py's direct agent calls (summary tab, assets tab) are unaffected.
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def _get_agents(self) -> dict[str, BaseAgent]:
        if not self._agents:
            generator = get_generator()
            self._agents = {
                "qa":        QAAgent(generator=generator),
                "summarize": SummarizationAgent(generator=generator),
                "explain":   ExplanationAgent(generator=generator),
                "formula":   FormulaAgent(generator=generator),
                "diagram":   DiagramAgent(generator=generator),
            }
        return self._agents

    def route(
        self,
        query: str,
        doc_id: Optional[str] = None,
        force_intent: Optional[str] = None,
        **kwargs,
    ) -> tuple[GenerationResult, IntentResult]:
        """
        Detect intent → run the corresponding agent (which internally
        calls RAGGenerator → LangGraph).

        Returns (GenerationResult, IntentResult) — identical to before.
        """
        agents = self._get_agents()

        if force_intent and force_intent in agents:
            intent = IntentResult(intent=force_intent, confidence=1.0,
                                  matched_patterns=[])
        else:
            intent = detect_intent(query)

        agent = agents[intent.intent]
        logger.info(
            f"[Router] intent='{intent.intent}' "
            f"(conf={intent.confidence:.2f}) → {agent.agent_name}"
        )

        result = agent(query, doc_id=doc_id, **kwargs)
        return result, intent

    def get_agent(self, intent: str) -> Optional[BaseAgent]:
        return self._get_agents().get(intent)

    def available_intents(self) -> list[str]:
        return list(self._get_agents().keys())


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_default_router: Optional[AgentRouter] = None


def get_router() -> AgentRouter:
    global _default_router
    if _default_router is None:
        _default_router = AgentRouter()
    return _default_router
