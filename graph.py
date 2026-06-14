"""
graph.py
=========
LangGraph-powered core engine for the Smart Academic Assistant.

Graph topology (StateGraph):
  detect_intent → retrieve → build_context → ensemble_generate
      → hallucination_check → format_result → END

  Conditional retry edge:
    hallucination_check → retrieve   (if confidence < threshold AND retries < 1)
    hallucination_check → format_result  (otherwise)

LangChain is used for:
  - ChatGroq / ChatOpenAI / ChatAnthropic model wrappers
  - ChatPromptTemplate for validated prompt construction
  - Each ensemble model is a distinct ChatGroq(model=...) instance

The graph is invoked from:
  - rag/generator.py   (RAGGenerator.generate)
  - agents/agent_router.py  (AgentRouter.route)

The Streamlit app (app.py) is NOT changed.
"""

import asyncio
import concurrent.futures
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import numpy as np
from langgraph.graph import StateGraph, END

# LangChain LLM wrappers
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL,
    RETRIEVAL_TOP_K, LLM_MAX_TOKENS, LLM_TEMPERATURE,
)
from config.prompts import (
    SYSTEM_QA, QA_PROMPT,
    SYSTEM_SUMMARIZER, SUMMARIZATION_PROMPT,
    SYSTEM_EXPLAINER, EXPLANATION_PROMPT,
    SYSTEM_FORMULA, FORMULA_PROMPT,
    SYSTEM_DIAGRAM, DIAGRAM_PROMPT,
)
from rag.retriever import get_retriever
from rag.context_builder import ContextBuilder
from evaluation.hallucination_detector import HallucinationDetector

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# AGENT CONFIG MAP
# ─────────────────────────────────────────────

AGENT_CONFIG: dict[str, dict] = {
    "qa":        {"system": SYSTEM_QA,        "template": QA_PROMPT,
                  "top_k": RETRIEVAL_TOP_K,   "chunk_types": ["text", "table"],
                  "extra_keys": ["question"]},
    "summarize": {"system": SYSTEM_SUMMARIZER, "template": SUMMARIZATION_PROMPT,
                  "top_k": 15,                "chunk_types": ["text", "table"],
                  "extra_keys": [],           "query_override":
                      "main objective contribution methodology results conclusions key findings",
                  "score_threshold": 0.0},
    "explain":   {"system": SYSTEM_EXPLAINER,  "template": EXPLANATION_PROMPT,
                  "top_k": 6,                 "chunk_types": ["text"],
                  "extra_keys": ["concept"]},
    "formula":   {"system": SYSTEM_FORMULA,    "template": FORMULA_PROMPT,
                  "top_k": 5,                 "chunk_types": ["text", "formula"],
                  "extra_keys": ["formula_latex"]},
    "diagram":   {"system": SYSTEM_DIAGRAM,    "template": DIAGRAM_PROMPT,
                  "top_k": 5,                 "chunk_types": ["text", "image"],
                  "extra_keys": ["caption"]},
}

# ─────────────────────────────────────────────
# ENSEMBLE MODEL DEFINITIONS  (LangChain ChatGroq)
# ─────────────────────────────────────────────

@dataclass
class EnsembleModelConfig:
    model: str
    temperature: float
    top_p: float
    top_k_chunks: int
    system_prompt: str


ENSEMBLE_CONFIGS: list[EnsembleModelConfig] = [
    EnsembleModelConfig(
        model="llama-3.3-70b-versatile", temperature=0.2, top_p=0.90,
        top_k_chunks=3,
        system_prompt=(
            "You are an academic assistant. "
            "Answer strictly from the provided context. Be concise and precise. "
            "Cite page numbers using [Page X] notation. "
            "If the answer is not in the context, say so explicitly."
        ),
    ),
    EnsembleModelConfig(
        model="llama-3.1-8b-instant", temperature=0.4, top_p=0.95,
        top_k_chunks=5,
        system_prompt=(
            "You are an academic assistant. "
            "Explain step-by-step using only the provided context. "
            "Structure your answer clearly with numbered steps where applicable. "
            "Always cite sources."
        ),
    ),
    EnsembleModelConfig(
        model="qwen/qwen3-32b", temperature=0.6, top_p=0.85,
        top_k_chunks=7,
        system_prompt=(
            "You are an academic assistant. "
            "Provide a structured bullet-point answer based solely on the context. "
            "Use '•' for main points and '  -' for sub-points. "
            "Include page citations."
        ),
    ),
    EnsembleModelConfig(
        model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7,
        top_p=0.90, top_k_chunks=5,
        system_prompt=(
            "You are an academic assistant. "
            "Focus on accuracy. Avoid assumptions or external knowledge. "
            "Answer only what the context supports. "
            "Flag any uncertainty explicitly."
        ),
    ),
    EnsembleModelConfig(
        model="qwen/qwen3-32b", temperature=0.5, top_p=0.80,
        top_k_chunks=5,
        system_prompt=(
            "You are an academic assistant. "
            "Give a detailed explanation with concrete examples drawn from the context. "
            "Be thorough but stay grounded in the provided material."
        ),
    ),
]

# ─────────────────────────────────────────────
# GRAPH STATE  (TypedDict required by LangGraph)
# ─────────────────────────────────────────────

class AssistantState(TypedDict):
    # Inputs
    query: str
    intent: str
    doc_id: Optional[str]
    extra_variables: dict

    # Pipeline intermediates
    retrieval_query: str
    chunks: list[dict]
    context_str: str
    sources: list[dict]
    citations_text: str
    user_prompt: str
    system_prompt_text: str

    # Outputs
    answer: str
    confidence_score: float
    confidence_label: str
    hallucination_flag: bool
    hallucination_warning: str
    model: str
    ensemble_result: Optional[dict]

    # Internal control
    retries: int
    generation_time_sec: float
    _start_time: float


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _build_context_from_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        header = f"[Source {i}] Page {page}"
        if section:
            header += f" | {section}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "No context available."


def _summarize_context(chunks: list[dict], max_chars: int = 1500) -> str:
    sentences = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        first = re.split(r"(?<=[.!?])\s+", text)
        if first:
            sentences.append(f"[Page {chunk.get('page', '?')}] {first[0]}")
    summary = " ".join(sentences)
    return summary[:max_chars]


def _prepare_context_for_model(chunks: list[dict], idx: int) -> str:
    if idx == 0:
        return _build_context_from_chunks(chunks[:3])
    elif idx == 1:
        return _build_context_from_chunks(chunks[:5])
    elif idx == 2:
        return _build_context_from_chunks(chunks[:7])
    elif idx == 3:
        shuffled = chunks[:5][:]
        random.shuffle(shuffled)
        return _build_context_from_chunks(shuffled)
    else:
        return _summarize_context(chunks[:5])


# ─────────────────────────────────────────────
# SCORING  (same logic as original ensemble_generator.py)
# ─────────────────────────────────────────────

def _score_token_overlap(response: str, context: str) -> float:
    from evaluation.hallucination_detector import _token_overlap_ratio
    return _token_overlap_ratio(response, context)


def _score_hallucination_safety(response: str, context: str) -> float:
    detector = HallucinationDetector()
    return round(1.0 - detector.score(context, response), 4)


def _score_semantic_similarity(response: str, context: str) -> float:
    try:
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        r_vec = embedder.embed(response)
        c_vec = embedder.embed(context[:2000])
        return round(float(max(0.0, min(1.0, np.dot(r_vec, c_vec)))), 4)
    except Exception as exc:
        logger.warning(f"Semantic similarity failed: {exc}")
        return 0.0


def _score_answer_quality(response: str) -> float:
    score = 0.0
    length = len(response.strip())
    if 100 <= length <= 800:
        score += 0.5
    elif 50 <= length < 100 or 800 < length <= 1500:
        score += 0.3
    elif length > 1500:
        score += 0.2
    if re.search(r"\[Page \d+\]", response):
        score += 0.3
    refusal_patterns = [
        r"i (don't|do not|cannot|can't) (know|find|answer)",
        r"not (found|available|present) in (the )?context",
        r"no (relevant|sufficient) (information|context)",
    ]
    if any(re.search(p, response.lower()) for p in refusal_patterns):
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 4)


def _compute_score(response: str, context: str) -> float:
    return round(
        0.4 * _score_token_overlap(response, context)
        + 0.3 * _score_hallucination_safety(response, context)
        + 0.2 * _score_semantic_similarity(response, context)
        + 0.1 * _score_answer_quality(response),
        4,
    )


def _compute_confidence(responses: list[str]) -> float:
    try:
        from embeddings.embedder import get_embedder
        embedder = get_embedder()
        vecs = np.array([embedder.embed(r) for r in responses])
        sim_matrix = vecs @ vecs.T
        n = len(responses)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(sim_matrix[i, j])
                count += 1
        return round(max(0.0, min(1.0, total / count if count > 0 else 0.0)), 4)
    except Exception as exc:
        logger.warning(f"Confidence computation failed: {exc}")
        return 0.5


# ─────────────────────────────────────────────
# LANGCHAIN ENSEMBLE CALL  (single model, async)
# ─────────────────────────────────────────────

async def _call_one_model(
    config: EnsembleModelConfig,
    context: str,
    query: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Call one Groq model via LangChain ChatGroq inside a thread executor."""
    async with semaphore:
        start = time.time()
        try:
            # Build prompt with LangChain ChatPromptTemplate
            lc_prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ])

            template_vars = defaultdict(str, {
                "context": context,
                "question": query,
                "concept": query,
                "sources": "See context above.",
                "formula_latex": "",
                "caption": "",
            })
            filled_user = prompt_template.format_map(template_vars)

            # LangChain ChatGroq wrapper — uses the same GROQ_API_KEY from .env
            llm = ChatGroq(
                model=config.model,
                api_key=GROQ_API_KEY,
                temperature=config.temperature,
                max_tokens=LLM_MAX_TOKENS,
            )

            chain = lc_prompt | llm

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chain.invoke({
                    "system_prompt": config.system_prompt,
                    "user_prompt": filled_user,
                }),
            )

            raw = response.content if hasattr(response, "content") else str(response)
            text = _strip_think_tags(raw.strip())
            elapsed = round(time.time() - start, 2)
            logger.info(
                f"[Graph/Ensemble] model={config.model} time={elapsed}s "
                f"len={len(text)}"
            )
            return {"model": config.model, "response": text,
                    "context": context, "error": None, "elapsed": elapsed}

        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            logger.error(f"[Graph/Ensemble] model={config.model} failed: {exc}")
            return {"model": config.model, "response": "",
                    "context": context, "error": str(exc), "elapsed": elapsed}


async def _run_ensemble(
    query: str,
    chunks: list[dict],
    prompt_template: str,
) -> dict[str, Any]:
    """Run all 5 LangChain ChatGroq models in parallel and score results."""
    semaphore = asyncio.Semaphore(5)
    tasks = [
        _call_one_model(cfg, _prepare_context_for_model(chunks, i),
                        query, prompt_template, semaphore)
        for i, cfg in enumerate(ENSEMBLE_CONFIGS)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=False)
    valid = [r for r in raw_results if r["response"] and not r["error"]]

    if not valid:
        return {
            "final_answer": "⚠️ All ensemble models failed. Please try again.",
            "confidence": 0.0, "best_model": "none",
            "all_responses": [], "scores": [], "models": [],
        }

    scores = [_compute_score(r["response"], r["context"]) for r in valid]
    best_idx = int(np.argmax(scores))
    confidence = _compute_confidence([r["response"] for r in valid])

    return {
        "final_answer": valid[best_idx]["response"],
        "confidence": confidence,
        "best_model": valid[best_idx]["model"],
        "all_responses": [r["response"] for r in valid],
        "scores": scores,
        "models": [r["model"] for r in valid],
    }


def _run_ensemble_sync(query: str, chunks: list[dict], prompt_template: str) -> dict:
    """Sync wrapper — safe to call from Streamlit (spawns its own event loop)."""
    def _in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _run_ensemble(query, chunks, prompt_template)
            )
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result(timeout=120)


# ─────────────────────────────────────────────
# LANGGRAPH NODE FUNCTIONS
# ─────────────────────────────────────────────

_retriever = None
_context_builder = ContextBuilder()
_hallucination_detector = HallucinationDetector()


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def node_detect_intent(state: AssistantState) -> dict:
    """
    Resolves agent config based on already-detected intent.
    Builds the effective retrieval query and picks system/template.
    """
    intent = state["intent"]
    cfg = AGENT_CONFIG.get(intent, AGENT_CONFIG["qa"])
    extra = state.get("extra_variables", {})

    retrieval_query = cfg.get("query_override") or state["query"]

    return {
        "retrieval_query": retrieval_query,
        "system_prompt_text": cfg["system"],
        "_start_time": time.time(),
    }


def node_retrieve(state: AssistantState) -> dict:
    """Semantic retrieval from ChromaDB via existing Retriever."""
    intent = state["intent"]
    cfg = AGENT_CONFIG.get(intent, AGENT_CONFIG["qa"])

    chunks = _get_retriever().retrieve(
        query=state["retrieval_query"],
        doc_id=state.get("doc_id"),
        top_k=cfg["top_k"],
        chunk_types=cfg.get("chunk_types"),
        score_threshold=cfg.get("score_threshold"),
    )
    return {"chunks": chunks}


def node_build_context(state: AssistantState) -> dict:
    """Format retrieved chunks into context string + sources list."""
    context_str, sources, confidence = _context_builder.build(state["chunks"])
    citations_text = _context_builder.build_sources_citation(sources)

    intent = state["intent"]
    cfg = AGENT_CONFIG.get(intent, AGENT_CONFIG["qa"])
    extra = state.get("extra_variables", {})

    template_vars = {
        "context": context_str,
        "sources": citations_text,
        "question": state["query"],
        "concept": state["query"],
        **extra,
    }
    try:
        user_prompt = cfg["template"].format(**template_vars)
    except KeyError as e:
        logger.warning(f"Prompt template missing key {e}; using raw template")
        user_prompt = cfg["template"]

    return {
        "context_str": context_str,
        "sources": sources,
        "citations_text": citations_text,
        "user_prompt": user_prompt,
        "confidence_score": confidence,
    }


def node_ensemble_generate(state: AssistantState) -> dict:
    """
    Run 5 LangChain ChatGroq models in parallel.
    Falls back to a single ChatGroq call if ensemble fails entirely.
    """
    intent = state["intent"]
    cfg = AGENT_CONFIG.get(intent, AGENT_CONFIG["qa"])

    try:
        ensemble = _run_ensemble_sync(
            query=state["retrieval_query"],
            chunks=state["chunks"],
            prompt_template=cfg["template"],
        )
        answer = ensemble["final_answer"]
        confidence = max(ensemble["confidence"], state["confidence_score"])
        model = ensemble["best_model"]
        ensemble_result = ensemble

    except Exception as exc:
        logger.error(f"[Graph] Ensemble failed, falling back to single model: {exc}")
        try:
            # LangChain fallback: single ChatGroq call
            lc_prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ])
            llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            chain = lc_prompt | llm
            response = chain.invoke({
                "system_prompt": state["system_prompt_text"],
                "user_prompt": state["user_prompt"],
            })
            answer = _strip_think_tags(response.content.strip())
            confidence = state["confidence_score"]
            model = GROQ_MODEL
            ensemble_result = None
        except Exception as fallback_exc:
            logger.error(f"[Graph] Fallback also failed: {fallback_exc}")
            answer = f"⚠️ Generation failed: {fallback_exc}"
            confidence = 0.0
            model = "none"
            ensemble_result = None

    conf_label = ContextBuilder.confidence_label(confidence)
    return {
        "answer": answer,
        "confidence_score": confidence,
        "confidence_label": conf_label,
        "model": model,
        "ensemble_result": ensemble_result,
    }


def node_hallucination_check(state: AssistantState) -> dict:
    """Heuristic hallucination detection — zero LLM cost."""
    if state["chunks"]:
        flag, warning = _hallucination_detector.check(
            context=state["context_str"],
            answer=state["answer"],
        )
    else:
        flag, warning = False, ""

    return {
        "hallucination_flag": flag,
        "hallucination_warning": warning,
        "retries": state.get("retries", 0),
    }


def node_format_result(state: AssistantState) -> dict:
    """Compute final timing and return — terminal node."""
    elapsed = round(time.time() - state.get("_start_time", time.time()), 2)
    logger.info(
        f"[Graph] Done in {elapsed}s | "
        f"intent={state['intent']} | "
        f"confidence={state['confidence_label']} | "
        f"hallucination={'yes' if state['hallucination_flag'] else 'no'}"
    )
    return {"generation_time_sec": elapsed}


# ─────────────────────────────────────────────
# CONDITIONAL EDGE: retry once if low confidence
# ─────────────────────────────────────────────

def should_retry(state: AssistantState) -> str:
    """
    Retry retrieval + generation once if confidence is Low and
    hallucination was detected, and we haven't retried yet.
    """
    if (
        state.get("retries", 0) < 1
        and state.get("hallucination_flag", False)
        and state.get("confidence_label", "Low") == "Low"
    ):
        logger.info("[Graph] Low confidence + hallucination → retrying retrieval")
        return "retry"
    return "done"


# ─────────────────────────────────────────────
# BUILD THE LANGGRAPH STATE MACHINE
# ─────────────────────────────────────────────

def _build_graph() -> Any:
    g = StateGraph(AssistantState)

    g.add_node("detect_intent",       node_detect_intent)
    g.add_node("retrieve",            node_retrieve)
    g.add_node("build_context",       node_build_context)
    g.add_node("ensemble_generate",   node_ensemble_generate)
    g.add_node("hallucination_check", node_hallucination_check)
    g.add_node("format_result",       node_format_result)

    g.set_entry_point("detect_intent")
    g.add_edge("detect_intent",     "retrieve")
    g.add_edge("retrieve",          "build_context")
    g.add_edge("build_context",     "ensemble_generate")
    g.add_edge("ensemble_generate", "hallucination_check")

    # Conditional: retry once on low-confidence + hallucination
    g.add_conditional_edges(
        "hallucination_check",
        should_retry,
        {
            "retry": "retrieve",    # widen retrieval, regenerate
            "done":  "format_result",
        },
    )
    g.add_edge("format_result", END)

    return g.compile()


# Module-level compiled graph singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ─────────────────────────────────────────────
# PUBLIC INTERFACE  (called by generator.py / agent_router.py)
# ─────────────────────────────────────────────

def run_graph(
    query: str,
    intent: str,
    doc_id: Optional[str] = None,
    extra_variables: Optional[dict] = None,
) -> AssistantState:
    """
    Execute the full LangGraph pipeline and return the final state.

    Args:
        query:           User's natural-language input
        intent:          Detected agent intent (qa/summarize/explain/formula/diagram)
        doc_id:          Document scope for retrieval
        extra_variables: Agent-specific prompt variables (concept, formula_latex, caption…)

    Returns:
        Final AssistantState dict with all output fields populated.
    """
    initial: AssistantState = {
        "query": query,
        "intent": intent,
        "doc_id": doc_id,
        "extra_variables": extra_variables or {},
        "retrieval_query": query,
        "chunks": [],
        "context_str": "",
        "sources": [],
        "citations_text": "",
        "user_prompt": "",
        "system_prompt_text": "",
        "answer": "",
        "confidence_score": 0.0,
        "confidence_label": "Low",
        "hallucination_flag": False,
        "hallucination_warning": "",
        "model": "",
        "ensemble_result": None,
        "retries": 0,
        "generation_time_sec": 0.0,
        "_start_time": time.time(),
    }
    graph = get_graph()
    final_state = graph.invoke(initial)
    return final_state
