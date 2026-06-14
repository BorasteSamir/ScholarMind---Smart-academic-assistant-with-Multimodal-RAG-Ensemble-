# Smart Academic Assistant — Solution Document

> **Version 2.0** — Refactored with LangChain + LangGraph

---

## 1. Problem Statement

Academic researchers, students, and professionals frequently work with dense, multimodal PDF documents — research papers, textbooks, and technical reports — that contain not just text but also tables, mathematical equations, and diagrams. Existing tools fall short in several ways:

- **Plain text search** ignores semantic meaning, returning irrelevant results.
- **Generic chatbots** hallucinate facts not present in the document.
- **Single-modality systems** cannot reason over figures, formulas, or tabular data.
- **Single-model generation** is brittle — one model's failure or bias directly affects the answer quality.
- **No intent awareness** — a single query pipeline cannot distinguish between a user wanting a summary vs. a formula explanation vs. a diagram analysis.
- **Opaque image content** — extracted figures are displayed without any explanation of what they show academically.
- **Brittle custom pipelines** — hand-rolled orchestration code is hard to trace, extend, and maintain; there is no built-in state, retry logic, or conditional branching.

The core problem is: *How do we build a system that can ingest any academic PDF in its full multimodal richness, understand user intent, retrieve the most relevant content, generate grounded and verifiable answers using an ensemble of models — explain every visual element automatically — and do all of this through a maintainable, observable, stateful pipeline?*

---

## 2. Proposed Solution

The Smart Academic Assistant is a **production-grade Multimodal Retrieval-Augmented Generation (RAG)** system that addresses the above problem through seven integrated capabilities:

1. **Multimodal Ingestion** — Parses PDFs to extract text, tables, images, and mathematical formulas as distinct, typed elements.
2. **Semantic Vector Search** — Embeds all content types into a shared vector space (ChromaDB) for unified semantic retrieval.
3. **Intent-Aware Multi-Agent Routing** — Detects user intent (QA, summarize, explain, formula, diagram) and dispatches to a specialized agent.
4. **LangGraph Stateful Pipeline** — A compiled `StateGraph` manages the full query lifecycle (retrieve → context → generate → validate → retry) with typed state, conditional edges, and automatic retry on low-confidence answers.
5. **5-Model LangChain Ensemble Generation** — Calls 5 `ChatGroq` LangChain models in parallel with context and prompt variation, scores all responses, and selects the best answer.
6. **Heuristic Hallucination Detection** — Validates every generated answer against the source context using zero-LLM-cost heuristics.
7. **Automatic Image Explanation** — Every extracted figure receives an AI-generated academic explanation via the DiagramAgent, displayed inline in the UI.

---

## 3. System Architecture

### 3.1 High-Level Architecture (Version 2.0)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                         │
│    Upload PDF │ Chat & Q/A Tab │ Summary Tab │ Extracted Assets Tab  │
│                                              │                        │
│                                    [Image + BLIP caption]            │
│                                    [AI Explanation — DiagramAgent]   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  (unchanged — calls same router interface)
             ┌──────────────▼──────────────┐
             │   Agent Router (unchanged)   │
             │   agents/agent_router.py     │
             │   Regex intent detection     │
             │   Lazy agent dispatch        │
             └──┬───┬───┬───┬───┬──────────┘
                │   │   │   │   │
       ┌────────┘   │   │   │   └──────────┐
       ▼            ▼   ▼   ▼              ▼
  QAAgent  SumAgent ExpAgent FormulaAgent DiagramAgent
       │            │   │   │              │
       └────────────┴───┴───┴──────────────┘
                            │  agent._run() → RAGGenerator.generate()
             ┌──────────────▼──────────────┐
             │    RAG Generator (thin)      │
             │    rag/generator.py          │
             │    Maps intent → run_graph() │
             └──────────────┬──────────────┘
                            │
             ┌──────────────▼──────────────────────────────────────────┐
             │              LangGraph StateGraph  (graph.py)            │
             │                                                          │
             │   ┌─────────────┐                                        │
             │   │detect_intent│  resolves agent config, sets           │
             │   │    node     │  retrieval_query, system_prompt        │
             │   └──────┬──────┘                                        │
             │          │                                               │
             │   ┌──────▼──────┐                                        │
             │   │  retrieve   │  ChromaDB semantic search              │
             │   │    node     │  via existing Retriever                │
             │   └──────┬──────┘          ▲                            │
             │          │                 │ retry edge                  │
             │   ┌──────▼──────┐          │                            │
             │   │build_context│  dedup, format context,               │
             │   │    node     │  build citations, compute              │
             │   └──────┬──────┘  confidence                           │
             │          │                                               │
             │   ┌──────▼──────────────────────────────┐               │
             │   │       ensemble_generate node         │               │
             │   │                                      │               │
             │   │  LangChain ChatPromptTemplate        │               │
             │   │       ↓                              │               │
             │   │  5 × ChatGroq in parallel            │               │
             │   │  (asyncio + ThreadPoolExecutor)      │               │
             │   │       ↓                              │               │
             │   │  Score + select best answer          │               │
             │   │  LangChain fallback on full failure  │               │
             │   └──────┬───────────────────────────────┘               │
             │          │                                               │
             │   ┌──────▼──────┐                                        │
             │   │hallucination│  heuristic checks (zero-LLM)          │
             │   │check node   │  sets flag + warning string            │
             │   └──────┬──────┘                                        │
             │          │                                               │
             │   should_retry()  ←── conditional edge function         │
             │     │         │                                          │
             │  "retry"    "done"                                       │
             │     │         │                                          │
             │  [retrieve] [format_result node]                         │
             │               │                                          │
             │              END                                         │
             └──────────────────────────────────────────────────────────┘
                            │
             ┌──────────────▼──────────────┐
             │      Ingestion Pipeline      │
             │  ingestion/pipeline.py       │
             │  1. PDFParser (PyMuPDF +     │
             │     pdfplumber)              │
             │  2. FormulaHandler (pix2tex/ │
             │     MathPix)                 │
             │  3. ImageCaptioner (BLIP)    │
             │  4. Chunker (sliding window) │
             └──────────────┬──────────────┘
                            │ DocumentBundle
             ┌──────────────▼──────────────┐
             │       Embeddings Layer       │
             │  sentence-transformers       │
             │  all-MiniLM-L6-v2 (384-dim) │
             │  + diskcache (skip recompute)│
             └──────────────┬──────────────┘
                            │ Vectors + Metadata
             ┌──────────────▼──────────────┐
             │     ChromaDB Vector Store    │
             │  HNSW cosine similarity      │
             │  Metadata: type, page, doc   │
             └─────────────────────────────┘
```

---

### 3.2 LangGraph State Machine — Detailed Node Map

```
AssistantState (TypedDict)
├── Inputs
│   ├── query             str
│   ├── intent            str   (qa/summarize/explain/formula/diagram)
│   ├── doc_id            Optional[str]
│   └── extra_variables   dict  (concept, formula_latex, caption …)
│
├── Pipeline intermediates
│   ├── retrieval_query   str   (may differ from query for summarize)
│   ├── chunks            list[dict]
│   ├── context_str       str
│   ├── sources           list[dict]
│   ├── citations_text    str
│   ├── user_prompt       str   (filled prompt template)
│   └── system_prompt_text str
│
├── Outputs
│   ├── answer            str
│   ├── confidence_score  float
│   ├── confidence_label  "High" | "Medium" | "Low"
│   ├── hallucination_flag bool
│   ├── hallucination_warning str
│   ├── model             str   (best ensemble model name)
│   └── ensemble_result   Optional[dict]
│
└── Control
    ├── retries           int   (max 1 automatic retry)
    ├── generation_time_sec float
    └── _start_time       float
```

```
NODE: detect_intent
  IN:  query, intent, extra_variables
  OUT: retrieval_query, system_prompt_text, _start_time
  DO:  Looks up AGENT_CONFIG[intent] to get top_k, chunk_types,
       system prompt, and template. Applies query_override for
       summarize (fixed broad query). Records wall-clock start time.

NODE: retrieve
  IN:  retrieval_query, doc_id, intent
  OUT: chunks
  DO:  Calls existing Retriever.retrieve() → ChromaDB cosine search.
       Applies per-intent top_k and chunk_types from AGENT_CONFIG.
       On retry: same call with same params (ChromaDB may return
       slightly different results due to internal HNSW scoring).

NODE: build_context
  IN:  chunks, intent, extra_variables, query
  OUT: context_str, sources, citations_text, user_prompt,
       confidence_score
  DO:  Calls existing ContextBuilder.build() for dedup + formatting.
       Fills ChatPromptTemplate variables (context, sources, question,
       concept, formula_latex, caption) to produce user_prompt.
       Computes initial confidence from top-3 chunk cosine scores.

NODE: ensemble_generate
  IN:  retrieval_query, chunks, system_prompt_text, user_prompt
  OUT: answer, confidence_score, confidence_label, model,
       ensemble_result
  DO:  Builds 5 × LangChain ChatPromptTemplate instances.
       Instantiates 5 × ChatGroq(model=…, api_key=GROQ_API_KEY).
       Runs asyncio.gather() in a thread-pool-isolated event loop.
       Scores each valid response: composite 4-axis score.
       Selects best; computes inter-model embedding confidence.
       On total failure: falls back to single ChatGroq call.

NODE: hallucination_check
  IN:  chunks, context_str, answer
  OUT: hallucination_flag, hallucination_warning, retries
  DO:  Runs existing HallucinationDetector.check():
       token overlap, numeric consistency, NER, overconfident phrasing.
       Sets retries += 0 (incremented by conditional edge logic).

CONDITIONAL EDGE: should_retry(state) → "retry" | "done"
  Triggers "retry" if ALL three conditions are true:
    1. retries < 1      (only one automatic retry allowed)
    2. hallucination_flag is True
    3. confidence_label == "Low"
  Otherwise routes to "done" → format_result.
  On retry: increments retries counter in state, loops back to
  retrieve node. This widens context on second pass because
  ChromaDB HNSW may surface different chunks.

NODE: format_result
  IN:  _start_time
  OUT: generation_time_sec
  DO:  Computes total wall-clock elapsed time. Logs final summary.
       Terminal node before END.
```

---

### 3.3 LangChain Integration Points

| LangChain Component | Where Used | What It Replaces |
|---|---|---|
| `ChatGroq(model, api_key, temperature, max_tokens)` | `graph.py` — ensemble node + fallback | Raw `Groq(api_key=…).chat.completions.create(…)` — new client per call |
| `ChatPromptTemplate.from_messages([("system",…),("human",…)])` | `graph.py` — each model call | Plain `f-string.format_map()` with silent KeyError swallowing |
| `chain = lc_prompt \| llm` | `graph.py` — LCEL pipe operator | Manual message list construction + client call |
| `chain.invoke({…})` | `graph.py` — inside thread executor | Direct `client.chat.completions.create(…)` |
| `response.content` | `graph.py` — response extraction | `response.choices[0].message.content.strip()` |

#### Why `ChatGroq` is better than the raw `Groq()` client

The original code instantiated a brand-new `Groq(api_key=...)` object on **every single model call** inside `_groq_call_sync`. This means 5 new HTTP connection objects per query.

`ChatGroq` from `langchain-groq`:
- Shares a connection pool across calls — one client object per model config
- Has built-in `tenacity`-based retry with exponential backoff on rate-limit errors (429)
- Raises typed `LangChainException` subclasses instead of raw API errors, making fallback logic cleaner
- Participates in LangSmith tracing automatically if `LANGCHAIN_API_KEY` is set in `.env`

#### Why `ChatPromptTemplate` is better than f-string formatting

The original `prompt_template.format_map(defaultdict(str, {...}))` would silently fill missing keys with empty strings, making bugs invisible. `ChatPromptTemplate`:
- Validates that all required variables are present at `.invoke()` time
- Separates system and human message roles cleanly
- Is compatible with LangSmith prompt versioning and tracing

---

### 3.4 File-by-File Change Summary

| File | Status | Change |
|---|---|---|
| `graph.py` | **NEW** | LangGraph StateGraph — the new core engine |
| `rag/generator.py` | **MODIFIED** | Delegates to `run_graph()`, wraps state in `GenerationResult` |
| `rag/ensemble_generator.py` | **MODIFIED** | Shim — delegates `_run_ensemble_sync` to `graph.py`, re-exports `EnsembleResult` |
| `agents/agent_router.py` | **MODIFIED** | Intent detection unchanged; routing now flows through graph via `RAGGenerator` |
| `requirements.txt` | **MODIFIED** | Added `langchain`, `langchain-core`, `langchain-groq`, `langchain-openai`, `langchain-anthropic`, `langgraph` |
| `app.py` | **UNTOUCHED** | Zero changes — same `router.route()` and `GenerationResult` interface |
| `agents/base_agent.py` | **UNTOUCHED** | — |
| `agents/qa_agent.py` | **UNTOUCHED** | — |
| `agents/summarization_agent.py` | **UNTOUCHED** | — |
| `agents/explanation_agent.py` | **UNTOUCHED** | — |
| `agents/formula_agent.py` | **UNTOUCHED** | — |
| `agents/diagram_agent.py` | **UNTOUCHED** | — |
| `rag/retriever.py` | **UNTOUCHED** | — |
| `rag/context_builder.py` | **UNTOUCHED** | — |
| `rag/llm_client.py` | **UNTOUCHED** | Still used for non-Groq providers (OpenAI, Anthropic, Ollama) |
| `config/prompts.py` | **UNTOUCHED** | — |
| `config/settings.py` | **UNTOUCHED** | `GROQ_API_KEY` still loaded from `.env` |
| All ingestion / embedding / evaluation files | **UNTOUCHED** | — |

---

### 3.5 Key Components Table (Updated)

| Component | File | Responsibility |
|---|---|---|
| **LangGraph State Machine** | `graph.py` | Stateful pipeline: detect → retrieve → context → ensemble → validate → retry |
| **LangChain Ensemble** | `graph.py` | 5 × `ChatGroq` + `ChatPromptTemplate` + LCEL chains in parallel |
| RAG Generator (thin) | `rag/generator.py` | Maps agent calls to `run_graph()`, wraps result in `GenerationResult` |
| Ensemble Shim | `rag/ensemble_generator.py` | Backward-compat re-export of `EnsembleResult` and `ensemble_generate_sync` |
| Agent Router | `agents/agent_router.py` | Regex intent detection + lazy agent dispatch (unchanged public API) |
| PDF Parser | `ingestion/pdf_parser.py` | Layout-aware text, table, image, link extraction |
| Chunker | `ingestion/chunker.py` | Sliding-window, token-aware chunking |
| Image Captioner | `multimodal/image_captioner.py` | BLIP auto-captioning of embedded figures |
| Formula Handler | `multimodal/formula_handler.py` | LaTeX extraction via pix2tex / MathPix API |
| Embedder | `embeddings/embedder.py` | SentenceTransformer batch embedding |
| Embedding Cache | `embeddings/cache.py` | SHA-256 keyed diskcache singleton |
| Vector Store | `embeddings/vector_store.py` | ChromaDB upsert, cosine search, metadata filtering |
| Hallucination Detector | `evaluation/hallucination_detector.py` | Heuristic answer validation (unchanged) |
| LLM Client | `rag/llm_client.py` | Fallback unified interface for OpenAI, Anthropic, Groq, Ollama |

---

## 4. Techniques Used

### 4.1 PDF Parsing & Multimodal Extraction

- **PyMuPDF (fitz)** — Layout-aware text extraction preserving reading order, page numbers, and section structure.
- **pdfplumber** — Extracts tables and converts them to Markdown format for structured representation.
- **Pillow** — Decodes and saves embedded raster images from PDF streams; filters by minimum pixel size (default 100px).
- **pix2tex / MathPix API** — Converts equation images to LaTeX strings. Local OCR via pix2tex is the default; MathPix is the cloud fallback.
- **BLIP (`Salesforce/blip-image-captioning-base`)** — Vision-language model that generates natural language captions for extracted figures, enabling semantic search over images.
- **tiktoken special token fix** — `disallowed_special=()` passed to all `encode()` calls so PDFs containing special token strings do not crash the chunker.

### 4.2 Text Chunking

- **Sliding-window chunking** with configurable token size (default 512 tokens) and overlap (default 64 tokens).
- **tiktoken** tokenizer ensures chunk boundaries respect token limits for the target LLM.
- Metadata (page number, section heading, document ID, chunk index, token count) is preserved per chunk.
- Tables and formulas are treated as **atomic chunks** — never split across boundaries.

### 4.3 Embeddings & Vector Search

- **`sentence-transformers/all-MiniLM-L6-v2`** — Lightweight transformer-based sentence embedding model producing 384-dimensional L2-normalized vectors. Unlike Word2Vec or GloVe (word-level, context-free), this model encodes full sentence meaning.
- **ChromaDB** — Persistent local vector database using HNSW indexing with cosine similarity. Stores vectors + metadata for text, image captions, formulas, and tables in a unified collection.
- **Diskcache singleton** — SHA-256 keyed cache opened once per process; avoids redundant SentenceTransformer inference on re-ingestion.

### 4.4 Intent Detection & Multi-Agent Routing

- **Regex-based intent classification** — Zero-LLM-cost pattern matching across five intent categories: `summarize`, `explain`, `formula`, `diagram`, `qa`.
- **Lazy agent initialization** — `AgentRouter.__init__` does nothing heavy; agents and the generator are built on first query, eliminating startup latency.
- Priority ordering: `summarize > formula > diagram > explain > qa`.
- Five specialized agents with tailored system prompts and prompt templates.
- **V2 change:** Routing now flows through the LangGraph graph inside `RAGGenerator.generate()`. The router and all agent classes are completely unchanged — the graph is an internal implementation detail invisible to them.

### 4.5 LangGraph Stateful Pipeline (New in V2)

The core architectural upgrade replacing the hand-rolled linear pipeline in `generator.py`:

**Why LangGraph:**

| Old approach | LangGraph approach |
|---|---|
| Linear function calls, no state object | Typed `AssistantState` TypedDict carries all intermediate data |
| If confidence low, answer returned as-is | Conditional edge: auto-retry retrieval + generation once |
| No built-in tracing or observability | Every node execution visible in LangSmith trace |
| Retry logic would require manual if/else | `should_retry()` edge function + state `retries` counter |
| Hard to add new pipeline steps | Add a node + two edges — rest of graph unchanged |

**Retry Logic:**

The `should_retry()` conditional edge triggers a second full pass through `retrieve → build_context → ensemble_generate → hallucination_check` when:
- `retries < 1` (cap at one retry to avoid infinite loops)
- `hallucination_flag is True`
- `confidence_label == "Low"`

On the retry, `retries` is incremented to 1 in the state so the condition is never satisfied again, guaranteeing termination.

**State isolation:** Each `graph.invoke(initial_state)` call creates a fresh state dict. There is no shared mutable state between concurrent Streamlit user sessions.

### 4.6 LangChain ChatGroq Ensemble (New in V2)

The 5-model ensemble is now built with LangChain components:

**Per-model chain construction:**
```python
lc_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human",  "{user_prompt}"),
])
llm = ChatGroq(
    model=config.model,
    api_key=GROQ_API_KEY,   # same key from .env, no change
    temperature=config.temperature,
    max_tokens=LLM_MAX_TOKENS,
)
chain = lc_prompt | llm   # LCEL pipe — validated composition
response = chain.invoke({"system_prompt": …, "user_prompt": …})
```

**Context Variation (unchanged from V1):**

| Model | LangChain Model ID | Context Strategy | Temperature |
|---|---|---|---|
| 1 | `ChatGroq("llama-3.3-70b-versatile")` | Top-3 chunks | 0.2 |
| 2 | `ChatGroq("llama-3.1-8b-instant")` | Top-5 chunks | 0.4 |
| 3 | `ChatGroq("qwen/qwen3-32b")` | Top-7 chunks | 0.6 |
| 4 | `ChatGroq("meta-llama/llama-4-scout-17b-16e-instruct")` | Shuffled chunks | 0.7 |
| 5 | `ChatGroq("qwen/qwen3-32b")` | Summarized context | 0.5 |

**Parallel Execution:** `asyncio.gather()` launches all 5 `chain.invoke()` calls simultaneously. Each blocking call runs in a thread pool via `loop.run_in_executor()`. The sync wrapper spawns a brand-new thread with its own event loop, completely isolated from Streamlit's internal event loop.

**Scoring Formula (unchanged from V1):**
```
score = 0.4 × token_overlap(response, context)
      + 0.3 × (1 − hallucination_risk_score)
      + 0.2 × cosine_similarity(embed(response), embed(context))
      + 0.1 × answer_quality(length + citations + no_refusals)
```

**Confidence:** Mean pairwise cosine similarity across all 5 response embeddings. High agreement = high confidence.

**Fallback:** If all ensemble models fail, a single `ChatGroq(GROQ_MODEL)` call is made using the LCEL chain before returning an error message.

### 4.7 Automatic Image Explanation

- Every image extracted from the PDF is automatically explained by the `DiagramAgent` when the Extracted Assets tab opens.
- Explanations are cached in `st.session_state.image_explanations` keyed by `{doc_id}_img_{index}` — generated once, reused on tab revisits.
- Cache is cleared when a new document is processed.
- The DiagramAgent uses the BLIP caption + page number to retrieve surrounding text context from ChromaDB, producing a grounded academic explanation rather than a generic description.
- **V2:** The DiagramAgent's `_run()` → `RAGGenerator.generate()` → LangGraph graph path applies automatically here too, including the retry edge.

### 4.8 Hallucination Detection (Zero-LLM Heuristics)

Four independent checks run on every generated answer (unchanged from V1):

| Check | Method | Trigger |
|---|---|---|
| Token Overlap | Jaccard-style overlap of answer tokens vs. context tokens (stop-words removed) | Overlap < 0.25 |
| Numeric Consistency | Set difference of numeric literals in answer vs. context | > 2 unsupported numbers |
| Named Entity Verification | Capitalized proper noun extraction; checks presence in context | > 3 unsupported entities |
| Overconfident Phrasing | Regex for "definitely", "certainly", "without doubt", etc. | Triggered when overlap also low |

**V2 integration:** The `hallucination_check` LangGraph node calls the same `HallucinationDetector` class. If the flag is set AND confidence is Low, the LangGraph conditional edge automatically retries retrieval + generation once — this was impossible in V1 without a major code change.

### 4.9 Performance Optimizations

- **Split `@st.cache_resource`** — ChromaDB connection and agent router cached independently.
- **Lazy LLM client** — `RAGGenerator` defers all initialization to first `generate()` call.
- **LangChain connection pooling** — `ChatGroq` reuses HTTP connections across calls within the ensemble instead of opening 5 new connections per query.
- **Graph singleton** — `get_graph()` compiles the `StateGraph` once and reuses the compiled object for all subsequent queries.
- **`disallowed_special=()`** — prevents tiktoken from crashing on PDFs containing special token strings.

---

## 5. Mapping with NLP / LLM Concepts

| Concept | Implementation in This Project |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | Core architecture: retrieve relevant chunks → inject into prompt → generate grounded answer |
| **Dense Retrieval** | SentenceTransformer embeddings + ChromaDB cosine similarity search |
| **Chunking & Sliding Window** | Token-aware chunking with overlap to preserve cross-boundary context |
| **Prompt Engineering** | `ChatPromptTemplate` per agent and per ensemble model; validated variable injection |
| **Zero-shot Prompting** | Zero-shot prompts with explicit output format instructions |
| **Grounding** | LLM instructed to answer only from provided context; explicit "I don't know" fallback |
| **Hallucination Mitigation** | Heuristic post-generation validation + LangGraph automatic retry on low-confidence answers |
| **Intent Classification** | Rule-based NLU using regex patterns — zero-LLM-cost routing |
| **Multi-Agent Systems** | Specialized agents with distinct roles, orchestrated by a central lazy router |
| **Model Ensembling / Bagging** | 5 `ChatGroq` models with context + prompt + parameter variation; best response selected by composite score |
| **Multimodal LLM Inputs** | BLIP captions and LaTeX formulas bridge vision/math modalities into text for the LLM |
| **Semantic Search** | Vector similarity over heterogeneous content types (text, image captions, formulas, tables) |
| **Context Window Management** | Top-k retrieval + token-aware chunking ensures context fits within LLM token limits |
| **Source Attribution** | Every answer includes page-level citations built from chunk metadata |
| **Confidence Estimation** | Ensemble inter-model agreement (pairwise cosine similarity) + retrieval similarity score |
| **Async Parallel Inference** | `asyncio.gather()` + `run_in_executor()` for non-blocking parallel `ChatGroq` calls |
| **Stateful Agentic Pipelines** | LangGraph `StateGraph` with typed state, conditional retry edges, and compiled graph |
| **LCEL (LangChain Expression Language)** | `prompt \| llm` pipe chains for each ensemble model; validated composition |
| **Conditional Branching in Agents** | `should_retry()` edge in LangGraph: routes to retry or termination based on runtime state |

---

## 6. Evaluation Metrics

### 6.1 QA Evaluation (SQuAD-style)

Implemented in `evaluation/qa_eval.py`:

- **Exact Match (EM)** — 1 if normalized prediction equals normalized reference, else 0.
- **Token-level F1** — Token overlap between prediction and reference (Precision, Recall, F1).
- **Best-of-N F1** — Maximum F1 across multiple reference answers.
- **Batch Evaluation** — Macro-averaged EM, F1, Precision, Recall.

### 6.2 Summarization Evaluation (ROUGE)

Implemented in `evaluation/rouge_eval.py`:

- **ROUGE-1** — Unigram overlap; vocabulary coverage.
- **ROUGE-2** — Bigram overlap; phrase-level fluency.
- **ROUGE-L** — Longest Common Subsequence; sentence-level structure.

### 6.3 Ensemble Response Scoring

Scoring runs inside the `ensemble_generate` LangGraph node:

| Component | Weight | Method |
|---|---|---|
| Token Overlap | 0.4 | Jaccard token overlap (answer vs. context) |
| Hallucination Safety | 0.3 | `1 − hallucination_risk_score` |
| Semantic Similarity | 0.2 | Cosine similarity of response and context embeddings |
| Answer Quality | 0.1 | Length heuristic + citation presence + no refusal phrases |

### 6.4 Ensemble Confidence

- **Inter-model confidence** = mean pairwise cosine similarity across all 5 response embeddings.
- High (≥ 0.70): models strongly agree.
- Medium (0.45–0.70): moderate agreement.
- Low (< 0.45): models diverge — LangGraph conditional edge triggers one automatic retry.

### 6.5 Retrieval Quality

- **Confidence Score** — Mean cosine similarity of top-k retrieved chunks.
- **Score Threshold** — Chunks below 0.3 similarity discarded before context building.

### Summary Table

| Metric | Scope | Method |
|---|---|---|
| Exact Match (EM) | QA accuracy | SQuAD normalization |
| Token F1 | QA soft match | Token overlap |
| ROUGE-1/2/L | Summarization quality | rouge-score library |
| Ensemble Score | Response selection | Composite 4-axis score (graph node) |
| Ensemble Confidence | Inter-model agreement | Pairwise cosine similarity |
| Hallucination Risk | Answer faithfulness | Token overlap heuristic (graph node) |
| Retry Trigger | Pipeline resilience | LangGraph conditional edge |
| Retrieval Confidence | Retrieval quality | Mean cosine similarity |

---

## 7. Applications and Future Scope

### Current Applications

- **Academic Research Assistance** — Upload papers, ask targeted questions, get summaries, understand formulas.
- **Student Learning Tool** — Concept explanations in simple terms with analogies, directly from course materials.
- **Literature Review Acceleration** — Extract key findings, methodology, and conclusions from multiple papers.
- **Technical Documentation Q&A** — Query technical manuals for precise answers with page citations.
- **Formula & Equation Understanding** — Symbol-by-symbol breakdowns of mathematical expressions.
- **Visual Content Understanding** — Every extracted figure automatically explained in academic context.

### Future Scope

| Area | Enhancement |
|---|---|
| **Multi-document RAG** | Add a `multi_retrieve` node to the LangGraph graph for cross-document synthesis |
| **LangGraph Persistence** | Add `MemorySaver` checkpointer to LangGraph for conversation memory across turns |
| **LangSmith Observability** | Set `LANGCHAIN_API_KEY` in `.env` to get automatic per-node tracing of every graph run |
| **Human-in-the-loop** | LangGraph `interrupt_before` to pause graph and request user confirmation on low-confidence answers |
| **Citation Graph Integration** | Link answers to external databases (Semantic Scholar, arXiv) |
| **Fine-tuned Embeddings** | Domain-specific models fine-tuned on scientific corpora (S2ORC) |
| **NLI Hallucination Check** | Add an NLI entailment node to the LangGraph graph after `hallucination_check` |
| **Table Reasoning** | TAPAS integration for numerical reasoning over extracted tables |
| **OCR for Scanned PDFs** | Tesseract or AWS Textract for non-digital documents |
| **Microservices Deployment** | Wrap `graph.py` behind FastAPI; use LangGraph Cloud for managed deployment |
| **Evaluation Dashboard** | Real-time ROUGE and EM tracking per document in the Streamlit UI |
| **User Feedback Loop** | Thumbs-up/down ratings fed back as LangSmith dataset for fine-tuning |
| **Multilingual Support** | `paraphrase-multilingual-MiniLM-L12-v2` for non-English papers |
| **Dynamic Ensemble** | Auto-select ChatGroq models based on query type and available Groq quota |

---

## 8. Technology Stack Summary

| Layer | Technology | Version Note |
|---|---|---|
| UI | Streamlit | Unchanged |
| **Agentic Pipeline** | **LangGraph (`StateGraph`)** | **New in V2** |
| **LLM Wrappers** | **LangChain (`ChatGroq`, `ChatPromptTemplate`, LCEL)** | **New in V2** |
| PDF Parsing | PyMuPDF (fitz), pdfplumber | Unchanged |
| Image Processing | Pillow, BLIP (`Salesforce/blip-image-captioning-base`) | Unchanged |
| Formula OCR | pix2tex (local), MathPix API (cloud) | Unchanged |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) | Unchanged |
| Vector Database | ChromaDB (persistent, HNSW cosine) | Unchanged |
| Embedding Cache | diskcache (SHA-256 keyed, singleton) | Unchanged |
| Ensemble LLMs | Groq API — llama-3.3-70b, llama-3.1-8b, qwen3-32b, llama-4-scout | Via `ChatGroq` in V2 |
| Fallback LLMs | OpenAI (GPT-4o-mini), Anthropic (Claude 3 Haiku), Ollama (local) | Via `rag/llm_client.py` |
| Async Execution | asyncio + ThreadPoolExecutor | Unchanged |
| Tokenization | tiktoken (cl100k_base) | Unchanged |
| Evaluation | rouge-score, custom SQuAD-style QA eval | Unchanged |
| Configuration | python-dotenv, environment variables | Unchanged |
| Logging | Python logging module | Unchanged |
| Language | Python 3.10+ | Unchanged |

---

## 9. V2 Dependency Additions

```
langchain           # Core LangChain abstractions
langchain-core      # LCEL, ChatPromptTemplate, message types
langchain-groq      # ChatGroq wrapper for Groq API
langchain-openai    # ChatOpenAI (available for future use)
langchain-anthropic # ChatAnthropic (available for future use)
langgraph           # StateGraph, conditional edges, END
```

All installed into the existing project virtualenv. The same `GROQ_API_KEY` from `.env` is passed directly to `ChatGroq(api_key=GROQ_API_KEY)` — no new keys or environment variables required.
