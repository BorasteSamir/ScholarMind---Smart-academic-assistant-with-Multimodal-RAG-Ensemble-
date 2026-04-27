# Smart Academic Assistant — Solution Document

---

## 1. Problem Statement

Academic researchers, students, and professionals frequently work with dense, multimodal PDF documents — research papers, textbooks, and technical reports — that contain not just text but also tables, mathematical equations, and diagrams. Existing tools fall short in several ways:

- **Plain text search** ignores semantic meaning, returning irrelevant results.
- **Generic chatbots** hallucinate facts not present in the document.
- **Single-modality systems** cannot reason over figures, formulas, or tabular data.
- **Single-model generation** is brittle — one model's failure or bias directly affects the answer quality.
- **No intent awareness** — a single query pipeline cannot distinguish between a user wanting a summary vs. a formula explanation vs. a diagram analysis.
- **Opaque image content** — extracted figures are displayed without any explanation of what they show academically.

The core problem is: *How do we build a system that can ingest any academic PDF in its full multimodal richness, understand user intent, retrieve the most relevant content, generate grounded and verifiable answers using an ensemble of models — and explain every visual element automatically?*

---

## 2. Proposed Solution

The Smart Academic Assistant is a **production-grade Multimodal Retrieval-Augmented Generation (RAG)** system that addresses the above problem through six integrated capabilities:

1. **Multimodal Ingestion** — Parses PDFs to extract text, tables, images, and mathematical formulas as distinct, typed elements.
2. **Semantic Vector Search** — Embeds all content types into a shared vector space (ChromaDB) for unified semantic retrieval.
3. **Intent-Aware Multi-Agent Routing** — Detects user intent (QA, summarize, explain, formula, diagram) and dispatches to a specialized agent.
4. **5-Model Ensemble Generation** — Calls 5 Groq LLMs in parallel with context and prompt variation, scores all responses, and selects the best answer.
5. **Heuristic Hallucination Detection** — Validates every generated answer against the source context using zero-LLM-cost heuristics.
6. **Automatic Image Explanation** — Every extracted figure receives an AI-generated academic explanation via the DiagramAgent, displayed inline in the UI.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                         │
│    Upload PDF │ Chat & Q/A Tab │ Summary Tab │ Extracted Assets Tab  │
│                                              │                        │
│                                    [Image + BLIP caption]            │
│                                    [AI Explanation — DiagramAgent]   │
└───────────────────────────┬──────────────────────────────────────────┘
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
             └──────────────┬──────────────┘
                            │
             ┌──────────────▼──────────────┐
             │       Agent Router           │
             │  Regex intent detection      │
             │  (zero-LLM-cost)             │
             └──┬───┬───┬───┬───┬──────────┘
                │   │   │   │   │
       ┌────────┘   │   │   │   └──────────┐
       ▼            ▼   ▼   ▼              ▼
  QAAgent  SumAgent ExpAgent FormulaAgent DiagramAgent
       │            │   │   │              │
       └────────────┴───┴───┴──────────────┘
                            │
             ┌──────────────▼──────────────┐
             │       RAG Generator          │
             │  rag/generator.py            │
             │  Retriever → ContextBuilder  │
             │       ↓                      │
             │  ensemble_generate_sync()    │
             └──────────────┬──────────────┘
                            │
             ┌──────────────▼──────────────────────────────┐
             │         Ensemble Generator                   │
             │  rag/ensemble_generator.py                   │
             │                                              │
             │  asyncio.gather() — 5 models in parallel     │
             │                                              │
             │  Model 1: llama-3.3-70b  top_k=3  temp=0.2  │
             │  Model 2: llama-3.1-8b   top_k=5  temp=0.4  │
             │  Model 3: qwen3-32b      top_k=7  temp=0.6  │
             │  Model 4: llama-4-scout  shuffled temp=0.7  │
             │  Model 5: qwen3-32b      summary  temp=0.5  │
             │                                              │
             │  Score each → select best → confidence       │
             └──────────────┬──────────────────────────────┘
                            │
             ┌──────────────▼──────────────┐
             │   Hallucination Detector     │
             │  Token overlap, NER,         │
             │  numeric consistency,        │
             │  overconfident phrasing      │
             └─────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|---|---|---|
| PDF Parser | `ingestion/pdf_parser.py` | Layout-aware text, table, image, link extraction |
| Chunker | `ingestion/chunker.py` | Sliding-window, token-aware chunking (`disallowed_special=()` fix) |
| Image Captioner | `multimodal/image_captioner.py` | BLIP auto-captioning of embedded figures |
| Formula Handler | `multimodal/formula_handler.py` | LaTeX extraction via pix2tex / MathPix API |
| Embedder | `embeddings/embedder.py` | SentenceTransformer batch embedding |
| Embedding Cache | `embeddings/cache.py` | SHA-256 keyed diskcache singleton |
| Vector Store | `embeddings/vector_store.py` | ChromaDB upsert, cosine search, metadata filtering |
| Agent Router | `agents/agent_router.py` | Lazy-init regex intent detection and agent dispatch |
| Ensemble Generator | `rag/ensemble_generator.py` | 5-model async parallel generation + scoring |
| RAG Generator | `rag/generator.py` | Orchestrates retrieval → ensemble → hallucination check |
| LLM Client | `rag/llm_client.py` | Unified interface for OpenAI, Anthropic, Groq, Ollama |
| Hallucination Detector | `evaluation/hallucination_detector.py` | Heuristic answer validation |

---

## 4. Techniques Used

### 4.1 PDF Parsing & Multimodal Extraction

- **PyMuPDF (fitz)** — Layout-aware text extraction preserving reading order, page numbers, and section structure.
- **pdfplumber** — Extracts tables and converts them to Markdown format for structured representation.
- **Pillow** — Decodes and saves embedded raster images from PDF streams; filters by minimum pixel size (default 100px).
- **pix2tex / MathPix API** — Converts equation images to LaTeX strings. Local OCR via pix2tex is the default; MathPix is the cloud fallback.
- **BLIP (`Salesforce/blip-image-captioning-base`)** — Vision-language model that generates natural language captions for extracted figures, enabling semantic search over images.
- **tiktoken special token fix** — `disallowed_special=()` passed to all `encode()` calls so PDFs containing special token strings (e.g. `<|endofprompt|>`) don't crash the chunker.

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

### 4.5 5-Model Ensemble Generation

The core innovation replacing the single LLM call:

**Context Variation:**
| Model | Strategy |
|---|---|
| Model 1 | Top-3 highest-scoring chunks |
| Model 2 | Top-5 highest-scoring chunks |
| Model 3 | Top-7 highest-scoring chunks |
| Model 4 | Top-5 chunks in shuffled order |
| Model 5 | Extractive summary of top-5 chunks |

**Prompt Variation:** Each model receives a different system prompt style (concise, step-by-step, bullet-point, accuracy-focused, detailed).

**Generation Parameter Variation:** Temperature [0.2, 0.4, 0.6, 0.7, 0.5], top_p [0.90, 0.95, 0.85, 0.90, 0.80].

**Parallel Execution:** `asyncio.gather()` launches all 5 API calls simultaneously. Each blocking Groq call runs in a thread pool via `loop.run_in_executor()` to avoid blocking the event loop.

**Streamlit Compatibility:** The sync wrapper always spawns a brand-new thread with its own event loop (`asyncio.new_event_loop()`), completely isolated from Streamlit's internal event loop.

**Scoring Formula:**
```
score = 0.4 × token_overlap(response, context)
      + 0.3 × (1 − hallucination_risk_score)
      + 0.2 × cosine_similarity(embed(response), embed(context))
      + 0.1 × answer_quality(length + citations + no_refusals)
```

**Confidence:** Mean pairwise cosine similarity across all 5 response embeddings. High agreement = high confidence.

**Fallback:** If all ensemble models fail, the system automatically falls back to the single configured LLM client.

### 4.6 Automatic Image Explanation

- Every image extracted from the PDF is automatically explained by the `DiagramAgent` when the Extracted Assets tab opens.
- Explanations are cached in `st.session_state.image_explanations` keyed by `{doc_id}_img_{index}` — generated once, reused on tab revisits.
- Cache is cleared when a new document is processed.
- The DiagramAgent uses the BLIP caption + page number to retrieve surrounding text context from ChromaDB, producing a grounded academic explanation rather than a generic description.

### 4.7 Hallucination Detection (Zero-LLM Heuristics)

Four independent checks run on every generated answer:

| Check | Method | Trigger |
|---|---|---|
| Token Overlap | Jaccard-style overlap of answer tokens vs. context tokens (stop-words removed) | Overlap < 0.25 |
| Numeric Consistency | Set difference of numeric literals in answer vs. context | > 2 unsupported numbers |
| Named Entity Verification | Capitalized proper noun extraction; checks presence in context | > 3 unsupported entities |
| Overconfident Phrasing | Regex for "definitely", "certainly", "without doubt", etc. | Triggered when overlap also low |

### 4.8 Performance Optimizations

- **Split `@st.cache_resource`** — ChromaDB connection and agent router cached independently; ChromaDB reconnect no longer re-triggers LLM client init.
- **Lazy LLM client** — `RAGGenerator.__init__` defers `get_llm_client()` to first `generate()` call.
- **Lazy hallucination detector** — instantiated on first use, not at construction time.
- **ChromaDB count log** demoted from INFO to DEBUG — removes a blocking DB round-trip from the startup path.
- **`disallowed_special=()`** — prevents tiktoken from crashing on PDFs containing special token strings.

---

## 5. Mapping with NLP / LLM Concepts

| Concept | Implementation in This Project |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | Core architecture: retrieve relevant chunks → inject into prompt → generate grounded answer |
| **Dense Retrieval** | SentenceTransformer embeddings + ChromaDB cosine similarity search |
| **Chunking & Sliding Window** | Token-aware chunking with overlap to preserve cross-boundary context |
| **Prompt Engineering** | Separate system prompts and user prompt templates per agent and per ensemble model |
| **Zero-shot Prompting** | Zero-shot prompts with explicit output format instructions |
| **Grounding** | LLM instructed to answer only from provided context; explicit "I don't know" fallback |
| **Hallucination Mitigation** | Heuristic post-generation validation (token overlap, NER, numeric checks) |
| **Intent Classification** | Rule-based NLU using regex patterns — zero-LLM-cost routing |
| **Multi-Agent Systems** | Specialized agents with distinct roles, orchestrated by a central lazy router |
| **Model Ensembling / Bagging** | 5 models with context + prompt + parameter variation; best response selected by composite score |
| **Multimodal LLM Inputs** | BLIP captions and LaTeX formulas bridge vision/math modalities into text for the LLM |
| **Semantic Search** | Vector similarity over heterogeneous content types (text, image captions, formulas, tables) |
| **Context Window Management** | Top-k retrieval + token-aware chunking ensures context fits within LLM token limits |
| **Source Attribution** | Every answer includes page-level citations built from chunk metadata |
| **Confidence Estimation** | Ensemble inter-model agreement (pairwise cosine similarity) + retrieval similarity score |
| **Async Parallel Inference** | `asyncio.gather()` + `run_in_executor()` for non-blocking parallel LLM calls |

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

Implemented in `rag/ensemble_generator.py`:

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
- Low (< 0.45): models diverge — treat answer with caution.

### 6.5 Retrieval Quality

- **Confidence Score** — Mean cosine similarity of top-k retrieved chunks.
- **Score Threshold** — Chunks below 0.3 similarity discarded before context building.

### Summary Table

| Metric | Scope | Method |
|---|---|---|
| Exact Match (EM) | QA accuracy | SQuAD normalization |
| Token F1 | QA soft match | Token overlap |
| ROUGE-1/2/L | Summarization quality | rouge-score library |
| Ensemble Score | Response selection | Composite 4-axis score |
| Ensemble Confidence | Inter-model agreement | Pairwise cosine similarity |
| Hallucination Risk | Answer faithfulness | Token overlap heuristic |
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
| **Multi-document RAG** | Cross-document retrieval and synthesis across multiple uploaded papers |
| **Citation Graph Integration** | Link answers to external databases (Semantic Scholar, arXiv) |
| **Fine-tuned Embeddings** | Domain-specific models fine-tuned on scientific corpora (S2ORC) |
| **NLI Hallucination Check** | Complement heuristics with Natural Language Inference entailment model |
| **Table Reasoning** | TAPAS integration for numerical reasoning over extracted tables |
| **OCR for Scanned PDFs** | Tesseract or AWS Textract for non-digital documents |
| **Microservices Deployment** | FastAPI + managed vector DB + Celery task queues |
| **Evaluation Dashboard** | Real-time ROUGE and EM tracking per document in the Streamlit UI |
| **User Feedback Loop** | Thumbs-up/down ratings to build fine-tuning dataset |
| **Multilingual Support** | `paraphrase-multilingual-MiniLM-L12-v2` for non-English papers |
| **Audio/Video Lectures** | Whisper transcription + indexing alongside PDFs |
| **Dynamic Ensemble** | Auto-select models based on query type and available Groq quota |

---

## 8. Technology Stack Summary

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (fitz), pdfplumber |
| Image Processing | Pillow, BLIP (`Salesforce/blip-image-captioning-base`) |
| Formula OCR | pix2tex (local), MathPix API (cloud) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Database | ChromaDB (persistent, HNSW cosine) |
| Embedding Cache | diskcache (SHA-256 keyed, singleton) |
| Ensemble LLMs | Groq API — llama-3.3-70b, llama-3.1-8b, qwen3-32b, llama-4-scout |
| Fallback LLMs | OpenAI (GPT-4o-mini), Anthropic (Claude 3 Haiku), Ollama (local) |
| Async Execution | asyncio + ThreadPoolExecutor |
| Tokenization | tiktoken (cl100k_base) |
| Evaluation | rouge-score, custom SQuAD-style QA eval |
| Configuration | python-dotenv, environment variables |
| Logging | Python logging module |
| Language | Python 3.10+ |
