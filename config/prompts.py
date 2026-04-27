"""
config/prompts.py
=================
Centralized prompt templates for all agents in the Smart Academic Assistant.

Design principles:
- Ground every answer in retrieved context
- Instruct model to say "I don't know" rather than hallucinate
- Always include source citations
- Keep instructions declarative and precise
"""

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_BASE = (
    "You are a highly accurate Academic AI Assistant. "
    "Your answers must be strictly grounded in the provided context. "
    "If the answer is not found in the context, state: "
    "'The provided documents do not contain enough information to answer this question.' "
    "Never fabricate information. Always cite the source page or section."
)

SYSTEM_QA = SYSTEM_BASE + (
    "\n\nYou are answering a specific question about an academic document. "
    "Provide a precise, factual answer with citations in the format [Page X] or [Section Y]."
)

SYSTEM_SUMMARIZER = SYSTEM_BASE + (
    "\n\nYou are summarizing an academic document. "
    "Provide a structured summary covering: Objective, Methodology, Key Findings, and Conclusions."
)

SYSTEM_EXPLAINER = SYSTEM_BASE + (
    "\n\nYou are explaining an academic concept in simple, accessible terms. "
    "Use analogies where helpful. Avoid jargon unless you define it."
)

SYSTEM_FORMULA = SYSTEM_BASE + (
    "\n\nYou are explaining a mathematical formula or equation found in an academic document. "
    "Describe: (1) what the formula represents, (2) each variable/symbol, (3) its practical significance."
)

SYSTEM_DIAGRAM = SYSTEM_BASE + (
    "\n\nYou are analyzing a diagram or image extracted from an academic document. "
    "Describe what the image shows, its academic relevance, and key insights."
)

# ─────────────────────────────────────────────
# USER PROMPT TEMPLATES
# ─────────────────────────────────────────────

QA_PROMPT = """CONTEXT (retrieved from the document):
{context}

QUESTION: {question}

Instructions:
- Answer ONLY based on the context above.
- Cite the source using [Page X] or [Section: Y] notation.
- If context is insufficient, say so explicitly.
- Provide a confidence note at the end: Confidence: High | Medium | Low

ANSWER:"""


SUMMARIZATION_PROMPT = """DOCUMENT CONTENT:
{context}

Provide a comprehensive academic summary in the following structure:

**1. Objective / Problem Statement**
(What is this paper/document trying to solve or study?)

**2. Methodology**
(What methods, techniques, or approaches are used?)

**3. Key Findings / Results**
(What are the main results or contributions?)

**4. Conclusions & Implications**
(What conclusions are drawn? What is the broader significance?)

**5. Key Terms**
(List 5–10 important technical terms with brief definitions.)

Source references: {sources}

SUMMARY:"""


EXPLANATION_PROMPT = """CONTEXT:
{context}

CONCEPT TO EXPLAIN: {concept}

Explain this concept clearly for someone new to the field:
1. Simple definition (1–2 sentences)
2. Detailed explanation with any sub-components
3. Real-world analogy or example
4. Why it matters in this document's context

Sources: {sources}

EXPLANATION:"""


FORMULA_PROMPT = """FORMULA / EQUATION:
{formula_latex}

SURROUNDING CONTEXT (from document):
{context}

Explain this formula:
1. **What it represents**: The high-level meaning
2. **Symbol breakdown**: Define each variable/symbol used
3. **Units** (if applicable)
4. **Practical significance**: How or where is it used?
5. **Intuition**: Non-mathematical interpretation

Sources: {sources}

EXPLANATION:"""


DIAGRAM_PROMPT = """IMAGE CAPTION (auto-generated):
{caption}

SURROUNDING CONTEXT (from document, same page):
{context}

Provide a detailed academic description:
1. **What is shown**: Describe the diagram/figure
2. **Key components**: Identify labels, axes, flows, or structures
3. **Interpretation**: What does this diagram convey academically?
4. **Relation to document**: How does this support the document's arguments?

Sources: {sources}

ANALYSIS:"""


HALLUCINATION_CHECK_PROMPT = """CONTEXT PROVIDED:
{context}

GENERATED ANSWER:
{answer}

Task: Analyze whether the generated answer contains any claims NOT supported by the context.
- List any unsupported claims.
- Rate hallucination risk: LOW | MEDIUM | HIGH
- Provide a revised, safer answer if hallucination is detected.

ANALYSIS:"""
