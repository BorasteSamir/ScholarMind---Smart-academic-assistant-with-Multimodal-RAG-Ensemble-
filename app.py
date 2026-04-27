"""
app.py
======
Main Streamlit Application for the Smart Academic Assistant.

Features:
  - Sidebar for PDF upload & ingestion state management
  - Main tabbed interface (Chat/QA, Summary, Extracted Assets)
  - Intent-based routing to specialized agents
  - Visual display of sources, images, and formulas
  - Hallucination warnings and confidence scores
"""

import streamlit as st
import time
from pathlib import Path

# --- Must be imported early for environment variable and logging setup ---
import config.settings
import evaluation.logger
import logging

from ingestion.pipeline import ingest_pdf
from embeddings.vector_store import get_vector_store
from agents.agent_router import get_router
from utils.file_utils import save_uploaded_file

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# INITIALIZATION & CACHING
# ─────────────────────────────────────────────

# Inject custom CSS for better aesthetics
st.set_page_config(
    page_title=config.settings.APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { font-family: 'Inter', sans-serif; }
    .source-box { 
        background-color: #1e2a3a;
        color: #e8edf2 !important;
        border-left: 4px solid #4a90e2; 
        padding: 12px 14px; 
        margin-bottom: 10px; 
        border-radius: 6px;
        font-size: 0.9em;
    }
    .source-box strong {
        color: #7eb8f7 !important;
        display: block;
        margin-bottom: 4px;
    }
    .source-box em {
        color: #a0b4c8 !important;
        font-style: normal;
        font-size: 0.85em;
    }
    .hallucination-warning {
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #f5c2c7;
        border-radius: 6px;
        color: #ffb3b8 !important;
        background-color: #3d1a1e;
    }
    .hallucination-warning strong {
        color: #ff6b75 !important;
    }
    .confidence-high { color: #4caf82; font-weight: bold; }
    .confidence-medium { color: #f0a500; font-weight: bold; }
    .confidence-low { color: #e05c5c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Connect to ChromaDB — fast, no model loading."""
    return get_vector_store()

@st.cache_resource(show_spinner=False)
def load_router():
    """Load LLM client + agent router — deferred until first query."""
    return get_router()

vstore = load_vector_store()
router = load_router()

# Show a one-time ready indicator on cold start
if "app_ready" not in st.session_state:
    st.session_state.app_ready = True
    st.toast("✅ Assistant ready!", icon="🎓")


# ─────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_bundle" not in st.session_state:
    st.session_state.doc_bundle = None
if "is_ingesting" not in st.session_state:
    st.session_state.is_ingesting = False


# ─────────────────────────────────────────────
# SIDEBAR: INGESTION PIPELINE
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🎓 Smart Academic Assistant")
    st.markdown("Upload a research paper or academic text to begin.")
    
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if we need to process a new file
        new_doc_id = Path(uploaded_file.name).stem
        if new_doc_id != st.session_state.doc_id:
            if st.button("Process Document", type="primary"):
                st.session_state.is_ingesting = True
                
                with st.spinner("Saving file..."):
                    pdf_path = save_uploaded_file(uploaded_file)
                
                if pdf_path:
                    try:
                        # 1. Parse and extract (Pipeline)
                        with st.spinner("Extracting text, images, and formulas..."):
                            start_t = time.time()
                            bundle = ingest_pdf(pdf_path)
                            ext_time = time.time() - start_t
                        
                        # 2. Embed and Index
                        with st.spinner("Generating embeddings & building index..."):
                            start_t = time.time()
                            num_chunks = vstore.add_from_bundle(bundle)
                            idx_time = time.time() - start_t
                        
                        # Update session state upon success
                        st.session_state.doc_id = bundle.doc_id
                        st.session_state.doc_bundle = bundle
                        st.session_state.image_explanations = {}  # clear cache for new doc
                        
                        st.success(f"Successfully processed {bundle.filename}!")
                        st.info(
                            f"- Pages: {bundle.metadata.get('page_count', 'Unknown')}\n"
                            f"- Extraction: {ext_time:.1f}s\n"
                            f"- Indexing ({num_chunks} chunks): {idx_time:.1f}s"
                        )
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
                        logger.exception("Ingestion failed")
                    finally:
                        st.session_state.is_ingesting = False

    st.divider()
    st.subheader("Settings")
    force_intent_ui = st.selectbox(
        "Force Agent (Optional)", 
        ["Auto-detect", "QA", "Summarize", "Explain", "Formula", "Diagram"]
    )
    
    if st.session_state.doc_id:
        st.caption(f"**Current Document:** {st.session_state.doc_id}")

# ─────────────────────────────────────────────
# MAIN APP BODY
# ─────────────────────────────────────────────

if not st.session_state.doc_id:
    st.info("👈 Please upload and process a PDF document first.")
    st.stop()

tab_chat, tab_summary, tab_assets = st.tabs(["💬 Chat & Q/A", "📝 Document Summary", "🖼️ Extracted Assets"])

# ── TAB: SUMMARY ─────────────────────────────
with tab_summary:
    st.header("Document Summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating structured summary..."):
            agent = router.get_agent("summarize")
            summary_result = agent("Summarize document", doc_id=st.session_state.doc_id)
            st.markdown(summary_result.answer)

# ── TAB: EXTRACTED ASSETS ─────────────────────
with tab_assets:
    st.header("Extracted Assets")
    bundle = st.session_state.doc_bundle
    
    col_img, col_math = st.columns(2)
    
    with col_img:
        st.subheader(f"Images & Diagrams ({len(bundle.image_elements)})")
        if bundle.image_elements:
            # Cache explanations in session state so they aren't regenerated on every rerender
            if "image_explanations" not in st.session_state:
                st.session_state.image_explanations = {}

            diagram_agent = router.get_agent("diagram")

            for img_idx, img_dict in enumerate(bundle.image_elements):
                if not img_dict.get("file_path"):
                    continue

                caption   = img_dict.get("caption", "Academic document image")
                page      = img_dict.get("page", "?")
                cache_key = f"{st.session_state.doc_id}_img_{img_idx}"

                # ── Image card container ──
                st.markdown(
                    f"<div style='"
                    f"background:#1e2a3a;"
                    f"border:1px solid #2d3f55;"
                    f"border-radius:10px;"
                    f"padding:14px;"
                    f"margin-bottom:18px;'"
                    f">",
                    unsafe_allow_html=True,
                )

                # Image
                st.image(
                    img_dict["file_path"],
                    use_container_width=True,
                )

                # Page + BLIP caption badge
                st.markdown(
                    f"<p style='color:#a0b4c8;font-size:0.8em;margin:6px 0 10px 0;'>"
                    f"📄 <strong style='color:#7eb8f7;'>Page {page}</strong>"
                    f" &nbsp;|&nbsp; "
                    f"🤖 <em style='color:#c8d8e8;'>{caption}</em>"
                    f"</p>",
                    unsafe_allow_html=True,
                )

                # ── Auto-generate explanation (cached) ──
                if cache_key not in st.session_state.image_explanations:
                    with st.spinner(f"Generating explanation for image on page {page}..."):
                        try:
                            res = diagram_agent(
                                input_data=caption,
                                doc_id=st.session_state.doc_id,
                                caption=caption,
                                page=page,
                            )
                            st.session_state.image_explanations[cache_key] = res.answer
                        except Exception as e:
                            st.session_state.image_explanations[cache_key] = (
                                f"⚠️ Could not generate explanation: {e}"
                            )

                explanation = st.session_state.image_explanations.get(cache_key, "")

                # Explanation box
                st.markdown(
                    f"<div style='"
                    f"background:#111d2b;"
                    f"border-left:4px solid #4a90e2;"
                    f"border-radius:6px;"
                    f"padding:12px 14px;"
                    f"margin-top:4px;'"
                    f">"
                    f"<p style='color:#7eb8f7;font-size:0.82em;font-weight:bold;margin:0 0 6px 0;'>"
                    f"🔍 AI Explanation"
                    f"</p>"
                    f"<p style='color:#e8edf2;font-size:0.88em;line-height:1.6;margin:0;'>"
                    f"{explanation}"
                    f"</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        else:
            st.info("No substantial images detected in document.")
            
    with col_math:
        st.subheader(f"Formulas ({len(bundle.formula_elements)})")
        if bundle.formula_elements:
            for f_dict in bundle.formula_elements:
                with st.expander(f"Equation - Page {f_dict['page']}"):
                    st.latex(f_dict.get('latex', f_dict.get('content')))
                    st.caption(f"Context: {f_dict.get('context', '')[:100]}...")
                    if st.button("Explain this", key=f"btn_math_{f_dict['formula_idx']}"):
                        with st.spinner(f"Analyzing formula..."):
                            f_agent = router.get_agent("formula")
                            res = f_agent(
                                input_data=f_dict.get('content', ''),
                                doc_id=st.session_state.doc_id,
                                latex=f_dict.get('latex', '')
                            )
                            st.markdown(res.answer)
        else:
            st.info("No mathematical formulas detected.")

# ── TAB: CHAT INTERFACE ──────────────────────
with tab_chat:
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display metadata on assistant responses
            if msg["role"] == "assistant" and "sources_html" in msg:
                # Hallucination warning
                if msg.get("hallucination_flag"):
                    st.markdown(
                        f"""<div class='hallucination-warning'>
                        <strong>⚠️ Hallucination Risk Detected:</strong> {msg['hallucination_warning']}
                        </div>""", 
                        unsafe_allow_html=True
                    )
                
                # Expandable sources box
                with st.expander(f"📄 Sources & Metadata — Confidence: {msg['confidence_label']}"):
                    st.markdown(msg["sources_html"], unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='color:#a0b4c8; font-size:0.82em; margin-top:6px;'>"
                        f"🤖 Agent: <strong style='color:#7eb8f7;'>{msg['agent_name']}</strong>"
                        f" &nbsp;|&nbsp; ⏱ Time: <strong style='color:#7eb8f7;'>{msg['time_sec']}s</strong></p>",
                        unsafe_allow_html=True
                    )

    # Chat input
    if prompt := st.chat_input("Ask a question, request an explanation, or analyze a formula..."):
        
        # Determine intent override
        force_intent_map = {
            "QA": "qa", "Summarize": "summarize", "Explain": "explain",
            "Formula": "formula", "Diagram": "diagram"
        }
        force_intent = force_intent_map.get(force_intent_ui, None)
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                result, intent_info = router.route(
                    query=prompt, 
                    doc_id=st.session_state.doc_id,
                    force_intent=force_intent
                )
            
            st.markdown(result.answer)
            
            # Build pretty sources HTML format
            sources_html = ""
            for src in result.sources:
                sources_html += (
                    f"<div class='source-box'>"
                    f"<strong style='color:#7eb8f7;'>{src['citation']}</strong> "
                    f"<span style='color:#e8edf2;'>(Page {src['page']})</span><br>"
                    f"<em style='color:#a0b4c8;'>Score: {src['score']:.3f} &nbsp;|&nbsp; "
                    f"Type: {src.get('type','text')} &nbsp;|&nbsp; "
                    f"Doc: {src.get('doc_id','')}</em>"
                    f"</div>"
                )
            if not sources_html:
                sources_html = "<div class='source-box'><span style='color:#e8edf2;'>No specific context chunks were retrieved.</span></div>"
            
            if result.hallucination_flag:
                st.markdown(
                    f"""<div class='hallucination-warning'>
                    <strong>⚠️ Hallucination Risk Detected:</strong> {result.hallucination_warning}
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with st.expander(f"📄 Sources & Metadata — Confidence: {result.confidence_label}"):
                st.markdown(sources_html, unsafe_allow_html=True)

                # Ensemble breakdown
                if result.ensemble_result:
                    er = result.ensemble_result
                    st.markdown(
                        f"<p style='color:#a0b4c8;font-size:0.82em;margin-top:8px;'>"
                        f"🏆 <strong style='color:#7eb8f7;'>Best Model:</strong> "
                        f"<span style='color:#e8edf2;'>{er.best_model}</span>"
                        f" &nbsp;|&nbsp; "
                        f"🤖 <strong style='color:#7eb8f7;'>Agent:</strong> "
                        f"<span style='color:#e8edf2;'>{intent_info.intent.upper()}</span>"
                        f" &nbsp;|&nbsp; "
                        f"⏱ <strong style='color:#7eb8f7;'>Time:</strong> "
                        f"<span style='color:#e8edf2;'>{result.generation_time_sec}s</span>"
                        f"</p>",
                        unsafe_allow_html=True
                    )
                    # Per-model scores table
                    score_rows = "".join(
                        f"<tr>"
                        f"<td style='color:#e8edf2;padding:3px 10px;'>{m}</td>"
                        f"<td style='color:#{'4caf82' if s == max(er.scores) else 'a0b4c8'};padding:3px 10px;font-weight:{'bold' if s == max(er.scores) else 'normal'};'>{s:.4f}</td>"
                        f"</tr>"
                        for m, s in zip(er.models, er.scores)
                    )
                    st.markdown(
                        f"<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
                        f"<thead><tr>"
                        f"<th style='color:#7eb8f7;text-align:left;padding:3px 10px;'>Model</th>"
                        f"<th style='color:#7eb8f7;text-align:left;padding:3px 10px;'>Score</th>"
                        f"</tr></thead><tbody>{score_rows}</tbody></table>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<p style='color:#a0b4c8;font-size:0.82em;margin-top:6px;'>"
                        f"🤖 Agent: <strong style='color:#7eb8f7;'>{intent_info.intent.upper()}</strong>"
                        f" &nbsp;|&nbsp; ⏱ Time: <strong style='color:#7eb8f7;'>{result.generation_time_sec}s</strong></p>",
                        unsafe_allow_html=True
                    )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources_html": sources_html,
            "confidence_label": result.confidence_label,
            "agent_name": intent_info.intent.upper(),
            "time_sec": result.generation_time_sec,
            "hallucination_flag": result.hallucination_flag,
            "hallucination_warning": result.hallucination_warning
        })
