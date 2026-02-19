from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.chunking import chunk_text  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import init_db, insert_chunk, list_subjects  # noqa: E402
from app.embedding_store import normalize, pack_f32  # noqa: E402
from app.gemini_service import GeminiClient, GeminiModels  # noqa: E402
from app.pdf_processor import extract_pdf_text_by_page  # noqa: E402
from app.rag_pipeline import answer_question  # noqa: E402


def _inject_tech_theme() -> None:
        st.markdown(
                """
<style>
    /* --- Tech theme (CSS injection) --- */
    :root {
        --bg0: #070b14;
        --bg1: #0b1220;
        --panel: rgba(17, 27, 46, 0.72);
        --panel2: rgba(17, 27, 46, 0.9);
        --stroke: rgba(34, 211, 238, 0.18);
        --stroke2: rgba(167, 139, 250, 0.18);
        --text: #e5e7eb;
        --muted: rgba(229, 231, 235, 0.72);
        --accent: #22d3ee;
        --accent2: #a78bfa;
        --good: #34d399;
        --warn: #fbbf24;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% 0%, rgba(34, 211, 238, 0.18), transparent 60%),
            radial-gradient(1000px 600px at 85% 10%, rgba(167, 139, 250, 0.18), transparent 55%),
            linear-gradient(180deg, var(--bg0), var(--bg1));
        color: var(--text);
    }

    /* Subtle animated grid */
    .stApp:before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(to right, rgba(34, 211, 238, 0.08) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(167, 139, 250, 0.06) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: radial-gradient(closest-side at 50% 10%, rgba(0,0,0,0.9), transparent 85%);
        opacity: 0.55;
        animation: gridShift 18s linear infinite;
    }
    @keyframes gridShift {
        from { transform: translate3d(0, 0, 0); }
        to { transform: translate3d(-48px, -48px, 0); }
    }

    /* Glassy blocks */
    div[data-testid="stVerticalBlock"],
    section[data-testid="stSidebar"] > div {
        backdrop-filter: blur(10px);
    }

    /* Sidebar panel */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 11, 20, 0.6), rgba(11, 18, 32, 0.85));
        border-right: 1px solid rgba(34, 211, 238, 0.12);
    }

    /* Main content max width polish */
    div[data-testid="stAppViewContainer"] > .main {
        padding-top: 1.0rem;
    }

    /* Headings */
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }

    /* Buttons */
    .stButton > button {
        border: 1px solid rgba(34, 211, 238, 0.25) !important;
        background: linear-gradient(180deg, rgba(34, 211, 238, 0.18), rgba(167, 139, 250, 0.10)) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.06), 0 18px 40px rgba(0, 0, 0, 0.35);
        transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.12), 0 22px 60px rgba(0, 0, 0, 0.42);
    }

    /* Inputs */
    input, textarea {
        border-radius: 12px !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: rgba(17, 27, 46, 0.60) !important;
        border: 1px solid rgba(34, 211, 238, 0.14) !important;
    }

    /* Tabs */
    button[role="tab"] {
        border-radius: 12px 12px 0 0 !important;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(34, 211, 238, 0.10);
        background: var(--panel);
        border-radius: 16px;
        padding: 0.25rem 0.25rem;
    }
    div[data-testid="stChatMessage"] a { color: var(--accent) !important; }

    /* Expander */
    details {
        border-radius: 14px !important;
        border: 1px solid rgba(167, 139, 250, 0.12) !important;
        background: rgba(17, 27, 46, 0.55) !important;
    }

    /* Captions */
    .stCaption, small {
        color: var(--muted) !important;
    }

    /* Remove some default white seams */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    /* Hero card */
    .hero {
        border: 1px solid rgba(34, 211, 238, 0.16);
        background: linear-gradient(180deg, rgba(17, 27, 46, 0.62), rgba(11, 18, 32, 0.20));
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
    }
    .heroTitle {
        margin: 0;
        font-size: 1.8rem;
        line-height: 1.15;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .heroSub {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
    }
    .chip {
        display: inline-block;
        padding: 6px 10px;
        margin-right: 8px;
        border-radius: 999px;
        border: 1px solid rgba(34, 211, 238, 0.18);
        background: rgba(17, 27, 46, 0.55);
        color: rgba(229, 231, 235, 0.86);
        font-size: 0.85rem;
    }
</style>
                """,
                unsafe_allow_html=True,
        )


def _get_api_key() -> str:
    # Prefer Streamlit Cloud secrets, then environment variables.
    try:
        # On Streamlit (especially local runs), st.secrets may raise
        # StreamlitSecretNotFoundError if no secrets file exists.
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def _ensure_ready():
    settings = get_settings()

    api_key = _get_api_key()
    settings.gemini_api_key = api_key

    # Allow overriding models from Streamlit secrets.
    try:
        if "GEMINI_MODEL" in st.secrets:
            settings.gemini_model = str(st.secrets["GEMINI_MODEL"]).strip()
        if "GEMINI_EMBEDDING_MODEL" in st.secrets:
            settings.gemini_embedding_model = str(st.secrets["GEMINI_EMBEDDING_MODEL"]).strip()
    except Exception:
        pass

    settings.data_path().mkdir(parents=True, exist_ok=True)
    init_db(settings)
    return settings


def ingest_notes(*, settings, subject: str, unit: Optional[str], topic: Optional[str], source_name: str, text: str) -> int:
    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeds = client.embed_texts(chunks)
    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]

    for ch, emb_blob in zip(chunks, blobs):
        insert_chunk(
            settings=settings,
            subject=subject,
            unit=unit,
            topic=topic,
            source_type="notes",
            source_name=source_name,
            source_page=None,
            content=ch,
            embedding=emb_blob,
        )

    return len(blobs)


def ingest_pdf(*, settings, subject: str, unit: Optional[str], topic: Optional[str], filename: str, pdf_bytes: bytes) -> int:
    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    pages = extract_pdf_text_by_page(pdf_bytes)
    all_texts: list[str] = []
    all_pages: list[int] = []

    for p in pages:
        for ch in chunk_text(p.text):
            all_texts.append(ch)
            all_pages.append(p.page)

    if not all_texts:
        return 0

    embeds = client.embed_texts(all_texts)
    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]

    for text, page, emb_blob in zip(all_texts, all_pages, blobs):
        insert_chunk(
            settings=settings,
            subject=subject,
            unit=unit,
            topic=topic,
            source_type="pdf",
            source_name=filename or "upload.pdf",
            source_page=page,
            content=text,
            embedding=emb_blob,
        )

    return len(blobs)


def ingest_image(*, settings, subject: str, unit: Optional[str], topic: Optional[str], filename: str, mime: str, img_bytes: bytes, kind: str) -> int:
    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    prompt = (
        "Extract the technical study text from this image. "
        "If it is a diagram, describe it technically for a B.Tech student. "
        "Return plain text only."
    )
    extracted = client.generate_text(system="You extract study notes from images.", user=prompt, images=[(img_bytes, mime)])
    chunks = chunk_text(extracted)
    if not chunks:
        return 0

    embeds = client.embed_texts(chunks)
    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]

    for ch, emb_blob in zip(chunks, blobs):
        insert_chunk(
            settings=settings,
            subject=subject,
            unit=unit,
            topic=topic,
            source_type=kind,
            source_name=filename or "upload.png",
            source_page=None,
            content=ch,
            embedding=emb_blob,
        )

    return len(blobs)


st.set_page_config(page_title="B.Tech Multimodal Study RAG Assistant", layout="wide")

settings = _ensure_ready()

_inject_tech_theme()

st.markdown(
        """
<div class="hero">
    <div class="chip">Multimodal RAG</div>
    <div class="chip">PDF • Notes • Handwriting • Diagrams</div>
    <div class="chip">Exam-mode answers</div>
    <h1 class="heroTitle">B.Tech Multimodal Study Assistant</h1>
    <p class="heroSub">Upload your material, then ask doubts. If context exists, answers cite sources; otherwise it answers from standard knowledge.</p>
</div>
        """,
        unsafe_allow_html=True,
)
st.write("")

api_key = _get_api_key()
if not api_key:
    st.error(
        "GEMINI_API_KEY not set. On Streamlit Cloud: App → Settings → Secrets → add GEMINI_API_KEY. "
        "Locally you can set an environment variable GEMINI_API_KEY."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "last_ingest" not in st.session_state:
    st.session_state.last_ingest = {"type": None, "added": 0, "source": None}

with st.sidebar:
    st.header("Workspace")

    with st.form("context_form", clear_on_submit=False):
        subject = st.text_input("Subject", value=st.session_state.get("subject", "Electrical Networks"))
        unit = st.text_input("Unit", value=st.session_state.get("unit", ""), help="Optional")
        topic = st.text_input("Topic", value=st.session_state.get("topic", ""), help="Optional")

        mode = st.selectbox(
            "Answer mode",
            options=["short", "5_mark", "10_mark", "detailed", "numerical"],
            index=["short", "5_mark", "10_mark", "detailed", "numerical"].index(st.session_state.get("mode", "5_mark")),
        )
        top_k = st.slider("Top-K context chunks", min_value=2, max_value=12, value=int(st.session_state.get("top_k", 6)))

        submitted = st.form_submit_button("Apply")
        if submitted:
            st.session_state.subject = subject
            st.session_state.unit = unit
            st.session_state.topic = topic
            st.session_state.mode = mode
            st.session_state.top_k = top_k

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", settings.gemini_model)
    with c2:
        st.metric("Embed", settings.gemini_embedding_model)

    st.caption("Known subjects")
    try:
        st.write(list_subjects(settings))
    except Exception:
        st.write([])

    st.caption("Session")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()


tab_chat, tab_ingest = st.tabs(["Chat", "Ingest"])

with tab_chat:
    st.subheader("Ask a doubt")
    st.caption("If you ingest notes/PDFs, answers will include citations.")

    # Render existing chat
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Ask your question…")
    if user_q:
        st.session_state.history.append(("user", user_q))

        with st.chat_message("assistant"):
            status = st.status("Retrieving + generating…", expanded=False)
            try:
                answer, chunks = answer_question(
                    settings=settings,
                    subject=st.session_state.get("subject", subject),
                    question=user_q,
                    mode=st.session_state.get("mode", mode),
                    top_k=int(st.session_state.get("top_k", top_k)),
                )
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                raise e

            st.markdown(answer)
            if chunks:
                with st.expander("Sources used", expanded=False):
                    st.json(chunks)

        st.session_state.history.append(("assistant", answer))


with tab_ingest:
    st.subheader("Build your subject knowledge base")
    st.caption("Upload notes/PDFs/images so answers can reference your material.")

    subject_live = st.session_state.get("subject", subject)
    unit_live = st.session_state.get("unit", unit) or ""
    topic_live = st.session_state.get("topic", topic) or ""

    left, mid, right = st.columns(3)

    with left:
        st.markdown("#### Typed notes")
        with st.form("ingest_notes_form"):
            notes_source = st.text_input("Source name", value="typed-notes")
            notes_text = st.text_area("Paste notes", height=180)
            do_notes = st.form_submit_button("Ingest notes")
        if do_notes:
            with st.status("Embedding notes…", expanded=False) as s:
                added = ingest_notes(
                    settings=settings,
                    subject=subject_live,
                    unit=unit_live or None,
                    topic=topic_live or None,
                    source_name=notes_source,
                    text=notes_text,
                )
                st.session_state.last_ingest = {"type": "notes", "added": added, "source": notes_source}
                s.update(label=f"Notes ingested ({added} chunks)", state="complete")

    with mid:
        st.markdown("#### PDF")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False, key="pdf_uploader")
        if pdf_file is not None:
            st.caption(f"Selected: {pdf_file.name}")
        if pdf_file and st.button("Ingest PDF", use_container_width=True):
            with st.status("Extracting + embedding PDF…", expanded=False) as s:
                added = ingest_pdf(
                    settings=settings,
                    subject=subject_live,
                    unit=unit_live or None,
                    topic=topic_live or None,
                    filename=pdf_file.name,
                    pdf_bytes=pdf_file.getvalue(),
                )
                st.session_state.last_ingest = {"type": "pdf", "added": added, "source": pdf_file.name}
                s.update(label=f"PDF ingested ({added} chunks)", state="complete")

    with right:
        st.markdown("#### Image (handwritten/diagram)")
        kind = st.selectbox("Kind", options=["handwritten", "diagram"], index=0, key="img_kind")
        img_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="img_uploader",
        )
        if img_file is not None:
            st.image(img_file, use_container_width=True)
        if img_file and st.button("Ingest image", use_container_width=True):
            with st.status("Extracting text + embedding image…", expanded=False) as s:
                mime = img_file.type or "image/png"
                added = ingest_image(
                    settings=settings,
                    subject=subject_live,
                    unit=unit_live or None,
                    topic=topic_live or None,
                    filename=img_file.name,
                    mime=mime,
                    img_bytes=img_file.getvalue(),
                    kind=kind,
                )
                st.session_state.last_ingest = {"type": kind, "added": added, "source": img_file.name}
                s.update(label=f"Image ingested ({added} chunks)", state="complete")

    st.divider()
    last = st.session_state.get("last_ingest") or {}
    if last.get("type"):
        st.info(f"Last ingest: type={last.get('type')} source={last.get('source')} chunks_added={last.get('added')}")
