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


def _get_api_key() -> str:
    # Prefer Streamlit Cloud secrets, then environment variables.
    if "GEMINI_API_KEY" in st.secrets:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def _ensure_ready():
    settings = get_settings()

    api_key = _get_api_key()
    settings.gemini_api_key = api_key

    # Allow overriding models from Streamlit secrets.
    if "GEMINI_MODEL" in st.secrets:
        settings.gemini_model = str(st.secrets["GEMINI_MODEL"]).strip()
    if "GEMINI_EMBEDDING_MODEL" in st.secrets:
        settings.gemini_embedding_model = str(st.secrets["GEMINI_EMBEDDING_MODEL"]).strip()

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

st.title("B.Tech Multimodal Study RAG Assistant")

api_key = _get_api_key()
if not api_key:
    st.error(
        "GEMINI_API_KEY not set. On Streamlit Cloud: Settings → Secrets → add GEMINI_API_KEY.\n"
        "Locally you can set an environment variable GEMINI_API_KEY."
    )
    st.stop()

with st.sidebar:
    st.header("Study Context")
    subject = st.text_input("Subject", value="Electrical Networks")
    unit = st.text_input("Unit (optional)", value="")
    topic = st.text_input("Topic (optional)", value="")

    st.caption("Existing subjects in DB")
    try:
        st.write(list_subjects(settings))
    except Exception:
        st.write([])

    st.divider()
    st.header("Answer Mode")
    mode = st.selectbox("Mode", options=["short", "5_mark", "10_mark", "detailed", "numerical"], index=1)

col_ingest, col_chat = st.columns([0.45, 0.55])

with col_ingest:
    st.subheader("Ingest")

    with st.expander("Typed notes", expanded=True):
        notes_source = st.text_input("Source name", value="typed-notes")
        notes_text = st.text_area("Paste notes", height=160)
        if st.button("Ingest notes"):
            added = ingest_notes(
                settings=settings,
                subject=subject,
                unit=unit or None,
                topic=topic or None,
                source_name=notes_source,
                text=notes_text,
            )
            st.success(f"Added {added} chunks")

    with st.expander("PDF", expanded=True):
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
        if pdf_file and st.button("Ingest PDF"):
            added = ingest_pdf(
                settings=settings,
                subject=subject,
                unit=unit or None,
                topic=topic or None,
                filename=pdf_file.name,
                pdf_bytes=pdf_file.getvalue(),
            )
            st.success(f"Added {added} chunks")

    with st.expander("Handwritten / Diagram image", expanded=False):
        kind = st.selectbox("Kind", options=["handwritten", "diagram"], index=0)
        img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        if img_file and st.button("Ingest image"):
            mime = img_file.type or "image/png"
            added = ingest_image(
                settings=settings,
                subject=subject,
                unit=unit or None,
                topic=topic or None,
                filename=img_file.name,
                mime=mime,
                img_bytes=img_file.getvalue(),
                kind=kind,
            )
            st.success(f"Added {added} chunks")

with col_chat:
    st.subheader("Ask")

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Ask a question…")
    if user_q:
        st.session_state.history.append(("user", user_q))

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, chunks = answer_question(
                    settings=settings,
                    subject=subject,
                    question=user_q,
                    mode=mode,
                    top_k=6,
                )
                st.markdown(answer)
                if chunks:
                    st.caption("Sources")
                    st.json(chunks)

        st.session_state.history.append(("assistant", answer))
