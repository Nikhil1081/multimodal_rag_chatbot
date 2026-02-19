from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
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

    @keyframes fadeUp {
        from { opacity: 0; transform: translate3d(0, 10px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    @keyframes glowPulse {
        0% { box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.08), 0 22px 60px rgba(0, 0, 0, 0.45); }
        50% { box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.18), 0 28px 80px rgba(0, 0, 0, 0.52); }
        100% { box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.08), 0 22px 60px rgba(0, 0, 0, 0.45); }
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
        animation: fadeUp 220ms ease-out;
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
        animation: fadeUp 650ms ease-out;
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

    .loginWrap {
        max-width: 560px;
        margin: 0 auto;
        padding-top: 1.5rem;
        animation: fadeUp 650ms ease-out;
    }
    .loginCard {
        position: relative;
        border: 1px solid rgba(34, 211, 238, 0.18);
        background: linear-gradient(180deg, rgba(17, 27, 46, 0.70), rgba(11, 18, 32, 0.30));
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.08), 0 22px 60px rgba(0, 0, 0, 0.45);
        animation: glowPulse 3.2s ease-in-out infinite;
        overflow: hidden;
    }
    .loginCard:before {
        content: "";
        position: absolute;
        inset: -2px;
        background: radial-gradient(900px 160px at 10% 0%, rgba(34, 211, 238, 0.22), transparent 55%),
                    radial-gradient(900px 160px at 90% 0%, rgba(167, 139, 250, 0.20), transparent 55%);
        opacity: 0.8;
        pointer-events: none;
    }
    .loginTitle {
        margin: 0;
        font-size: 1.25rem;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .loginHint {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
    }

    /* Auth tabs look more app-like */
    div[data-testid="stTabs"] button[role="tab"] {
        border: 1px solid rgba(34, 211, 238, 0.12) !important;
        background: rgba(17, 27, 46, 0.35) !important;
        margin-right: 0.35rem;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        border-color: rgba(34, 211, 238, 0.28) !important;
        background: rgba(17, 27, 46, 0.65) !important;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _auth_db_path() -> Path:
    return REPO_ROOT / "data" / "auth.db"


def _auth_init_db() -> None:
    db_path = _auth_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                display_name TEXT,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                recovery_question TEXT,
                recovery_salt TEXT,
                recovery_hash TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _norm_username(username: str) -> str:
    return username.strip().lower()


def _pbkdf2_hash(*, secret_value: str, salt_hex: str, iterations: int = 200_000) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", secret_value.encode("utf-8"), salt, iterations)
    return dk.hex()


def _hash_new_secret(*, secret_value: str) -> tuple[str, str]:
    salt = secrets.token_bytes(16)
    salt_hex = salt.hex()
    hash_hex = _pbkdf2_hash(secret_value=secret_value, salt_hex=salt_hex)
    return salt_hex, hash_hex


def _verify_secret(*, secret_value: str, salt_hex: str, expected_hash_hex: str) -> bool:
    candidate = _pbkdf2_hash(secret_value=secret_value, salt_hex=salt_hex)
    return hmac.compare_digest(candidate, expected_hash_hex)


def _auth_get_user(username: str) -> Optional[dict]:
    u = _norm_username(username)
    with sqlite3.connect(str(_auth_db_path())) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (u,),
        ).fetchone()
    return dict(row) if row else None


def _auth_user_count() -> int:
    with sqlite3.connect(str(_auth_db_path())) as conn:
        row = conn.execute("SELECT COUNT(1) FROM users").fetchone()
        return int(row[0] or 0)


def _auth_create_user(
    *,
    username: str,
    display_name: str,
    password: str,
    recovery_question: str,
    recovery_answer: str,
) -> None:
    u = _norm_username(username)
    if not u:
        raise ValueError("Username is required")
    if len(u) < 3:
        raise ValueError("Username must be at least 3 characters")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    pw_salt, pw_hash = _hash_new_secret(secret_value=password)
    rec_salt, rec_hash = _hash_new_secret(secret_value=recovery_answer.strip().lower())
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(str(_auth_db_path())) as conn:
        conn.execute(
            """
            INSERT INTO users (username, display_name, password_salt, password_hash, recovery_question, recovery_salt, recovery_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                u,
                display_name.strip() or None,
                pw_salt,
                pw_hash,
                recovery_question.strip() or None,
                rec_salt,
                rec_hash,
                created_at,
            ),
        )
        conn.commit()


def _auth_update_password(*, username: str, new_password: str) -> None:
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters")
    u = _norm_username(username)
    pw_salt, pw_hash = _hash_new_secret(secret_value=new_password)
    with sqlite3.connect(str(_auth_db_path())) as conn:
        conn.execute(
            "UPDATE users SET password_salt = ?, password_hash = ? WHERE username = ?",
            (pw_salt, pw_hash, u),
        )
        conn.commit()


def _auth_attempt_login(*, username: str, password: str) -> bool:
    user = _auth_get_user(username)
    if not user:
        return False
    return _verify_secret(secret_value=password, salt_hex=user["password_salt"], expected_hash_hex=user["password_hash"])


def _auth_attempt_recovery(*, username: str, recovery_answer: str) -> bool:
    user = _auth_get_user(username)
    if not user:
        return False
    if not user.get("recovery_salt") or not user.get("recovery_hash"):
        return False
    return _verify_secret(
        secret_value=recovery_answer.strip().lower(),
        salt_hex=user["recovery_salt"],
        expected_hash_hex=user["recovery_hash"],
    )


def _render_auth_page() -> None:
    st.markdown(
        """
<div class="loginWrap">
  <div class="loginCard">
    <h2 class="loginTitle">Welcome</h2>
    <p class="loginHint">Login, or create an account to access the Study Assistant.</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        tab_login, tab_register, tab_forgot = st.tabs(["Login", "Register", "Forgot password"])

        with tab_login:
            with st.form("auth_login_form", clear_on_submit=False):
                username = st.text_input("Username", value=str(st.session_state.get("login_username", "")))
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

            if submitted:
                try:
                    ok = _auth_attempt_login(username=username, password=password)
                except Exception:
                    ok = False

                if ok:
                    st.session_state.authenticated = True
                    st.session_state.login_username = _norm_username(username)
                    user = _auth_get_user(username)
                    st.session_state.display_name = (user or {}).get("display_name") or _norm_username(username)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.caption("Tip: If you are new, open the Register tab.")

        with tab_register:
            with st.form("auth_register_form", clear_on_submit=False):
                username = st.text_input("Choose a username")
                display_name = st.text_input("Display name (optional)")
                password = st.text_input("Create password", type="password")
                password2 = st.text_input("Confirm password", type="password")
                recovery_question = st.text_input("Recovery question (e.g., Your first school?)")
                recovery_answer = st.text_input("Recovery answer", type="password")
                submitted = st.form_submit_button("Create account")

            if submitted:
                if password != password2:
                    st.error("Passwords do not match.")
                elif not recovery_question.strip() or not recovery_answer.strip():
                    st.error("Recovery question and answer are required (used for Forgot password).")
                else:
                    try:
                        _auth_create_user(
                            username=username,
                            display_name=display_name,
                            password=password,
                            recovery_question=recovery_question,
                            recovery_answer=recovery_answer,
                        )
                        st.success("Account created. You can login now.")
                    except sqlite3.IntegrityError:
                        st.error("That username is already taken.")
                    except Exception as e:
                        st.error(str(e))

        with tab_forgot:
            username = st.text_input("Username", key="forgot_username")
            user = None
            if username.strip():
                try:
                    user = _auth_get_user(username)
                except Exception:
                    user = None

            if user and user.get("recovery_question"):
                st.info(f"Recovery question: {user['recovery_question']}")
                with st.form("auth_forgot_form", clear_on_submit=False):
                    recovery_answer = st.text_input("Recovery answer", type="password")
                    new_password = st.text_input("New password", type="password")
                    new_password2 = st.text_input("Confirm new password", type="password")
                    submitted = st.form_submit_button("Reset password")

                if submitted:
                    if new_password != new_password2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            ok = _auth_attempt_recovery(username=username, recovery_answer=recovery_answer)
                        except Exception:
                            ok = False
                        if not ok:
                            st.error("Recovery answer is incorrect.")
                        else:
                            try:
                                _auth_update_password(username=username, new_password=new_password)
                                st.success("Password updated. You can login now.")
                            except Exception as e:
                                st.error(str(e))
            else:
                st.caption("Enter a valid username to see the recovery question.")

        st.divider()
        if st.button("Continue as guest"):
            st.session_state.authenticated = True
            st.session_state.login_username = "guest"
            st.session_state.display_name = "Guest"
            st.rerun()


def _require_auth() -> None:
    if st.session_state.get("authenticated") is True:
        return

    try:
        _auth_init_db()
    except Exception:
        st.error("Auth storage is unavailable right now. Continuing as guest.")
        st.session_state.authenticated = True
        st.session_state.login_username = "guest"
        st.session_state.display_name = "Guest"
        return

    _render_auth_page()
    st.stop()


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
_inject_tech_theme()

_require_auth()

settings = _ensure_ready()

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

    if st.session_state.get("authenticated") is True:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop("history", None)
            st.session_state.pop("display_name", None)
            st.rerun()

    who = st.session_state.get("display_name") or st.session_state.get("login_username")
    if who:
        st.caption(f"Signed in as: {who}")

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
