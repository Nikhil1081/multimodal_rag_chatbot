from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .config import Settings
from .db import fetch_chunks_by_ids, iter_embeddings
from .gemini_service import GeminiClient, GeminiModels
from .embedding_store import normalize, top_k_cosine


SYSTEM_PROMPT_WITH_CONTEXT = """You are a B.Tech professor and exam coach.

Rules:
- Use ONLY the provided context. If the context is insufficient, say what is missing and ask a clarifying question.
- Do not invent facts, formulas, or definitions not present in the context.
- Write in clear, structured university-exam format.
- When you use a fact from context, add a citation in this format: [source: <name> p.<page>]. If page is missing, omit it.

Output format:
1) Definition
2) Explanation
3) Step-by-step / Working
4) Example (if applicable)
5) Exam Answer (based on requested mode)
6) Quiz: 5 MCQs + 2 numericals, with answer key
""".strip()


SYSTEM_PROMPT_NO_CONTEXT = """You are a B.Tech professor and exam coach.

Rules:
- No study context was provided. Answer using standard engineering knowledge.
- Do NOT fabricate citations or pretend a source was provided.
- Be clear and structured in university-exam format.
- If the question depends on a specific syllabus/teacher's notes, ask a clarifying question at the end.

Output format:
1) Definition
2) Explanation
3) Step-by-step / Working
4) Example (if applicable)
5) Exam Answer (based on requested mode)
6) Quiz: 5 MCQs + 2 numericals, with answer key
""".strip()


@dataclass
class RetrievedChunk:
    score: float
    text: str
    meta: Dict[str, Any]


def _render_context(chunks: List[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        meta = ch.meta
        src = meta.get("source_name", "unknown")
        page = meta.get("page")
        src_tag = f"{src} p.{page}" if page else src
        lines.append(f"[Chunk {i} | {src_tag} | type={meta.get('source_type')} | subject={meta.get('subject')}]\n{ch.text}")
    return "\n\n".join(lines)


def answer_question(
    *,
    settings: Settings,
    subject: str,
    question: str,
    mode: str,
    top_k: int = 6,
) -> tuple[str, list[dict[str, Any]]]:
    settings.require_gemini_key()
    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    q_emb = client.embed_texts([question])[0]
    q_unit = normalize(q_emb)

    rows = list(iter_embeddings(settings, subject))
    top = top_k_cosine(query_unit=q_unit, rows=rows, k=top_k)
    ids = [row_id for _, row_id in top]
    chunks = fetch_chunks_by_ids(settings, ids)
    chunk_by_id = {c["id"]: c for c in chunks}
    retrieved: list[RetrievedChunk] = []
    for score, row_id in top:
        c = chunk_by_id.get(row_id)
        if not c:
            continue
        retrieved.append(
            RetrievedChunk(
                score=float(score),
                text=c["content"],
                meta={
                    "subject": c["subject"],
                    "unit": c["unit"],
                    "topic": c["topic"],
                    "source_type": c["source_type"],
                    "source_name": c["source_name"],
                    "page": c["source_page"],
                },
            )
        )

    context = _render_context(retrieved)

    if retrieved:
        system = SYSTEM_PROMPT_WITH_CONTEXT
        user_prompt = f"""Mode: {mode}

Subject: {subject}

Context:\n{context}

Question: {question}
""".strip()
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT
        user_prompt = f"""Mode: {mode}

Subject: {subject}

Question: {question}
""".strip()

    answer = client.generate_text(system=system, user=user_prompt)

    chunk_refs: list[dict[str, Any]] = []
    for ch in retrieved:
        chunk_refs.append(
            {
                "source_name": ch.meta.get("source_name"),
                "source_type": ch.meta.get("source_type"),
                "page": ch.meta.get("page"),
                "subject": ch.meta.get("subject"),
                "unit": ch.meta.get("unit"),
                "topic": ch.meta.get("topic"),
            }
        )

    return answer, chunk_refs


def answer_question_stream(
    *,
    settings: Settings,
    subject: str,
    question: str,
    mode: str,
    top_k: int = 6,
):
    """Return (generator, chunk_refs)."""

    settings.require_gemini_key()
    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    q_emb = client.embed_texts([question])[0]
    q_unit = normalize(q_emb)

    rows = list(iter_embeddings(settings, subject))
    top = top_k_cosine(query_unit=q_unit, rows=rows, k=top_k)
    ids = [row_id for _, row_id in top]
    chunks = fetch_chunks_by_ids(settings, ids)
    chunk_by_id = {c["id"]: c for c in chunks}
    retrieved: list[RetrievedChunk] = []
    for score, row_id in top:
        c = chunk_by_id.get(row_id)
        if not c:
            continue
        retrieved.append(
            RetrievedChunk(
                score=float(score),
                text=c["content"],
                meta={
                    "subject": c["subject"],
                    "unit": c["unit"],
                    "topic": c["topic"],
                    "source_type": c["source_type"],
                    "source_name": c["source_name"],
                    "page": c["source_page"],
                },
            )
        )

    context = _render_context(retrieved)
    if retrieved:
        system = SYSTEM_PROMPT_WITH_CONTEXT
        user_prompt = f"""Mode: {mode}

Subject: {subject}

Context:\n{context}

Question: {question}
""".strip()
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT
        user_prompt = f"""Mode: {mode}

Subject: {subject}

Question: {question}
""".strip()

    chunk_refs: list[dict[str, Any]] = []
    for ch in retrieved:
        chunk_refs.append(
            {
                "source_name": ch.meta.get("source_name"),
                "source_type": ch.meta.get("source_type"),
                "page": ch.meta.get("page"),
                "subject": ch.meta.get("subject"),
                "unit": ch.meta.get("unit"),
                "topic": ch.meta.get("topic"),
            }
        )

    return client.generate_text_stream(system=system, user=user_prompt), chunk_refs
