from __future__ import annotations

import json
import mimetypes
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from .chunking import chunk_text
from .config import get_settings
from .db import init_db, insert_chunk, list_subjects as db_list_subjects
from .pdf_processor import extract_pdf_text_by_page
from .rag_pipeline import answer_question, answer_question_stream
from .schemas import ChatRequest, ChatResponse, IngestResponse
from .gemini_service import GeminiClient, GeminiModels
from .embedding_store import normalize, pack_f32


app = FastAPI(title="B.Tech Multimodal Study RAG Assistant", version="0.1.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    settings.data_path().mkdir(parents=True, exist_ok=True)
    init_db(settings)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/subjects")
def list_subjects():
    return {"subjects": db_list_subjects(settings)}


@app.get("/debug/settings")
def debug_settings():
    # Intentionally excludes secrets.
    return {
        "gemini_model": settings.gemini_model,
        "gemini_embedding_model": settings.gemini_embedding_model,
        "cors_origins": settings.cors_origins_list(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        settings.require_gemini_key()
        answer, chunk_refs = answer_question(
            settings=settings,
            subject=req.subject,
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
        )
        return ChatResponse(answer=answer, chunks_used=chunk_refs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chat/stream")
def chat_stream(subject: str, question: str, mode: str = "5_mark", top_k: int = 6):
    try:
        settings.require_gemini_key()
        stream, chunk_refs = answer_question_stream(
            settings=settings,
            subject=subject,
            question=question,
            mode=mode,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def event_iter():
        # Send chunks as: event: token\ndata: ...
        for piece in stream:
            if not piece:
                continue
            data = piece.replace("\r", "").replace("\n", "\ndata: ")
            yield f"event: token\ndata: {data}\n\n"

        yield "event: done\n" + "data: " + json.dumps(chunk_refs) + "\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    subject: str = Form(...),
    unit: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    source_name: str = Form("typed-notes"),
    text: str = Form(...),
):
    try:
        settings.require_gemini_key()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found to ingest")

    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )
    embeds = client.embed_texts(chunks)

    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]
    added = len(blobs)

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

    return IngestResponse(
        subject=subject,
        chunks_added=added,
        source_name=source_name,
        source_type="notes",
    )


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    subject: str = Form(...),
    unit: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    try:
        settings.require_gemini_key()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    pdf_bytes = await file.read()

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
        return IngestResponse(subject=subject, chunks_added=0, source_name=file.filename or "upload.pdf", source_type="pdf")

    embeds = client.embed_texts(all_texts)
    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]
    added = len(blobs)

    # Persist to SQL (optional but useful)
    for text, page, emb_blob in zip(all_texts, all_pages, blobs):
        insert_chunk(
            settings=settings,
            subject=subject,
            unit=unit,
            topic=topic,
            source_type="pdf",
            source_name=file.filename or "upload.pdf",
            source_page=page,
            content=text,
            embedding=emb_blob,
        )

    return IngestResponse(
        subject=subject,
        chunks_added=added,
        source_name=file.filename or "upload.pdf",
        source_type="pdf",
        details={"pages": len(pages)},
    )


@app.post("/ingest/image", response_model=IngestResponse)
async def ingest_image(
    subject: str = Form(...),
    unit: Optional[str] = Form(None),
    topic: Optional[str] = Form(None),
    kind: str = Form("handwritten"),  # handwritten | diagram
    file: UploadFile = File(...),
):
    try:
        settings.require_gemini_key()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    img_bytes = await file.read()
    mime = file.content_type or mimetypes.guess_type(file.filename or "image")[0] or "image/png"

    client = GeminiClient(
        api_key=settings.gemini_api_key,
        models=GeminiModels(text_model=settings.gemini_model, embedding_model=settings.gemini_embedding_model),
    )

    prompt = (
        "Extract the technical text from this handwritten note. "
        "If it is a diagram, describe it technically for a B.Tech student. "
        "Return plain text only."
    )
    extracted = client.generate_text(system="You extract study notes from images.", user=prompt, images=[(img_bytes, mime)])

    chunks = chunk_text(extracted)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any text from image")

    embeds = client.embed_texts(chunks)
    normalized = [normalize(e) for e in embeds]
    blobs = [pack_f32(v) for v in normalized]
    added = len(blobs)

    for ch, emb_blob in zip(chunks, blobs):
        insert_chunk(
            settings=settings,
            subject=subject,
            unit=unit,
            topic=topic,
            source_type=kind,
            source_name=file.filename or "upload.png",
            source_page=None,
            content=ch,
            embedding=emb_blob,
        )

    return IngestResponse(
        subject=subject,
        chunks_added=added,
        source_name=file.filename or "upload.png",
        source_type=kind,
        details={"mime": mime},
    )
