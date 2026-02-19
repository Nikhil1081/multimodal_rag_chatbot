# B.Tech Multimodal Study RAG Assistant

Multimodal RAG web app for B.Tech subjects (PDFs, typed notes, handwritten notes/images, diagrams) using the **Gemini API**.

This repo is intentionally Windows-friendly: the backend uses a pure-Python stack (FastAPI + `requests` + `sqlite3` + `pypdf`) to avoid compiled dependency issues.

## Quick start (backend)

1. Create a `.env` in `backend/` (copy from `.env.example`) and set `GEMINI_API_KEY`.
2. Create a venv and install deps:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run API:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`.

If port `8000` is busy on your machine, use `--port 8001` and update the frontend `VITE_BACKEND_URL` accordingly.

## Frontend

The `frontend/` folder contains a Vite + React + Tailwind UI that supports uploads and streaming chat.

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_BACKEND_URL=http://localhost:8000` in `frontend/.env` (or use `http://localhost:8001` if you started the backend on 8001).
