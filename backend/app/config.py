from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    class Config:
        # Load backend/.env regardless of where uvicorn is launched from.
        env_file = str((Path(__file__).resolve().parent.parent / ".env"))
        env_file_encoding = "utf-8"

    # Keep default empty so the app can boot and show /docs even before configuring.
    # Endpoints that need Gemini will error clearly if this isn't set.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"
    # Default to the broadly-supported embedding model; client will attempt newer models too.
    gemini_embedding_model: str = "embedding-001"

    data_dir: str = "data"
    database_url: str = "sqlite:///./data/app.db"

    cors_origins: str = "http://localhost:5173"

    def require_gemini_key(self) -> None:
        key = (self.gemini_api_key or "").strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create backend/.env from backend/.env.example and set GEMINI_API_KEY."
            )
        # Keep the stripped version so downstream HTTP requests don't include whitespace.
        self.gemini_api_key = key

    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
