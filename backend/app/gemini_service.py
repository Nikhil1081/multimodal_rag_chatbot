from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import requests


@dataclass
class GeminiModels:
    text_model: str
    embedding_model: str


class GeminiClient:
    """Gemini REST API client.

    This avoids heavy SDK dependencies (grpc/cryptography) that often fail on
    non-standard Python distributions on Windows.

    API: Google Generative Language API (Gemini).
    """

    def __init__(self, *, api_key: str, models: GeminiModels, timeout_s: int = 120):
        # Defensive: keys are sometimes pasted with trailing whitespace/newlines.
        self.api_key = (api_key or "").strip()
        self.models = models
        self.timeout_s = timeout_s
        # Embeddings and text generation have historically differed in which API versions
        # support which models. We keep both and try the most likely first.
        self.base_generate = "https://generativelanguage.googleapis.com/v1"
        self.base_generate_fallback = "https://generativelanguage.googleapis.com/v1beta"
        self.base_embed = "https://generativelanguage.googleapis.com/v1"
        self.base_embed_fallback = "https://generativelanguage.googleapis.com/v1beta"

    def _url(self, base: str, path: str) -> str:
        sep = "&" if "?" in path else "?"
        return f"{base}{path}{sep}key={self.api_key}"

    def _post(self, *, base: str, path: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            self._url(base, path),
            json=payload,
            timeout=self.timeout_s,
        )

    def _get(self, *, base: str, path: str) -> requests.Response:
        return requests.get(
            self._url(base, path),
            timeout=self.timeout_s,
        )

    def _list_models(self, *, base: str) -> list[dict[str, Any]]:
        resp = self._get(base=base, path="/models")
        if resp.status_code >= 400:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        models = data.get("models")
        return models if isinstance(models, list) else []

    def _discover_embedding_models(self) -> list[str]:
        """Return model IDs (without the 'models/' prefix) likely to support embeddings."""

        discovered: list[str] = []
        for base in (self.base_embed, self.base_embed_fallback):
            for m in self._list_models(base=base):
                name = (m.get("name") or "").strip()  # e.g. models/embedding-001
                methods = m.get("supportedGenerationMethods") or []
                if not isinstance(methods, list):
                    methods = []
                methods_l = [str(x).strip() for x in methods]

                if not name.startswith("models/"):
                    continue
                model_id = name.split("/", 1)[1]

                # Heuristic: include models that explicitly support embedContent, or look like embedding models.
                if "embedcontent" in "".join(x.lower() for x in methods_l) or "embed" in model_id.lower():
                    if model_id not in discovered:
                        discovered.append(model_id)

        return discovered

    def _discover_text_models(self) -> list[str]:
        """Return model IDs likely to support generateContent."""

        discovered: list[str] = []
        for base in (self.base_generate, self.base_generate_fallback):
            for m in self._list_models(base=base):
                name = (m.get("name") or "").strip()  # e.g. models/gemini-1.5-flash
                methods = m.get("supportedGenerationMethods") or []
                if not isinstance(methods, list):
                    methods = []
                methods_l = [str(x).strip().lower() for x in methods]

                if not name.startswith("models/"):
                    continue
                model_id = name.split("/", 1)[1]

                if "generatecontent" in methods_l:
                    # Prefer gemini models.
                    if "gemini" in model_id.lower() and model_id not in discovered:
                        discovered.append(model_id)

        return discovered

    @staticmethod
    def _is_model_not_found(resp: requests.Response) -> bool:
        # 404 frequently indicates either the model doesn't exist for that API version
        # OR the method (batchEmbedContents/embedContent) isn't supported for the model.
        if resp.status_code == 404:
            return True
        if resp.status_code != 400:
            return False
        try:
            data = resp.json()
        except Exception:
            return False
        msg = ((data.get("error") or {}).get("message") or "").lower()
        return "not found" in msg or "not supported" in msg or "unsupported" in msg

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Try configured model first, then common fallback.
        candidates = [self.models.embedding_model, "embedding-001"]
        models_to_try: list[str] = []
        for m in candidates:
            m = (m or "").strip()
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_err: str | None = None

        embed_bases = [self.base_embed, self.base_embed_fallback]

        for model in models_to_try:
            for base in embed_bases:
                # Try batch endpoint first.
                reqs = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts]
                resp = self._post(
                    base=base,
                    path=f"/models/{model}:batchEmbedContents",
                    payload={"requests": reqs},
                )

                if resp.status_code < 400:
                    data = resp.json()
                    embeddings = data.get("embeddings") or []
                    out: list[list[float]] = []
                    for e in embeddings:
                        values = e.get("values")
                        out.append([float(x) for x in values] if values else [])
                    return out

                # Fall back to single-item endpoint if batch isn't supported.
                if self._is_model_not_found(resp):
                    single: list[list[float]] = []
                    ok = True
                    for t in texts:
                        r2 = self._post(
                            base=base,
                            path=f"/models/{model}:embedContent",
                            payload={"content": {"parts": [{"text": t}]}, "model": f"models/{model}"},
                        )
                        if r2.status_code >= 400:
                            ok = False
                            last_err = f"{r2.status_code} {r2.text}"
                            break
                        d2 = r2.json()
                        emb = (d2.get("embedding") or {}).get("values")
                        single.append([float(x) for x in emb] if emb else [])
                    if ok:
                        return single

                last_err = f"{resp.status_code} {resp.text}"

        # As a last resort, discover available models and retry a few embedding-capable ones.
        discovered = self._discover_embedding_models()
        for model in discovered[:5]:
            for base in embed_bases:
                r2 = self._post(
                    base=base,
                    path=f"/models/{model}:embedContent",
                    payload={"content": {"parts": [{"text": texts[0]}]}, "model": f"models/{model}"},
                )
                if r2.status_code >= 400:
                    last_err = f"{r2.status_code} {r2.text}"
                    continue

                # If a single embed works, use this model for the full batch.
                single: list[list[float]] = []
                ok = True
                for t in texts:
                    r3 = self._post(
                        base=base,
                        path=f"/models/{model}:embedContent",
                        payload={"content": {"parts": [{"text": t}]}, "model": f"models/{model}"},
                    )
                    if r3.status_code >= 400:
                        ok = False
                        last_err = f"{r3.status_code} {r3.text}"
                        break
                    d3 = r3.json()
                    emb = (d3.get("embedding") or {}).get("values")
                    single.append([float(x) for x in emb] if emb else [])
                if ok:
                    return single

        raise RuntimeError(f"Gemini embeddings error: {last_err}")

    def generate_text(self, *, system: str, user: str, images: Optional[list[tuple[bytes, str]]] = None) -> str:
        images = images or []

        parts: list[dict[str, Any]] = [{"text": user}]
        for data, mime in images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(data).decode("utf-8"),
                    }
                }
            )

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
        }

        model = (self.models.text_model or "").strip()
        bases = [self.base_generate, self.base_generate_fallback]

        resp: requests.Response | None = None
        for base in bases:
            resp = self._post(base=base, path=f"/models/{model}:generateContent", payload=payload)
            if resp.status_code < 400:
                break
            if not self._is_model_not_found(resp):
                break

        if resp is None or resp.status_code >= 400:
            # Discover and try other text models that support generateContent.
            for fallback_model in self._discover_text_models()[:5]:
                for base in bases:
                    r2 = self._post(base=base, path=f"/models/{fallback_model}:generateContent", payload=payload)
                    if r2.status_code < 400:
                        resp = r2
                        break
                if resp is not None and resp.status_code < 400:
                    break

        if resp is None or resp.status_code >= 400:
            raise RuntimeError(f"Gemini generate error: {resp.status_code if resp is not None else 'unknown'} {resp.text if resp is not None else ''}")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        content = candidates[0].get("content") or {}
        parts_out = content.get("parts") or []
        texts = []
        for p in parts_out:
            if "text" in p:
                texts.append(p["text"])
        return "".join(texts).strip()

    def generate_text_stream(self, *, system: str, user: str) -> Iterable[str]:
        # Simple fallback: yield once.
        yield self.generate_text(system=system, user=user)
