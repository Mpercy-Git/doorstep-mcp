"""Speech-to-text: ElevenLabs Scribe, or any OpenAI-compatible /audio/transcriptions endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Config


class STTError(Exception):
    pass


@dataclass
class Transcript:
    text: str
    language: str | None


class STT:
    def __init__(
        self, cfg: Config, elevenlabs_key: str | None, openai_key: str | None, client: httpx.AsyncClient
    ):
        self.cfg = cfg
        self.elevenlabs_key = elevenlabs_key
        self.openai_key = openai_key
        self.client = client

    async def transcribe(self, wav: bytes) -> Transcript:
        if self.cfg.stt.provider == "openai_compatible":
            return await self._openai(wav)
        return await self._scribe(wav)

    async def _scribe(self, wav: bytes) -> Transcript:
        s = self.cfg.stt
        if not self.elevenlabs_key:
            raise STTError("ELEVENLABS_API_KEY is not set")
        data = {"model_id": s.model_id}
        if s.language_code:
            data["language_code"] = s.language_code
        r = await self.client.post(
            f"{s.api_url}/v1/speech-to-text",
            headers={"xi-api-key": self.elevenlabs_key},
            data=data,
            files={"file": ("turn.wav", wav, "audio/wav")},
            timeout=s.timeout_s,
        )
        if r.status_code != 200:
            raise STTError(f"Scribe failed: HTTP {r.status_code} {r.text[:200]}")
        body = r.json()
        return Transcript((body.get("text") or "").strip(), body.get("language_code"))

    async def _openai(self, wav: bytes) -> Transcript:
        s = self.cfg.stt
        if not s.base_url:
            raise STTError("stt.base_url is required for the openai_compatible provider")
        data = {"model": s.model_id, "response_format": "verbose_json"}
        if s.language_code:
            data["language"] = s.language_code
        headers = {"Authorization": f"Bearer {self.openai_key}"} if self.openai_key else {}
        r = await self.client.post(
            f"{s.base_url.rstrip('/')}/audio/transcriptions",
            headers=headers,
            data=data,
            files={"file": ("turn.wav", wav, "audio/wav")},
            timeout=s.timeout_s,
        )
        if r.status_code != 200:
            raise STTError(f"transcription failed: HTTP {r.status_code} {r.text[:200]}")
        try:
            body = r.json()
        except ValueError:
            return Transcript(r.text.strip(), None)
        return Transcript((body.get("text") or "").strip(), body.get("language"))
