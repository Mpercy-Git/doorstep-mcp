"""ElevenLabs text-to-speech, the phrase cache and the clip store."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Config

log = logging.getLogger(__name__)

CLIP_TTL_S = 120


class TTSError(Exception):
    pass


def mp3_duration(data: bytes, fallback_kbps: int = 32) -> float:
    """Clip length in seconds, read from the MP3 frames."""
    try:
        from mutagen.mp3 import MP3

        return float(MP3(io.BytesIO(data)).info.length)
    except Exception:
        return len(data) * 8 / (fallback_kbps * 1000)


@dataclass
class Clip:
    data: bytes
    duration_s: float
    cached: bool


class ElevenLabsTTS:
    def __init__(self, cfg: Config, api_key: str | None, client: httpx.AsyncClient):
        self.cfg = cfg
        self.api_key = api_key
        self.client = client

    def cache_key(self, text: str, voice_id: str) -> str:
        t = self.cfg.tts
        material = json.dumps(
            {
                "voice": voice_id,
                "model": t.model_id,
                "settings": t.voice_settings,
                "format": t.output_format,
                "text": text,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    async def synthesise(self, text: str, voice_id: str) -> bytes:
        if not self.api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set")
        if not voice_id:
            raise TTSError("tts.voice_id is not configured")
        t = self.cfg.tts
        body: dict[str, Any] = {"text": text, "model_id": t.model_id}
        if t.voice_settings:
            body["voice_settings"] = t.voice_settings
        r = await self.client.post(
            f"{t.api_url}/v1/text-to-speech/{voice_id}",
            params={"output_format": t.output_format},
            headers={"xi-api-key": self.api_key, "accept": "audio/mpeg"},
            json=body,
            timeout=15.0,
        )
        if r.status_code != 200:
            raise TTSError(f"ElevenLabs text-to-speech failed: HTTP {r.status_code} {r.text[:200]}")
        return r.content


class PhraseCache:
    """Rendered clips on disk, keyed by a hash of voice, model, settings and text."""

    def __init__(self, directory: Path, tts: ElevenLabsTTS):
        self.dir = directory
        self.tts = tts
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.mp3"

    async def get(self, text: str, voice_id: str) -> Clip:
        key = self.tts.cache_key(text, voice_id)
        p = self._path(key)
        if p.exists():
            data = p.read_bytes()
            return Clip(data, mp3_duration(data), cached=True)
        data = await self.tts.synthesise(text, voice_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        return Clip(data, mp3_duration(data), cached=False)

    async def preload(self, phrases: list[str], voice_id: str) -> None:
        for text in phrases:
            try:
                await self.get(text, voice_id)
            except Exception as e:
                log.warning("could not preload phrase %r: %s", text, e)


class ClipStore:
    """Short-lived clips served to go2rtc at /clips/<128-bit random id>.mp3."""

    def __init__(self, ttl_s: float = CLIP_TTL_S):
        self.ttl_s = ttl_s
        self._clips: dict[str, tuple[bytes, float]] = {}

    def put(self, data: bytes) -> str:
        self.expire()
        clip_id = secrets.token_hex(16)
        self._clips[clip_id] = (data, time.monotonic() + self.ttl_s)
        return clip_id

    def get(self, clip_id: str) -> bytes | None:
        self.expire()
        item = self._clips.get(clip_id)
        return item[0] if item else None

    def expire(self) -> None:
        now = time.monotonic()
        for k in [k for k, (_, exp) in self._clips.items() if exp <= now]:
            del self._clips[k]
