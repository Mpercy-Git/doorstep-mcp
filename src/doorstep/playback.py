"""go2rtc client and the speaker: one line at a time across the whole server."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from .config import Config
from .speech import Clip, ClipStore, PhraseCache

log = logging.getLogger(__name__)


class PlaybackError(Exception):
    pass


class Go2rtc:
    def __init__(self, cfg: Config, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client

    async def play_url(self, url: str) -> None:
        """Ask go2rtc's ffmpeg to fetch the clip, transcode to PCMU and push it to the doorbell."""
        src = f"ffmpeg:{url}#audio=pcmu#input=file"
        r = await self.client.post(
            f"{self.cfg.frigate.go2rtc_url}/api/streams",
            params={"dst": self.cfg.twoway_stream, "src": src},
            timeout=self.cfg.frigate.timeout_s,
        )
        if r.status_code >= 400:
            raise PlaybackError(f"go2rtc refused playback: HTTP {r.status_code} {r.text[:200]}")

    async def streams(self) -> dict:
        r = await self.client.get(
            f"{self.cfg.frigate.go2rtc_url}/api/streams", timeout=self.cfg.frigate.timeout_s
        )
        r.raise_for_status()
        return r.json()


@dataclass
class Spoken:
    said: str
    duration_s: float
    cached: bool


class Speaker:
    """Renders, hosts and plays lines, and tells the listener when the doorbell is talking."""

    def __init__(self, cfg: Config, cache: PhraseCache, clips: ClipStore, go2rtc: Go2rtc):
        self.cfg = cfg
        self.cache = cache
        self.clips = clips
        self.go2rtc = go2rtc
        self._lock = asyncio.Lock()
        self.speaking = False
        # Monotonic time before which the microphone is treated as echo.
        self.gate_until = 0.0

    def gated(self, at: float | None = None) -> bool:
        at = time.monotonic() if at is None else at
        return self.speaking or at < self.gate_until

    async def render(self, text: str, voice: str | None = None) -> Clip:
        return await self.cache.get(text, voice or self.cfg.tts.voice_id)

    async def play(self, clips: list[tuple[str, Clip]], wait: bool = True) -> Spoken:
        """Play pre-rendered clips back to back. Holds the speaker until playback ends."""
        if wait:
            return await self._play_locked(clips)
        task = asyncio.create_task(self._play_locked(clips))
        task.add_done_callback(_log_task_error)
        total = sum(c.duration_s for _, c in clips)
        return Spoken(" ".join(t for t, _ in clips), total, all(c.cached for _, c in clips))

    async def _play_locked(self, clips: list[tuple[str, Clip]]) -> Spoken:
        tail = self.cfg.audio.playback_tail_ms / 1000
        guard = self.cfg.audio.echo_guard_ms / 1000
        async with self._lock:
            self.speaking = True
            try:
                for _, clip in clips:
                    clip_id = self.clips.put(clip.data)
                    started = time.monotonic()
                    await self.go2rtc.play_url(f"{self.cfg.public_clip_base}/clips/{clip_id}.mp3")
                    # go2rtc may return before or after playback; either way wait out the clip.
                    remaining = clip.duration_s + tail - (time.monotonic() - started)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
            finally:
                self.speaking = False
                self.gate_until = time.monotonic() + guard
        return Spoken(
            " ".join(t for t, _ in clips),
            sum(c.duration_s for _, c in clips),
            all(c.cached for _, c in clips),
        )


def _log_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        log.error("background playback failed: %s", task.exception())
