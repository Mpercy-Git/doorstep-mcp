from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pytest
import pytest_asyncio

from doorstep.config import Config, Secrets
from doorstep.core import Doorstep
from doorstep.listen import FRAME_BYTES, FRAME_SAMPLES

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


@pytest.fixture(scope="session")
def mp3_clip() -> bytes:
    if not HAVE_FFMPEG:
        return b"\x00" * 400  # ~0.1 s at the 32 kbps fallback estimate
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
         "-ar", "22050", "-b:a", "32k", "-f", "mp3", "pipe:1"],
        capture_output=True, check=True,
    )  # fmt: skip
    return out.stdout


class FakeWorld:
    """Fake Frigate, go2rtc, ElevenLabs and Frona, recording every request."""

    def __init__(self, mp3: bytes):
        self.mp3 = mp3
        self.requests: list[httpx.Request] = []
        self.transcript = "Hello, I have a parcel for number 12."
        self.frona_status = 202
        self.person_events: list[dict] = []
        self.in_progress: list[dict] = []
        self.webhook_bodies: list[dict] = []

    def of(self, host: str, path_prefix: str = "") -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host and r.url.path.startswith(path_prefix)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        u = request.url
        if u.host == "api.elevenlabs.io" and u.path.startswith("/v1/text-to-speech/"):
            return httpx.Response(200, content=self.mp3, headers={"content-type": "audio/mpeg"})
        if u.host == "api.elevenlabs.io" and u.path == "/v1/speech-to-text":
            return httpx.Response(200, json={"text": self.transcript, "language_code": "eng"})
        if u.host == "frigate" and u.port == 1984:
            if u.path == "/api/streams" and request.method == "POST":
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"FrontDoorbell_twoway": {}})
        if u.host == "frigate" and u.path.endswith("/latest.jpg"):
            return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
        if u.host == "frigate" and u.path == "/api/events":
            q = parse_qs(urlparse(str(u)).query)
            return httpx.Response(200, json=self.in_progress if "in_progress" in q else self.person_events)
        if u.host == "frigate" and u.path.startswith("/api/events/") and u.path.endswith("snapshot.jpg"):
            return httpx.Response(200, content=JPEG)
        if u.host == "frigate" and u.path.startswith("/api/events/"):
            return httpx.Response(200, json={"id": u.path.split("/")[3], "camera": "FrontDoorbell",
                                             "label": "person", "start_time": 1.0})  # fmt: skip
        if u.host == "frona":
            return httpx.Response(self.frona_status, json={"chat_id": "chat-1", "message_id": "m-1"})
        if u.host == "hooks":
            self.webhook_bodies.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.Response(404)


class FakeMic:
    """Frames pushed by the test stand in for Frigate's RTSP audio."""

    def __init__(self):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

    def source(self):
        async def gen() -> AsyncIterator[bytes]:
            while True:
                yield await self.queue.get()

        return gen

    def push(self, loud: bool, frames: int = 1) -> None:
        amp = 8000 if loud else 0
        for _ in range(frames):
            self.queue.put_nowait(np.full(FRAME_SAMPLES, amp, dtype=np.int16).tobytes())
        assert FRAME_BYTES == FRAME_SAMPLES * 2


class LoudnessVad:
    """Speech when a frame is loud: lets tests script exactly when the visitor talks."""

    def reset(self) -> None:
        pass

    def prob(self, frame: np.ndarray) -> float:
        return 1.0 if np.abs(frame).mean() > 1000 else 0.0


def make_config(tmp_path: Path, **over) -> Config:
    raw = {
        "data_dir": str(tmp_path / "data"),
        "house_rules_path": str(tmp_path / "house-rules.md"),
        "tts": {"voice_id": "voice-gb", "allowed_voices": ["voice-alt"]},
        "frigate": {"api_url": "http://frigate:5000", "go2rtc_url": "http://frigate:1984"},
        "session": {"agent_timeout_s": 5, "idle_timeout_s": 180, "greet_on_ring": True},
        "limits": {"quiet_hours": None},
        "audio": {"playback_tail_ms": 0, "echo_guard_ms": 50, "end_silence_ms": 300},
        "relay": {"frona": {"url": "http://frona:3001", "agent_id": "agent-1"}},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return Config.model_validate(raw)


SECRETS = Secrets(
    elevenlabs_api_key="el-key",
    doorstep_mcp_token="mcp-token",
    doorstep_ring_token="ring-token",
    frona_trigger_token="frona-token",
)


@pytest.fixture
def world(mp3_clip) -> FakeWorld:
    return FakeWorld(mp3_clip)


@pytest.fixture
def mic() -> FakeMic:
    return FakeMic()


@pytest_asyncio.fixture
async def make_door(tmp_path, world, mic):
    created: list[Doorstep] = []

    def make(**over) -> Doorstep:
        cfg = make_config(tmp_path, **over)
        http = httpx.AsyncClient(transport=httpx.MockTransport(world.handler))
        door = Doorstep(cfg, SECRETS, http, vad_factory=LoudnessVad, frame_source=mic.source())
        created.append(door)
        return door

    yield make
    for door in created:
        await door.stop()
        await door.http.aclose()
