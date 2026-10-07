"""RTSP audio reader, voice activity detection and the echo gate."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
import wave
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from importlib import resources
from typing import Protocol

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # Silero VAD's window at 16 kHz: 32 ms
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE
FRAME_BYTES = FRAME_SAMPLES * 2


class Vad(Protocol):
    def reset(self) -> None: ...
    def prob(self, frame: np.ndarray) -> float: ...


class SileroVad:
    """Silero VAD on onnxruntime, without torch."""

    CONTEXT = 64

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        if model_path is None:
            ref = resources.files("doorstep") / "models" / "silero_vad.onnx"
            with resources.as_file(ref) as p:
                self.session = ort.InferenceSession(
                    str(p), sess_options=opts, providers=["CPUExecutionProvider"]
                )
        else:
            self.session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = (frame.astype(np.float32) / 32768.0).reshape(1, -1)
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self.session.run(
            None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        )
        self._context = x[:, -self.CONTEXT :]
        return float(out[0][0])


@dataclass
class Frame:
    t: float  # monotonic arrival time
    pcm: np.ndarray  # int16, FRAME_SAMPLES long


FrameSource = Callable[[], AsyncIterator[bytes]]


def ffmpeg_source(rtsp_url: str) -> FrameSource:
    async def gen() -> AsyncIterator[bytes]:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-i", rtsp_url,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        assert proc.stdout is not None
        try:
            while True:
                try:
                    yield await proc.stdout.readexactly(FRAME_BYTES)
                except asyncio.IncompleteReadError:
                    return
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 3)
                except TimeoutError:
                    proc.kill()

    return gen


class AudioReader:
    """One long-running reader of Frigate's re-stream, with a short rolling buffer."""

    def __init__(self, source: FrameSource, preroll_s: float = 2.0):
        self.source = source
        self.buffer: deque[Frame] = deque(maxlen=max(1, int(preroll_s / FRAME_S)))
        self._subscribers: set[asyncio.Queue[Frame]] = set()
        self._task: asyncio.Task[None] | None = None
        self.keep_until = 0.0  # monotonic; warm-up keepalive without a session
        self.frames_seen = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        self.buffer.clear()

    async def _run(self) -> None:
        backoff = 0.5
        while True:
            try:
                async for chunk in self.source():
                    backoff = 0.5
                    f = Frame(time.monotonic(), np.frombuffer(chunk, dtype=np.int16))
                    self.frames_seen += 1
                    self.buffer.append(f)
                    for q in list(self._subscribers):
                        if q.qsize() < 2000:
                            q.put_nowait(f)
                log.warning("audio reader stream ended; restarting")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("audio reader failed; restarting")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    @contextlib.contextmanager
    def subscribe(self):
        q: asyncio.Queue[Frame] = asyncio.Queue()
        self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)


@dataclass
class Utterance:
    heard: bool
    pcm: np.ndarray | None
    speech_seconds: float
    reason: str  # "speech" | "no_speech" | "too_short" | "no_audio"


def pcm_to_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class Listener:
    """Records one turn: waits for speech, then for end_silence_ms of silence."""

    PAD_FRAMES = 10  # keep ~0.3 s before the first speech frame

    def __init__(self, cfg: Config, reader: AudioReader, vad: Vad, gated: Callable[[float], bool]):
        self.cfg = cfg
        self.reader = reader
        self.vad = vad
        self.gated = gated
        self._lock = asyncio.Lock()

    async def record(
        self,
        max_seconds: float,
        start_timeout_s: float,
        end_silence_ms: int,
        speaking: Callable[[], bool] = lambda: False,
    ) -> Utterance:
        async with self._lock:
            # Never listen while the doorbell is talking.
            while speaking():  # noqa: ASYNC110 - a 50 ms poll is fine here
                await asyncio.sleep(0.05)
            self.reader.start()
            return await self._record(max_seconds, start_timeout_s, end_silence_ms)

    async def _record(self, max_seconds: float, start_timeout_s: float, end_silence_ms: int) -> Utterance:
        a = self.cfg.audio
        on = a.vad_threshold
        off = max(0.0, on - 0.15)
        end_frames = max(1, round(end_silence_ms / 1000 / FRAME_S))
        max_frames = max(1, round(max_seconds / FRAME_S))
        self.vad.reset()

        t0 = time.monotonic()
        start_deadline = t0 + start_timeout_s
        hard_deadline = start_deadline + max_seconds + end_silence_ms / 1000 + 1

        pre: deque[np.ndarray] = deque(maxlen=self.PAD_FRAMES)
        captured: list[np.ndarray] = []
        in_speech = False
        started = False
        speech_frames = 0
        silence_run = 0
        got_audio = False

        with self.reader.subscribe() as q:
            backlog = [f for f in self.reader.buffer if f.t >= t0 - a.preroll_s]
            last_t = backlog[-1].t if backlog else float("-inf")
            while True:
                if backlog:
                    f = backlog.pop(0)
                else:
                    now = time.monotonic()
                    deadline = hard_deadline if started else start_deadline
                    if now >= deadline:
                        break
                    try:
                        f = await asyncio.wait_for(q.get(), deadline - now)
                    except TimeoutError:
                        break
                    if f.t <= last_t:
                        continue  # already taken from the rolling buffer
                got_audio = True
                if self.gated(f.t):
                    # Echo gate: the doorbell's own voice. Forget anything before it.
                    pre.clear()
                    if not started:
                        self.vad.reset()
                    continue
                p = self.vad.prob(f.pcm)
                if not started:
                    if p >= on:
                        started = True
                        in_speech = True
                        captured.extend(pre)
                        captured.append(f.pcm)
                        speech_frames = 1
                    else:
                        pre.append(f.pcm)
                        if f.t >= start_deadline:
                            break
                    continue
                captured.append(f.pcm)
                if in_speech:
                    if p < off:
                        in_speech = False
                        silence_run = 1
                    else:
                        speech_frames += 1
                else:
                    if p >= on:
                        in_speech = True
                        silence_run = 0
                        speech_frames += 1
                    else:
                        silence_run += 1
                if silence_run >= end_frames or len(captured) >= max_frames + self.PAD_FRAMES:
                    break

        speech_s = speech_frames * FRAME_S
        if not got_audio:
            return Utterance(False, None, 0.0, "no_audio")
        if not started:
            return Utterance(False, None, 0.0, "no_speech")
        if speech_s < a.min_speech_s:
            return Utterance(False, None, round(speech_s, 2), "too_short")
        # Trim trailing silence beyond a short tail.
        if silence_run > self.PAD_FRAMES:
            captured = captured[: len(captured) - (silence_run - self.PAD_FRAMES)]
        return Utterance(True, np.concatenate(captured), round(speech_s, 2), "speech")
