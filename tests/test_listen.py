import asyncio
import shutil
import subprocess
import time

import numpy as np
import pytest

from doorstep.listen import FRAME_SAMPLES, SAMPLE_RATE, AudioReader, Listener, SileroVad, pcm_to_wav

from .conftest import LoudnessVad, make_config


async def _listener(tmp_path, mic, gated=lambda t: False, **audio):
    cfg = make_config(tmp_path, audio={"end_silence_ms": 300, "min_speech_s": 0.3, **audio})
    reader = AudioReader(mic.source(), preroll_s=cfg.audio.preroll_s)
    reader.start()
    return Listener(cfg, reader, LoudnessVad(), gated), reader


async def test_turn_ends_after_silence(tmp_path, mic):
    listener, reader = await _listener(tmp_path, mic)
    mic.push(False, 5)
    mic.push(True, 20)  # 0.64 s of speech
    mic.push(False, 40)
    utt = await listener.record(max_seconds=5, start_timeout_s=2, end_silence_ms=300)
    await reader.stop()
    assert utt.heard and utt.reason == "speech"
    assert utt.speech_seconds == pytest.approx(20 * 0.032, abs=0.01)
    # padding before speech plus a short tail, but trailing silence is trimmed
    assert len(utt.pcm) <= (5 + 20 + 10) * FRAME_SAMPLES


async def test_no_speech_times_out(tmp_path, mic):
    listener, reader = await _listener(tmp_path, mic)
    for _ in range(20):
        mic.push(False, 1)
    t0 = time.monotonic()
    utt = await listener.record(max_seconds=5, start_timeout_s=0.4, end_silence_ms=300)
    await reader.stop()
    assert not utt.heard and utt.reason == "no_speech"
    assert time.monotonic() - t0 < 1.5


async def test_short_noise_is_ignored(tmp_path, mic):
    listener, reader = await _listener(tmp_path, mic)
    mic.push(True, 4)  # 0.13 s: a car door
    mic.push(False, 40)
    utt = await listener.record(max_seconds=5, start_timeout_s=1, end_silence_ms=300)
    await reader.stop()
    assert not utt.heard and utt.reason == "too_short"


async def test_echo_gate_discards_doorbell_voice(tmp_path, mic):
    gate = {"until": float("inf")}
    listener, reader = await _listener(tmp_path, mic, gated=lambda t: t < gate["until"])
    task = asyncio.create_task(listener.record(max_seconds=5, start_timeout_s=1.5, end_silence_ms=300))
    mic.push(True, 30)  # our own voice coming back through the microphone
    await asyncio.sleep(0.1)
    gate["until"] = time.monotonic()
    mic.push(False, 40)
    utt = await task
    await reader.stop()
    assert not utt.heard


async def test_no_audio_reported(tmp_path, mic):
    listener, reader = await _listener(tmp_path, mic)
    utt = await listener.record(max_seconds=1, start_timeout_s=0.3, end_silence_ms=300)
    await reader.stop()
    assert utt.reason == "no_audio"


async def test_max_seconds_caps_turn(tmp_path, mic):
    listener, reader = await _listener(tmp_path, mic)
    mic.push(True, 200)
    utt = await listener.record(max_seconds=1, start_timeout_s=1, end_silence_ms=300)
    await reader.stop()
    assert utt.heard
    assert utt.speech_seconds <= 1.4


def test_wav_header():
    wav = pcm_to_wav(np.zeros(1600, dtype=np.int16))
    assert wav[:4] == b"RIFF" and len(wav) == 44 + 3200


def _flite(text: str) -> np.ndarray | None:
    if not shutil.which("ffmpeg"):
        return None
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"flite=text='{text}'",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
        capture_output=True,
    )  # fmt: skip
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.int16)


def _max_prob(vad: SileroVad, pcm: np.ndarray) -> float:
    vad.reset()
    n = len(pcm) // FRAME_SAMPLES
    return max(vad.prob(pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(n))


def test_silero_tells_speech_from_noise():
    speech = _flite("Hello, I have a parcel for you. Could you sign for it please?")
    if speech is None:
        pytest.skip("ffmpeg flite filter not available")
    vad = SileroVad()
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 600, SAMPLE_RATE * 2)).astype(np.int16)  # steady hiss, like rain
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    assert _max_prob(vad, speech) > 0.5
    assert _max_prob(vad, noise) < 0.5
    assert _max_prob(vad, silence) < 0.5
