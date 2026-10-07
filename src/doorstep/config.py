"""Configuration: one YAML file for settings, environment variables for secrets."""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrigateConfig(BaseModel):
    api_url: str = "http://frigate:5000"
    rtsp_url: str | None = None  # defaults to rtsp://<frigate host>:8554/<camera>
    go2rtc_url: str = "http://frigate:1984"
    timeout_s: float = 5.0


class TTSConfig(BaseModel):
    voice_id: str = ""
    model_id: str = "eleven_flash_v2_5"
    output_format: str = "mp3_22050_32"
    voice_settings: dict[str, Any] = Field(default_factory=dict)
    # Extra voices an agent may pick with door_say(voice=...). voice_id is always allowed.
    allowed_voices: list[str] = Field(default_factory=list)
    api_url: str = "https://api.elevenlabs.io"


class STTConfig(BaseModel):
    provider: Literal["elevenlabs", "openai_compatible"] = "elevenlabs"
    model_id: str = "scribe_v1"
    language_code: str | None = None
    # openai_compatible only: base URL ending in /v1 (or wherever /audio/transcriptions lives).
    base_url: str | None = None
    api_url: str = "https://api.elevenlabs.io"
    timeout_s: float = 20.0


class SessionConfig(BaseModel):
    idle_timeout_s: float = 180
    agent_timeout_s: float = 8
    allow_unprompted_speech: bool = False
    greet_on_ring: bool = True
    # A ring this soon after the previous one is counted on the open session, never a new task.
    ring_merge_s: float = 20


class LimitsConfig(BaseModel):
    max_chars: int = 300
    lines_per_minute: int = 12
    quiet_hours: str | None = "22:30-07:00"
    timezone: str = "Europe/London"

    @field_validator("quiet_hours")
    @classmethod
    def _check_quiet_hours(cls, v: str | None) -> str | None:
        if v:
            parse_quiet_hours(v)
        return v


class PhrasesConfig(BaseModel):
    disclosure: str = "Hi, you're speaking to an automated assistant."
    greeting: str = "One moment please."
    fallback: str = "Thanks for ringing. Please leave any parcels in the porch box."
    preload: list[str] = Field(default_factory=list)


class AudioConfig(BaseModel):
    echo_guard_ms: int = 600
    end_silence_ms: int = 1200
    vad_threshold: float = 0.5
    playback_tail_ms: int = 400
    min_speech_s: float = 0.3
    preroll_s: float = 2.0
    # Keep a warmed-up reader (approach hint) alive this long without a session.
    warm_keepalive_s: float = 60
    vad_model_path: str | None = None  # defaults to the bundled Silero model


class FronaConfig(BaseModel):
    url: str
    agent_id: str
    snapshot_max_height: int = 720
    title: str = "Doorbell {time}"
    message: str = (
        "Doorbell rang at {time}. Session {session_id}. Frigate recognised: {recognised}. "
        "Follow the rules in doorstep://house-rules."
    )
    approach_message: str = (
        "Someone is approaching the front door at {time} (no ring yet). Session {session_id}. "
        "Frigate recognised: {recognised}. Follow the rules in doorstep://house-rules."
    )
    timeout_s: float = 10.0


class MQTTConfig(BaseModel):
    host: str
    port: int = 1883
    username: str | None = None
    ring_topic: str | None = None
    frigate_events_topic: str = "frigate/events"


class ApproachConfig(BaseModel):
    enabled: bool = False
    zone: str | None = "porch"  # None: any person on the camera counts
    open_session_on_approach: bool = False


class RelayConfig(BaseModel):
    frona: FronaConfig | None = None
    webhook_url: str | None = None  # generic ring webhook: same payload as the Frona trigger
    after_webhook_url: str | None = None  # receives the closed session record from door_end
    mqtt_topic: str | None = None  # shorthand for mqtt.ring_topic


class Config(BaseModel):
    camera: str = "FrontDoorbell"
    twoway_stream: str = "FrontDoorbell_twoway"
    frigate: FrigateConfig = Field(default_factory=FrigateConfig)
    public_clip_base: str = "http://doorstep:8765"
    listen_host: str = "0.0.0.0"
    listen_port: int = 8765
    data_dir: Path = Path("/data")
    house_rules_path: Path = Path("/config/house-rules.md")
    retention_days: int = 30
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    phrases: PhrasesConfig = Field(default_factory=PhrasesConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    relay: RelayConfig = Field(default_factory=RelayConfig)
    mqtt: MQTTConfig | None = None
    approach: ApproachConfig = Field(default_factory=ApproachConfig)

    @model_validator(mode="after")
    def _derive(self) -> Config:
        if self.frigate.rtsp_url is None:
            host = self.frigate.api_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
            self.frigate.rtsp_url = f"rtsp://{host}:8554/{self.camera}"
        self.public_clip_base = self.public_clip_base.rstrip("/")
        return self

    @property
    def ring_topic(self) -> str | None:
        if self.mqtt and self.mqtt.ring_topic:
            return self.mqtt.ring_topic
        return self.relay.mqtt_topic

    @property
    def allowed_voices(self) -> set[str]:
        return {v for v in [self.tts.voice_id, *self.tts.allowed_voices] if v}


class Secrets(BaseSettings):
    """Secrets come only from the environment, never from the YAML file."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore", case_sensitive=False)

    elevenlabs_api_key: SecretStr | None = None
    doorstep_mcp_token: SecretStr | None = None
    doorstep_ring_token: SecretStr | None = None
    frona_trigger_token: SecretStr | None = None
    frigate_token: SecretStr | None = None
    mqtt_password: SecretStr | None = None
    stt_api_key: SecretStr | None = None  # openai_compatible provider only


def parse_quiet_hours(spec: str) -> tuple[time, time]:
    try:
        start_s, end_s = spec.split("-")
        start = time.fromisoformat(start_s.strip())
        end = time.fromisoformat(end_s.strip())
    except ValueError as e:
        raise ValueError(f"quiet_hours must look like '22:30-07:00', got {spec!r}") from e
    return start, end


_SECRET_KEYS = {"api_key", "token", "password", "xi-api-key"}


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    path = path or os.environ.get("DOORSTEP_CONFIG", "/config/config.yaml")
    p = Path(path)
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
    _refuse_secrets(raw)
    return Config.model_validate(raw)


def _refuse_secrets(node: Any, where: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() in _SECRET_KEYS:
                raise ValueError(
                    f"config key {where}{k} looks like a secret; put secrets in environment variables instead"
                )
            _refuse_secrets(v, f"{where}{k}.")
