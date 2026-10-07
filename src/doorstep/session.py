"""Session state, timeouts, rate limits, quiet hours and the audit log."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from .config import Config, parse_quiet_hours

log = logging.getLogger(__name__)

Outcome = Literal["delivery", "visitor", "sales_or_canvassing", "no_answer", "other"]
OUTCOMES: tuple[str, ...] = ("delivery", "visitor", "sales_or_canvassing", "no_answer", "other")


class Refused(Exception):
    """A request the server will not carry out. The message is shown to the agent."""


@dataclass
class Session:
    id: str
    opened_at: float
    reason: str  # "ring" | "approach" | "dry_run"
    last_activity: float
    rings: list[float] = field(default_factory=list)
    turns: int = 0
    lines_spoken: int = 0
    disclosed: bool = False
    agent_seen: bool = False
    chat_id: str | None = None
    ring_snapshot: bytes | None = None
    ring_snapshot_meta: dict[str, Any] = field(default_factory=dict)
    closed_at: float | None = None
    outcome: str | None = None
    summary: str | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    def public(self, tz: ZoneInfo) -> dict[str, Any]:
        return {
            "id": self.id,
            "reason": self.reason,
            "opened_at": iso(self.opened_at, tz),
            "turns": self.turns,
            "rings": len(self.rings),
            "lines_spoken": self.lines_spoken,
            "chat_id": self.chat_id,
            "closed_at": iso(self.closed_at, tz) if self.closed_at else None,
            "outcome": self.outcome,
            "summary": self.summary,
        }


def iso(ts: float | None, tz: ZoneInfo) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz).isoformat(timespec="seconds")


class RateLimiter:
    """Sliding one-minute window, shared by the whole server."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self._hits: deque[float] = deque()

    def _trim(self) -> None:
        cutoff = self.clock() - 60
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()

    def check(self) -> None:
        self._trim()
        if len(self._hits) >= self.per_minute:
            wait = 60 - (self.clock() - self._hits[0])
            raise Refused(f"rate limit: at most {self.per_minute} lines a minute; try again in {wait:.0f}s")

    def hit(self) -> None:
        self._trim()
        self._hits.append(self.clock())


class QuietHours:
    def __init__(self, spec: str | None, tz: ZoneInfo):
        self.window = parse_quiet_hours(spec) if spec else None
        self.tz = tz

    def active(self, at: float | None = None) -> bool:
        if not self.window:
            return False
        now = datetime.fromtimestamp(at if at is not None else time.time(), self.tz).time()
        start, end = self.window
        if start <= end:
            return start <= now < end
        return now >= start or now < end


class AuditLog:
    """Every session event, line spoken and transcript, as JSON lines in one file per day."""

    def __init__(self, directory: Path, tz: ZoneInfo, retention_days: int):
        self.dir = directory
        self.tz = tz
        self.retention_days = retention_days
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, d: date) -> Path:
        return self.dir / f"{d.isoformat()}.jsonl"

    def write(self, event: str, session_id: str | None, **data: Any) -> None:
        now = time.time()
        record = {"ts": iso(now, self.tz), "event": event, "session_id": session_id, **data}
        line = json.dumps(record, ensure_ascii=False)
        with self._path(datetime.fromtimestamp(now, self.tz).date()).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        log.info(event, extra={"session_id": session_id, "data": data})

    def read_day(self, d: date) -> list[dict[str, Any]]:
        p = self._path(d)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def closed_sessions(self, since: datetime) -> list[dict[str, Any]]:
        out = []
        d = since.astimezone(self.tz).date()
        today = datetime.now(self.tz).date()
        while d <= today:
            for rec in self.read_day(d):
                if rec.get("event") == "session_closed" and datetime.fromisoformat(rec["ts"]) >= since:
                    out.append(rec)
            d += timedelta(days=1)
        return out

    def prune(self) -> int:
        cutoff = datetime.now(self.tz).date() - timedelta(days=self.retention_days)
        removed = 0
        for p in self.dir.glob("*.jsonl"):
            try:
                d = date.fromisoformat(p.stem)
            except ValueError:
                continue
            if d < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        return removed


SessionHook = Callable[[Session], Awaitable[None]]


class SessionManager:
    def __init__(self, cfg: Config, audit: AuditLog, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.audit = audit
        self.clock = clock
        self.tz = ZoneInfo(cfg.limits.timezone)
        self.current: Session | None = None
        self.last_closed: Session | None = None
        self.last_ring_at: float | None = None
        self.rate = RateLimiter(cfg.limits.lines_per_minute)
        self.quiet = QuietHours(cfg.limits.quiet_hours, self.tz)
        self.on_open: list[SessionHook] = []
        self.on_close: list[SessionHook] = []
        self._reaper: asyncio.Task[None] | None = None

    # --- lifecycle -------------------------------------------------------------------------

    def open_session(self) -> Session | None:
        s = self.current
        return s if s and s.is_open else None

    async def open(self, reason: str) -> Session:
        if self.open_session():
            raise RuntimeError("a session is already open")
        now = self.clock()
        s = Session(id=f"d-{secrets.token_hex(3)}", opened_at=now, reason=reason, last_activity=now)
        self.current = s
        self.audit.write("session_opened", s.id, reason=reason)
        for hook in self.on_open:
            try:
                await hook(s)
            except Exception:
                log.exception("session open hook failed")
        return s

    async def close(self, outcome: str, summary: str, session: Session | None = None) -> Session:
        s = session or self.open_session()
        if not s or not s.is_open:
            raise Refused("no session is open")
        if outcome not in OUTCOMES:
            raise Refused(f"outcome must be one of {', '.join(OUTCOMES)}")
        s.closed_at = self.clock()
        s.outcome = outcome
        s.summary = summary[:200]
        if self.current is s:
            self.current = None
        self.last_closed = s
        self.audit.write("session_closed", s.id, **{k: v for k, v in s.public(self.tz).items() if k != "id"})
        for hook in self.on_close:
            try:
                await hook(s)
            except Exception:
                log.exception("session close hook failed")
        return s

    def touch(self, agent: bool = True) -> Session | None:
        s = self.open_session()
        if s:
            s.last_activity = self.clock()
            if agent:
                s.agent_seen = True
        return s

    def record_ring(self) -> None:
        self.last_ring_at = self.clock()

    async def reap_once(self) -> None:
        s = self.open_session()
        if s and self.clock() - s.last_activity > self.cfg.session.idle_timeout_s:
            await self.close("other" if s.agent_seen else "no_answer", "Closed after inactivity.", s)

    async def run_reaper(self, interval: float = 5.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_once()
            except Exception:
                log.exception("session reaper failed")

    # --- speech policy ---------------------------------------------------------------------

    def check_speech(self, text: str, voice: str | None) -> Session | None:
        """Refuse before anything is spent. Returns the open session, if any."""
        text = text.strip()
        if not text:
            raise Refused("text is empty")
        if len(text) > self.cfg.limits.max_chars:
            raise Refused(f"text is {len(text)} characters; the limit is {self.cfg.limits.max_chars}")
        if voice and voice not in self.cfg.allowed_voices:
            raise Refused("voice is not on the allowlist")
        s = self.open_session()
        if s is None:
            if not self.cfg.session.allow_unprompted_speech:
                raise Refused("no session is open: speech is only allowed after a ring")
            if self.quiet.active():
                raise Refused("quiet hours: no unprompted speech")
        elif s.reason == "approach" and not s.rings and self.quiet.active():
            raise Refused("quiet hours: nobody has rung, so no speech")
        self.rate.check()
        return s
