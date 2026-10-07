"""Frigate client: snapshots and person events on the doorbell camera."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config


class FrigateError(Exception):
    pass


@dataclass
class Snapshot:
    jpeg: bytes
    captured_at: float
    meta: dict[str, Any] = field(default_factory=dict)


def _sub_label(ev: dict[str, Any]) -> str | None:
    sub = ev.get("sub_label")
    # Frigate returns either "Name" or ["Name", score] depending on version.
    if isinstance(sub, list | tuple):
        return sub[0] if sub else None
    return sub or None


def event_summary(ev: dict[str, Any]) -> dict[str, Any]:
    data = ev.get("data") or {}
    score = data.get("top_score") or ev.get("top_score") or data.get("score") or ev.get("score")
    return {
        "event_id": ev.get("id"),
        "label": ev.get("label"),
        "sub_label": _sub_label(ev),
        "score": round(score, 2) if isinstance(score, int | float) else None,
        "start": ev.get("start_time"),
        "end": ev.get("end_time"),
        "has_snapshot": bool(ev.get("has_snapshot")),
        "zones": ev.get("zones") or [],
        "description": data.get("description") or ev.get("description") or None,
    }


class Frigate:
    def __init__(self, cfg: Config, token: str | None, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def _get(self, path: str, **params: Any) -> httpx.Response:
        try:
            r = await self.client.get(
                f"{self.cfg.frigate.api_url}{path}",
                params={k: v for k, v in params.items() if v is not None},
                headers=self.headers,
                timeout=self.cfg.frigate.timeout_s,
            )
        except httpx.HTTPError as e:
            raise FrigateError(f"Frigate unreachable: {e}") from e
        if r.status_code != 200:
            raise FrigateError(f"Frigate {path}: HTTP {r.status_code}")
        return r

    async def latest(self, max_height: int = 720) -> Snapshot:
        r = await self._get(f"/api/{self.cfg.camera}/latest.jpg", h=max_height)
        return Snapshot(r.content, time.time(), {"source": "latest"})

    async def event(self, event_id: str) -> dict[str, Any]:
        r = await self._get(f"/api/events/{event_id}")
        return r.json()

    async def event_snapshot(self, event_id: str, max_height: int = 720) -> Snapshot:
        ev = await self.event(event_id)
        if ev.get("camera") and ev["camera"] != self.cfg.camera:
            raise FrigateError("that event is not from the doorbell camera")
        r = await self._get(f"/api/events/{event_id}/snapshot.jpg", h=max_height, bbox=0)
        return Snapshot(
            r.content, ev.get("start_time") or time.time(), {"source": "event", **event_summary(ev)}
        )

    async def person_events(self, after: float | None = None, limit: int = 10, in_progress: bool = False):
        r = await self._get(
            "/api/events",
            camera=self.cfg.camera,
            labels="person",
            after=after,
            limit=limit,
            in_progress=1 if in_progress else None,
        )
        return [event_summary(e) for e in r.json()]

    async def person_present(self) -> bool:
        return bool(await self.person_events(limit=1, in_progress=True))

    async def latest_person(self, within_s: float = 60) -> dict[str, Any] | None:
        """The most recent person event on the doorbell, for labelling a fresh snapshot."""
        try:
            events = await self.person_events(after=time.time() - within_s, limit=1)
        except FrigateError:
            return None
        return events[0] if events else None
