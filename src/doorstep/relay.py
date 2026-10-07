"""The ring relay: /ring, MQTT, the Frona trigger, the generic webhook, the fallback and the approach hint."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from typing import Any

from .core import Doorstep
from .frigate import FrigateError
from .session import Session

log = logging.getLogger(__name__)


class Relay:
    def __init__(self, door: Doorstep):
        self.door = door
        self.cfg = door.cfg
        self._last_ring_mono = float("-inf")
        self._watchdogs: dict[str, asyncio.Task] = {}
        self._approach_seen: set[str] = set()
        self._lock = asyncio.Lock()
        door.sessions.on_close.append(self._cancel_watchdog)
        door.sessions.on_close.append(self._after_hook)

    # --- rings -----------------------------------------------------------------------------

    async def ring(self, source: str, dry_run: bool = False) -> dict[str, Any]:
        async with self._lock:
            return await self._ring(source, dry_run)

    async def _ring(self, source: str, dry_run: bool) -> dict[str, Any]:
        door = self.door
        now = time.monotonic()
        door.sessions.record_ring()
        s = door.sessions.open_session()
        if s is not None:
            s.rings.append(time.time())
            door.sessions.touch(agent=False)
            door.audit.write("ring_counted", s.id, source=source)
            self._last_ring_mono = now
            return {"status": "counted", "session_id": s.id, "rings": len(s.rings)}
        if now - self._last_ring_mono < self.cfg.session.ring_merge_s:
            door.audit.write("ring_ignored", None, source=source, why="duplicate within ring_merge_s")
            return {"status": "duplicate"}
        self._last_ring_mono = now

        s = await door.sessions.open("dry_run" if dry_run else "ring")
        s.rings.append(time.time())
        door.audit.write("ring", s.id, source=source, dry_run=dry_run)
        if self.cfg.session.greet_on_ring and not dry_run:
            s.disclosed = True
            door.spawn(self._greet(s))
        await self._capture(s)
        payload = self.payload(s, "ring")

        if dry_run:
            shown = {**payload, "images": [{"media_type": i["media_type"], "bytes": len(i["data"]) * 3 // 4}
                                           for i in payload["images"]]}  # fmt: skip
            return {"status": "opened", "session_id": s.id, "dry_run": True, "would_send": shown}

        delivered = await self._wake(s, payload)
        if not delivered and (self.cfg.relay.frona or self.cfg.relay.webhook_url):
            door.spawn(self._fallback(s, "trigger failed"))
        else:
            self._watchdogs[s.id] = door.spawn(self._watchdog(s))
        return {"status": "opened", "session_id": s.id, "chat_id": s.chat_id, "agent_woken": delivered}

    async def _greet(self, s: Session) -> None:
        try:
            await self.door.announce(
                [self.cfg.phrases.disclosure, self.cfg.phrases.greeting], s, "greet_on_ring"
            )
        except Exception as e:
            s.disclosed = False  # the agent's first line will carry the disclosure instead
            log.error("greeting failed: %s", e)

    async def _capture(self, s: Session) -> None:
        h = self.cfg.relay.frona.snapshot_max_height if self.cfg.relay.frona else 720
        try:
            snap = await self.door.frigate.latest(h)
            s.ring_snapshot = snap.jpeg
        except FrigateError as e:
            log.warning("ring snapshot failed: %s", e)
        person = await self.door.frigate.latest_person()
        if person:
            s.ring_snapshot_meta = {
                k: person[k] for k in ("label", "sub_label", "score", "description", "event_id")
            }

    def payload(self, s: Session, kind: str) -> dict[str, Any]:
        frona = self.cfg.relay.frona
        when = datetime.fromtimestamp(s.opened_at, self.door.tz).strftime("%H:%M")
        fields = {
            "time": when,
            "session_id": s.id,
            "recognised": s.ring_snapshot_meta.get("sub_label") or "none",
            "camera": self.cfg.camera,
        }
        if frona:
            template = frona.approach_message if kind == "approach" else frona.message
            message = template.format(**fields)
            title = frona.title.format(**fields)
        else:
            message = (
                f"Doorbell {'approach' if kind == 'approach' else 'rang'} at {when}. Session {s.id}. "
                f"Frigate recognised: {fields['recognised']}. Follow the rules in doorstep://house-rules."
            )
            title = f"Doorbell {when}"
        images = []
        if s.ring_snapshot:
            images.append({"media_type": "image/jpeg", "data": base64.b64encode(s.ring_snapshot).decode()})
        return {"message": message, "title": title, "images": images}

    async def _wake(self, s: Session, payload: dict[str, Any]) -> bool:
        door = self.door
        delivered = False
        frona = self.cfg.relay.frona
        token = door.secrets.frona_trigger_token
        if frona and token:
            try:
                r = await door.http.post(
                    f"{frona.url.rstrip('/')}/api/agents/{frona.agent_id}/trigger",
                    headers={"Authorization": f"Bearer {token.get_secret_value()}"},
                    json=payload,
                    timeout=frona.timeout_s,
                )
                if r.status_code in (200, 201, 202):
                    try:
                        s.chat_id = r.json().get("chat_id")
                    except (ValueError, AttributeError):
                        pass
                    delivered = True
                    door.audit.write("agent_triggered", s.id, chat_id=s.chat_id)
                else:
                    door.audit.write("agent_trigger_failed", s.id, status=r.status_code, body=r.text[:200])
            except Exception as e:
                door.audit.write("agent_trigger_failed", s.id, error=str(e))
        elif frona:
            log.error("relay.frona is configured but FRONA_TRIGGER_TOKEN is not set")
        if self.cfg.relay.webhook_url:
            try:
                r = await door.http.post(
                    self.cfg.relay.webhook_url, json={**payload, "session_id": s.id}, timeout=10
                )
                ok = r.status_code < 400
                delivered = delivered or ok
                door.audit.write("webhook_sent", s.id, status=r.status_code)
            except Exception as e:
                door.audit.write("webhook_failed", s.id, error=str(e))
        return delivered

    async def _watchdog(self, s: Session) -> None:
        await asyncio.sleep(self.cfg.session.agent_timeout_s)
        if s.is_open and not s.agent_seen:
            await self._fallback(s, "agent_timeout")

    async def _fallback(self, s: Session, why: str) -> None:
        door = self.door
        if not s.is_open:
            return
        try:
            await door.announce([self.cfg.phrases.fallback], s, f"fallback: {why}")
        except Exception as e:
            log.error("fallback phrase failed: %s", e)
        if s.is_open and not (why == "agent_timeout" and s.agent_seen):
            await door.sessions.close("no_answer", f"Fallback played ({why}).", s)

    async def _cancel_watchdog(self, s: Session) -> None:
        t = self._watchdogs.pop(s.id, None)
        if t and t is not asyncio.current_task():
            t.cancel()

    # --- after hook ------------------------------------------------------------------------

    async def _after_hook(self, s: Session) -> None:
        if self.cfg.relay.after_webhook_url:
            self.door.spawn(self.after(s.public(self.door.tz)))

    async def after(self, record: dict[str, Any]) -> None:
        url = self.cfg.relay.after_webhook_url
        if not url:
            return
        try:
            await self.door.http.post(url, json=record, timeout=10)
        except Exception as e:
            log.warning("after hook failed: %s", e)

    # --- approach hint ---------------------------------------------------------------------

    async def frigate_event(self, ev: dict[str, Any]) -> None:
        ap = self.cfg.approach
        if not ap.enabled or ev.get("type") not in ("new", "update"):
            return
        after = ev.get("after") or {}
        if after.get("camera") != self.cfg.camera or after.get("label") != "person":
            return
        zones = set(after.get("current_zones") or []) | set(after.get("entered_zones") or [])
        if ap.zone and ap.zone not in zones:
            return
        eid = str(after.get("id"))
        if eid in self._approach_seen:
            return
        self._approach_seen.add(eid)
        if len(self._approach_seen) > 500:
            self._approach_seen.clear()
        await self.warm()
        if ap.open_session_on_approach:
            async with self._lock:
                if self.door.sessions.open_session() is None:
                    await self._approach_session()

    async def warm(self) -> None:
        """Start the audio reader and render the greeting, so a ring a few seconds later is quick."""
        door = self.door
        door.reader.keep_until = time.monotonic() + self.cfg.audio.warm_keepalive_s
        door.reader.start()
        door.audit.write("approach_warm", None)
        for t in (self.cfg.phrases.disclosure, self.cfg.phrases.greeting):
            if t:
                door.spawn(door.cache.get(t, self.cfg.tts.voice_id))

    async def _approach_session(self) -> None:
        door = self.door
        s = await door.sessions.open("approach")
        await self._capture(s)
        delivered = await self._wake(s, self.payload(s, "approach"))
        if delivered:
            self._watchdogs[s.id] = door.spawn(self._approach_watchdog(s))
        else:
            await door.sessions.close("no_answer", "Approach: no agent to wake.", s)

    async def _approach_watchdog(self, s: Session) -> None:
        # Nobody rang, so stay quiet: just close the session if the agent never shows up.
        await asyncio.sleep(self.cfg.session.agent_timeout_s)
        if s.is_open and not s.agent_seen:
            await self.door.sessions.close("no_answer", "Approach: agent did not respond.", s)

    # --- MQTT ------------------------------------------------------------------------------

    async def run_mqtt(self) -> None:
        m = self.cfg.mqtt
        if m is None:
            return
        ring_topic = self.cfg.ring_topic
        events_topic = m.frigate_events_topic if self.cfg.approach.enabled else None
        if not ring_topic and not events_topic:
            return
        import aiomqtt

        pw = self.door.secrets.mqtt_password
        backoff = 1.0
        while True:
            try:
                async with aiomqtt.Client(
                    m.host, m.port, username=m.username, password=pw.get_secret_value() if pw else None,
                    identifier="doorstep",
                ) as client:  # fmt: skip
                    if ring_topic:
                        await client.subscribe(ring_topic)
                    if events_topic:
                        await client.subscribe(events_topic)
                    backoff = 1.0
                    log.info("MQTT connected to %s:%s", m.host, m.port)
                    async for msg in client.messages:
                        topic = str(msg.topic)
                        if ring_topic and msg.topic.matches(ring_topic):
                            self.door.spawn(self.ring(f"mqtt:{topic}"))
                        elif events_topic and msg.topic.matches(events_topic):
                            try:
                                ev = json.loads(msg.payload)
                            except (ValueError, TypeError):
                                continue
                            self.door.spawn(self.frigate_event(ev))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("MQTT connection failed: %s; retrying in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
