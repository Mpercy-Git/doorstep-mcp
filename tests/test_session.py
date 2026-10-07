from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from doorstep.config import load_config
from doorstep.session import AuditLog, QuietHours, RateLimiter, Refused, SessionManager

from .conftest import make_config

TZ = ZoneInfo("Europe/London")


def at(hh: int, mm: int) -> float:
    return datetime(2026, 10, 7, hh, mm, tzinfo=TZ).timestamp()


def test_quiet_hours_overnight_window():
    q = QuietHours("22:30-07:00", TZ)
    assert q.active(at(23, 0))
    assert q.active(at(3, 0))
    assert q.active(at(22, 30))
    assert not q.active(at(7, 0))
    assert not q.active(at(14, 0))


def test_quiet_hours_same_day_window_and_off():
    q = QuietHours("13:00-14:00", TZ)
    assert q.active(at(13, 30))
    assert not q.active(at(14, 30))
    assert not QuietHours(None, TZ).active(at(3, 0))


def test_bad_quiet_hours_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_config(tmp_path, limits={"quiet_hours": "late"})


def test_rate_limiter_window():
    now = [1000.0]
    r = RateLimiter(2, clock=lambda: now[0])
    r.check()
    r.hit()
    r.hit()
    with pytest.raises(Refused, match="rate limit"):
        r.check()
    now[0] += 61
    r.check()


@pytest.fixture
def manager(tmp_path):
    cfg = make_config(tmp_path)
    return SessionManager(cfg, AuditLog(tmp_path / "log", TZ, 30))


def test_speech_refused_outside_session(manager):
    with pytest.raises(Refused, match="no session"):
        manager.check_speech("hello", None)


async def test_speech_rules_inside_session(manager):
    await manager.open("ring")
    assert manager.check_speech("hello", None) is not None
    with pytest.raises(Refused, match="limit is 300"):
        manager.check_speech("x" * 301, None)
    with pytest.raises(Refused, match="empty"):
        manager.check_speech("   ", None)
    with pytest.raises(Refused, match="allowlist"):
        manager.check_speech("hello", "somebody-famous")
    assert manager.check_speech("hello", "voice-alt")


def test_unprompted_speech_respects_quiet_hours(tmp_path):
    cfg = make_config(
        tmp_path, session={"allow_unprompted_speech": True}, limits={"quiet_hours": "00:00-23:59"}
    )
    m = SessionManager(cfg, AuditLog(tmp_path / "log", TZ, 30))
    with pytest.raises(Refused, match="quiet hours"):
        m.check_speech("hello", None)


async def test_session_close_and_idle_reap(manager):
    now = [1000.0]
    manager.clock = lambda: now[0]
    s = await manager.open("ring")
    with pytest.raises(RuntimeError):
        await manager.open("ring")
    now[0] += 179
    await manager.reap_once()
    assert s.is_open
    now[0] += 2
    await manager.reap_once()
    assert not s.is_open and s.outcome == "no_answer"
    with pytest.raises(Refused):
        await manager.close("visitor", "x")


async def test_bad_outcome_refused(manager):
    await manager.open("ring")
    with pytest.raises(Refused, match="outcome"):
        await manager.close("burglar", "x")


def test_audit_log_prune_and_search(tmp_path):
    log = AuditLog(tmp_path / "log", TZ, retention_days=30)
    log.write("session_closed", "d-1", outcome="visitor", summary="hi")
    old = date.today() - timedelta(days=40)
    (tmp_path / "log" / f"{old.isoformat()}.jsonl").write_text("{}\n")
    assert log.prune() == 1
    recs = log.closed_sessions(datetime.now(TZ) - timedelta(hours=1))
    assert [r["session_id"] for r in recs] == ["d-1"]


def test_config_refuses_secrets_in_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tts:\n  api_key: sk-123\n")
    with pytest.raises(ValueError, match="secret"):
        load_config(p)


def test_config_derives_rtsp_url(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera: Porch\nfrigate:\n  api_url: http://nvr:5000\n")
    assert load_config(p).frigate.rtsp_url == "rtsp://nvr:8554/Porch"


async def test_approach_session_silent_in_quiet_hours_until_ring(tmp_path):
    cfg = make_config(tmp_path, limits={"quiet_hours": "00:00-23:59"})
    m = SessionManager(cfg, AuditLog(tmp_path / "log", TZ, 30))
    s = await m.open("approach")
    with pytest.raises(Refused, match="nobody has rung"):
        m.check_speech("hello", None)
    s.rings.append(1.0)
    assert m.check_speech("hello", None) is s
