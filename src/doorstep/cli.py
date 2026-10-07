"""The doorstep command: run the server, and drive a running one on site."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any

from .config import Config, Secrets, load_config


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("session_id", "data"):
            if hasattr(record, key):
                out[key] = getattr(record, key)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    for name in ("httpx", "httpx2", "mcp"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_serve(cfg: Config, secrets: Secrets, args: argparse.Namespace) -> int:
    import uvicorn

    from .web import build_app

    app = build_app(cfg, secrets)
    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port, log_config=None, access_log=False)
    return 0


def _base_url(cfg: Config, args: argparse.Namespace) -> str:
    return (args.url or os.environ.get("DOORSTEP_URL") or f"http://127.0.0.1:{cfg.listen_port}").rstrip("/")


async def _call(cfg: Config, secrets: Secrets, args: argparse.Namespace, tool: str, arguments: dict) -> Any:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    token = secrets.doorstep_mcp_token
    if token is None:
        raise SystemExit("DOORSTEP_MCP_TOKEN is not set")
    http = httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token.get_secret_value()}"},
        timeout=httpx2.Timeout(30, read=120),
    )
    async with (
        http,
        Client(streamable_http_client(f"{_base_url(cfg, args)}/mcp", http_client=http)) as client,
    ):
        return await client.call_tool(tool, arguments)


def _print_result(result: Any, image_out: str | None = None) -> int:
    for block in result.content:
        if block.type == "text":
            print(block.text)
        elif block.type == "image":
            data = base64.b64decode(block.data)
            path = image_out or f"doorstep-{int(time.time())}.jpg"
            with open(path, "wb") as f:
                f.write(data)
            print(f"saved {len(data)} bytes to {path}")
    return 1 if result.is_error else 0


def cmd_tool(cfg: Config, secrets: Secrets, args: argparse.Namespace) -> int:
    if args.cmd == "say":
        tool, a = "door_say", {"text": args.text}
    elif args.cmd == "listen":
        tool, a = "door_listen", {"max_seconds": args.max_seconds}
    elif args.cmd == "converse":
        tool, a = "door_converse", {"text": args.text, "max_seconds": args.max_seconds}
    elif args.cmd == "snapshot":
        tool, a = "door_snapshot", {"source": args.source}
        if args.event_id:
            a["event_id"] = args.event_id
    elif args.cmd == "status":
        tool, a = "door_status", {}
    elif args.cmd == "visitors":
        tool, a = "door_visitors", {"hours": args.hours}
    elif args.cmd == "end":
        tool, a = "door_end", {"outcome": args.outcome, "summary": args.summary}
    else:
        raise AssertionError(args.cmd)
    result = asyncio.run(_call(cfg, secrets, args, tool, a))
    return _print_result(result, getattr(args, "output", None))


def cmd_ring(cfg: Config, secrets: Secrets, args: argparse.Namespace) -> int:
    import httpx

    token = secrets.doorstep_ring_token
    if token is None:
        raise SystemExit("DOORSTEP_RING_TOKEN is not set")
    r = httpx.post(
        f"{_base_url(cfg, args)}/ring",
        params={"dry_run": "1" if args.dry_run else "0", "source": "cli"},
        headers={"Authorization": f"Bearer {token.get_secret_value()}"},
        timeout=30,
    )
    print(json.dumps(r.json(), indent=2))
    return 0 if r.status_code < 300 else 1


def cmd_check(cfg: Config, secrets: Secrets, args: argparse.Namespace) -> int:
    """Check Frigate, go2rtc and the RTSP audio from wherever Doorstep runs."""
    import subprocess

    import httpx

    ok = True
    headers = {}
    if secrets.frigate_token:
        headers["Authorization"] = f"Bearer {secrets.frigate_token.get_secret_value()}"

    def report(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {detail}")

    try:
        r = httpx.get(f"{cfg.frigate.api_url}/api/{cfg.camera}/latest.jpg", params={"h": 720},
                      headers=headers, timeout=5)  # fmt: skip
        report("frigate snapshot", r.status_code == 200 and r.content[:2] == b"\xff\xd8",
               f"HTTP {r.status_code}, {len(r.content)} bytes")  # fmt: skip
    except httpx.HTTPError as e:
        report("frigate snapshot", False, str(e))
    try:
        r = httpx.get(f"{cfg.frigate.go2rtc_url}/api/streams", timeout=5)
        streams = r.json() if r.status_code == 200 else {}
        report("go2rtc two-way stream", cfg.twoway_stream in streams,
               f"{cfg.twoway_stream} {'found' if cfg.twoway_stream in streams else 'missing'}")  # fmt: skip
    except (httpx.HTTPError, ValueError) as e:
        report("go2rtc two-way stream", False, str(e))
    t0 = time.monotonic()
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", cfg.frigate.rtsp_url or "",
         "-t", "2", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        capture_output=True, timeout=20,
    )  # fmt: skip
    secs = len(proc.stdout) / 32000
    report("rtsp audio", secs >= 1.5, f"{secs:.1f}s of audio in {time.monotonic() - t0:.1f}s "
           f"{proc.stderr.decode(errors='replace').strip()[:200]}")  # fmt: skip
    report(
        "elevenlabs key",
        secrets.elevenlabs_api_key is not None,
        "set" if secrets.elevenlabs_api_key else "missing",
    )
    report("house rules", cfg.house_rules_path.exists(), str(cfg.house_rules_path))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="doorstep", description=__doc__)
    p.add_argument("-c", "--config", help="config file (default $DOORSTEP_CONFIG or /config/config.yaml)")
    p.add_argument("--url", help="a running Doorstep (default $DOORSTEP_URL or http://127.0.0.1:<port>)")
    p.add_argument("--log-level", default=os.environ.get("DOORSTEP_LOG_LEVEL", "INFO"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the server")
    sub.add_parser("check", help="check Frigate, go2rtc and RTSP audio")
    s = sub.add_parser("say", help="speak a line (needs an open session, e.g. from ring --dry-run)")
    s.add_argument("text")
    s = sub.add_parser("listen", help="listen for one visitor turn")
    s.add_argument("--max-seconds", type=float, default=10)
    s = sub.add_parser("converse", help="say a line, then listen")
    s.add_argument("text")
    s.add_argument("--max-seconds", type=float, default=10)
    s = sub.add_parser("snapshot", help="save a picture of the door")
    s.add_argument("--source", choices=["latest", "ring", "event"], default="latest")
    s.add_argument("--event-id")
    s.add_argument("-o", "--output")
    sub.add_parser("status", help="door_status")
    s = sub.add_parser("visitors", help="door_visitors")
    s.add_argument("--hours", type=float, default=24)
    s = sub.add_parser("end", help="close the open session")
    s.add_argument("--outcome", default="other",
                   choices=["delivery", "visitor", "sales_or_canvassing", "no_answer", "other"])  # fmt: skip
    s.add_argument("--summary", default="Closed from the CLI.")
    s = sub.add_parser("ring", help="simulate a ring")
    s.add_argument("--dry-run", action="store_true",
                   help="open a session and show the trigger payload without waking the agent or speaking")  # fmt: skip
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    secrets = Secrets()
    if args.cmd == "serve":
        return cmd_serve(cfg, secrets, args)
    if args.cmd == "check":
        return cmd_check(cfg, secrets, args)
    if args.cmd == "ring":
        return cmd_ring(cfg, secrets, args)
    return cmd_tool(cfg, secrets, args)


if __name__ == "__main__":
    sys.exit(main())
