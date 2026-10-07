"""The HTTP app: /mcp, /ring, /clips/{id}.mp3 and /healthz on one port."""

from __future__ import annotations

import contextlib
import hmac
import logging

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Config, Secrets
from .core import Doorstep
from .relay import Relay
from .server import build_mcp

log = logging.getLogger(__name__)


def _bearer_ok(header: str | None, token: str) -> bool:
    if not header or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode(), token.encode())


class RequireBearer:
    """Protects every path under a prefix with one bearer token."""

    def __init__(self, app: ASGIApp, prefix: str, token: str):
        self.app = app
        self.prefix = prefix
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(self.prefix):
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if not _bearer_ok(headers.get("authorization"), self.token):
                resp = JSONResponse({"error": "unauthorised"}, 401, headers={"WWW-Authenticate": "Bearer"})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(
    cfg: Config,
    secrets: Secrets,
    http: httpx.AsyncClient | None = None,
    door: Doorstep | None = None,
    run_background: bool = True,
) -> Starlette:
    mcp_token = secrets.doorstep_mcp_token
    ring_token = secrets.doorstep_ring_token
    if mcp_token is None or ring_token is None:
        raise SystemExit("DOORSTEP_MCP_TOKEN and DOORSTEP_RING_TOKEN must both be set")

    own_http = http is None
    http = http or httpx.AsyncClient(timeout=10)
    door = door or Doorstep(cfg, secrets, http)
    relay = Relay(door)
    mcp = build_mcp(door)
    mcp_app = mcp.streamable_http_app(host="0.0.0.0")

    async def ring(request: Request) -> Response:
        if not _bearer_ok(request.headers.get("authorization"), ring_token.get_secret_value()):
            return JSONResponse({"error": "unauthorised"}, 401, headers={"WWW-Authenticate": "Bearer"})
        dry_run = request.query_params.get("dry_run", "").lower() in ("1", "true", "yes")
        source = request.query_params.get("source") or "http"
        result = await relay.ring(source, dry_run=dry_run)
        return JSONResponse(result, 202 if result["status"] == "opened" else 200)

    async def clip(request: Request) -> Response:
        data = door.clips.get(request.path_params["clip_id"])
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    async def healthz(request: Request) -> Response:
        s = door.sessions.open_session()
        return JSONResponse(
            {
                "ok": True,
                "session": s.id if s else None,
                "audio_reader": door.reader.running,
                "speaking": door.speaker.speaking,
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            if run_background:
                await door.start()
                door.spawn(relay.run_mqtt())
            try:
                yield
            finally:
                await door.stop()
                if own_http:
                    await http.aclose()

    app = Starlette(
        routes=[
            Route("/ring", ring, methods=["POST"]),
            Route("/clips/{clip_id:str}.mp3", clip, methods=["GET", "HEAD"]),
            Route("/healthz", healthz, methods=["GET"]),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
    )
    app.state.door = door
    app.state.relay = relay
    app.state.mcp = mcp
    app.add_middleware(RequireBearer, prefix="/mcp", token=mcp_token.get_secret_value())
    return app
