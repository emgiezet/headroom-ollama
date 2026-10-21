"""Headroom proxy extension: serve a local Ollama under its own path prefix.

Requests under ``/ollama`` (optionally behind Headroom's ``/p/<project>``
prefix) are handled in two ways:

* ``/v1/chat/completions`` and ``/v1/responses`` are handed to Headroom's own
  OpenAI handlers, so they get compressed. The prefix is stripped and the
  request carries ``x-headroom-base-url`` (Headroom's per-request OpenAI
  upstream override) pointing at Ollama, plus ``x-client: ollama`` so the
  dashboard's Agent Usage panel shows an "Ollama" row.
* Everything else (``/v1/models``, ``/api/tags``, ``/api/show``, ``/api/pull``,
  ...) is streamed straight to Ollama. Headroom's generic routes ignore
  ``x-headroom-base-url`` for these and would forward them to the default
  OpenAI upstream instead.

The Ollama native API at the proxy root (``/api/*``) is streamed to Ollama
too: Headroom defines no ``/api`` routes (its catch-all would send them to the
OpenAI upstream), and Ollama-discovery clients such as Oh My Pi probe
``/api/tags`` and ``/api/show`` on the origin, ignoring any base-URL path.

Headroom's default OpenAI upstream (``OPENAI_TARGET_API_URL``) is untouched,
so other OpenAI-format clients of the same proxy keep working.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from typing import Any

import httpx

log = logging.getLogger(__name__)

PREFIX_ENV = "HEADROOM_OLLAMA_PREFIX"
UPSTREAM_ENV = "HEADROOM_OLLAMA_URL"
DEFAULT_PREFIX = "/ollama"
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
CLIENT_NAME = "ollama"

# Headroom handlers that honor ``x-headroom-base-url`` and compress the body.
COMPRESSED_PATHS = frozenset({"/v1/chat/completions", "/v1/responses"})

_BASE_URL_HEADER = b"x-headroom-base-url"
_CLIENT_HEADER = b"x-client"
_PROJECT_PREFIX = "/p/"
_NATIVE_API = "/api"
# RFC 9110 §7.6.1 connection-specific headers, plus ones httpx recomputes.
_HOP_BY_HOP = frozenset(
    {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
        b"host",
    }
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def normalize_prefix(raw: str) -> str:
    prefix = "/" + raw.strip().strip("/")
    if prefix == "/":
        raise ValueError(f"{PREFIX_ENV} must name a path segment, got {raw!r}")
    return prefix


def normalize_upstream(raw: str) -> str:
    upstream = raw.strip().rstrip("/")
    if not upstream.startswith(("http://", "https://")):
        raise ValueError(f"{UPSTREAM_ENV} must be an http(s) URL, got {raw!r}")
    return upstream


def split_route(path: str, prefix: str) -> tuple[str, str] | None:
    """Split ``[/p/<project>]<prefix>/<rest>`` into ``("/p/<project>" | "", "/<rest>")``.

    Returns ``None`` when ``path`` is not under ``prefix``.
    """
    project = ""
    if path.startswith(_PROJECT_PREFIX):
        name, sep, rest = path[len(_PROJECT_PREFIX) :].partition("/")
        if not name or not sep:
            return None
        project = f"{_PROJECT_PREFIX}{name}"
        path = "/" + rest
    if path == prefix:
        return project, "/"
    if path.startswith(prefix + "/"):
        return project, path[len(prefix) :]
    return None


class OllamaRouteMiddleware:
    """Pure ASGI middleware; see module docstring for the routing contract."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        prefix: str,
        upstream: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.app = app
        self.prefix = prefix
        self.upstream = upstream
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        route = split_route(path, self.prefix)
        if route is None:
            if scope["type"] == "http" and (path == _NATIVE_API or path.startswith(_NATIVE_API + "/")):
                await self._forward(scope, receive, send, path)
                return
            await self.app(scope, receive, send)
            return
        project, rest = route

        if scope["type"] == "websocket":
            # Headroom's Responses WebSocket ignores x-headroom-base-url and would
            # connect to the default OpenAI upstream; refuse instead of leaking.
            await send({"type": "websocket.close", "code": 1008, "reason": "ollama: use HTTP"})
            return

        if rest in COMPRESSED_PATHS:
            await self.app(self._to_headroom(scope, project + rest), receive, send)
            return
        await self._forward(scope, receive, send, rest)

    def _to_headroom(self, scope: Scope, path: str) -> Scope:
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != _BASE_URL_HEADER]
        headers.append((_BASE_URL_HEADER, self.upstream.encode("latin-1")))
        if not any(k.lower() == _CLIENT_HEADER for k, _ in headers):
            headers.append((_CLIENT_HEADER, CLIENT_NAME.encode("latin-1")))
        rewritten = dict(scope)
        rewritten["path"] = path
        rewritten["raw_path"] = path.encode("utf-8")
        rewritten["headers"] = headers
        return rewritten

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Pulls and long generations stream for minutes: only bound connect.
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(None, connect=5.0),
            )
        return self._client

    async def _forward(self, scope: Scope, receive: Receive, send: Send, path: str) -> None:
        query = scope.get("query_string", b"").decode("latin-1")
        url = self.upstream + path + (f"?{query}" if query else "")
        headers = [(k, v) for k, v in scope["headers"] if k.lower() not in _HOP_BY_HOP]
        has_body = any(k.lower() in (b"content-length", b"transfer-encoding") for k, _ in scope["headers"])
        request = self._http().build_request(
            scope["method"],
            url,
            headers=headers,
            content=_request_body(receive) if has_body else None,
        )
        try:
            response = await self._http().send(request, stream=True)
        except httpx.HTTPError as exc:
            log.warning("ollama upstream %s unreachable: %s", self.upstream, exc)
            await _send_json(send, 502, {"error": {"message": f"Ollama upstream unreachable: {exc}", "type": "upstream_error"}})
            return
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": [(k, v) for k, v in response.headers.raw if k.lower() not in _HOP_BY_HOP],
                }
            )
            async for chunk in response.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await response.aclose()


async def _request_body(receive: Receive) -> AsyncIterator[bytes]:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        chunk = message.get("body", b"")
        if chunk:
            yield chunk
        if not message.get("more_body", False):
            return


async def _send_json(send: Send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def install(app: Any, config: Any) -> None:
    """``headroom.proxy_extension`` entry point. Performs no network I/O."""
    prefix = normalize_prefix(os.environ.get(PREFIX_ENV, DEFAULT_PREFIX))
    upstream = normalize_upstream(os.environ.get(UPSTREAM_ENV, DEFAULT_UPSTREAM))
    app.add_middleware(OllamaRouteMiddleware, prefix=prefix, upstream=upstream)
    log.info("headroom-ollama: %s/* -> %s", prefix, upstream)
