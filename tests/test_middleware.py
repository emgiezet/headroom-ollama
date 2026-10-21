"""Routing contract of OllamaRouteMiddleware, with a fake Headroom app and a mocked Ollama."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from headroom_ollama.proxy import OllamaRouteMiddleware, install

UPSTREAM = "http://ollama.test:11434"


async def headroom_echo(request: Request) -> JSONResponse:
    """Stands in for Headroom: reports what reached it."""
    return JSONResponse(
        {
            "reached": "headroom",
            "path": request.url.path,
            "base_url": request.headers.get("x-headroom-base-url"),
            "client": request.headers.get("x-client"),
        }
    )


async def headroom_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text(websocket.url.path)
    await websocket.close()


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


def streamed(status: int, parts: list[bytes], headers: dict[str, str] | None = None) -> httpx.Response:
    """A response with an unread body stream, as a real network transport returns."""
    return httpx.Response(status, headers=headers, content=_chunks(parts))


def streamed_json(status: int, payload: object, headers: dict[str, str] | None = None) -> httpx.Response:
    return streamed(status, [json.dumps(payload).encode()], {"content-type": "application/json", **(headers or {})})


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail = False
        self.respond = lambda: streamed_json(200, {"models": []})

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("connection refused", request=request)
        await request.aread()
        self.requests.append(request)
        return self.respond()


@pytest.fixture
def ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def client(ollama: FakeOllama) -> TestClient:
    app = Starlette(
        routes=[
            Route("/{path:path}", headroom_echo, methods=["GET", "POST"]),
            WebSocketRoute("/{path:path}", headroom_ws),
        ]
    )
    app.add_middleware(
        OllamaRouteMiddleware,
        prefix="/ollama",
        upstream=UPSTREAM,
        transport=httpx.MockTransport(ollama.handler),
    )
    return TestClient(app)


@pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/chat/completions"])
def test_compressible_endpoints_go_to_headroom_with_ollama_upstream_and_client_tag(client, ollama, endpoint):
    body = client.post(f"/ollama{endpoint}", json={"model": "qwen3:8b"}).json()

    assert body == {"reached": "headroom", "path": endpoint, "base_url": UPSTREAM, "client": "ollama"}
    assert ollama.requests == []


def test_project_prefix_is_kept_for_headroom_attribution(client):
    body = client.post("/p/workspace/ollama/v1/responses", json={}).json()

    assert body["path"] == "/p/workspace/v1/responses"
    assert body["base_url"] == UPSTREAM


def test_caller_cannot_redirect_ollama_route_to_another_upstream(client):
    body = client.post(
        "/ollama/v1/responses", json={}, headers={"x-headroom-base-url": "https://api.openai.com"}
    ).json()

    assert body["base_url"] == UPSTREAM


def test_caller_supplied_client_tag_wins(client):
    body = client.post("/ollama/v1/responses", json={}, headers={"x-client": "aider"}).json()

    assert body["client"] == "aider"


def test_metadata_endpoints_bypass_headroom_and_hit_ollama(client, ollama):
    ollama.respond = lambda: streamed_json(404, {"error": "nope"}, {"x-upstream": "1"})

    response = client.get("/p/workspace/ollama/v1/models?limit=2")

    assert response.status_code == 404
    assert response.json() == {"error": "nope"}
    assert response.headers["x-upstream"] == "1"
    [sent] = ollama.requests
    assert (sent.method, str(sent.url)) == ("GET", f"{UPSTREAM}/v1/models?limit=2")
    assert sent.headers["host"] == "ollama.test:11434"


def test_bare_prefix_maps_to_ollama_root(client, ollama):
    client.get("/ollama")

    assert str(ollama.requests[0].url) == f"{UPSTREAM}/"


def test_request_body_is_forwarded_unchanged(client, ollama):
    payload = {"model": "kimi-k3:cloud", "verbose": False}

    client.post("/ollama/api/show", json=payload)

    [sent] = ollama.requests
    assert sent.method == "POST"
    assert json.loads(sent.content) == payload


def test_streamed_ndjson_arrives_complete_and_in_order(client, ollama):
    chunks = [json.dumps({"status": f"step {i}"}).encode() + b"\n" for i in range(50)]
    ollama.respond = lambda: streamed(200, chunks, {"content-type": "application/x-ndjson"})

    response = client.post("/ollama/api/pull", json={"model": "qwen3:8b"})

    assert response.content == b"".join(chunks)


def test_unreachable_ollama_returns_502_json(client, ollama):
    ollama.fail = True

    response = client.get("/ollama/api/tags")

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


@pytest.mark.parametrize(("method", "path"), [("GET", "/api/tags"), ("POST", "/api/show")])
def test_root_native_api_goes_to_ollama(client, ollama, method, path):
    """Ollama-discovery clients probe the origin and drop the base-URL path."""
    response = client.request(method, path, json={"model": "x"} if method == "POST" else None)

    assert response.json() == {"models": []}
    assert [(r.method, str(r.url)) for r in ollama.requests] == [(method, f"{UPSTREAM}{path}")]


@pytest.mark.parametrize(
    "path",
    ["/v1/responses", "/ollamax/v1/models", "/p/workspace/v1/messages", "/apix/tags"],
)
def test_paths_outside_prefix_are_untouched(client, ollama, path):
    body = client.post(path, json={}).json()

    assert body == {"reached": "headroom", "path": path, "base_url": None, "client": None}
    assert ollama.requests == []


def test_websocket_under_prefix_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ollama/v1/responses"):
            pass

    assert excinfo.value.code == 1008


def test_websocket_outside_prefix_passes_through(client):
    with client.websocket_connect("/v1/responses") as ws:
        assert ws.receive_text() == "/v1/responses"


@pytest.mark.parametrize(
    ("env", "value"),
    [("HEADROOM_OLLAMA_PREFIX", "/"), ("HEADROOM_OLLAMA_URL", "127.0.0.1:11434")],
)
def test_install_rejects_invalid_configuration(monkeypatch, env, value):
    monkeypatch.setenv(env, value)

    with pytest.raises(ValueError, match=env):
        install(Starlette(), config=None)


def _installed(app: Starlette) -> bool:
    return any(m.cls is OllamaRouteMiddleware for m in app.user_middleware)


def test_install_refuses_loopback_upstream_headroom_would_reject(monkeypatch):
    """Otherwise Headroom drops the override and sends Ollama prompts to OpenAI."""
    monkeypatch.delenv("HEADROOM_ALLOWED_BASE_URLS", raising=False)
    monkeypatch.setenv("HEADROOM_OLLAMA_URL", "http://127.0.0.1:11434")
    app = Starlette()

    with pytest.raises(RuntimeError, match="HEADROOM_ALLOWED_BASE_URLS=http://127.0.0.1:11434"):
        install(app, config=None)
    assert not _installed(app)


def test_install_accepts_allowlisted_loopback_upstream(monkeypatch):
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "http://127.0.0.1:11434")
    monkeypatch.setenv("HEADROOM_OLLAMA_URL", "http://127.0.0.1:11434")
    app = Starlette()

    install(app, config=None)

    assert _installed(app)
