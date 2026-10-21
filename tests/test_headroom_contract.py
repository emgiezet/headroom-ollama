"""Contract with the installed Headroom: the internals this extension relies on.

Runs the real Headroom app with the extension enabled against a fake Ollama on a
local port. Fails when a Headroom upgrade stops honoring ``x-headroom-base-url``
on the compressed handlers or stops labelling ``x-client`` traffic.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

AUTH = {"authorization": "Bearer test"}


class FakeOllamaServer:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/v1/responses", self.responses, methods=["POST"]),
            ]
        )

    async def chat(self, request: Request) -> JSONResponse:
        body = await request.json()
        self.seen.append((request.url.path, body["model"]))
        return JSONResponse(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        )

    async def responses(self, request: Request) -> JSONResponse:
        body = await request.json()
        self.seen.append((request.url.path, body["model"]))
        return JSONResponse(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            }
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def fake_ollama() -> Iterator[tuple[FakeOllamaServer, str]]:
    fake = FakeOllamaServer()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(fake.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("fake Ollama did not start")
        time.sleep(0.05)
    yield fake, f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def headroom(fake_ollama, tmp_path_factory) -> Iterator[TestClient]:
    _, upstream = fake_ollama
    state = tmp_path_factory.mktemp("headroom")
    with pytest.MonkeyPatch.context() as mp:
        # Keep the developer's real ~/.headroom stats out of the test run.
        mp.setenv("HEADROOM_WORKSPACE_DIR", str(state))
        mp.setenv("HEADROOM_CONFIG_DIR", str(state))
        mp.setenv("HEADROOM_TELEMETRY", "off")
        mp.setenv("HEADROOM_OLLAMA_URL", upstream)

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        # Default OpenAI upstream deliberately unreachable: nothing under the
        # prefix may depend on it. One attempt keeps the negative test fast.
        config = ProxyConfig(
            openai_api_url="http://127.0.0.1:9",
            proxy_extensions=["ollama"],
            retry_max_attempts=1,
        )
        with TestClient(create_app(config)) as client:
            yield client


def test_chat_completions_reach_ollama_through_headroom(headroom, fake_ollama):
    fake, _ = fake_ollama
    fake.seen.clear()

    response = headroom.post(
        "/ollama/v1/chat/completions",
        headers=AUTH,
        json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert fake.seen == [("/v1/chat/completions", "qwen3:8b")]


def test_responses_reach_ollama_through_headroom_under_project_prefix(headroom, fake_ollama):
    fake, _ = fake_ollama
    fake.seen.clear()

    response = headroom.post(
        "/p/workspace/ollama/v1/responses",
        headers=AUTH,
        json={"model": "kimi-k3:cloud", "input": "hi"},
    )

    assert response.status_code == 200, response.text
    assert fake.seen == [("/v1/responses", "kimi-k3:cloud")]


def test_ollama_traffic_gets_its_own_agent_usage_row(headroom):
    headroom.post(
        "/ollama/v1/chat/completions",
        headers=AUTH,
        json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}]},
    )

    agents = headroom.get("/stats").json()["agent_usage"]["agents"]

    assert {"label": "Ollama", "source": "client"} in [
        {"label": a["label"], "source": a["source"]} for a in agents
    ]


def test_unprefixed_openai_traffic_keeps_the_default_upstream(headroom, fake_ollama):
    fake, _ = fake_ollama
    fake.seen.clear()

    headroom.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert fake.seen == []
