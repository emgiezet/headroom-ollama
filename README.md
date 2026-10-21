# headroom-ollama

[![CI](https://github.com/emgiezet/headroom-ollama/actions/workflows/ci.yml/badge.svg)](https://github.com/emgiezet/headroom-ollama/actions/workflows/ci.yml)

[Headroom](https://github.com/chopratejas/headroom) proxy extension that serves a local
[Ollama](https://ollama.com) under its own route, so Ollama traffic (including `:cloud`
models) is compressed by Headroom without repointing the proxy's default OpenAI upstream.

| Request under `/ollama` (or `/p/<project>/ollama`) | Handling |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/responses` | Headroom's OpenAI handlers (compressed), upstream overridden to Ollama via `x-headroom-base-url`, tagged `x-client: ollama` → **Ollama** row in the dashboard's Agent Usage |
| everything else (`/v1/models`, `/api/tags`, `/api/show`, `/api/pull`, …) | streamed straight to Ollama, uncompressed |
| WebSocket | refused (close 1008): Headroom's Responses WebSocket ignores the upstream override and would connect to OpenAI |

Ollama's native API at the proxy root (`/api/*`) is streamed to Ollama as well. Headroom has no
`/api` routes, and Ollama-discovery clients probe `/api/tags` / `/api/show` on the origin,
ignoring the base-URL path.

## Install

Into the same environment as Headroom, then opt in (Headroom only runs extensions that are
explicitly enabled). Headroom ≥ 0.36.5 refuses client-chosen upstreams on loopback/private
addresses unless the operator allowlists them, so the Ollama URL goes into
`HEADROOM_ALLOWED_BASE_URLS`; without it the extension refuses to install rather than let
Headroom send Ollama prompts to the default OpenAI upstream.

```bash
uv tool install 'headroom-ai[all]>=0.36.5,<0.37' --with headroom-ollama   # or --with-editable <path>
HEADROOM_PROXY_EXTENSIONS=ollama HEADROOM_ALLOWED_BASE_URLS=http://127.0.0.1:11434 headroom proxy
```

`HEADROOM_ALLOWED_BASE_URLS` is an exclusive allowlist: once set, Headroom accepts
`x-headroom-base-url` overrides only for the listed hosts/origins, so add any other BYOK
upstreams your clients use.

Persistent deployment:

```bash
headroom install apply --profile <profile> ... \
  --env HEADROOM_PROXY_EXTENSIONS=ollama \
  --env HEADROOM_ALLOWED_BASE_URLS=http://127.0.0.1:11434
```

Check the route:

```bash
headroom ollama status            # probes Ollama directly and via http://127.0.0.1:8787/ollama
```

## Configuration

| Env var (proxy process) | Default |
|---|---|
| `HEADROOM_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `HEADROOM_OLLAMA_PREFIX` | `/ollama` |

## Client setup

Point the client's OpenAI-compatible Ollama base URL at the proxy route, e.g. Oh My Pi
`~/.omp/agent/models.yml` (omp calls `{baseUrl}/responses`, hence the `/v1`):

```yaml
providers:
  ollama:
    baseUrl: http://127.0.0.1:8787/p/workspace/ollama/v1   # or http://127.0.0.1:8787/ollama/v1
    api: openai-responses
    auth: none
    discovery:
      type: ollama
```

## Limits

- Pinned to `headroom-ai` 0.36.x (≥ 0.36.5): the extension depends on Headroom internals
  (`x-headroom-base-url`, `x-client` classification, `upstream_guard`) beyond the stable
  `install(app, config)` hook. `tests/test_headroom_contract.py` checks them against the
  installed Headroom; CI reruns it weekly against the newest allowed patch release.
- Dashboard USD figures for Ollama models come from Headroom's LiteLLM pricing lookup, which
  does not know Ollama models; they are not real costs.

## Development

```bash
uv sync --group dev
uv run pytest
```
