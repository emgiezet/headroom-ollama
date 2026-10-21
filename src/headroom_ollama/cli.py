"""``headroom ollama ...`` subcommands (``headroom.cli_extension`` entry point)."""

from __future__ import annotations

import click
import httpx

from headroom_ollama.proxy import DEFAULT_PREFIX, DEFAULT_UPSTREAM


@click.group("ollama")
def ollama() -> None:
    """Ollama routing through the Headroom proxy (headroom-ollama)."""


@ollama.command("status")
@click.option("--proxy", "proxy_url", default="http://127.0.0.1:8787", show_default=True, help="Headroom proxy base URL.")
@click.option("--prefix", default=DEFAULT_PREFIX, show_default=True, help="Route prefix the extension serves.")
@click.option("--upstream", default=DEFAULT_UPSTREAM, show_default=True, help="Ollama base URL.")
def status(proxy_url: str, prefix: str, upstream: str) -> None:
    """Check Ollama directly and through the proxy route."""
    direct = _probe(f"{upstream.rstrip('/')}/api/version")
    routed = _probe(f"{proxy_url.rstrip('/')}/{prefix.strip('/')}/api/version")
    click.echo(f"ollama  {upstream}: {direct}")
    click.echo(f"proxy   {proxy_url}/{prefix.strip('/')}: {routed}")
    if routed.startswith("ok"):
        return
    if direct.startswith("ok"):
        raise click.ClickException(
            "route not served: install headroom-ollama into the proxy's environment "
            "and set HEADROOM_PROXY_EXTENSIONS=ollama on the proxy"
        )
    raise click.ClickException("Ollama itself is unreachable")


def _probe(url: str) -> str:
    try:
        response = httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        return f"unreachable ({exc})"
    if response.status_code != 200:
        return f"HTTP {response.status_code}"
    try:
        version = response.json().get("version")
    except ValueError:
        return "HTTP 200 but not an Ollama response"
    return f"ok (ollama {version})" if version else "HTTP 200 but not an Ollama response"


def register(main: click.Group) -> None:
    main.add_command(ollama)
