from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

MAX_PAGE_CHARS = 20_000
MAX_PAGE_BYTES = 2_000_000
USER_AGENT = "Artinium/0.2 local coding agent"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._ignored = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "div", "article", "section", "li", "h1", "h2", "h3", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._ignored:
            self._ignored -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        text = data.strip()
        if text:
            self.parts.append(text)
            if self._in_title:
                self.title += text

    def text(self) -> str:
        value = " ".join(self.parts)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\s*\n\s*", "\n", value)
        return value.strip()


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            href = values.get("href") or ""
            parsed = urlparse(href)
            target = parse_qs(parsed.query).get("uddg", [href])[0]
            self._current = {"title": "", "url": unquote(target), "snippet": ""}
            self.results.append(self._current)
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div"}:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] += html.unescape(data).strip() + " "


async def web_search(query: str, limit: int = 5) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    limit = max(1, min(int(limit), 10))
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        response = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
        response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    if not parser.results:
        raise ValueError("search returned no parseable results (the provider may have changed its markup)")
    results = []
    for result in parser.results[:limit]:
        results.append({key: value.strip() for key, value in result.items()})
    return {
        "query": query,
        "results": results,
        "provider": "DuckDuckGo",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


async def _validate_public_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("url has no hostname")
    if host.lower() == "localhost":
        raise ValueError("local addresses are not allowed")
    try:
        addresses = await asyncio.to_thread(socket.getaddrinfo, host, parsed.port or 443)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve host: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("private or local network addresses are not allowed")


async def fetch_url(url: str) -> dict[str, Any]:
    current = url.strip()
    async with httpx.AsyncClient(timeout=25, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(6):
            await _validate_public_url(current)
            # Stream with a byte budget: a hostile 100MB page must never be
            # fully loaded into RAM before truncation.
            async with client.stream("GET", current) as streamed:
                if streamed.is_redirect:
                    location = streamed.headers.get("location")
                    if not location:
                        raise ValueError("redirect has no location")
                    current = urljoin(current, location)
                    continue
                try:
                    declared = int(streamed.headers.get("content-length") or 0)
                except ValueError:
                    declared = 0
                if declared > MAX_PAGE_BYTES:
                    raise ValueError(f"page too large ({declared} bytes declared)")
                chunks: list[bytes] = []
                received = 0
                async for chunk in streamed.aiter_bytes(65_536):
                    received += len(chunk)
                    if received > MAX_PAGE_BYTES:
                        raise ValueError(f"page too large (over {MAX_PAGE_BYTES} bytes)")
                    chunks.append(chunk)
                response = httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    content=b"".join(chunks),
                    request=streamed.request,
                )
            response.raise_for_status()
            break
        else:
            raise ValueError("too many redirects")
    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        parser = _TextExtractor()
        parser.feed(response.text)
        text = parser.text()
        title = parser.title.strip()
    elif content_type.startswith("text/") or "json" in content_type:
        text = response.text
        title = ""
    else:
        raise ValueError(f"unsupported content type: {content_type or 'unknown'}")
    truncated = len(text) > MAX_PAGE_CHARS
    return {
        "url": str(response.url),
        "title": title,
        "content": text[:MAX_PAGE_CHARS],
        "truncated": truncated,
        "content_type": content_type,
        "status_code": response.status_code,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verified": True,
    }
