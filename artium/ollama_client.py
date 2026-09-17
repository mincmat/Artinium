from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


class OllamaUnavailable(RuntimeError):
    pass


class OllamaError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelInfo:
    name: str
    size: int = 0
    family: str = ""
    parameter_size: str = ""
    capabilities: tuple[str, ...] = ()
    context_length: int = 32768
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def supports_reasoning(self) -> bool:
        return "thinking" in self.capabilities or "reasoning" in self.capabilities

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.capabilities


class OllamaClient:
    """Tiny async wrapper around Ollama's native HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=2.0))
        # Keep a reference only so we can detect an immediate startup failure.
        # Ollama is deliberately not terminated by ``close``: it is a shared
        # local service and may also be used outside Artinium.
        self._server_process: asyncio.subprocess.Process | None = None

    async def close(self) -> None:
        await self._http.aclose()

    async def start_server(self, *, wait_seconds: float = 12.0) -> None:
        """Launch ``ollama serve`` and wait until its HTTP API is ready.

        This is intentionally safe to call only after a connection attempt
        failed. If another process wins the race and starts Ollama first, the
        readiness probe succeeds just the same.
        """
        if shutil.which("ollama") is None:
            raise OllamaUnavailable("Ollama is not installed or is not in PATH")

        try:
            self._server_process = await asyncio.create_subprocess_exec(
                "ollama",
                "serve",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise OllamaUnavailable(f"could not start Ollama: {exc}") from exc

        await self._wait_until_ready(wait_seconds)

    async def _wait_until_ready(self, wait_seconds: float) -> None:
        attempts = max(1, int(wait_seconds / 0.2))
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = await self._http.get("/api/tags", timeout=1.0)
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last_error = exc

            if self._server_process is not None and self._server_process.returncode is not None:
                raise OllamaUnavailable("Ollama stopped immediately after it was started")
            await asyncio.sleep(0.2)

        detail = f": {last_error}" if last_error else ""
        raise OllamaUnavailable(f"Ollama did not become ready after {wait_seconds:g} seconds{detail}")

    async def unload_model(self, model: str) -> None:
        """Release one model from Ollama's memory immediately.

        An empty, non-streaming chat request is sufficient for the chat API,
        while ``keep_alive: 0`` tells Ollama not to retain the loaded weights.
        Keep this deliberately short: application shutdown must not hang if
        Ollama has already stopped.
        """
        try:
            response = await self._http.post(
                "/api/chat",
                json={"model": model, "messages": [], "stream": False, "keep_alive": 0},
                timeout=5.0,
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailable("connection to Ollama was lost") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not unload {model}: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._http.get("/api/tags")
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailable("could not connect to 127.0.0.1:11434") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"/api/tags failed: {exc}") from exc

        result: list[ModelInfo] = []
        for raw in response.json().get("models", []):
            name = raw.get("name") or raw.get("model")
            if not name:
                continue
            details = raw.get("details") or {}
            info = ModelInfo(
                name=name,
                size=int(raw.get("size") or 0),
                family=str(details.get("family") or ""),
                parameter_size=str(details.get("parameter_size") or ""),
                details=details,
            )
            try:
                shown = await self.show_model(name)
                info.capabilities = tuple(shown.get("capabilities") or ())
                info.context_length = self._context_length(shown) or info.context_length
            except OllamaError:
                pass
            result.append(info)
        return result

    async def show_model(self, name: str) -> dict[str, Any]:
        try:
            response = await self._http.post("/api/show", json={"model": name})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not inspect {name}: {exc}") from exc

    async def suggest_title(self, model: str, prompt: str) -> str:
        """Ask the local model for a concise session title without adding chat history."""
        messages = [
            {
                "role": "system",
                "content": "Create a concise 3 to 6 word title for this coding session. Return only the title, without quotes or punctuation at the end. Use the request's language.",
            },
            {"role": "user", "content": prompt[:2_000]},
        ]
        try:
            response = await self._http.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 16},
                },
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailable("connection to Ollama was lost") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"could not generate session title: {exc}") from exc
        return str((response.json().get("message") or {}).get("content") or "").strip()

    @staticmethod
    def _context_length(data: dict[str, Any]) -> int | None:
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None

    @staticmethod
    def choose_model(models: list[ModelInfo]) -> ModelInfo:
        if not models:
            raise ValueError("no models")
        if len(models) == 1:
            return models[0]

        def score(model: ModelInfo) -> tuple[int, int]:
            text = f"{model.name} {model.family} {model.parameter_size}".lower()
            points = 100 if model.supports_tools else 0
            points += 20 if any(f in text for f in ("qwen", "llama", "mistral", "gemma")) else 0
            points -= 80 if any(f in text for f in ("embed", "vision", "rerank")) else 0
            points += 10 if any(size in text for size in ("7b", "8b", "9b", "12b", "14b")) else 0
            return points, model.context_length

        return max(models, key=score)

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        num_ctx: int | None = None,
        options: dict[str, int | float] | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "keep_alive": "10m",
        }
        request_options = dict(options or {})
        if num_ctx and "num_ctx" not in request_options:
            request_options["num_ctx"] = num_ctx
        if request_options:
            body["options"] = request_options
        if think is not None:
            body["think"] = think
        for attempt in range(2):
            received_content = False
            try:
                async with self._http.stream("POST", "/api/chat", json=body) as response:
                    if response.is_error:
                        # Consume the diagnostic while the response stream is
                        # still open. Reading it in an outer exception handler
                        # raises httpx.StreamClosed and hides Ollama's real error.
                        detail = (await response.aread()).decode(errors="replace")[:500]
                        raise OllamaError(f"Ollama returned {response.status_code}: {detail}")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            received_content = True
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise OllamaError("Ollama returned an invalid JSON line") from exc
                return
            except httpx.ReadTimeout as exc:
                seconds = int(self.timeout_seconds)
                duration = f"{seconds // 60} minutes" if seconds >= 60 and seconds % 60 == 0 else f"{seconds} seconds"
                raise OllamaError(
                    f"response timed out after {duration}. Ollama was still processing a large context; "
                    "reduce the context window or choose a faster/smaller model."
                ) from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # Retrying before any streamed content is safe; once output
                # has begun, a retry could duplicate a partially completed turn.
                if attempt == 0 and not received_content:
                    await asyncio.sleep(0.4)
                    continue
                raise OllamaUnavailable("connection to Ollama was lost") from exc
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed: {exc}") from exc
