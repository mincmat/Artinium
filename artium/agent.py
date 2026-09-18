from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .context import ContextManager
from .ollama_client import ModelInfo, OllamaClient
from .model_settings import ModelSettings
from .tools import ToolRegistry

SYSTEM_PROMPT = """You are Artinium, a concise local coding agent. Inspect before editing. Use tools only when useful, make minimal changes, and work inside the workspace by default; only act outside it when the user explicitly asks. Keep answers short and report what changed. Use glob to find paths, grep to search with regular expressions, and question only when a missing user choice materially changes the result.

For questions that depend on current public information, use web_search. For time-sensitive or important facts, follow a relevant primary source with fetch_url before answering. State only facts supported by the returned search results or page text; do not invent dates, versions, features, or quotes. If the sources conflict or are insufficient, say so plainly."""
INTERRUPTION_NOTE = "[Previous task was interrupted before completion. Re-check state; do not assume unfinished actions succeeded.]"


@dataclass(slots=True)
class SessionStats:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    last_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class AgentEvent:
    kind: str
    data: dict[str, Any]


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        model: ModelInfo,
        tools: ToolRegistry,
        *,
        max_iterations: int = 12,
        settings: ModelSettings | None = None,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_iterations = max_iterations
        self.settings = settings or ModelSettings()
        self.context = ContextManager(self._context_limit())
        self.history: list[dict[str, Any]] = []
        self.stats = SessionStats()

    def clear(self) -> None:
        self.history.clear()
        self.stats = SessionStats()

    def restore(self, history: list[dict[str, Any]], stats: dict[str, Any]) -> None:
        """Restore a locally stored conversation after validating its basic shape."""
        self.history = [item for item in history if isinstance(item, dict) and isinstance(item.get("role"), str)]
        self.stats = SessionStats(
            input_tokens=int(stats.get("input_tokens") or 0),
            output_tokens=int(stats.get("output_tokens") or 0),
            tool_calls=int(stats.get("tool_calls") or 0),
            last_prompt_tokens=int(stats.get("last_prompt_tokens") or 0),
        )

    def mark_interrupted(self) -> None:
        """Preserve the fact that the previous task was deliberately stopped."""
        if not self.history or self.history[-1].get("content") != INTERRUPTION_NOTE:
            self.history.append({"role": "system", "content": INTERRUPTION_NOTE})

    def set_model(self, model: ModelInfo) -> None:
        self.model = model
        self.context = ContextManager(self._context_limit())

    def set_settings(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.context = ContextManager(self._context_limit())

    def compact_context(self) -> bool:
        """Summarize older history while preserving the latest coherent turn."""
        before = self.history
        compacted = self.context.compact(before, force=True)
        if compacted is before:
            return False
        self.history = compacted
        return True

    @property
    def last_compaction(self) -> dict[str, Any] | None:
        return self.context.last_compaction

    def _auto_compact_context(self) -> bool:
        threshold = self.settings.auto_compact_threshold
        if threshold is None:
            return False
        # Ollama's last prompt count is exact; the local estimate accounts for
        # new messages added since that request.
        used = max(self.stats.last_prompt_tokens, self.context.usage(self.history))
        if used * 100 < self.context.max_tokens * threshold:
            return False
        return self.compact_context()

    def _context_limit(self) -> int:
        requested = self.settings.context_window or min(self.model.context_length, 32_768)
        return min(max(requested, 4_096), self.model.context_length)

    @property
    def context_tokens(self) -> int:
        """Exact input-token count reported by Ollama for the latest request."""
        return self.stats.last_prompt_tokens

    async def run(self, prompt: str, *, images: list[str] | None = None) -> AsyncIterator[AgentEvent]:
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images
        self.history.append(user_message)
        for iteration in range(1, self.max_iterations + 1):
            if self._auto_compact_context():
                yield AgentEvent("compacted", {"automatic": True, **(self.last_compaction or {})})
            self.history = self.context.compact(self.history)
            payload = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]
            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            final_meta: dict[str, Any] = {}
            # Keep the streamed answer in history as it arrives. This lets the
            # UI persist an in-progress response instead of only saving it at
            # the end of a request.
            assistant: dict[str, Any] = {"role": "assistant", "content": ""}
            self.history.append(assistant)
            yield AgentEvent("assistant_start", {"iteration": iteration})

            try:
                async for chunk in self.client.chat_stream(
                    model=self.model.name,
                    messages=payload,
                    # Ollama rejects a tools payload for models that do not
                    # advertise tool calling (for example dolphin3:latest).
                    tools=self.tools.schemas if self.model.supports_tools else [],
                    options=self.settings.options_for(self.model.context_length),
                    think=self.settings.think_value(self.model.supports_reasoning),
                ):
                    message = chunk.get("message") or {}
                    text = message.get("content") or ""
                    if text:
                        content_parts.append(text)
                        assistant["content"] = "".join(content_parts)
                        yield AgentEvent("delta", {"text": text})
                    if message.get("tool_calls"):
                        tool_calls.extend(message["tool_calls"])
                    if chunk.get("done"):
                        final_meta = chunk
            except asyncio.CancelledError:
                if not content_parts and self.history and self.history[-1] is assistant:
                    self.history.pop()
                raise
            except Exception:
                if self.history and self.history[-1] is assistant:
                    self.history.pop()
                raise

            self.stats.input_tokens += int(final_meta.get("prompt_eval_count") or 0)
            self.stats.output_tokens += int(final_meta.get("eval_count") or 0)
            self.stats.last_prompt_tokens = int(final_meta.get("prompt_eval_count") or 0)
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            yield AgentEvent("stats", self._stats_data())

            if not tool_calls:
                yield AgentEvent("done", {"iteration": iteration})
                return

            for index, call in enumerate(tool_calls):
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = self._arguments(function.get("arguments"))
                call_id = str(call.get("id") or f"call_{iteration}_{index}")
                self.stats.tool_calls += 1
                yield AgentEvent("tool_start", {"name": name, "arguments": arguments})
                result = await self.tools.execute(name, arguments)
                self.history.append({
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "content": result.compact_json(),
                })
                yield AgentEvent("tool_end", {
                    "name": name,
                    "arguments": arguments,
                    "result": result.data,
                    "error": result.error,
                })
                yield AgentEvent("stats", self._stats_data())

        warning = f"Task stopped after {self.max_iterations} iterations to prevent a loop."
        self.history.append({"role": "assistant", "content": warning})
        yield AgentEvent("limit", {"text": warning})

    @staticmethod
    def _arguments(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _stats_data(self) -> dict[str, Any]:
        return {
            "input_tokens": self.stats.input_tokens,
            "output_tokens": self.stats.output_tokens,
            "total_tokens": self.stats.total_tokens,
            "tool_calls": self.stats.tool_calls,
            "context_tokens": self.context_tokens,
            "context_limit": self.context.max_tokens,
        }
