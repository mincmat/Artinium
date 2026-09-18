from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx

from artium.agent import Agent
from artium.context import ContextManager
from artium.model_settings import ModelSettings
from artium.ollama_client import ModelInfo, OllamaClient, OllamaError
from artium.tools import ToolRegistry
from artium.updater import check_for_update
from artium.workspace import Workspace


class FakeClient:
    def __init__(self):
        self.calls = 0

    async def chat_stream(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            yield {"message": {"content": "Voy a mirar. "}, "done": False}
            yield {"message": {"content": "", "tool_calls": [{"id": "1", "function": {
                "name": "write_file", "arguments": {"path": "made.txt", "content": "ok"}
            }}]}, "done": True, "prompt_eval_count": 25, "eval_count": 8}
        else:
            yield {"message": {"content": "Listo."}, "done": True, "prompt_eval_count": 40, "eval_count": 3}


class SlowClient:
    async def chat_stream(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"message": {"content": "inicio"}, "done": False}
        await asyncio.sleep(30)


class CapturingClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    async def chat_stream(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.request = kwargs
        yield {"message": {"content": "done"}, "done": True}


class ThinkingClient:
    async def chat_stream(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"message": {"thinking": "I should inspect the file. "}, "done": False}
        yield {"message": {"thinking": "Then make the smallest change."}, "done": False}
        yield {"message": {"content": "Done."}, "done": True}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_check_uses_artiniums_github_source(self) -> None:
        original_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/commits/main"):
                return httpx.Response(200, json={"sha": "new-revision"})
            if request.url.path.endswith("/pyproject.toml"):
                return httpx.Response(200, text='[project]\nversion = "0.2.1"\n')
            return httpx.Response(404)

        class MockClient:
            def __init__(self, **_: Any) -> None:
                self.client = original_client(transport=httpx.MockTransport(handler), base_url="https://test")

            async def __aenter__(self) -> httpx.AsyncClient:
                return self.client

            async def __aexit__(self, *_: Any) -> None:
                await self.client.aclose()

        with patch("artium.updater.httpx.AsyncClient", MockClient), patch(
            "artium.updater.installed_commit", return_value="old-revision"
        ):
            info = await check_for_update("0.2.1")
        self.assertIsNotNone(info)
        self.assertTrue(info.available)  # type: ignore[union-attr]
        self.assertEqual(info.latest_commit, "new-revision")  # type: ignore[union-attr]

    async def test_ollama_client_explains_generation_timeouts(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow response", request=request)

        client = OllamaClient(timeout=120)
        await client.close()
        client._http = httpx.AsyncClient(
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaisesRegex(OllamaError, "timed out after 2 minutes.*reduce the context window"):
                async for _ in client.chat_stream(model="test", messages=[], tools=[]):
                    pass
        finally:
            await client.close()

    async def test_ollama_client_starts_a_server_when_requested(self) -> None:
        client = OllamaClient()
        client._wait_until_ready = AsyncMock()  # type: ignore[method-assign]
        try:
            with patch("artium.ollama_client.shutil.which", return_value="/usr/bin/ollama"), patch(
                "artium.ollama_client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=object())
            ) as start:
                await client.start_server(wait_seconds=3)
        finally:
            await client.close()

        start.assert_awaited_once_with(
            "ollama", "serve", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        client._wait_until_ready.assert_awaited_once_with(3)  # type: ignore[attr-defined]

    async def test_ollama_client_unloads_model_with_zero_keep_alive(self) -> None:
        requests: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True})

        client = OllamaClient()
        await client.close()
        client._http = httpx.AsyncClient(
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.unload_model("qwen-test:8b")
        finally:
            await client.close()

        self.assertEqual(requests, [{"model": "qwen-test:8b", "messages": [], "stream": False, "keep_alive": 0}])

    async def test_agent_streams_calls_tool_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))
            client = FakeClient()
            agent = Agent(client, ModelInfo("test:8b", context_length=8192), ToolRegistry(workspace))  # type: ignore[arg-type]
            events = [event async for event in agent.run("crea el archivo")]
            self.assertEqual(client.calls, 2)
            self.assertEqual((Path(directory) / "made.txt").read_text(), "ok")
            self.assertEqual([e.kind for e in events].count("tool_start"), 1)
            self.assertEqual(agent.stats.tool_calls, 1)
            self.assertEqual(agent.stats.input_tokens, 65)
            self.assertEqual(agent.context_tokens, 40)

    async def test_generation_is_cancellable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(SlowClient(), ModelInfo("test"), ToolRegistry(Workspace(Path(directory))))  # type: ignore[arg-type]

            async def consume() -> None:
                async for _ in agent.run("wait"):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_model_settings_are_sent_to_ollama_and_limit_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = CapturingClient()
            settings = ModelSettings(
                reasoning="enabled",
                temperature=0.2,
                max_output_tokens=1024,
                context_window=16384,
            )
            model = ModelInfo("reasoning-test", capabilities=("completion", "thinking"), context_length=32768)
            agent = Agent(client, model, ToolRegistry(Workspace(Path(directory))), settings=settings)  # type: ignore[arg-type]
            _ = [event async for event in agent.run("hello")]
            self.assertEqual(agent.context.max_tokens, 16384)
            self.assertEqual(client.request["options"], {"num_ctx": 16384, "temperature": 0.2, "num_predict": 1024})  # type: ignore[index]
            self.assertTrue(client.request["think"])  # type: ignore[index]

    async def test_non_tool_model_is_not_sent_a_tools_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = CapturingClient()
            agent = Agent(
                client, ModelInfo("plain-chat", capabilities=("completion",)),
                ToolRegistry(Workspace(Path(directory))),
            )  # type: ignore[arg-type]
            _ = [event async for event in agent.run("hello")]
            self.assertEqual(client.request["tools"], [])  # type: ignore[index]

    async def test_agent_streams_reasoning_separately_from_the_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                ThinkingClient(), ModelInfo("reasoning-test", capabilities=("completion", "thinking")),
                ToolRegistry(Workspace(Path(directory))),
            )  # type: ignore[arg-type]
            events = [event async for event in agent.run("hello")]
            thinking = [event.data["text"] for event in events if event.kind == "thinking_delta"]
            self.assertEqual(thinking, ["I should inspect the file. ", "Then make the smallest change."])
            self.assertEqual([event.data["text"] for event in events if event.kind == "delta"], ["Done."])

    async def test_images_are_sent_in_the_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = CapturingClient()
            model = ModelInfo("vision-test", capabilities=("completion", "vision"))
            agent = Agent(client, model, ToolRegistry(Workspace(Path(directory))))  # type: ignore[arg-type]
            _ = [event async for event in agent.run("inspect this", images=["aW1hZ2U="])]
            self.assertEqual(client.request["messages"][1]["images"], ["aW1hZ2U="])  # type: ignore[index]

    def test_context_compacts_old_tool_results(self) -> None:
        manager = ContextManager(max_tokens=4096, reserve_tokens=1000)
        messages = [{"role": "user", "content": "x" * 2000}]
        messages += [{"role": "tool", "content": "y" * 3000} for _ in range(4)]
        messages += [{"role": "assistant", "content": "final"}]
        compacted = manager.compact(messages)
        self.assertLess(len(compacted), len(messages))
        self.assertEqual(compacted[-1]["content"], "final")

    async def test_auto_compact_runs_before_a_context_window_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = ModelSettings(context_window=4096, auto_compact_threshold=70)
            agent = Agent(
                CapturingClient(), ModelInfo("test", context_length=4096),
                ToolRegistry(Workspace(Path(directory))), settings=settings,
            )  # type: ignore[arg-type]
            agent.history = [
                {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 2200}
                for index in range(6)
            ]
            events = [event async for event in agent.run("continue")]
            self.assertIn("compacted", [event.kind for event in events])
            self.assertLess(len(agent.history), 8)

    def test_interruption_is_preserved_for_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(FakeClient(), ModelInfo("test"), ToolRegistry(Workspace(Path(directory))))  # type: ignore[arg-type]
            agent.history.append({"role": "user", "content": "do work"})
            agent.mark_interrupted()
            self.assertIn("interrupted before completion", agent.history[-1]["content"])


class ModelSelectionTests(unittest.TestCase):
    def test_prefers_tool_capable_coding_size(self) -> None:
        models = [
            ModelInfo("embed:latest", capabilities=("embedding",)),
            ModelInfo("qwen3:8b", capabilities=("completion", "tools")),
        ]
        self.assertEqual(OllamaClient.choose_model(models).name, "qwen3:8b")
