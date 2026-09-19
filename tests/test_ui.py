from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, AsyncIterator

from textual.widgets import Collapsible, Input, ListItem, ListView, Static

from artium.ollama_client import ModelInfo
from artium.model_settings import ModelSettings, ModelSettingsStore
from artium.ui.app import (
    ArtiumApp, CommandMenu, ContextSettingsScreen, DeleteSessionScreen, EditQueueScreen,
    ChangeHistoryScreen, ModelHubScreen, ModelScreen, ModelSettingsScreen, QueueScreen, SessionScreen, StopScreen,
    QuestionScreen, _search_matches,
)
from artium.sessions import SessionRecord
from artium.tools import QuestionRequest
from artium.ui.widgets import ActivityPulse, AssistantMessage, ChatLog, ThinkingBlock, UserMessage, WorkspaceTree


class FakeUIClient:
    def __init__(self) -> None:
        self.round = 0
        self.unloaded_models: list[str] = []

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("qwen-test:8b", capabilities=("completion", "tools"), context_length=8192)]

    @staticmethod
    def choose_model(models: list[ModelInfo]) -> ModelInfo:
        return models[0]

    async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
        self.round += 1
        if self.round == 1:
            yield {"message": {"content": "Reviso. "}, "done": False}
            yield {"message": {"content": "", "tool_calls": [{"function": {
                "name": "read_file", "arguments": {"path": "sample.txt"}
            }}]}, "done": True, "prompt_eval_count": 8, "eval_count": 3}
        else:
            yield {"message": {"content": "Listo."}, "done": True, "prompt_eval_count": 12, "eval_count": 2}

    async def close(self) -> None:
        pass

    async def unload_model(self, model: str) -> None:
        self.unloaded_models.append(model)


class UISmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_question_custom_answer_is_reachable_with_arrow_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 28)) as pilot:
                await pilot.pause(0.15)
                app.push_screen(QuestionScreen(QuestionRequest(
                    question="Pick one", options=["First", "Second"],
                )))
                await pilot.pause(0.05)
                choices = app.screen.query_one("#question-list", ListView)
                answer = app.screen.query_one("#question-answer", Input)
                self.assertIs(app.screen.focused, choices)
                await pilot.press("down", "down")
                self.assertIs(app.screen.focused, answer)
                await pilot.press("up")
                self.assertIs(app.screen.focused, choices)
                self.assertEqual(choices.index, 1)

    def test_turn_duration_format(self) -> None:
        self.assertEqual(ArtiumApp._format_duration(173), "2m 53s")

    def test_tool_badges_are_portable_and_distinct(self) -> None:
        self.assertEqual(ChatLog.tool_badge("read_file"), "▧")
        self.assertEqual(ChatLog.tool_badge("run_command"), "▻")
        self.assertEqual(ChatLog.tool_badge("unknown_tool"), "◇")

    def test_model_list_describes_supported_capabilities(self) -> None:
        model = ModelInfo("capable", parameter_size="8.0B", capabilities=("completion", "tools", "thinking", "vision"))
        self.assertEqual(ModelScreen._capabilities(model), "◆ thinking  ⊞ tools  ◉ vision  ·  8b")

    def test_context_choices_expand_for_long_context_models(self) -> None:
        screen = ContextSettingsScreen(
            ModelSettings(),
            ModelInfo("long-context", capabilities=("completion",), context_length=1_048_576),
            lambda _: None,
            lambda: None,
        )
        self.assertEqual(screen._context_options()[-1], 1_048_576)
        self.assertIn(131_072, screen._context_options())
        self.assertIn(524_288, screen._context_options())

    def test_file_change_details_show_written_source_and_edit_diff(self) -> None:
        write_detail = ChatLog.tool_detail(
            "write_file",
            {"path": "hello.py", "content": "print('hello')\nprint('bye')\n"},
            {"path": "hello.py", "chars": 29, "created": True},
            False,
        )
        self.assertIn("+ print('hello')", write_detail)
        self.assertIn("+ print('bye')", write_detail)

        edit_detail = ChatLog.tool_detail(
            "edit_file",
            {"path": "hello.py"},
            {"diff": "--- a/hello.py\n+++ b/hello.py\n@@ -1 +1 @@\n-old\n+new"},
            False,
        )
        self.assertIn("- old", edit_detail)
        self.assertIn("+ new", edit_detail)
        self.assertNotIn("--- a/hello.py", edit_detail)

    async def test_ui_initializes_streams_and_renders_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "sample.txt").write_text("hello")
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                self.assertFalse(prompt.disabled)
                prompt.value = "lee sample.txt"
                await pilot.press("enter")
                await pilot.pause(0.5)
                self.assertIsNone(app._generation_task)
                self.assertEqual(app.agent.stats.tool_calls, 1)  # type: ignore[union-attr]
                self.assertEqual(len(app.query(Collapsible)), 1)
                self.assertEqual(app.active_session.title, "lee sample.txt")  # type: ignore[union-attr]
                response = app.query_one(AssistantMessage)
                self.assertIsInstance(response, AssistantMessage)
                self.assertTrue(response.read_only)
                self.assertTrue(any("Listo." in item.text for item in app.query(AssistantMessage)))

    async def test_startup_recovers_session_draft_queue_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = ArtiumApp(root)
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.15)
                app.agent.history.append({"role": "user", "content": "remember this"})  # type: ignore[union-attr]
                app._queued_prompts = ["run this next"]
                app.active_session.draft = "unfinished draft"  # type: ignore[union-attr]
                app._save_active_session()

            recovered = ArtiumApp(root)
            await recovered.client.close()
            recovered.client = FakeUIClient()  # type: ignore[assignment]
            async with recovered.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.15)
                self.assertEqual(recovered.agent.history[0]["content"], "remember this")  # type: ignore[union-attr]
                self.assertEqual(recovered._queued_prompts, ["run this next"])
                self.assertEqual(recovered.query_one("#prompt", Input).value, "unfinished draft")

    async def test_change_history_opens_for_session_undo_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause(0.15)
                record = app.undo_manager.snapshot("write_file", {"path": "draft.py"})
                app.undo_manager.commit(record)
                app.action_changes()
                await pilot.pause(0.05)
                self.assertIsInstance(app.screen, ChangeHistoryScreen)

    async def test_sent_message_cards_follow_composer_width_and_scrollbar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 20)) as pilot:
                await pilot.pause(0.2)
                chat = app.query_one("#chat", ChatLog)
                composer = app.query_one("#composer-shell")
                chat.add_user("short message")
                await pilot.pause(0.1)
                card = app.query_one(UserMessage)
                self.assertEqual((card.region.x, card.region.width), (composer.region.x, composer.region.width))

                for _ in range(20):
                    chat.add_user("long message " * 20)
                await pilot.pause(0.2)
                card = list(app.query(UserMessage))[-1]
                self.assertTrue(chat.show_vertical_scrollbar)
                self.assertTrue(card.has_class("-scroll-gutter"))
                self.assertEqual(card.region.x, composer.region.x)
                self.assertEqual(
                    card.region.x + card.region.width,
                    composer.region.x + composer.region.width - chat.scrollbar_size_vertical - 1,
                )

    async def test_shutdown_releases_models_selected_by_artinium(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            client = FakeUIClient()
            app.client = client  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
            self.assertEqual(client.unloaded_models, ["qwen-test:8b"])

    async def test_command_shortcut_opens_action_menu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+p")
                await pilot.pause(0.2)
                self.assertIsInstance(app.screen, CommandMenu)
                self.assertIsNotNone(app.screen.query_one("#command-model", ListItem))

    async def test_context_settings_group_compaction_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.action_context_settings()
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, ContextSettingsScreen)
                self.assertIsNotNone(app.screen.query_one("#context-context_window", Static))
                self.assertIsNotNone(app.screen.query_one("#context-auto_compact", Static))
                self.assertIsNotNone(app.screen.query_one("#context-compact_now", Static))

                await pilot.press("down", "down", "enter")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, ContextSettingsScreen)

    async def test_slash_commands_configure_auto_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            app.model_settings_store = ModelSettingsStore(Path(directory) / "settings.json")
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "/autocompact off"
                await pilot.press("enter")
                self.assertIsNone(app.model_settings.auto_compact_threshold)
                prompt.value = "/autocompact 90"
                await pilot.press("enter")
                self.assertEqual(app.model_settings.auto_compact_threshold, 90)

    async def test_slash_compact_reduces_a_large_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.agent.history = [  # type: ignore[union-attr]
                    {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 2_000}
                    for index in range(10)
                ]
                prompt = app.query_one("#prompt", Input)
                prompt.value = "/compact"
                await pilot.press("enter")
                await pilot.pause(0.05)
                self.assertLess(len(app.agent.history), 10)  # type: ignore[union-attr]
                self.assertIsNotNone(app.agent.last_compaction)  # type: ignore[union-attr]

    async def test_escape_from_menu_subsection_returns_to_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+p")
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, SessionScreen)
                await pilot.press("escape")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, CommandMenu)

    async def test_model_settings_open_from_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            app.model_settings_store = ModelSettingsStore(Path(directory) / "settings.json")
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.action_model_settings()
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, ModelSettingsScreen)
                self.assertEqual(len(app.screen.query("#setting-context_window")), 0)
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertEqual(app.model_settings.temperature, 0.0)

    async def test_model_hub_groups_selection_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.action_model_hub()
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, ModelHubScreen)
                self.assertIsNotNone(app.screen.query_one("#model-hub-change", ListItem))
                self.assertIsNotNone(app.screen.query_one("#model-hub-settings", ListItem))
                self.assertIsNotNone(app.screen.query_one("#model-hub-context", ListItem))

    async def test_main_screen_keeps_keyboard_focus_in_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                self.assertIs(app.focused, prompt)
                self.assertIn("qwen-test:8b", str(app.query_one("#model-status", Static).renderable))
                self.assertIn("CONTEXT[/]  0%", str(app.query_one("#session-info", Static).renderable))
                self.assertFalse(app.query_one(WorkspaceTree).can_focus)
                self.assertFalse(app.query_one(ChatLog).can_focus)
                chat = app.query_one("#chat", ChatLog)
                await chat.begin_assistant()
                await chat.add_delta("Response that was clicked with the mouse.")
                await pilot.pause(0.05)
                app.query_one(AssistantMessage).focus()
                await pilot.pause(0.05)
                self.assertIs(app.focused, prompt)

    async def test_thinking_is_available_in_a_collapsible_transcript_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                chat = app.query_one("#chat", ChatLog)
                chat.add_thinking("First thought. ")
                chat.add_thinking("Second thought.")
                await pilot.pause(0.05)
                block = app.query_one(ThinkingBlock)
                self.assertTrue(block.collapsed)
                self.assertIn("First thought. Second thought.", str(block.query_one("#thinking-detail-body", Static).renderable))
                chat.finish_thinking()
                self.assertIn("Thinking", block.title)

    async def test_session_manager_creates_and_restores_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                original = app.active_session
                self.assertIsNotNone(original)
                self.assertEqual(len(app.session_store.sessions), 0)
                app.agent.history.append({"role": "user", "content": "first message"})  # type: ignore[union-attr]
                app._save_active_session()
                self.assertEqual(len(app.session_store.sessions), 1)
                app.action_sessions()
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, SessionScreen)
                app.screen.dismiss("new")
                await pilot.pause(0.1)
                self.assertIsNot(app.active_session, original)
                self.assertEqual(len(app.session_store.sessions), 1)
                app._session_selected(f"select:{original.id}")  # type: ignore[union-attr]
                await pilot.pause(0.1)
                self.assertIs(app.active_session, original)

                other = SessionRecord.new()
                other.title = "Other session"
                app.session_store.sessions.append(other)
                app.session_store.save()
                app._session_selected(f"pin:{original.id}")  # type: ignore[union-attr]
                await pilot.pause(0.1)
                self.assertTrue(original.pinned)  # type: ignore[union-attr]
                self.assertIs(app.session_store.sessions[0], original)
                self.assertIsInstance(app.screen, SessionScreen)
                app.screen.dismiss(None)
                await pilot.pause(0.1)

                app._session_selected(f"delete:{other.id}")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, DeleteSessionScreen)
                app.screen.dismiss(True)
                await pilot.pause(0.1)
                self.assertNotIn(other, app.session_store.sessions)

    async def test_exit_requires_two_ctrl_c_presses_without_a_dialog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+c")
                await pilot.pause(0.1)
                self.assertTrue(app._exit_armed)
                self.assertIs(app.focused, app.query_one("#prompt", Input))

    async def test_ctrl_c_exit_prompt_works_while_a_menu_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+p")
                self.assertIsInstance(app.screen, CommandMenu)
                await pilot.press("ctrl+c")
                await pilot.pause(0.1)
                self.assertTrue(app._exit_armed)
                self.assertIsInstance(app.screen, CommandMenu)

    async def test_ctrl_c_cancels_generation(self) -> None:
        class SlowUIClient(FakeUIClient):
            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                yield {"message": {"content": "empezando"}, "done": False}
                await asyncio.sleep(30)

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = SlowUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "tarea larga"
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertIsNotNone(app._generation_task)
                await pilot.press("ctrl+c")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, StopScreen)
                app.screen.dismiss(True)
                await pilot.pause(0.2)
                self.assertIsNone(app._generation_task)
                self.assertFalse(prompt.disabled)

    async def test_command_menu_opens_while_generation_is_running(self) -> None:
        class SlowUIClient(FakeUIClient):
            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                yield {"message": {"content": "working"}, "done": False}
                await asyncio.sleep(30)

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = SlowUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "long task"
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertIsNotNone(app._generation_task)
                await pilot.press("ctrl+p")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, CommandMenu)
                self.assertFalse(app.screen.query_one("#command-model", ListItem).disabled)
                self.assertFalse(app.screen.query_one("#command-files", ListItem).disabled)
                saved = next(item for item in app.session_store.sessions if item is app.active_session)
                self.assertEqual(saved.history[-1]["content"], "working")
                app.screen.dismiss(None)
                await pilot.pause(0.1)
                app._generation_task.cancel()  # type: ignore[union-attr]
                await pilot.pause(0.1)

    async def test_model_change_queues_until_active_task_finishes(self) -> None:
        class SwitchingClient(FakeUIClient):
            def __init__(self) -> None:
                super().__init__()
                self.release = asyncio.Event()

            async def list_models(self) -> list[ModelInfo]:
                return [
                    ModelInfo("first", capabilities=("completion", "tools"), context_length=8192),
                    ModelInfo("second", capabilities=("completion", "tools", "thinking", "vision"), context_length=16384),
                ]

            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                await self.release.wait()
                yield {"message": {"content": "done"}, "done": True, "prompt_eval_count": 2, "eval_count": 1}

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            client = SwitchingClient()
            app.client = client  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.query_one("#prompt", Input).value = "work"
                await pilot.press("enter")
                await pilot.pause(0.1)
                await pilot.press("ctrl+p", "enter", "enter", "down", "enter")
                await pilot.pause(0.2)
                self.assertIsInstance(app.screen, ModelHubScreen)
                self.assertEqual(app.agent.model.name, "first")  # type: ignore[union-attr]
                self.assertEqual(app._pending_model.name, "second")  # type: ignore[union-attr]
                self.assertIn("after task", str(app._main_static("#model-status").renderable))

                client.release.set()
                await pilot.pause(0.3)
                self.assertIsNone(app._pending_model)
                self.assertEqual(app.agent.model.name, "second")  # type: ignore[union-attr]

    async def test_ctrl_p_does_not_stack_command_menus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+p")
                await pilot.press("ctrl+p")
                await pilot.press("ctrl+p")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, CommandMenu)
                self.assertEqual(
                    sum(isinstance(screen, CommandMenu) for screen in app.screen_stack),
                    1,
                )

    async def test_escape_closes_menu_without_interrupting_active_task(self) -> None:
        class SlowUIClient(FakeUIClient):
            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                yield {"message": {"content": "working"}, "done": False}
                await asyncio.sleep(30)

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = SlowUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                app.query_one("#prompt", Input).value = "long task"
                await pilot.press("enter")
                await pilot.pause(0.1)
                task = app._generation_task
                self.assertIsNotNone(task)
                await pilot.press("ctrl+p")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, CommandMenu)
                await pilot.press("escape")
                await pilot.pause(0.1)
                self.assertFalse(isinstance(app.screen, CommandMenu))
                self.assertIs(app._generation_task, task)
                self.assertFalse(task.done())  # type: ignore[union-attr]
                task.cancel()  # type: ignore[union-attr]
                await pilot.pause(0.1)

    async def test_escape_requests_stop_during_generation(self) -> None:
        class SlowUIClient(FakeUIClient):
            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                yield {"message": {"content": "empezando"}, "done": False}
                await asyncio.sleep(30)

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = SlowUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "tarea larga"
                await pilot.press("enter")
                await pilot.pause(0.1)
                await pilot.press("escape")
                await pilot.pause(0.1)
                self.assertNotIsInstance(app.screen, StopScreen)
                self.assertTrue(app._stop_armed)
                await pilot.press("escape")
                await pilot.pause(0.2)
                self.assertIsNone(app._generation_task)

    async def test_escape_does_not_exit_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("escape")
                await pilot.pause(0.1)
                self.assertFalse(app._exit_armed)

    async def test_prompt_history_navigates_with_arrow_keys_and_restores_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                app._remember_prompt("first message")
                app._remember_prompt("second message")
                prompt.value = "unfinished draft"
                await pilot.press("up")
                self.assertEqual(prompt.value, "second message")
                await pilot.press("up")
                self.assertEqual(prompt.value, "first message")
                await pilot.press("down")
                self.assertEqual(prompt.value, "second message")
                await pilot.press("down")
                self.assertEqual(prompt.value, "unfinished draft")

    async def test_assistant_renders_markdown_bold_lists_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                chat = app.query_one("#chat", ChatLog)
                await chat.begin_assistant()
                source = "**negrita**\n\n- item 1\n- item 2\n\n| A | B |\n|---|---|\n| 1 | 2 |"
                await chat.add_delta(source)
                await pilot.pause(0.2)
                from textual.widgets import Markdown as _Markdown

                message = app.query_one(AssistantMessage)
                self.assertIsInstance(message, _Markdown)
                self.assertEqual(message.text, source)

    async def test_session_single_letters_filter_instead_of_triggering_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                first = SessionRecord.new()
                first.title = "Primera"
                second = SessionRecord.new()
                second.title = "Segunda"
                results: list[str | None] = []
                app.push_screen(SessionScreen([first, second], first.id), results.append)
                await pilot.pause(0.2)
                search = app.screen.query_one("#session-search", Input)
                self.assertIs(app.screen.focused, search)
                await pilot.press("n")
                await pilot.pause(0.2)
                self.assertIsInstance(app.screen, SessionScreen)
                self.assertEqual(search.value, "n")
                self.assertEqual(results, [])
                await pilot.press("ctrl+n")
                await pilot.pause(0.2)
                self.assertEqual(results, ["new"])

    def test_global_search_matches_subsections_across_languages(self) -> None:
        self.assertTrue(_search_matches("contexto", "Context settings", "window and compaction", "context contexto"))
        self.assertTrue(_search_matches("modelo", "Model", "select a model", "model modelo"))
        self.assertFalse(_search_matches("xyz123", "Model", "select", "model"))
        hits = [key for key, title, detail, keywords in CommandMenu.SEARCH_INDEX if _search_matches("contexto", title, detail, keywords)]
        self.assertIn("model-context", hits)
        # Global index must not contain dynamic session/model names.
        blob = " ".join(f"{title} {detail} {keywords}" for _, title, detail, keywords in CommandMenu.SEARCH_INDEX)
        self.assertNotIn("qwen", blob.lower())

    async def test_menu_search_filters_and_selects_with_keyboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press("ctrl+p")
                await pilot.pause(0.2)
                search = app.screen.query_one("#command-search", Input)
                self.assertIs(app.screen.focused, search)
                for char in "contexto":
                    await pilot.press(char)
                await pilot.pause(0.3)
                items = list(app.screen.query_one("#command-list", ListView).query(ListItem))
                ids = [item.id or "" for item in items]
                self.assertTrue(any("context" in item_id or "compact" in item_id for item_id in ids))

    def test_prompt_paste_converts_paths_to_opencode_style_pills(self) -> None:
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "Captura de pantalla_20260918_225649.png"
            image.write_bytes(b"fake")
            note = Path(directory) / "nota.txt"
            note.write_text("hola")
            app = ArtiumApp(Path(directory))
            app.agent = MagicMock()  # type: ignore[assignment]
            app.agent.model.supports_vision = True
            app.notify = lambda *_, **__: None  # type: ignore[method-assign]

            single = app.prepare_prompt_paste(f"'{image}'")
            self.assertEqual(single, "[Image 1]")
            self.assertIn("[Image 1]", app._attachment_refs)

            mixed = app.prepare_prompt_paste(f"'{image}' que ves?")
            self.assertEqual(mixed, "[Image 2] que ves?")
            self.assertNotIn(str(image), mixed or "")

            encoded = app.prepare_prompt_paste(f"file://{note}")
            # nota.txt has no spaces so file:// form survives shlex as one chunk
            if " " not in str(note):
                self.assertEqual(encoded, "[File 3]")

            self.assertIsNone(app.prepare_prompt_paste("hola que tal"))
            expanded = app._expand_tokens_for_display("[Image 1] que ves?")
            self.assertIn(image.name, expanded)
            self.assertIn("Image 1", expanded)

    async def test_prompt_selection_copies_and_pastes_without_triggering_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            app.client = FakeUIClient()  # type: ignore[assignment]
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "copy this"
                prompt.cursor_position = len(prompt.value)
                await pilot.press("shift+home")
                await pilot.press("ctrl+c")
                self.assertEqual(app.clipboard, "copy this")
                self.assertFalse(app._exit_armed)
                prompt.value = ""
                await pilot.press("ctrl+v")
                self.assertEqual(prompt.value, "copy this")

    async def test_message_queue_can_be_edited_deleted_and_runs_next(self) -> None:
        class QueuedUIClient(FakeUIClient):
            def __init__(self) -> None:
                super().__init__()
                self.release = asyncio.Event()

            async def chat_stream(self, **_: Any) -> AsyncIterator[dict[str, Any]]:
                self.round += 1
                current = self.round
                if current == 1:
                    await self.release.wait()
                yield {
                    "message": {"content": f"answer {current}"},
                    "done": True,
                    "prompt_eval_count": 2,
                    "eval_count": 2,
                }

        with tempfile.TemporaryDirectory() as directory:
            app = ArtiumApp(Path(directory))
            await app.client.close()
            client = QueuedUIClient()
            app.client = client  # type: ignore[assignment]
            async with app.run_test(size=(110, 32)) as pilot:
                await pilot.pause(0.2)
                prompt = app.query_one("#prompt", Input)
                prompt.value = "first"
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertFalse(prompt.disabled)
                self.assertEqual(app.query_one(ActivityPulse).label, "Thinking…")

                prompt.value = "second"
                await pilot.press("enter")
                self.assertEqual(app._queued_prompts, ["second"])

                await pilot.press("ctrl+e")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, QueueScreen)
                await pilot.press("enter")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, EditQueueScreen)
                queue_input = app.screen.query_one(Input)
                self.assertEqual(queue_input.value, "second")
                queue_input.value = "edited second"
                await pilot.press("enter")
                self.assertEqual(app._queued_prompts, ["edited second"])

                await pilot.press("ctrl+d")
                await pilot.pause(0.1)
                self.assertIsInstance(app.screen, QueueScreen)
                await pilot.press("d")
                await pilot.pause(0.1)
                self.assertEqual(app._queued_prompts, [])
                prompt.value = "final second"
                await pilot.press("enter")
                prompt.value = "third"
                await pilot.press("enter")
                prompt.value = "fourth"
                await pilot.press("enter")
                self.assertEqual(app._queued_prompts, ["final second", "third", "fourth"])

                client.release.set()
                await pilot.pause(0.9)
                self.assertEqual(client.round, 4)
                self.assertIsNone(app._generation_task)
                users = [item["content"] for item in app.agent.history if item.get("role") == "user"]  # type: ignore[union-attr]
                self.assertEqual(users, ["first", "final second", "third", "fourth"])
