from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from artium.tools.files import edit_file, glob_files, grep_files, list_files, read_file, search_files, write_file
from artium.undo import UndoManager
from artium.tools.shell import command_risk, is_dangerous, run_command
from artium.tools.web import _DuckDuckGoParser, _TextExtractor, fetch_url
from artium.tools import ToolRegistry
from artium.workspace import Workspace, WorkspaceError


class WorkspaceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = Workspace(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_rejects_path_traversal_and_symlink_escape(self) -> None:
        with self.assertRaises(WorkspaceError):
            self.workspace.resolve("../../etc/passwd")
        (self.root / "outside").symlink_to("/etc")
        with self.assertRaises(WorkspaceError):
            self.workspace.resolve("outside/passwd")

    def test_file_tool_round_trip(self) -> None:
        result = write_file(self.workspace, "src/demo.py", "one\ntwo\nthree\n")
        self.assertTrue(result["created"])
        read = read_file(self.workspace, "src/demo.py", 2, 3)
        self.assertEqual(read["content"], "2: two\n3: three")
        edited = edit_file(self.workspace, "src/demo.py", "two", "TWO")
        self.assertEqual(edited["replacements"], 1)
        self.assertIn("-two", edited["diff"])
        self.assertIn("+TWO", edited["diff"])
        found = search_files(self.workspace, "TWO")
        self.assertEqual(found["matches"][0]["line"], 2)
        self.assertIn("src/", list_files(self.workspace)["entries"])

    def test_glob_grep_and_undo(self) -> None:
        manager = UndoManager(self.workspace)
        record = manager.snapshot("write_file", {"path": "src/demo.py"})
        write_file(self.workspace, "src/demo.py", "alpha\nbeta\n")
        manager.commit(record)
        self.assertEqual(glob_files(self.workspace, "**/*.py")["matches"], ["src/demo.py"])
        matches = grep_files(self.workspace, r"^beta$", glob="**/*.py")["matches"]
        self.assertEqual(matches[0]["line"], 2)
        self.assertEqual(manager.undo(), {"path": "src/demo.py", "action": "removed", "tool": "write_file"})
        self.assertFalse((self.root / "src/demo.py").exists())

    def test_undo_records_can_be_restored_for_a_session(self) -> None:
        manager = UndoManager(self.workspace)
        record = manager.snapshot("write_file", {"path": "saved.txt"})
        write_file(self.workspace, "saved.txt", "saved")
        manager.commit(record)
        restored = UndoManager(self.workspace)
        restored.restore(manager.dump())
        self.assertEqual(restored.undo(), {"path": "saved.txt", "action": "removed", "tool": "write_file"})

    async def test_shell_output_and_confirmation(self) -> None:
        result = await run_command(self.workspace, "printf hello")
        self.assertEqual(result["stdout"], "hello")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(is_dangerous("git reset --hard HEAD"))
        self.assertTrue(is_dangerous("dd if=a of=b"))
        self.assertTrue(is_dangerous("rm -r generated"))
        self.assertTrue(is_dangerous("find . -name '*.log' -delete"))
        self.assertFalse(is_dangerous("sudo apt update"))
        self.assertIn("recursively deletes", command_risk("sudo rm -rf build") or "")
        denied = await run_command(self.workspace, "rm -rf generated")
        self.assertTrue(denied["cancelled"])

    async def test_shell_caps_large_output(self) -> None:
        result = await run_command(self.workspace, "yes x | head -c 50000")
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["stdout"]), 16_000)

    async def test_web_fetch_rejects_local_addresses(self) -> None:
        with self.assertRaises(ValueError):
            await fetch_url("http://127.0.0.1:11434/api/tags")

    def test_web_tools_are_registered_and_parse_content(self) -> None:
        names = {schema["function"]["name"] for schema in ToolRegistry(self.workspace).schemas}
        self.assertIn("web_search", names)
        self.assertIn("fetch_url", names)
        self.assertIn("glob", names)
        self.assertIn("grep", names)
        self.assertIn("question", names)

        search = _DuckDuckGoParser()
        search.feed(
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example</a>'
            '<a class="result__snippet">Useful result</a>'
        )
        self.assertEqual(search.results[0]["url"], "https://example.com")
        self.assertIn("Example", search.results[0]["title"])

        page = _TextExtractor()
        page.feed("<title>Example</title><p>Hello <b>world</b></p><script>ignored()</script>")
        self.assertEqual(page.title, "Example")
        self.assertIn("Hello world", page.text())

    async def test_tool_errors_include_a_clear_type(self) -> None:
        result = await ToolRegistry(self.workspace).execute("missing_tool", {})
        self.assertTrue(result.error)
        self.assertEqual(result.data["error_type"], "ValueError")
        self.assertIn("does not provide", result.data["error"])
        self.assertIn("unknown tool", result.data["technical_detail"])

    async def test_failed_commands_are_marked_as_failures(self) -> None:
        result = await ToolRegistry(self.workspace).execute("run_command", {"command": "exit 7"})
        self.assertTrue(result.error)
        self.assertEqual(result.data["error_type"], "CommandFailed")
        self.assertIn("exit code 7", result.data["error"])

    async def test_transient_web_failures_retry_once(self) -> None:
        operation = AsyncMock(side_effect=[httpx.ConnectError("offline"), {"results": []}])
        result = await ToolRegistry._retry_network(operation)
        self.assertTrue(result["retried"])
        self.assertEqual(operation.await_count, 2)

    async def test_permissions_and_question_callbacks(self) -> None:
        decisions: list[str] = []

        async def authorize(request):
            decisions.append(request.tool)
            return "allow_once"

        async def question(request):
            self.assertEqual(request.options, ["A", "B"])
            return "B"

        registry = ToolRegistry(
            self.workspace,
            authorize=authorize,
            ask_question=question,
            permission_policy=lambda _: "ask",
        )
        written = await registry.execute("write_file", {"path": "allowed.txt", "content": "ok"})
        self.assertFalse(written.error)
        self.assertEqual(decisions, ["write_file"])
        answer = await registry.execute("question", {"question": "Choose", "options": ["A", "B"]})
        self.assertEqual(answer.data["answer"], "B")

        denied = ToolRegistry(self.workspace, permission_policy=lambda _: "deny")
        result = await denied.execute("write_file", {"path": "denied.txt", "content": "no"})
        self.assertTrue(result.data["cancelled"])
        self.assertFalse((self.root / "denied.txt").exists())
