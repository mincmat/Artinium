from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from ..error_messages import explain, technical_detail
from ..workspace import Workspace
from ..undo import UndoManager
from . import files
from . import web
from .shell import ConfirmCallback, run_command


@dataclass(slots=True)
class PermissionRequest:
    tool: str
    summary: str


@dataclass(slots=True)
class QuestionRequest:
    question: str
    options: list[str]


PermissionCallback = Callable[[PermissionRequest], Awaitable[str]]
QuestionCallback = Callable[[QuestionRequest], Awaitable[str | None]]


@dataclass(slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    data: dict[str, Any]
    error: bool = False

    def compact_json(self, limit: int = 18_000) -> str:
        value = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        if len(value) <= limit:
            return value
        return json.dumps({"truncated": True, "preview": value[:limit]}, ensure_ascii=False, separators=(",", ":"))


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        confirm: ConfirmCallback | None = None,
        authorize: PermissionCallback | None = None,
        ask_question: QuestionCallback | None = None,
        permission_policy: Callable[[str], str] | None = None,
        undo: UndoManager | None = None,
    ):
        self.workspace = workspace
        self.confirm = confirm
        self.authorize = authorize
        self.ask_question = ask_question
        self.permission_policy = permission_policy
        self.undo = undo or UndoManager(workspace)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
            return {"type": "function", "function": {"name": name, "description": description, "parameters": {
                "type": "object", "properties": properties, "required": required or []
            }}}

        string = {"type": "string"}
        integer = {"type": "integer"}
        boolean = {"type": "boolean"}
        array = {"type": "array", "items": string}
        return [
            tool("list_files", "List a small workspace tree.", {"path": string, "depth": integer}),
            tool("read_file", "Read a text file or line range.", {"path": string, "start_line": integer, "end_line": integer}, ["path"]),
            tool("write_file", "Create or replace a text file.", {"path": string, "content": string}, ["path", "content"]),
            tool("edit_file", "Replace exact text in a file.", {"path": string, "old_text": string, "new_text": string, "replace_all": boolean}, ["path", "old_text", "new_text"]),
            tool("search_files", "Find text in workspace files.", {"query": string, "path": string, "case_sensitive": boolean}, ["query"]),
            tool("glob", "Find files and directories by glob pattern. Use ** for recursive matching.", {"pattern": string, "path": string, "limit": integer}, ["pattern"]),
            tool("grep", "Search workspace text with a regular expression and optional file glob.", {"pattern": string, "path": string, "glob": string, "case_sensitive": boolean, "limit": integer}, ["pattern"]),
            tool("question", "Ask the user one focused question when a required choice is missing. Provide short options when useful.", {"question": string, "options": array}, ["question"]),
            tool("run_command", "Run a shell command in the workspace.", {"command": string, "timeout": integer}, ["command"]),
            tool("web_search", "Search the public web with DuckDuckGo and return titles, links, and snippets. Use it for current information, then verify important facts with fetch_url.", {"query": string, "limit": integer}, ["query"]),
            tool("fetch_url", "Read the visible text of one specific public HTTP or HTTPS page. Prefer an official or primary source and cite its URL in the final answer.", {"url": string}, ["url"]),
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name in {"write_file", "edit_file", "run_command"}:
                policy = self.permission_policy(name) if self.permission_policy else "allow"
                if policy == "deny":
                    return ToolResult(name, arguments, {"cancelled": True, "reason": "denied by tool permissions"})
                if policy == "ask":
                    if self.authorize is None:
                        return ToolResult(name, arguments, {"cancelled": True, "reason": "permission was not granted"})
                    decision = await self.authorize(PermissionRequest(name, self._summary(name, arguments)))
                    if decision not in {"allow_once", "always_allow"}:
                        return ToolResult(name, arguments, {"cancelled": True, "reason": "denied by the user"})
            snapshot = self.undo.snapshot(name, arguments)
            if name == "run_command":
                data = await run_command(self.workspace, confirm=self.confirm, **arguments)
            elif name == "web_search":
                data = await self._retry_network(lambda: web.web_search(**arguments))
            elif name == "fetch_url":
                data = await self._retry_network(lambda: web.fetch_url(**arguments))
            elif name == "question":
                if self.ask_question is None:
                    data = {"cancelled": True, "reason": "interactive questions are unavailable"}
                else:
                    raw_options = arguments.get("options") or []
                    options = raw_options if isinstance(raw_options, list) else []
                    answer = await self.ask_question(QuestionRequest(
                        str(arguments.get("question") or ""),
                        [str(item) for item in options if str(item).strip()][:8],
                    ))
                    data = {"answer": answer} if answer is not None else {"cancelled": True}
            else:
                function = {
                    "list_files": files.list_files,
                    "read_file": files.read_file,
                    "write_file": files.write_file,
                    "edit_file": files.edit_file,
                    "search_files": files.search_files,
                    "glob": files.glob_files,
                    "grep": files.grep_files,
                }.get(name)
                if function is None:
                    raise ValueError(f"unknown tool: {name}")
                data = function(self.workspace, **arguments)
            if not data.get("cancelled"):
                self.undo.commit(snapshot)
            if name == "run_command" and data.get("exit_code") not in {None, 0}:
                exit_code = data.get("exit_code")
                message = (
                    "The command timed out and was stopped."
                    if exit_code == 124
                    else f"The command failed with exit code {exit_code}."
                )
                data.update({"error": message, "error_type": "CommandFailed", "retryable": False})
                return ToolResult(name, arguments, data, error=True)
            return ToolResult(name, arguments, data)
        except Exception as exc:
            return ToolResult(
                name,
                arguments,
                {
                    "error": explain(exc, area="web" if name in {"web_search", "fetch_url"} else "tool"),
                    "error_type": exc.__class__.__name__,
                    "technical_detail": technical_detail(exc),
                    "retryable": name in {"web_search", "fetch_url"},
                },
                error=True,
            )

    @staticmethod
    async def _retry_network(operation: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """Retry one transient web failure without repeating file or shell work."""
        try:
            return await operation()
        except (httpx.TimeoutException, httpx.NetworkError):
            await asyncio.sleep(0.4)
            data = await operation()
            data["retried"] = True
            return data

    @staticmethod
    def _summary(name: str, arguments: dict[str, Any]) -> str:
        if name in {"write_file", "edit_file"}:
            return str(arguments.get("path") or "file")
        return " ".join(str(arguments.get("command") or "").split())[:160] or name
