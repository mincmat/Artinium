from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Collapsible, DirectoryTree, Markdown, Static

from ..workspace import IGNORED_DIRS


class WorkspaceTree(DirectoryTree):
    """Workspace-only file tree with a restrained, useful surface."""

    # Textual defaults to coloured emoji folders/files. Keep the tree strictly
    # terminal-like and monochrome instead.
    ICON_NODE_EXPANDED = "▾ "
    ICON_NODE = "▸ "
    ICON_FILE = "· "

    # The tree remains fully clickable, but typing should always go to the prompt.
    can_focus = False

    def on_mount(self) -> None:
        self.show_root = False
        self.show_horizontal_scrollbar = False

    def filter_paths(self, paths: Any) -> list[Path]:
        return [path for path in paths if path.name not in IGNORED_DIRS and not path.is_symlink()]


class ActivityPulse(Static):
    """Small animated state indicator; animation stops while idle."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    frame = reactive(0)
    label = reactive("Starting…")
    active = reactive(True)
    tone = reactive("working")

    def on_mount(self) -> None:
        self.set_interval(0.09, self._tick)
        self._render_state()

    def _tick(self) -> None:
        if self.active:
            self.frame = (self.frame + 1) % len(self.FRAMES)
            self._render_state()

    def _render_state(self) -> None:
        if self.active:
            icon = self.FRAMES[self.frame]
        elif self.tone == "error":
            icon = "×"
        elif self.tone in {"warning", "confirm"}:
            icon = "!"
        else:
            icon = "●"
        self.update(f"{icon}  {escape(self.label)}")

    def set_state(self, label: str, *, active: bool = False, tone: str = "ready") -> None:
        self.label = label
        self.active = active
        self.tone = tone
        self.set_class(not active and tone == "ready", "-idle")
        self.set_class(tone == "error", "-error")
        self.set_class(tone == "warning", "-warning")
        self.set_class(tone == "confirm", "-confirm")
        self.set_class(active, "-active")
        self._render_state()


class ThinkingBlock(Collapsible):
    """A compact, click-to-expand record of a model's streamed reasoning."""

    FRAMES = ActivityPulse.FRAMES
    frame = reactive(0)

    def __init__(self) -> None:
        self._detail = Static("", markup=False, id="thinking-detail-body", classes="thinking-detail")
        self._active = True
        self._started = time.monotonic()
        super().__init__(
            self._detail,
            title="Thinking…",
            collapsed=True,
            classes="thinking-block",
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, round(seconds))
        minutes, remaining = divmod(total, 60)
        if minutes:
            return f"{minutes}m {remaining}s"
        return f"{remaining}s"

    def on_mount(self) -> None:
        self.set_interval(0.09, self._tick)
        self._render_title()

    def _tick(self) -> None:
        if self._active:
            self.frame = (self.frame + 1) % len(self.FRAMES)
            self._render_title()

    def _render_title(self) -> None:
        if self._active:
            self.title = f"{self.FRAMES[self.frame]}  Thinking"
        else:
            elapsed = self._format_duration(time.monotonic() - self._started)
            self.title = f"◇  Thought  ·  {elapsed}"

    def append(self, text: str) -> None:
        self._detail.update(f"{self._detail.renderable}{text}")

    def finish(self) -> None:
        self._active = False
        self._render_title()


class ContextGauge(Static):
    """Compact context meter designed for the top bar."""

    def set_usage(self, used: int, limit: int) -> None:
        ratio = min(1.0, used / limit) if limit else 0.0
        self.update(f"ctx {ratio:.0%}")
        self.set_class(ratio >= 0.8, "-danger")
        self.tooltip = f"Approximate context: {used:,} / {limit:,} tokens"


class UserMessage(Static):
    def __init__(self, text: str):
        super().__init__(escape(text), classes="message user-message", markup=True)


class AssistantMessage(Markdown):
    """A read-only response with real markdown: bold, lists, code, tables.

    Previously a plain TextArea, so ``**bold**`` showed literally.
    Textual's Markdown renders like OpenCode (headings, bold/italic,
    lists, quotes, code blocks, comparison tables) with no new deps.
    """

    can_focus = False

    def __init__(self):
        super().__init__(
            "",
            classes="message assistant-message",
        )
        self._source = ""

    @property
    def text(self) -> str:
        """Raw markdown source (compat for tests and session restore)."""
        return self._source

    @property
    def read_only(self) -> bool:
        return True

    async def set_markdown(self, text: str) -> None:
        self._source = text
        await self.update(text)


class SystemNotice(Static):
    def __init__(self, value: str, tone: str = "info"):
        super().__init__(value, classes=f"system-notice -{tone}", markup=True)


class ChatLog(VerticalScroll):
    """Animated transcript with markdown responses and expandable tool cards."""

    # Keep keyboard input in the composer; mouse scrolling remains available.
    can_focus = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._assistant: AssistantMessage | None = None
        self._assistant_text = ""
        self._assistant_pending = False
        self._tool: Collapsible | None = None
        self._tool_detail: Static | None = None
        self._tool_entries: list[Collapsible] = []
        self._activity: ActivityPulse | None = None
        self._thinking: ThinkingBlock | None = None

    # Basic monochrome Unicode, intentionally avoiding private-use Nerd Font
    # glyphs and emojis. These render in the same terminal fonts that already
    # provide the disclosure arrows and spinner used by Artinium.
    TOOL_BADGES = {
        "list_files": "▤",
        "read_file": "▧",
        "write_file": "✎",
        "edit_file": "✎",
        "search_files": "⌕",
        "glob": "◇",
        "grep": "⌕",
        "question": "?",
        "run_command": "▻",
        "web_search": "◌",
        "fetch_url": "↗",
    }

    def _mount_animated(self, widget: Widget) -> None:
        widget.styles.opacity = 0.0
        self.mount(widget)
        self.call_after_refresh(widget.styles.animate, "opacity", 1.0, duration=0.22, easing="out_cubic")
        self.call_after_refresh(self._sync_user_card_gutter)
        self.call_after_refresh(self.scroll_end, animate=False)

    def _sync_user_card_gutter(self) -> None:
        """Leave a one-column air gap beside sent cards and a visible scrollbar."""
        has_scrollbar = self.show_vertical_scrollbar
        for card in self.query(UserMessage):
            card.set_class(has_scrollbar, "-scroll-gutter")

    def on_resize(self) -> None:
        self.call_after_refresh(self._sync_user_card_gutter)

    def write(self, value: str, *, tone: str = "info", **_: Any) -> None:
        self._mount_animated(SystemNotice(value, tone))

    def add_user(self, text: str) -> None:
        self._mount_animated(UserMessage(text))

    async def begin_assistant(self) -> None:
        self.clear_activity()
        self._assistant_text = ""
        self._assistant_pending = False
        self._assistant = AssistantMessage()
        self._assistant.styles.opacity = 0.0
        await self.mount(self._assistant)
        self._assistant.styles.animate("opacity", 1.0, duration=0.14, easing="out_cubic")
        self.call_after_refresh(self.scroll_end, animate=False)

    async def add_delta(self, text: str) -> None:
        self.finish_thinking()
        if self._assistant_pending:
            await self.begin_assistant()
        self._assistant_text += text
        if self._assistant is not None:
            await self._assistant.set_markdown(self._assistant_text)
        self.call_after_refresh(self._sync_user_card_gutter)
        self.call_after_refresh(self.scroll_end, animate=False)

    @staticmethod
    def tool_label(name: str, arguments: dict[str, Any]) -> str:
        path = str(arguments.get("path") or "")
        if name == "list_files":
            return f"List {path}" if path and path != "." else "List files"
        if name == "read_file":
            return f"Read {path or 'file'}"
        if name == "write_file":
            return f"Write {path or 'file'}"
        if name == "edit_file":
            return f"Edit {path or 'file'}"
        if name == "search_files":
            query = str(arguments.get("query") or "")
            return f'Search "{query}"' if query else "Search files"
        if name == "glob":
            pattern = str(arguments.get("pattern") or "")
            return f'Glob "{pattern[:64]}"' if pattern else "Glob files"
        if name == "grep":
            pattern = str(arguments.get("pattern") or "")
            return f'Grep "{pattern[:64]}"' if pattern else "Grep files"
        if name == "question":
            question = " ".join(str(arguments.get("question") or "").split())
            return f"Ask {question[:68]}" if question else "Ask user"
        if name == "run_command":
            command = " ".join(str(arguments.get("command") or "").split())
            return f"Run {command[:72]}" if command else "Run command"
        if name == "web_search":
            query = " ".join(str(arguments.get("query") or "").split())
            return f'Search web “{query[:64]}”' if query else "Search web"
        if name == "fetch_url":
            url = str(arguments.get("url") or "")
            return f"Read {url[:72]}" if url else "Read web page"
        return name.replace("_", " ").capitalize()

    @classmethod
    def tool_badge(cls, name: str) -> str:
        """Return a portable visual marker for a tool invocation."""
        return cls.TOOL_BADGES.get(name, "◇")

    @staticmethod
    def tool_summary(name: str, arguments: dict[str, Any], result: dict[str, Any], error: bool) -> str:
        if error:
            kind = str(result.get("error_type") or "Tool error")
            detail = str(result.get("error") or "The action failed.")
            return f"Failed · {kind}\n{detail}"

        path = str(result.get("path") or arguments.get("path") or "")
        if result.get("cancelled"):
            return "Cancelled."
        if name == "read_file":
            start = result.get("start_line")
            end = result.get("end_line")
            total = result.get("total_lines")
            lines = f"lines {start}–{end}" if start and end else "file"
            suffix = f" of {total}" if total else ""
            return f"Read {lines}{suffix} from {path}."
        if name == "list_files":
            entries = result.get("entries") or []
            preview = "\n".join(str(item) for item in entries[:8])
            header = f"{len(entries)} items" + (f" in {path}" if path else "")
            return f"{header}." + (f"\n{preview}" if preview else "")
        if name == "write_file":
            action = "Created" if result.get("created") else "Updated"
            return f"{action} {path} · {result.get('chars', 0)} characters."
        if name == "edit_file":
            diff = str(result.get("diff") or "")
            if diff:
                return diff
            return f"Updated {path} · {result.get('replacements', 0)} replacement(s)."
        if name == "search_files":
            matches = result.get("matches") or []
            preview = "\n".join(
                f"{item.get('path')}:{item.get('line')}  {item.get('text', '')}"
                for item in matches[:6]
            )
            return f"{len(matches)} matches." + (f"\n{preview}" if preview else "")
        if name == "glob":
            matches = result.get("matches") or []
            return f"{len(matches)} paths.\n" + "\n".join(str(item) for item in matches)
        if name == "grep":
            matches = result.get("matches") or []
            preview = "\n".join(
                f"{item.get('path')}:{item.get('line')}  {item.get('text', '')}" for item in matches
            )
            return f"{len(matches)} matches." + (f"\n{preview}" if preview else "")
        if name == "question":
            return f"Answer: {result.get('answer')}" if result.get("answer") is not None else "No answer."
        if name == "run_command":
            stdout = str(result.get("stdout") or "").strip()
            stderr = str(result.get("stderr") or "").strip()
            if stdout and stderr:
                return f"{stdout}\n\n[stderr]\n{stderr}"
            return stdout or stderr or "No terminal output."
        if name == "web_search":
            results = result.get("results") or []
            header = f"{result.get('provider', 'Search')} · {result.get('retrieved_at', '')}".rstrip(" ·")
            entries = "\n\n".join(
                f"{index}. {item.get('title', '')}\n{item.get('url', '')}\n{item.get('snippet', '')}".strip()
                for index, item in enumerate(results, 1)
            ) or "No results."
            return f"{header}\n\n{entries}" if header else entries
        if name == "fetch_url":
            title = str(result.get("title") or result.get("url") or "Page")
            content = str(result.get("content") or "")
            source = str(result.get("url") or "")
            retrieved = str(result.get("retrieved_at") or "")
            return f"{title}\n{source}\nRetrieved {retrieved}\n\n{content}".strip()
        return "Done."

    @staticmethod
    def _render_diff(diff: str) -> str:
        """Turn unified diff data into a compact, line-numbered change view."""
        rendered: list[str] = []
        old_line = new_line = 1
        hunk = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

        for line in diff.splitlines():
            # File and hunk headers are useful for a patch, but duplicate the
            # tool title here. The visual view should show only source changes.
            if line.startswith(("---", "+++")):
                continue
            match = hunk.match(line)
            if match:
                old_line, new_line = map(int, match.groups())
                continue

            if line.startswith("-"):
                rendered.append(
                    f"[#eeeeee on #3b2025]{old_line:>4} - {escape(line[1:])}[/]"
                )
                old_line += 1
            elif line.startswith("+"):
                rendered.append(
                    f"[#eeeeee on #16353a]{new_line:>4} + {escape(line[1:])}[/]"
                )
                new_line += 1
            elif line.startswith(" "):
                rendered.append(f"[dim]{old_line:>4}   {escape(line[1:])}[/]")
                old_line += 1
                new_line += 1
            elif line:
                # Gracefully display non-standard diff output too.
                rendered.append(f"[dim]{escape(line)}[/]")

        return "\n".join(rendered) or "[dim]No textual changes.[/]"

    @classmethod
    def tool_detail(cls, name: str, arguments: dict[str, Any], result: dict[str, Any], error: bool) -> str:
        """Return the expandable tool body, preserving source changes verbatim."""
        if not error and name == "write_file":
            path = str(result.get("path") or arguments.get("path") or "file")
            content = str(arguments.get("content") or "")
            lines = content.splitlines()
            header = "Created" if result.get("created") else "Updated"
            diff_lines = [
                "--- /dev/null" if result.get("created") else f"--- a/{path}",
                f"+++ b/{path}",
                f"@@ -0,0 +1,{max(1, len(lines))} @@",
            ]
            # A write always exposes every line it was asked to write. This is
            # deliberately derived from the tool arguments rather than a short
            # result summary, so it remains useful when the tool is collapsed.
            diff_lines.extend(f"+{line}" for line in lines) if lines else diff_lines.append("+")
            return f"[dim]{header} {escape(path)} · {result.get('chars', 0)} characters.[/]\n" + cls._render_diff(
                "\n".join(diff_lines)
            )

        if not error and name == "edit_file":
            diff = str(result.get("diff") or "")
            if diff:
                return cls._render_diff(diff)

        return f"[dim]{escape(cls.tool_summary(name, arguments, result, error))}[/]"

    @staticmethod
    def tool_outcome(name: str, result: dict[str, Any], error: bool) -> str:
        if error:
            return "failed"
        if result.get("cancelled"):
            return "cancelled"
        if name == "read_file":
            start, end = result.get("start_line"), result.get("end_line")
            return f"{end - start + 1} lines" if isinstance(start, int) and isinstance(end, int) else "done"
        if name == "list_files":
            return f"{len(result.get('entries') or [])} items"
        if name == "write_file":
            return "created" if result.get("created") else "updated"
        if name == "edit_file":
            return f"{result.get('replacements', 0)} replacement(s)"
        if name == "search_files":
            return f"{len(result.get('matches') or [])} matches"
        if name == "glob":
            return f"{len(result.get('matches') or [])} paths"
        if name == "grep":
            return f"{len(result.get('matches') or [])} matches"
        if name == "question":
            return "answered" if result.get("answer") is not None else "cancelled"
        if name == "run_command":
            return ""
        if name == "web_search":
            return f"{len(result.get('results') or [])} results"
        if name == "fetch_url":
            return "read"
        return "done"

    def add_tool(self, name: str, arguments: dict[str, Any]) -> None:
        self.finish_thinking()
        self.clear_activity()
        self._assistant_pending = False
        self._tool_detail = Static("Waiting for result…", markup=True, classes="tool-detail")
        self._tool = Collapsible(
            self._tool_detail,
            title=f"{self.tool_badge(name)}  {self.tool_label(name, arguments)}",
            collapsed=True,
            classes="tool-call -running",
        )
        self._tool_entries.append(self._tool)
        self._mount_animated(self._tool)

    def finish_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        error: bool,
        show_details: bool,
    ) -> None:
        label = self.tool_label(name, arguments)
        if self._tool is not None:
            outcome = self.tool_outcome(name, result, error)
            self._tool.title = (
                f"{'x  ' if error else ''}{self.tool_badge(name)}  {label}"
                + (f"  ·  {outcome}" if outcome else "")
            )
            self._tool.collapsed = not show_details
            self._tool.remove_class("-running")
            self._tool.set_class(error, "-failed")
            self._tool.set_class(not error, "-complete")
        if self._tool_detail is not None:
            self._tool_detail.update(self.tool_detail(name, arguments, result, error))
            self._tool_detail.set_class(
                not error and name in {"write_file", "edit_file"}, "-diff"
            )
        self.call_after_refresh(self.scroll_end, animate=False)

    def stop_pending_tool(self) -> None:
        """Leave an honest, compact record when cancellation interrupts a tool."""
        if self._tool is None:
            return
        self._tool.title = f"x  {self._tool.title}  ·  interrupted"
        self._tool.remove_class("-running")
        self._tool.add_class("-failed")
        if self._tool_detail is not None:
            self._tool_detail.update("[dim]Task interrupted before this action completed.[/]")
        self.call_after_refresh(self.scroll_end, animate=False)

    def show_activity(self, label: str, *, active: bool = True, tone: str = "ready") -> None:
        """Show transient agent progress exactly where the next event will land."""
        # Ollama opens an assistant turn before it may decide to call a tool.
        # Do not leave an empty reply line above the transient status.
        if self._assistant is not None and not self._assistant_text:
            self._assistant.remove()
            self._assistant = None
        if self._activity is None:
            self._activity = ActivityPulse(classes="agent-activity")
            self._mount_animated(self._activity)
        self._activity.set_state(label, active=active, tone=tone)

    def clear_activity(self) -> None:
        if self._activity is not None:
            self._activity.remove()
            self._activity = None

    def add_thinking(self, text: str) -> None:
        """Append a reasoning chunk that the user may expand while it streams."""
        if self._thinking is None:
            self.clear_activity()
            self._thinking = ThinkingBlock()
            self._mount_animated(self._thinking)
        self._thinking.append(text)
        self.call_after_refresh(self.scroll_end, animate=False)

    async def promote_thinking_to_response(self, text: str) -> None:
        """Recover a final reply accidentally streamed through ``thinking``."""
        if self._thinking is not None:
            self._thinking.remove()
            self._thinking = None
        await self.add_delta(text)

    def finish_thinking(self) -> None:
        if self._thinking is not None:
            self._thinking.finish()
            self._thinking = None

    def set_tool_details(self, visible: bool) -> None:
        for entry in self._tool_entries:
            entry.collapsed = not visible

    def clear(self) -> None:
        self.remove_children()
        self._assistant = None
        self._assistant_text = ""
        self._tool = None
        self._tool_detail = None
        self._tool_entries.clear()
        self._activity = None
        self._thinking = None
        self._assistant_pending = False

    def prepare_assistant(self) -> None:
        """Reserve the next transcript slot without hiding a thinking state."""
        self._assistant_pending = True
