from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import time
import unicodedata
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, ListItem, ListView, Static, TextArea

from ..agent import INTERRUPTION_NOTE, Agent
from .. import __version__
from ..error_messages import explain
from ..model_settings import ModelSettings, ModelSettingsStore, total_memory_gib
from ..ollama_client import ModelInfo, OllamaClient, OllamaError, OllamaUnavailable
from ..preferences import AppPreferences, MUTATING_TOOLS, PERMISSION_VALUES, PreferencesStore
from ..sessions import SessionRecord, SessionStore
from ..tools import PermissionRequest, QuestionRequest, ToolRegistry
from ..tools.shell import ConfirmationRequest
from ..undo import UndoManager
from ..updater import UpdateInfo, check_for_update, install_update
from ..workspace import Workspace, WorkspaceError
from .widgets import ChatLog, WorkspaceTree


CONTEXT_WINDOWS = (None, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)


def _normalize_search(text: str) -> str:
    """Lowercase + strip accents so 'contexto' matches 'context' settings."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in folded if not unicodedata.combining(char))


def _system_memory() -> tuple[float, float] | None:
    """System RAM as (used_gb, total_gb) via /proc; None when unavailable.

    Deliberately no psutil dependency: a single tiny file read per topbar
    refresh, which already happens on stats/model/session changes.
    """
    try:
        total = available = 0
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])  # kB
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1])
            if total and available:
                break
    except (OSError, ValueError, IndexError):
        return None
    if not total:
        return None
    return ((total - available) / 1024**2, total / 1024**2)


def _system_load() -> float | None:
    """1-minute load average; None when unavailable."""
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def _system_lines() -> str:
    """Extra sidebar rows (RAM + load), kept within the 24-char content width."""
    lines = ""
    memory = _system_memory()
    if memory:
        used_gb, total_gb = memory
        percent = round(used_gb / total_gb * 100) if total_gb else 0
        lines += f"\n[dim]RAM[/]    {percent}% {used_gb:.1f}/{total_gb:.1f}G"
    load = _system_load()
    if load is not None:
        lines += f"\n[dim]LOAD[/]   {load:.2f}"
    return lines


def _search_matches(query: str, *haystacks: str) -> bool:
    """Substring or bidirectional prefix match per word (EN/ES tolerant)."""
    q = _normalize_search(query).strip()
    if not q:
        return True
    hay = " ".join(_normalize_search(h) for h in haystacks)
    if q in hay:
        return True
    q_words = q.split()
    hay_words = hay.split()
    for qw in q_words:
        matched = False
        for hw in hay_words:
            if qw in hw or hw in qw or hw.startswith(qw[:4]) or qw.startswith(hw[:4]):
                # The [:4] prefix rule makes 'contexto' match 'context',
                # 'modelo' match 'model', 'sesion' match 'sessions', etc.
                # Guard against tiny words matching everything.
                if len(qw) >= 3 and len(hw) >= 3:
                    matched = True
                    break
        if not matched:
            return False
    return True


class PromptInput(Input):
    """Input that keeps terminal-style history navigation local to the composer."""

    def key_up(self) -> None:
        self.app.recall_prompt_history(self, direction=-1)  # type: ignore[attr-defined]

    def key_down(self) -> None:
        self.app.recall_prompt_history(self, direction=1)  # type: ignore[attr-defined]

    def action_copy(self) -> None:
        """Copy a selection first; retain Ctrl+C's terminal stop behaviour otherwise."""
        if self.selected_text:
            super().action_copy()
        else:
            self.app.action_interrupt_or_exit()  # type: ignore[attr-defined]

    def _spaced_for_insert(self, text: str) -> str:
        """Pad pills with a single space so they never glue to typed text."""
        try:
            if self.selection.is_empty:
                cursor = self.cursor_position
                before = self.value[:cursor]
                after = self.value[cursor:]
                if before and not before[-1].isspace() and not text[:1].isspace():
                    text = " " + text
                if after and not after[:1].isspace() and not text[-1:].isspace():
                    text = text + " "
        except Exception:
            pass
        return text

    def _on_paste(self, event: events.Paste) -> None:
        # Terminal bracketed paste. The insert always goes through the base
        # handler exactly once, so a token can never duplicate the raw path.
        replacement = self.app.prepare_prompt_paste(event.text)  # type: ignore[attr-defined]
        if not replacement:
            super()._on_paste(event)
            return
        event.text = self._spaced_for_insert(replacement)
        super()._on_paste(event)

    def action_paste(self) -> None:
        # Ctrl+V bypasses Paste events (base replaces directly), so tokenize
        # here too — otherwise a pasted path stays raw next to the pill.
        try:
            clipboard = self.app.clipboard
        except Exception:
            clipboard = ""
        replacement = None
        if clipboard:
            try:
                replacement = self.app.prepare_prompt_paste(clipboard)  # type: ignore[attr-defined]
            except Exception:
                replacement = None
        if not replacement:
            super().action_paste()
            return
        start, end = self.selection
        self.replace(self._spaced_for_insert(replacement), start, end)


class EscapeModalScreen(ModalScreen[Any]):
    """A modal owns Esc before it can reach the active chat beneath it."""

    BINDINGS = [
        Binding("escape", "close_modal", show=False, priority=True),
    ]

    def action_close_modal(self) -> None:
        self.dismiss(None)


class ModelHubScreen(EscapeModalScreen):
    """Entry point for selection and the two kinds of model configuration."""

    CSS = """
    ModelHubScreen { align: center middle; background: #000000 75%; }
    #model-hub-box { width: 58; height: auto; padding: 1 2; background: $surface; border: none; }
    #model-hub-title { height: 2; color: $text; text-style: bold; }
    #model-hub-help { height: 2; color: $text-muted; margin-bottom: 1; }
    #model-hub-list { height: auto; background: $surface; border: none; }
    #model-hub-list:focus { border: none; }
    #model-hub-list ListItem { height: 3; padding: 1; color: #bdbdbd; background: $surface; }
    #model-hub-list:focus > ListItem.-highlight, #model-hub-list > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    #model-hub-list ListItem.-locked { color: #5f5f5f; }
    """

    def __init__(self, *, working: bool = False):
        super().__init__()
        self.working = working

    def compose(self) -> ComposeResult:
        with Vertical(id="model-hub-box"):
            yield Static("Model", id="model-hub-title")
            if self.working:
                yield Static(
                    "Change model applies after this task",
                    id="model-hub-help",
                )
            yield ListView(
                ListItem(Label("Change model\n[dim]Select a detected model[/]"), id="model-hub-change"),
                ListItem(
                    Label("Model settings\n[dim]Reasoning, temperature, and output[/]"),
                    id="model-hub-settings",
                    disabled=self.working,
                    classes="-locked" if self.working else "",
                ),
                ListItem(
                    Label("Context settings\n[dim]Window and compaction[/]"),
                    id="model-hub-context",
                    disabled=self.working,
                    classes="-locked" if self.working else "",
                ),
                id="model-hub-list",
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("model-hub-"))

    def key_escape(self) -> None:
        self.dismiss(None)


class ModelScreen(EscapeModalScreen):
    CSS = """
    ModelScreen { align: center middle; background: #000000 75%; }
    #model-box {
        width: 68; height: auto; max-height: 76%; padding: 1 2;
        border: none; background: $surface;
    }
    .modal-kicker { color: $text-muted; height: 1; }
    .modal-title { color: $text; text-style: bold; height: 2; }
    .modal-help { color: #666666; margin-bottom: 1; height: 1; }
    #model-search { margin-bottom: 1; border: tall #333333; background: $background; }
    #model-search:focus { border: tall #a0a0a0; }
    #model-list { height: auto; max-height: 18; background: $surface; border: none; }
    #model-list:focus { border: none; }
    #model-list ListItem { height: 4; padding: 1; color: $text-muted; background: $surface; }
    #model-list > ListItem:hover { background: #303030; color: #ffffff; }
    #model-list > ListItem.-highlight, #model-list:focus > ListItem.-highlight {
        background: #303030; color: #ffffff; text-style: none;
    }
    """

    def __init__(self, models: list[ModelInfo], current: str, pending: str | None = None):
        super().__init__()
        self.models = models
        self.current = current
        self.pending = pending
        self._filtered: list[ModelInfo] = list(models)

    @staticmethod
    def _capabilities(model: ModelInfo) -> str:
        capabilities: list[str] = []
        if model.supports_reasoning:
            capabilities.append("◆ thinking")
        if model.supports_tools:
            capabilities.append("⊞ tools")
        if "vision" in model.capabilities:
            capabilities.append("◉ vision")
        if not capabilities:
            capabilities.append("○ chat")
        size = ""
        if model.parameter_size:
            size = model.parameter_size.strip()
            integer_size = re.fullmatch(r"(\d+)(?:\.0+)?[bB]", size)
            size = f"{integer_size.group(1)}b" if integer_size else size
        return "  ".join(capabilities) + (f"  ·  {size}" if size else "")

    def compose(self) -> ComposeResult:
        with Vertical(id="model-box"):
            yield Label("Select model", classes="modal-title")
            yield Input(placeholder="Type to filter models…  e.g. qwen", id="model-search")
            yield ListView(*[
                ListItem(Label(
                    model.display_name + f"\n[dim]{self._capabilities(model)}[/]"
                ), id=f"model-{index}")
                for index, model in enumerate(self._filtered)
            ], id="model-list")

    def on_mount(self) -> None:
        box = self.query_one("#model-box")
        box.styles.opacity = 0.0
        box.styles.animate("opacity", 1.0, duration=0.18, easing="out_cubic")
        self.query_one("#model-search", Input).focus()
        self._select_current()

    def _select_current(self) -> None:
        try:
            view = self.query_one("#model-list", ListView)
        except Exception:
            return
        selected = self.pending or self.current
        names = [model.name for model in self._filtered]
        if selected in names:
            view.index = names.index(selected)
        elif names:
            view.index = 0

    async def _rebuild_models(self, query: str) -> None:
        q = _normalize_search(query)
        if q:
            self._filtered = [
                model for model in self.models
                if q in _normalize_search(f"{model.display_name} {model.name}")
            ]
        else:
            self._filtered = list(self.models)
        view = self.query_one("#model-list", ListView)
        await view.clear()
        for index, model in enumerate(self._filtered):
            await view.append(
                ListItem(
                    Label(model.display_name + f"\n[dim]{self._capabilities(model)}[/]"),
                    id=f"model-{index}",
                )
            )
        self._select_current()

    @on(Input.Changed, "#model-search")
    async def _filter_changed(self, event: Input.Changed) -> None:
        await self._rebuild_models(event.value)

    @on(Input.Submitted, "#model-search")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        view = self.query_one("#model-list", ListView)
        item = view.highlighted_child
        if item is not None and item.id:
            try:
                self.dismiss(self._filtered[int(item.id.removeprefix("model-"))].name)
                return
            except (ValueError, IndexError):
                pass
        if len(self._filtered) == 1:
            self.dismiss(self._filtered[0].name)

    def _move_highlight(self, direction: int) -> None:
        try:
            view = self.query_one("#model-list", ListView)
        except Exception:
            return
        if not self._filtered:
            return
        view.index = (view.index + direction) % len(self._filtered)

    def key_up(self) -> None:
        self._move_highlight(-1)

    def key_down(self) -> None:
        self._move_highlight(1)

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            try:
                self.dismiss(self._filtered[int(event.item.id.removeprefix("model-"))].name)
            except (ValueError, IndexError):
                return

    def key_escape(self) -> None:
        self.dismiss(None)


class CommandMenu(EscapeModalScreen):
    """Minimal command menu opened from Ctrl+P, with global static search."""

    CSS = """
    CommandMenu { align: center middle; background: #000000 75%; }
    #command-menu {
        width: 58; height: auto; max-height: 80%; padding: 1 2; background: $surface;
        border: none;
    }
    #command-title { height: 2; color: $text; text-style: bold; }
    #command-search { margin-bottom: 1; border: tall #333333; background: $background; }
    #command-search:focus { border: tall #a0a0a0; }
    #command-help { height: 1; color: #666666; margin-bottom: 1; }
    #command-list { height: auto; max-height: 21; background: $surface; border: none; }
    #command-list:focus { border: none; }
    #command-list ListItem { height: 3; padding: 1 1; color: #bdbdbd; background: $surface; }
    #command-list ListItem:hover { background: #303030; color: #ffffff; }
    #command-list:focus > ListItem.-highlight, #command-list > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    #command-list ListItem.-locked { color: #5f5f5f; background: $surface; }
    .command-key { color: $text-muted; }
    """

    ITEMS = (
        ("model", "Model", "select a model or change its settings"),
        ("sessions", "Sessions", "new, switch, rename, pin, or delete"),
        ("permissions", "Permissions", "per-session allow, ask, or deny"),
        ("workspace", "Change workspace", "open another folder"),
        ("files", "Show / hide sidebar", "files and session information"),
        ("commands", "Commands", "slash commands and keyboard shortcuts"),
        ("artinium", "Artinium", "themes, updates, and info"),
    )

    # Static searchable index: top-level + subsections. Intentionally excludes
    # dynamic names (session titles, model names) — those have their own local
    # search inside SessionScreen / ModelScreen.
    SEARCH_INDEX: tuple[tuple[str, str, str, str], ...] = (
        ("model", "Model", "select a model or change its settings", "model modelo cambiar"),
        ("model-change", "Change model", "select a detected model", "model modelo cambiar change select"),
        ("model-settings", "Model settings", "reasoning, temperature, and output", "model settings razonamiento temperatura salida razonar"),
        ("model-context", "Context settings", "window and compaction", "context contexto ventana compactar compaction ventana"),
        ("sessions", "Sessions", "new, switch, rename, pin, or delete", "session sesion sesiones cambiar nueva renombrar"),
        ("sessions-new", "New session", "start a fresh conversation", "session sesion nueva new fresh"),
        ("permissions", "Permissions", "per-session allow, ask, or deny", "permissions permisos herramientas session sesion"),
        ("workspace", "Change workspace", "open another folder", "workspace carpeta folder proyecto"),
        ("files", "Show / hide sidebar", "files and session information", "sidebar files archivos barra lateral mostrar ocultar"),
        ("commands", "Commands", "slash commands and keyboard shortcuts", "commands comandos slash atajos shortcuts"),
        ("compact", "Compact now", "summarize older messages", "compact context contexto resumir compactar"),
        ("autocompact", "Auto compact", "off, 70, 80, or 90", "autocompact automatico contexto"),
        ("undo", "Undo", "revert latest file change", "undo deshacer revertir cambio"),
        ("changes", "Changes", "review session file changes", "changes cambios historial"),
        ("artinium", "Artinium", "themes, permissions, updates, and info", "artinium config info"),
        ("themes", "Themes", "change interface theme", "theme tema apariencia"),
        ("updates", "Updates", "check for updates", "update actualizar version"),
    )

    _ALLOWED_WHILE_WORKING = {"files", "model", "model-change", "commands"}

    def __init__(self, *, working: bool = False):
        super().__init__()
        self.working = working
        self._filtered: list[tuple[str, str, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="command-menu"):
            yield Static("Menu", id="command-title")
            yield Input(placeholder="Type to search…  e.g. contexto", id="command-search")
            if self.working:
                yield Static("Model selection and the sidebar remain available", id="command-help")
            yield ListView(*self._build_items(""), id="command-list")

    def _build_items(self, query: str) -> list[ListItem]:
        query = query.strip()
        if not query:
            self._filtered = [(key, title, detail) for key, title, detail in self.ITEMS]
        else:
            self._filtered = [
                (key, title, detail)
                for key, title, detail, keywords in self.SEARCH_INDEX
                if _search_matches(query, title, detail, keywords)
            ]
        items: list[ListItem] = []
        seen_keys: set[str] = set()
        for key, title, detail in self._filtered:
            # A menu action has one stable widget ID. Keep the runtime safe if
            # a future search-index entry accidentally repeats an action.
            if key in seen_keys:
                continue
            seen_keys.add(key)
            locked = self.working and key not in self._ALLOWED_WHILE_WORKING
            label = (
                f"{title}\n[dim]{'Wait for current task to finish' if locked else detail}[/]"
            )
            items.append(
                ListItem(
                    Label(label),
                    id=f"command-{key}",
                    classes="command-item -locked" if locked else "command-item",
                    disabled=locked,
                )
            )
        return items

    async def _rebuild(self, query: str) -> None:
        view = self.query_one("#command-list", ListView)
        await view.clear()
        for item in self._build_items(query):
            await view.append(item)
        if self._filtered:
            view.index = 0

    def on_mount(self) -> None:
        menu = self.query_one("#command-menu")
        menu.styles.opacity = 0.0
        menu.styles.animate("opacity", 1.0, duration=0.14, easing="out_cubic")
        self.query_one("#command-search", Input).focus()

    @on(Input.Changed, "#command-search")
    async def _search_changed(self, event: Input.Changed) -> None:
        await self._rebuild(event.value)

    @on(Input.Submitted, "#command-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        view = self.query_one("#command-list", ListView)
        item = view.highlighted_child
        if item is not None and item.id and not item.disabled:
            self.dismiss(item.id.removeprefix("command-"))
        elif self._filtered:
            # Single unambiguous match: select it even without highlight.
            for key, _, _ in self._filtered:
                if key not in self._ALLOWED_WHILE_WORKING and self.working:
                    continue
                self.dismiss(key)
                return

    def _move_highlight(self, direction: int) -> None:
        try:
            view = self.query_one("#command-list", ListView)
        except Exception:
            return
        if not self._filtered:
            return
        view.index = (view.index + direction) % len(self._filtered)

    def key_up(self) -> None:
        self._move_highlight(-1)

    def key_down(self) -> None:
        self._move_highlight(1)

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("command-"))

    def key_escape(self) -> None:
        self.dismiss(None)


class ModelSettingsScreen(EscapeModalScreen):
    """Keyboard-first, safe configuration of Ollama request behaviour."""

    CSS = """
    ModelSettingsScreen { align: center middle; background: #000000 75%; }
    #settings-box { width: 64; height: auto; padding: 1 2; background: $surface; border: none; }
    #settings-title { height: 2; color: $text; text-style: bold; }
    #settings-help { height: 2; color: $text-muted; margin-bottom: 1; }
    .setting-row { height: 3; padding: 1; color: $text-muted; background: $surface; }
    .setting-row.-selected { background: #303030; color: #ffffff; }
    """

    _fields = ("reasoning", "temperature", "max_output_tokens")
    _temperatures = (None, 0.0, 0.2, 0.7, 1.0)
    _outputs = (None, 256, 512, 1024, 2048, 4096)

    def __init__(
        self,
        settings: ModelSettings,
        model: ModelInfo,
        on_change: Callable[[ModelSettings], None],
    ):
        super().__init__()
        self.settings = ModelSettings(
            reasoning=settings.reasoning,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            context_window=settings.context_window,
            auto_compact_threshold=settings.auto_compact_threshold,
        )
        self.model = model
        self.on_change = on_change
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-box"):
            yield Static("Model settings", id="settings-title")
            yield Static("←→ change  ·  R reset", id="settings-help")
            for field in self._fields:
                yield Static("", id=f"setting-{field}", classes="setting-row", markup=True)

    def on_mount(self) -> None:
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        reasoning = (
            {"auto": "Automatic", "enabled": "Enabled", "disabled": "Disabled"}[self.settings.reasoning]
            if self.model.supports_reasoning else "Not supported by this model"
        )
        max_output = "Default" if self.settings.max_output_tokens is None else f"{self.settings.max_output_tokens:,}"
        values = {
            "reasoning": reasoning,
            "temperature": "Default" if self.settings.temperature is None else str(self.settings.temperature),
            "max_output_tokens": max_output,
        }
        labels = {
            "reasoning": "Reasoning",
            "temperature": "Temperature",
            "max_output_tokens": "Max output tokens",
        }
        for index, field in enumerate(self._fields):
            row = self.query_one(f"#setting-{field}", Static)
            row.update(f"{labels[field]:<22}[#dedede]{values[field]}[/]")
            row.set_class(index == self.selected, "-selected")

    @staticmethod
    def _cycle(values: tuple[Any, ...], current: Any, direction: int) -> Any:
        try:
            index = values.index(current)
        except ValueError:
            index = 0
        return values[(index + direction) % len(values)]

    def key_up(self) -> None:
        self.selected = (self.selected - 1) % len(self._fields)
        self._refresh_rows()

    def key_down(self) -> None:
        self.selected = (self.selected + 1) % len(self._fields)
        self._refresh_rows()

    def key_left(self) -> None:
        self._change(-1)

    def key_right(self) -> None:
        self._change(1)

    def _change(self, direction: int) -> None:
        field = self._fields[self.selected]
        if field == "reasoning":
            if not self.model.supports_reasoning:
                return
            choices = ("auto", "enabled", "disabled")
            self.settings.reasoning = self._cycle(choices, self.settings.reasoning, direction)
        elif field == "temperature":
            self.settings.temperature = self._cycle(self._temperatures, self.settings.temperature, direction)
        elif field == "max_output_tokens":
            self.settings.max_output_tokens = self._cycle(self._outputs, self.settings.max_output_tokens, direction)
        self.on_change(self.settings)
        self._refresh_rows()

    def key_r(self) -> None:
        self.settings.reasoning = "auto"
        self.settings.temperature = None
        self.settings.max_output_tokens = None
        self.on_change(self.settings)
        self._refresh_rows()

    def key_enter(self) -> None:
        self._change(1)

    def key_escape(self) -> None:
        self.dismiss(None)


class AutoCompactScreen(EscapeModalScreen):
    """Choose when Artinium should summarize older context automatically."""

    CSS = """
    AutoCompactScreen { align: center middle; background: #000000 75%; }
    #auto-compact-box { width: 58; height: auto; padding: 1 2; background: $surface; border: none; }
    #auto-compact-title { height: 2; color: $text; text-style: bold; }
    #auto-compact-help { height: 2; color: $text-muted; margin-bottom: 1; }
    #auto-compact-list { height: auto; background: $surface; border: none; }
    #auto-compact-list:focus { border: none; }
    #auto-compact-list ListItem { height: 3; padding: 1; color: #bdbdbd; background: $surface; }
    #auto-compact-list:focus > ListItem.-highlight, #auto-compact-list > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    """

    OPTIONS = (None, 70, 80, 90)

    def __init__(self, current: int | None):
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="auto-compact-box"):
            yield Static("Auto compact", id="auto-compact-title")
            yield ListView(*[
                ListItem(
                    Label(
                        f"{'●  ' if value == self.current else '   '}{'Off' if value is None else f'{value}% of context window'}"
                    ),
                    id=f"auto-compact-{'off' if value is None else value}",
                )
                for value in self.OPTIONS
            ], id="auto-compact-list")

    def on_mount(self) -> None:
        view = self.query_one(ListView)
        view.focus()
        try:
            view.index = self.OPTIONS.index(self.current)
        except ValueError:
            view.index = self.OPTIONS.index(80)

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if not event.item.id:
            return
        value = event.item.id.removeprefix("auto-compact-")
        self.dismiss("off" if value == "off" else int(value))

    def key_escape(self) -> None:
        self.dismiss(None)


class ContextSettingsScreen(EscapeModalScreen):
    """All controls that change how conversation context is retained."""

    CSS = """
    ContextSettingsScreen { align: center middle; background: #000000 75%; }
    #context-settings-box { width: 64; height: auto; padding: 1 2; background: $surface; border: none; }
    #context-settings-title { height: 2; color: $text; text-style: bold; }
    #context-settings-help { height: 2; color: $text-muted; margin-bottom: 1; }
    .context-setting-row { height: 3; padding: 1; color: $text-muted; background: $surface; }
    .context-setting-row.-selected { background: #303030; color: #ffffff; }
    """

    _fields = ("context_window", "auto_compact", "compact_now")

    def __init__(
        self,
        settings: ModelSettings,
        model: ModelInfo,
        on_change: Callable[[ModelSettings], None],
        on_compact: Callable[[], None],
    ):
        super().__init__()
        self.settings = ModelSettings(
            reasoning=settings.reasoning,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            context_window=settings.context_window,
            auto_compact_threshold=settings.auto_compact_threshold,
        )
        self.model = model
        self.on_change = on_change
        self.on_compact = on_compact
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="context-settings-box"):
            yield Static("Context settings", id="context-settings-title")
            yield Static("←→ change  ·  R reset", id="context-settings-help")
            for field in self._fields:
                yield Static("", id=f"context-{field}", classes="context-setting-row", markup=True)

    def on_mount(self) -> None:
        self._refresh_rows()

    def _context_options(self) -> tuple[int | None, ...]:
        options = [value for value in CONTEXT_WINDOWS if value is None or value <= self.model.context_length]
        if self.model.context_length >= 4096 and self.model.context_length not in options:
            options.append(self.model.context_length)
        return tuple(options)

    @staticmethod
    def _cycle(values: tuple[Any, ...], current: Any, direction: int) -> Any:
        try:
            index = values.index(current)
        except ValueError:
            index = 0
        return values[(index + direction) % len(values)]

    def _refresh_rows(self) -> None:
        if self.settings.context_window is None:
            selected = self.settings.automatic_context_window(self.model.context_length)
            ram_gib = total_memory_gib()
            ram = f" · {ram_gib:.0f} GB RAM" if ram_gib is not None else ""
            context = f"Automatic ({selected:,}{ram})"
        else:
            context = f"{min(self.settings.context_window, self.model.context_length):,} tokens"
        automatic = "Off" if self.settings.auto_compact_threshold is None else f"At {self.settings.auto_compact_threshold}%"
        values = {
            "context_window": context,
            "auto_compact": automatic,
            "compact_now": "Summarize older messages",
        }
        labels = {
            "context_window": "Context window",
            "auto_compact": "Automatic compaction",
            "compact_now": "Compact now",
        }
        for index, field in enumerate(self._fields):
            row = self.query_one(f"#context-{field}", Static)
            row.update(f"{labels[field]:<24}[#dedede]{values[field]}[/]")
            row.set_class(index == self.selected, "-selected")

    def key_up(self) -> None:
        self.selected = (self.selected - 1) % len(self._fields)
        self._refresh_rows()

    def key_down(self) -> None:
        self.selected = (self.selected + 1) % len(self._fields)
        self._refresh_rows()

    def _change(self, direction: int) -> None:
        field = self._fields[self.selected]
        if field == "context_window":
            self.settings.context_window = self._cycle(self._context_options(), self.settings.context_window, direction)
        elif field == "auto_compact":
            self.settings.auto_compact_threshold = self._cycle(AutoCompactScreen.OPTIONS, self.settings.auto_compact_threshold, direction)
        else:
            self.on_compact()
            return
        self.on_change(self.settings)
        self._refresh_rows()

    def key_left(self) -> None:
        self._change(-1)

    def key_right(self) -> None:
        self._change(1)

    def key_enter(self) -> None:
        self._change(1)

    def key_r(self) -> None:
        self.settings.context_window = None
        self.settings.auto_compact_threshold = 80
        self.on_change(self.settings)
        self._refresh_rows()

    def key_escape(self) -> None:
        self.dismiss(None)


class WorkspaceScreen(EscapeModalScreen):
    CSS = """
    WorkspaceScreen { align: center middle; background: #000000 75%; }
    #workspace-box { width: 72; height: 13; border: none; background: $surface; padding: 1 2; }
    .modal-kicker { color: $text-muted; height: 1; }
    .modal-title { color: $text; text-style: bold; margin-top: 1; height: 2; }
    .modal-help { color: #666666; height: 2; }
    #workspace-path { margin-top: 1; border: tall #333333; background: $background; }
    #workspace-path:focus { border: tall #a0a0a0; }
    #open-workspace { margin-top: 1; min-width: 16; align-horizontal: right; background: #dddddd; color: $background; }
    """

    def __init__(self, current: Path):
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="workspace-box"):
            yield Label("WORKSPACE", classes="modal-kicker")
            yield Label("Change workspace", classes="modal-title")
            yield Label("The conversation is cleared to avoid mixing projects.", classes="modal-help")
            yield Input(value=str(self.current), id="workspace-path")
            yield Button("Open workspace  →", variant="primary", id="open-workspace")

    def on_mount(self) -> None:
        box = self.query_one("#workspace-box")
        box.styles.opacity = 0.0
        box.styles.animate("opacity", 1.0, duration=0.18, easing="out_cubic")
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    @on(Button.Pressed, "#open-workspace")
    def open_pressed(self) -> None:
        self.dismiss(self.query_one(Input).value)

    def key_escape(self) -> None:
        self.dismiss(None)


class SessionScreen(EscapeModalScreen):
    """A compact session manager inspired by terminal-first coding agents."""

    CSS = """
    SessionScreen { align: center middle; background: #000000 75%; }
    #session-box { width: 72; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #session-title { height: 2; color: $text; text-style: bold; }
    #session-search { margin-bottom: 1; border: tall #333333; background: $background; }
    #session-search:focus { border: tall #a0a0a0; }
    #session-help { height: 1; color: $text-muted; margin-bottom: 1; }
    #session-list { height: auto; max-height: 20; background: $surface; border: none; }
    #session-list:focus { border: none; }
    #session-list ListItem { height: 3; padding: 1; color: #bdbdbd; background: $surface; }
    #session-list:focus > ListItem.-highlight, #session-list > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    #session-list .session-group { height: 2; padding: 1 1 0 1; color: #666666; background: $surface; text-style: bold; }
    """

    BINDINGS = [
        # Single letters must keep typing in the filter; actions use Ctrl.
        Binding("ctrl+n", "new_session", "New", show=False, priority=True),
        Binding("ctrl+r", "rename_session", "Rename", show=False, priority=True),
        Binding("ctrl+s", "pin_session", "Pin", show=False, priority=True),
        Binding("ctrl+x", "delete_session", "Delete", show=False, priority=True),
    ]

    def __init__(self, sessions: list[SessionRecord], current_id: str):
        super().__init__()
        self.sessions = sessions
        self.current_id = current_id
        self._filtered: list[SessionRecord] = list(sessions)

    def _entries(self, sessions: list[SessionRecord], *, grouped: bool) -> list[ListItem]:
        if not grouped:
            return [
                ListItem(
                    Label(
                        f"{'●' if session.id == self.current_id else ' '}"
                        f"  {'★  ' if session.pinned else '    '}{session.title}\n"
                        f"[dim]{len(session.history)} messages[/]"
                    ),
                    id=f"session-{session.id}",
                )
                for session in sessions
            ]
        pinned = [session for session in sessions if session.pinned]
        regular = [session for session in sessions if not session.pinned]
        entries: list[ListItem] = []
        if pinned:
            entries.append(ListItem(Label("PINNED"), classes="session-group", disabled=True))
            entries.extend(
                ListItem(
                    Label(
                        f"{'●' if session.id == self.current_id else ' '}  ★  {session.title}\n"
                        f"[dim]{len(session.history)} messages[/]"
                    ),
                    id=f"session-{session.id}",
                )
                for session in pinned
            )
        if regular:
            entries.append(ListItem(Label("SESSIONS"), classes="session-group", disabled=True))
            entries.extend(
                ListItem(
                    Label(
                        f"{'●' if session.id == self.current_id else ' '}     {session.title}\n"
                        f"[dim]{len(session.history)} messages[/]"
                    ),
                    id=f"session-{session.id}",
                )
                for session in regular
            )
        return entries

    def compose(self) -> ComposeResult:
        with Vertical(id="session-box"):
            yield Static("Sessions", id="session-title")
            yield Input(placeholder="Type to filter sessions…", id="session-search")
            yield Static("^N new  ·  ^R rename  ·  ^S pin  ·  ^X delete", id="session-help")
            yield ListView(*self._entries(self.sessions, grouped=True), id="session-list")

    def on_mount(self) -> None:
        self.query_one("#session-search", Input).focus()

    async def _rebuild(self, query: str) -> None:
        q = _normalize_search(query)
        if q:
            self._filtered = [s for s in self.sessions if q in _normalize_search(s.title)]
        else:
            self._filtered = list(self.sessions)
        view = self.query_one("#session-list", ListView)
        await view.clear()
        for item in self._entries(self._filtered, grouped=not bool(q)):
            await view.append(item)
        if self._filtered:
            # Highlight current session if visible, else first match.
            ids = [f"session-{s.id}" for s in self._filtered]
            current = f"session-{self.current_id}"
            try:
                view.index = ids.index(current) if not q else 0
                if q:
                    # When filtering, first match is the expected target.
                    view.index = 0
            except ValueError:
                view.index = 0

    @on(Input.Changed, "#session-search")
    async def _search_changed(self, event: Input.Changed) -> None:
        await self._rebuild(event.value)

    @on(Input.Submitted, "#session-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        view = self.query_one("#session-list", ListView)
        item = view.highlighted_child
        if item is not None and item.id and not item.disabled:
            self.dismiss(f"select:{item.id.removeprefix('session-')}")

    def _move_highlight(self, direction: int) -> None:
        try:
            view = self.query_one("#session-list", ListView)
        except Exception:
            return
        # Count only selectable rows (skip PINNED/SESSIONS headers).
        selectable = sum(1 for _ in view.query(ListItem) if not _.disabled)
        if selectable == 0:
            return
        view.index = (view.index + direction) % len(list(view.query(ListItem)))

    def key_up(self) -> None:
        self._move_highlight(-1)

    def key_down(self) -> None:
        self._move_highlight(1)

    def _selected_id(self) -> str | None:
        view = self.query_one(ListView)
        item = view.highlighted_child
        if item and item.id:
            return item.id.removeprefix("session-")
        return self._filtered[0].id if self._filtered else None

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(f"select:{event.item.id.removeprefix('session-')}")

    def action_new_session(self) -> None:
        self.dismiss("new")

    def action_rename_session(self) -> None:
        if selected := self._selected_id():
            self.dismiss(f"rename:{selected}")

    def action_pin_session(self) -> None:
        if selected := self._selected_id():
            self.dismiss(f"pin:{selected}")

    def action_delete_session(self) -> None:
        if selected := self._selected_id():
            self.dismiss(f"delete:{selected}")

    def key_escape(self) -> None:
        self.dismiss(None)


class RenameSessionScreen(EscapeModalScreen):
    CSS = """
    RenameSessionScreen { align: center middle; background: #000000 75%; }
    #rename-box { width: 58; height: 10; padding: 1 2; background: $surface; border: none; }
    #rename-title { height: 2; color: $text; text-style: bold; }
    #rename-input { border: tall #555555; background: $background; }
    #rename-input:focus { border: tall #a0a0a0; }
    """

    def __init__(self, title: str):
        super().__init__()
        self.title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-box"):
            yield Static("Rename session", id="rename-title")
            yield Input(value=self.title, placeholder="Session name", id="rename-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def key_escape(self) -> None:
        self.dismiss(None)


class QueueScreen(EscapeModalScreen):
    """Select any queued prompt to edit or remove without stopping work."""

    CSS = """
    QueueScreen { align: center middle; background: #000000 75%; }
    #queue-box { width: 76; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #queue-title { height: 2; color: $text; text-style: bold; }
    #queue-help { height: 1; color: $text-muted; margin-bottom: 1; }
    #queued-list { height: auto; max-height: 20; background: $surface; border: none; }
    #queued-list:focus { border: none; }
    #queued-list ListItem { height: 3; padding: 1; color: #bdbdbd; background: $surface; }
    #queued-list ListItem:hover { background: #303030; color: #ffffff; }
    #queued-list:focus > ListItem.-highlight, #queued-list > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    """

    BINDINGS = [
        Binding("d", "delete_selected", "Delete", show=False, priority=True),
        Binding("delete", "delete_selected", "Delete", show=False, priority=True),
    ]

    def __init__(self, prompts: list[str]):
        super().__init__()
        self.prompts = prompts

    def compose(self) -> ComposeResult:
        with Vertical(id="queue-box"):
            yield Static("Queued messages", id="queue-title")
            yield Static("D delete", id="queue-help")
            yield ListView(*[
                ListItem(Label(f"{index}.  {prompt}"), id=f"queue-{index - 1}")
                for index, prompt in enumerate(self.prompts, 1)
            ], id="queued-list")

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def _selected_index(self) -> int | None:
        item = self.query_one(ListView).highlighted_child
        if item and item.id:
            return int(item.id.removeprefix("queue-"))
        return 0 if self.prompts else None

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(f"edit:{event.item.id.removeprefix('queue-')}")

    def action_delete_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            self.dismiss(f"delete:{index}")

    def key_escape(self) -> None:
        self.dismiss(None)


class EditQueueScreen(EscapeModalScreen):
    CSS = """
    EditQueueScreen { align: center middle; background: #000000 75%; }
    #edit-queue-box { width: 76; height: 10; padding: 1 2; background: $surface; border: none; }
    #edit-queue-title { height: 2; color: $text; text-style: bold; }
    #edit-queue-help { height: 1; color: $text-muted; }
    #edit-queue-input { margin-top: 1; border: tall #555555; background: $background; }
    #edit-queue-input:focus { border: tall #a0a0a0; }
    """

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-queue-box"):
            yield Static("Edit queued message", id="edit-queue-title")
            yield Input(value=self.prompt, id="edit-queue-input")

    def on_mount(self) -> None:
        input_widget = self.query_one(Input)
        input_widget.focus()
        input_widget.cursor_position = len(input_widget.value)

    @on(Input.Submitted)
    def submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def key_escape(self) -> None:
        self.dismiss(None)


class ChangeHistoryScreen(EscapeModalScreen):
    """Read-only session change history; only the newest entry is reversible."""

    CSS = """
    ChangeHistoryScreen { align: center middle; background: #000000 75%; }
    #changes-box { width: 68; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #changes-title { height: 2; color: $text; text-style: bold; }
    #changes-help { height: 2; color: $text-muted; }
    #changes-list { height: auto; max-height: 20; background: $surface; border: none; }
    #changes-list ListItem { height: 4; padding: 1; color: $text; background: $surface; }
    #changes-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    #changes-list ListItem.-locked { color: #666666; }
    """

    def __init__(self, records: list[Any]):
        super().__init__()
        self.records = list(reversed(records))

    def compose(self) -> ComposeResult:
        with Vertical(id="changes-box"):
            yield Static("Changes", id="changes-title")
            if not self.records:
                yield Static("No file changes in this session.")
                return
            yield ListView(*[
                ListItem(
                    Label(
                        f"{record.tool.replace('_', ' ').capitalize()}  {record.path}\n"
                        f"[dim]{'Latest change · can be reverted' if index == 0 else 'Earlier change · restore newer changes first'}[/]"
                    ),
                    id=f"change-{record.id}",
                    disabled=index != 0,
                    classes="-locked" if index else "",
                )
                for index, record in enumerate(self.records)
            ], id="changes-list")

    def on_mount(self) -> None:
        if self.records:
            self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("change-"))

    def key_escape(self) -> None:
        self.dismiss(None)


class ArrowConfirmScreen(EscapeModalScreen):
    """Confirmation controls designed for arrow keys, not tab traversal."""

    BINDINGS = [
        Binding("left", "focus_previous_choice", show=False, priority=True),
        Binding("right", "focus_next_choice", show=False, priority=True),
        Binding("tab", "ignore_tab", show=False, priority=True),
    ]

    def _move_button_focus(self, direction: int) -> None:
        buttons = list(self.query(Button))
        if not buttons:
            return
        focused = self.app.focused
        try:
            index = buttons.index(focused)
        except ValueError:
            index = 0
        buttons[(index + direction) % len(buttons)].focus()

    def action_focus_previous_choice(self) -> None:
        self._move_button_focus(-1)

    def action_focus_next_choice(self) -> None:
        self._move_button_focus(1)

    def action_ignore_tab(self) -> None:
        """Keep Tab from turning these two choices into a form."""


class DeleteSessionScreen(ArrowConfirmScreen):
    CSS = """
    DeleteSessionScreen { align: center middle; background: #000000 80%; }
    #delete-session-box { width: 58; height: auto; padding: 1 2; background: $surface; border: none; }
    #delete-session-title { height: 2; color: $text; text-style: bold; }
    #delete-session-detail { height: 2; color: $text-muted; }
    #delete-session-actions { height: 4; align-horizontal: right; padding-top: 1; }
    #delete-session-actions Button { margin-left: 1; min-width: 13; border: none; background: transparent; color: $text-muted; }
    #delete-session-actions Button:focus { background: #dddddd; color: $background; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-session-box"):
            yield Static("Delete this session?", id="delete-session-title")
            yield Static("This cannot be undone.", id="delete-session-detail")
            with Horizontal(id="delete-session-actions"):
                yield Button("Cancel", id="delete-session-deny")
                yield Button("Delete", id="delete-session-allow")

    def on_mount(self) -> None:
        self.query_one("#delete-session-deny", Button).focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "delete-session-allow")

    def key_escape(self) -> None:
        self.dismiss(False)


class ConfirmScreen(ArrowConfirmScreen):
    CSS = """
    ConfirmScreen { align: center middle; background: #000000 80%; }
    #confirm-box { width: 72; height: auto; border: none; background: $surface; padding: 1 2; }
    #danger-kicker { color: #999999; text-style: bold; height: 1; }
    #danger-title { color: $text; text-style: bold; margin: 1 0; }
    #command-preview { background: $background; border: tall #333333; padding: 1 2; margin: 1 0; color: #dddddd; height: auto; }
    #danger-reason { color: $text-muted; margin-bottom: 1; }
    #confirm-actions { height: 4; align-horizontal: right; padding-top: 1; }
    #confirm-actions Button { margin-left: 1; min-width: 15; border: none; background: transparent; color: $text-muted; }
    #confirm-actions Button:focus { background: #dddddd; color: $background; text-style: bold; }
    """

    def __init__(self, request: ConfirmationRequest):
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label("PROTECTED ACTION", id="danger-kicker")
            yield Label("This command may modify the system", id="danger-title")
            yield Static(self.request.command, id="command-preview")
            yield Label(self.request.reason, id="danger-reason")
            with Horizontal(id="confirm-actions"):
                yield Button("Cancel  Esc", id="deny")
                yield Button("Run anyway", id="allow")

    def on_mount(self) -> None:
        box = self.query_one("#confirm-box")
        box.styles.opacity = 0.0
        box.styles.animate("opacity", 1.0, duration=0.18, easing="out_cubic")
        self.query_one("#deny", Button).focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def key_escape(self) -> None:
        self.dismiss(False)


class StopScreen(ArrowConfirmScreen):
    CSS = """
    StopScreen { align: center middle; background: #000000 80%; }
    #stop-box { width: 58; height: auto; padding: 1 2; background: $surface; border: none; }
    #stop-title { height: 2; color: $text; text-style: bold; }
    #stop-detail { height: 2; color: $text-muted; }
    #stop-actions { height: 4; align-horizontal: right; padding-top: 1; }
    #stop-actions Button { margin-left: 1; min-width: 13; border: none; background: transparent; color: $text-muted; }
    #stop-actions Button:focus { background: #dddddd; color: $background; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-box"):
            yield Static("Stop current task?", id="stop-title")
            yield Static("The current generation and command will be stopped.", id="stop-detail")
            with Horizontal(id="stop-actions"):
                yield Button("Keep working", id="stop-deny")
                yield Button("Stop task", id="stop-allow")

    def on_mount(self) -> None:
        self.query_one("#stop-deny", Button).focus()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "stop-allow")

    def key_escape(self) -> None:
        self.dismiss(False)


class PermissionPromptScreen(EscapeModalScreen):
    """One clear decision for a mutating tool invocation."""

    CSS = """
    PermissionPromptScreen { align: center middle; background: #000000 80%; }
    #permission-prompt-box { width: 72; height: auto; padding: 1 2; background: $surface; border: none; }
    #permission-prompt-title { height: 2; color: $text; text-style: bold; }
    #permission-prompt-scope { height: 1; color: $text-muted; margin-bottom: 1; }
    #permission-prompt-detail { height: auto; max-height: 8; padding: 1 2; background: #151515; color: $text; }
    #permission-prompt-help { height: 2; margin-top: 1; color: $text-muted; }
    #permission-prompt-list { height: auto; border: none; background: $surface; }
    #permission-prompt-list ListItem { height: 3; padding: 1; background: $surface; color: $text; }
    #permission-prompt-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    """

    def __init__(self, request: PermissionRequest):
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-prompt-box"):
            yield Static(f"Allow {self.request.tool}?", id="permission-prompt-title")
            yield Static("This action only · change rules in Permissions", id="permission-prompt-scope")
            yield Static(self.request.summary, id="permission-prompt-detail")
            yield ListView(
                ListItem(Label("Allow once"), id="permission-allow-once"),
                ListItem(Label("Deny"), id="permission-deny"),
                id="permission-prompt-list",
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("permission-"))

    def key_escape(self) -> None:
        self.dismiss("deny")


class QuestionList(ListView):
    """Choice list that hands keyboard navigation to the custom answer field."""

    def action_cursor_down(self) -> None:
        if self.index is not None and self.index >= len(self) - 1:
            self.screen.query_one("#question-answer", Input).focus()
            return
        super().action_cursor_down()


class QuestionAnswer(Input):
    """Free-form answer field that can return to the choices with the keyboard."""

    BINDINGS = [Binding("up", "return_to_choices", show=False, priority=True)]

    def action_return_to_choices(self) -> None:
        if self.value:
            return
        if not self.screen.query(QuestionList):
            return
        choices = self.screen.query_one(QuestionList)
        choices.index = len(choices) - 1
        choices.focus()


class QuestionScreen(EscapeModalScreen):
    """Interactive model question with optional choices and a free-form answer."""

    CSS = """
    QuestionScreen { align: center middle; background: #000000 80%; }
    #question-box { width: 72; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #question-title { height: 2; color: $text; text-style: bold; }
    #question-text { height: auto; padding: 1 0; color: #d8d8d8; }
    #question-list { height: auto; max-height: 16; border: none; background: $surface; }
    #question-list ListItem { height: 3; padding: 1; color: $text; background: $surface; }
    #question-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    #question-answer { margin-top: 1; border: tall #333333; background: $background; }
    #question-help { height: 2; color: $text-muted; }
    """

    def __init__(self, request: QuestionRequest):
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="question-box"):
            yield Static("Artinium needs your input", id="question-title")
            yield Static(self.request.question, id="question-text")
            if self.request.options:
                yield QuestionList(*[
                    ListItem(Label(option), id=f"question-option-{index}")
                    for index, option in enumerate(self.request.options)
                ], id="question-list")
            yield QuestionAnswer(placeholder="Type another answer…", id="question-answer")

    def on_mount(self) -> None:
        if self.request.options:
            self.query_one(ListView).focus()
        else:
            self.query_one(Input).focus()

    @on(ListView.Selected)
    def option_selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            index = int(event.item.id.removeprefix("question-option-"))
            self.dismiss(self.request.options[index])

    @on(Input.Submitted)
    def answer_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def key_escape(self) -> None:
        self.dismiss(None)


class CommandsScreen(EscapeModalScreen):
    COMMANDS = (
        ("/model", "models and model settings"),
        ("/sessions", "manage conversations"),
        ("/workspace", "change working folder"),
        ("/sidebar", "show or hide sidebar"),
        ("/compact", "compact context now"),
        ("/context-summary", "show the current compacted memory"),
        ("/autocompact", "automatic compaction"),
        ("/undo", "revert the latest file change"),
        ("/changes", "review file changes in this session"),
        ("/permissions", "tool permissions"),
        ("/theme", "select interface theme"),
        ("/update", "check for updates"),
    )
    CSS = """
    CommandsScreen { align: center middle; background: #000000 75%; }
    #commands-box { width: 64; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #commands-title { height: 2; color: $text; text-style: bold; }
    #commands-help { height: 2; color: $text-muted; }
    #commands-list { height: auto; max-height: 22; border: none; background: $surface; }
    #commands-list ListItem { height: 3; padding: 1; color: $text; background: $surface; }
    #commands-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="commands-box"):
            yield Static("Commands", id="commands-title")
            yield ListView(*[
                ListItem(Label(f"{command}\n[dim]{description}[/]"), id=f"slash-{index}")
                for index, (command, description) in enumerate(self.COMMANDS)
            ], id="commands-list")

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(self.COMMANDS[int(event.item.id.removeprefix("slash-"))][0])


class ArtiniumScreen(EscapeModalScreen):
    ITEMS = (
        ("info", "About Artinium", "version and runtime information"),
        ("updates", "Updates", "check or install a newer version"),
        ("themes", "Themes", "automatic, dark, and light themes"),
    )
    CSS = """
    ArtiniumScreen { align: center middle; background: #000000 75%; }
    #artinium-box { width: 64; height: auto; padding: 1 2; background: $surface; border: none; }
    #artinium-title { height: 2; color: $text; text-style: bold; }
    #artinium-list { height: auto; border: none; background: $surface; }
    #artinium-list ListItem { height: 3; padding: 1; color: $text; background: $surface; }
    #artinium-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="artinium-box"):
            yield Static("Artinium", id="artinium-title")
            yield ListView(*[
                ListItem(Label(f"{title}\n[dim]{detail}[/]"), id=f"artinium-{key}")
                for key, title, detail in self.ITEMS
            ], id="artinium-list")

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("artinium-"))


class PermissionsScreen(EscapeModalScreen):
    LABELS = {"write_file": "Write files", "edit_file": "Edit files", "run_command": "Run shell commands"}
    CSS = """
    PermissionsScreen { align: center middle; background: #000000 75%; }
    #permissions-box { width: 66; height: auto; padding: 1 2; background: $surface; border: none; }
    #permissions-title { height: 2; color: $text; text-style: bold; }
    #permissions-help { height: 1; color: $text-muted; margin-bottom: 1; }
    #permissions-scope { height: 1; color: $text-muted; margin-top: 1; }
    .permission-row { height: 3; padding: 1; color: $text; }
    .permission-row.-selected { background: #303030; color: #ffffff; }
    """

    def __init__(self, session: SessionRecord, on_change: Callable[[], None]):
        super().__init__()
        self.session = session
        self.on_change = on_change
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="permissions-box"):
            yield Static("Tool permissions", id="permissions-title")
            yield Static("←→ change", id="permissions-help")
            for tool in MUTATING_TOOLS:
                yield Static("", id=f"policy-{tool}", classes="permission-row", markup=True)
            yield Static(
                "Session only · new sessions start at ask · reads always allowed",
                id="permissions-scope",
            )

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        for index, tool in enumerate(MUTATING_TOOLS):
            value = self.session.tool_permissions.get(tool, "ask").capitalize()
            row = self.query_one(f"#policy-{tool}", Static)
            row.update(f"{self.LABELS[tool]:<28}[#dedede]{value}[/]")
            row.set_class(index == self.selected, "-selected")

    def _change(self, direction: int) -> None:
        tool = MUTATING_TOOLS[self.selected]
        current = self.session.tool_permissions.get(tool, "ask")
        index = PERMISSION_VALUES.index(current) if current in PERMISSION_VALUES else 0
        self.session.tool_permissions[tool] = PERMISSION_VALUES[(index + direction) % len(PERMISSION_VALUES)]
        self.on_change()
        self._refresh()

    def key_up(self) -> None:
        self.selected = (self.selected - 1) % len(MUTATING_TOOLS)
        self._refresh()

    def key_down(self) -> None:
        self.selected = (self.selected + 1) % len(MUTATING_TOOLS)
        self._refresh()

    def key_left(self) -> None: self._change(-1)
    def key_right(self) -> None: self._change(1)
    def key_enter(self) -> None: self._change(1)


class ThemeScreen(EscapeModalScreen):
    CSS = """
    ThemeScreen { align: center middle; background: #000000 75%; }
    #theme-box { width: 60; height: auto; max-height: 78%; padding: 1 2; background: $surface; border: none; }
    #theme-title { height: 2; color: $text; text-style: bold; }
    #theme-help { height: 2; color: $text-muted; }
    #theme-list { height: auto; max-height: 22; border: none; background: $surface; }
    #theme-list ListItem { height: 3; padding: 1; color: $text; background: $surface; }
    #theme-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    """

    def __init__(self, themes: list[str], current: str):
        super().__init__()
        self.themes = ["automatic", *themes]
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-box"):
            yield Static("Theme", id="theme-title")
            yield ListView(*[
                ListItem(Label(f"{'●  ' if theme == self.current else '   '}{theme}"), id=f"theme-{index}")
                for index, theme in enumerate(self.themes)
            ], id="theme-list")

    def on_mount(self) -> None:
        view = self.query_one(ListView)
        view.focus()
        if self.current in self.themes:
            view.index = self.themes.index(self.current)

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(self.themes[int(event.item.id.removeprefix("theme-"))])


class UpdateScreen(EscapeModalScreen):
    CSS = """
    UpdateScreen { align: center middle; background: #000000 75%; }
    #update-box { width: 66; height: auto; padding: 1 2; background: $surface; border: none; }
    #update-title { height: 2; color: $text; text-style: bold; }
    #update-detail { height: auto; color: $text-muted; margin-bottom: 1; }
    #update-list { height: auto; border: none; background: $surface; }
    #update-list ListItem { height: 3; padding: 1; color: $text; background: $surface; }
    #update-list > ListItem.-highlight { background: #303030; color: #ffffff; }
    """

    def __init__(self, info: UpdateInfo | None):
        super().__init__()
        self.info = info

    def compose(self) -> ComposeResult:
        with Vertical(id="update-box"):
            yield Static("Updates", id="update-title")
            if self.info is None:
                detail = f"Installed: {__version__}\nCould not reach GitHub. You may be offline."
                actions = (("later", "Close"),)
            elif self.info.available:
                detail = f"Installed: {self.info.current}\nAvailable: {self.info.latest}"
                actions = (("install", "Install now"), ("later", "Later"), ("ignore", "Ignore this version"))
            else:
                detail = f"Artinium {self.info.current} is up to date."
                actions = (("later", "Close"),)
            yield Static(detail, id="update-detail")
            yield ListView(*[ListItem(Label(label), id=f"update-{key}") for key, label in actions], id="update-list")

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected)
    def selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            self.dismiss(event.item.id.removeprefix("update-"))


class InfoScreen(EscapeModalScreen):
    CSS = """
    InfoScreen { align: center middle; background: #000000 75%; }
    #info-box { width: 62; height: auto; padding: 1 2; background: $surface; border: none; }
    #info-title { height: 2; color: $text; text-style: bold; }
    #info-detail { height: auto; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="info-box"):
            yield Static("Artinium", id="info-title")
            yield Static(f"Version {__version__}\nLocal coding agent for Ollama", id="info-detail")


class ArtiumApp(App[None]):
    TITLE = "Artinium"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    $bg: $background;
    $line: $panel;
    $muted: $text-muted;

    Screen { background: $bg; color: $text; }
    ListView {
        scrollbar-color: #3a3a3a;
        scrollbar-background: $surface;
    }
    ListView:focus > ListItem.-highlight, ListView > ListItem.-highlight { background: #303030; color: #ffffff; text-style: none; }
    DirectoryTree > .tree--cursor { background: #303030; color: #ffffff; text-style: none; }
    #body { height: 1fr; }
    #sidebar {
        width: 28; min-width: 24; max-width: 34; background: $bg;
        border-right: solid #202020; transition: width 160ms out_cubic;
    }
    #sidebar.-hidden { width: 0; display: none; }
    #files-label { height: 3; padding: 1 2; color: #bdbdbd; text-style: bold; }
    DirectoryTree {
        height: 7fr; background: $bg; padding: 0 1;
        scrollbar-color: #333333; scrollbar-background: $bg;
        overflow-x: hidden;
    }
    DirectoryTree:hover { background: #0b0b0b; }
    #session-info {
        height: 3fr; padding: 1 2; background: $bg;
        border-top: solid #242424; color: #a8a8a8;
    }
    #chat-column { width: 1fr; background: $bg; }
    #chat {
        width: 1fr; height: 1fr; margin: 0 6 0 3; padding: 1 0; background: $bg;
        scrollbar-color: #333333; scrollbar-background: $bg;
    }
    .message { width: 1fr; height: auto; max-width: 100%; margin: 0; padding: 0; }
    .user-message {
        margin: 0 0 1 0; padding: 1 2; color: $text;
        max-width: 100%; width: 1fr; background: $boost; border: none;
    }
    .user-message.-scroll-gutter { margin-right: 1; }
    .assistant-message {
        width: 1fr; height: auto; margin: 0 0 1 0; padding: 0; color: #d4d4d4;
        background: $bg; border: none;
    }
    .assistant-message:focus { border: none; }
    .system-notice { width: 1fr; height: auto; margin: 0 0 1 0; padding: 0; color: #666666; background: $bg; border: none; }
    .system-notice.-warning { color: $text-muted; }
    .system-notice.-error { color: $text; text-style: bold; }
    .system-notice.-stats { color: #666666; }
    .tool-call {
        width: 1fr; height: auto; margin: 0 0 1 0; padding: 0; background: $bg;
        border: none; color: $text-muted;
    }
    .tool-call:focus-within { background: $bg; background-tint: transparent; }
    .tool-call CollapsibleTitle { padding: 0; background: $bg; color: $text-muted; }
    .tool-call CollapsibleTitle:hover { background: #101010; color: $text; }
    .tool-call CollapsibleTitle:focus { background: $boost; color: #d0d0d0; text-style: none; }
    .tool-call Contents { padding: 0; }
    .tool-call.-running { color: $text; }
    .tool-call.-complete { color: #888888; }
    .tool-call.-failed { color: $text; }
    .tool-detail { height: auto; padding: 0 2; background: $surface; color: #888888; }
    .tool-detail.-diff { padding: 1 2 0 2; background: #151515; }
    .thinking-block { width: 1fr; height: auto; margin: 0 0 1 0; padding: 0; background: $bg; color: $text-muted; }
    .thinking-block CollapsibleTitle { padding: 0; background: $bg; color: $text-muted; }
    .thinking-block CollapsibleTitle:hover { background: #101010; color: $text; }
    .thinking-block CollapsibleTitle:focus { background: $boost; color: #d0d0d0; text-style: none; }
    .thinking-block Contents { padding: 0; }
    .thinking-detail { height: auto; padding: 0 2; background: transparent; color: #636363; }
    #composer-shell {
        height: 6; margin: 0 6 2 3; padding: 1 2; background: $boost;
        border: none;
    }
    #queue-preview {
        display: none; height: auto; padding: 0; color: #b0b0b0;
        background: $boost; overflow: hidden;
    }
    #queue-preview.-visible { display: block; }
    #attachments-preview {
        display: none; height: auto; padding: 0 0 1 0; color: #b0b0b0;
        background: $boost; overflow: hidden;
    }
    #attachments-preview.-visible { display: block; }
    #composer-shell:focus-within { border: none; }
    #prompt { height: 3; border: none; background: transparent; padding: 0; color: $text; }
    #prompt:focus { border: none; }
    #composer-meta { height: 1; color: #555555; }
    #model-status { width: 1fr; color: $text-muted; }
    .agent-activity { width: 1fr; height: 1; margin: 0 0 1 0; color: $text; }
    .agent-activity.-warning { color: $text-muted; }
    .agent-activity.-confirm { color: #ff9f43; text-style: bold; }
    .agent-activity.-error { color: $text; }
    #send-hint { width: 1fr; text-align: right; color: #555555; }
    """
    BINDINGS = [
        Binding("ctrl+p", "menu", "Menu", priority=True),
        Binding("ctrl+o", "workspace", "Workspace", priority=True),
        Binding("ctrl+t", "toggle_tools", "Tools", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        # Capture Ctrl+C above modals too. The action first copies a selected
        # prompt or response, and only otherwise performs the terminal stop.
        Binding("ctrl+c", "interrupt_or_exit", "Stop", show=True, priority=True),
        Binding("ctrl+e", "edit_queue", "Edit queued message", show=False, priority=True),
        Binding("ctrl+d", "delete_queue", "Delete queued message", show=False, priority=True),
        Binding("escape", "escape", "", show=False),
    ]

    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = Workspace(workspace)
        self.client = OllamaClient()
        self.models: list[ModelInfo] = []
        self.model_settings_store = ModelSettingsStore()
        self.model_settings = self.model_settings_store.load()
        self.preferences_store = PreferencesStore()
        self.preferences = self.preferences_store.load()
        self.undo_manager = UndoManager(self.workspace)
        self.update_info: UpdateInfo | None = None
        self._update_task: asyncio.Task[None] | None = None
        self.agent: Agent | None = None
        self.session_store = SessionStore(self.workspace.root)
        self.active_session: SessionRecord | None = None
        self._generation_task: asyncio.Task[Any] | None = None
        self._queued_prompts: list[str] = []
        self._prompt_history: list[str] = []
        self._prompt_history_index: int | None = None
        self._prompt_history_draft = ""
        self._attachment_refs: dict[str, Path] = {}
        self._attachment_counter = 0
        self._title_seed: str | None = None
        self._title_tasks: set[asyncio.Task[None]] = set()
        self._last_autosave = 0.0
        self._tool_active = False
        self._loaded_models: set[str] = set()
        self._pending_model: ModelInfo | None = None
        self.show_tool_details = False
        self._stop_confirmation_open = False
        self._stop_armed = False
        self._exit_armed = False
        self._shutting_down = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("Files", id="files-label")
                yield WorkspaceTree(str(self.workspace.root), id="tree")
                yield Static("[bold]New session[/]\n\n[dim]CONTEXT[/]  0%\n[dim]TOKENS[/]   0 / —\n[dim]TOOLS[/]    0" + _system_lines(), id="session-info", markup=True)
            with Vertical(id="chat-column"):
                yield ChatLog(id="chat")
                with Vertical(id="composer-shell"):
                    yield Static("", id="queue-preview", markup=False)
                    yield Static("", id="attachments-preview", markup=True)
                    yield PromptInput(placeholder="Write a message…  ·  drop or paste images/files", id="prompt", disabled=True)
                    with Horizontal(id="composer-meta"):
                        yield Static("", id="model-status")
                        yield Static("Enter send  ·  Ctrl+P menu  ·  Ctrl+C exit", id="send-hint")

    def on_mount(self) -> None:
        self._apply_theme(self.preferences.theme, save=False)
        self.initialize()
        if self.preferences.check_updates and isinstance(self.client, OllamaClient):
            self._update_task = asyncio.create_task(self._check_updates(announce=True))

    @on(events.DescendantFocus)
    def keep_composer_focused(self, event: events.DescendantFocus) -> None:
        """Keep typing ready after mouse interaction with the transcript or tools."""
        if isinstance(self.screen, ModalScreen) or event.widget.id == "prompt":
            return
        self.call_after_refresh(self._restore_composer_focus)

    def _restore_composer_focus(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        prompt = self.query_one("#prompt", Input)
        if not prompt.disabled:
            prompt.focus()

    @on(events.Resize)
    def adapt_layout(self, event: events.Resize) -> None:
        """Keep the hierarchy useful on smaller terminal windows."""
        width = event.size.width
        if width < 88:
            self.query_one("#sidebar").add_class("-hidden")

    @work(exclusive=True)
    async def initialize(self) -> None:
        try:
            try:
                self.models = await self.client.list_models()
            except OllamaUnavailable:
                if not isinstance(self.client, OllamaClient):
                    raise
                self.query_one("#chat", ChatLog).write("Starting Ollama…", tone="warning")
                await self.client.start_server()
                self.models = await self.client.list_models()
            if not self.models:
                self.query_one("#chat", ChatLog).write("No models are installed. Run [bold]ollama pull qwen3.5:4b[/].", tone="warning")
                return
            sessions = self.session_store.load()
            restored_session = next(
                (session for session in sessions if session.id == self.session_store.active_session_id),
                sessions[0] if sessions else None,
            )
            model = next(
                (item for item in self.models if restored_session and item.name == restored_session.model_name),
                self.client.choose_model(self.models),
            )
            tools = ToolRegistry(
                self.workspace,
                self.confirm_command,
                authorize=self.authorize_tool,
                ask_question=self.ask_question,
                permission_policy=self._session_permission,
                undo=self.undo_manager,
            )
            self.agent = Agent(self.client, model, tools, settings=self.model_settings)
            self._loaded_models.add(model.name)
            self.active_session = restored_session or SessionRecord.new()
            recovered = bool(
                self.active_session.history
                or self.active_session.queued_prompts
                or self.active_session.draft
                or self.active_session.undo_records
            )
            if recovered:
                self.agent.restore(self.active_session.history, self.active_session.stats)
                self._queued_prompts = list(self.active_session.queued_prompts)
                self.undo_manager.restore(self.active_session.undo_records)
            await self._render_active_session()
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.value = self.active_session.draft
            prompt.focus()
            self._update_queue_ui()
            if recovered:
                self.query_one("#chat", ChatLog).write(
                    "Recovered the previous session. Any unfinished task was stopped; review its last action before continuing.",
                    tone="warning",
                )
            self._update_model_status()
            self._warn_if_no_tools()
            self.update_topbar()
        except (OllamaError, OllamaUnavailable) as exc:
            self.query_one("#chat", ChatLog).write(explain(exc, area="ollama"), tone="error")
        except Exception as exc:
            self.query_one("#chat", ChatLog).write(f"Startup error: {exc}", tone="error")

    @on(Input.Submitted, "#prompt")
    def prompt_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.agent is None:
            return
        event.input.value = ""
        if self.active_session:
            self.active_session.draft = ""
        if prompt.startswith("/"):
            self._run_slash_command(prompt)
            return
        self._remember_prompt(prompt)
        if self._generation_task and not self._generation_task.done():
            self._queued_prompts.append(prompt)
            self._update_queue_ui()
            self._save_active_session()
            return
        self._start_prompt(prompt)

    @on(Input.Changed, "#prompt")
    def prompt_changed(self, event: Input.Changed) -> None:
        if self.active_session:
            self.active_session.draft = event.value
            self._autosave_active_session()
        self._refresh_attachments_preview()

    def _run_slash_command(self, value: str) -> None:
        command, _, argument = value.partition(" ")
        command = command.lower()
        argument = argument.strip().lower()
        chat = self.query_one("#chat", ChatLog)
        working = bool(self._generation_task and not self._generation_task.done())
        idle_only = {"/settings", "/context", "/sessions", "/workspace", "/compact", "/autocompact"}
        if command in idle_only and working:
            chat.write("Wait for the current task to finish before changing this.", tone="warning")
            return
        if command in {"/help", "/commands"}:
            chat.write(
                "Commands\n"
                "/menu  open actions\n/model  change model\n/settings  model settings\n"
                "/context  context and compaction settings\n"
                "/sessions  manage sessions\n/workspace  change workspace\n"
                "/sidebar  show or hide sidebar\n"
                "/compact  summarize older context\n"
                "/context-summary  show the compacted context memory\n"
                "/autocompact [off|70|80|90]  set automatic compaction\n"
                "/undo  revert latest file change\n/changes  review session file changes\n/permissions  tool permissions\n"
                "/theme  interface theme\n/update  check for updates"
            )
        elif command == "/menu":
            self.action_menu()
        elif command == "/model":
            self.action_model_hub()
        elif command in {"/settings", "/model-settings"}:
            self.action_model_settings()
        elif command in {"/context", "/context-settings"}:
            self.action_context_settings()
        elif command == "/sessions":
            self.action_sessions()
        elif command == "/workspace":
            self.action_workspace()
        elif command in {"/sidebar", "/files"}:
            self.action_toggle_sidebar()
        elif command == "/compact":
            self.action_compact_context()
        elif command in {"/context-summary", "/summary"}:
            self.action_context_summary()
        elif command == "/autocompact":
            if argument:
                self._set_auto_compact_from_text(argument)
            else:
                self.action_auto_compact()
        elif command == "/undo":
            self.action_undo()
        elif command in {"/changes", "/history"}:
            self.action_changes()
        elif command == "/permissions":
            self.action_permissions()
        elif command == "/theme":
            self.action_themes()
        elif command in {"/update", "/updates"}:
            self.action_updates()
        elif command in {"/artinium", "/about"}:
            self.action_artinium()
        else:
            chat.write(f"Unknown command: {command}. Use /help to see commands.", tone="warning")

    def _set_auto_compact_from_text(self, value: str) -> None:
        if value == "off":
            self._apply_auto_compact(None)
            return
        try:
            threshold = int(value.removesuffix("%"))
        except ValueError:
            threshold = -1
        if threshold not in AutoCompactScreen.OPTIONS:
            self.query_one("#chat", ChatLog).write("Use /autocompact off, 70, 80, or 90.", tone="warning")
            return
        self._apply_auto_compact(threshold)

    def _remember_prompt(self, prompt: str) -> None:
        self._prompt_history.append(prompt)
        self._prompt_history_index = None
        self._prompt_history_draft = ""

    def recall_prompt_history(self, prompt: Input, *, direction: int) -> None:
        """Move through submitted prompts, preserving an unfinished draft on return."""
        if not self._prompt_history:
            return
        if direction < 0:
            if self._prompt_history_index is None:
                self._prompt_history_draft = prompt.value
                self._prompt_history_index = len(self._prompt_history) - 1
            else:
                self._prompt_history_index = max(0, self._prompt_history_index - 1)
            prompt.value = self._prompt_history[self._prompt_history_index]
        elif self._prompt_history_index is not None:
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                prompt.value = self._prompt_history[self._prompt_history_index]
            else:
                self._prompt_history_index = None
                prompt.value = self._prompt_history_draft
        prompt.cursor_position = len(prompt.value)

    def _start_prompt(self, prompt: str) -> None:
        if self.active_session and self.active_session.title == "New session" and self._title_seed is None:
            self._title_seed = prompt
        # Transcript shows friendly filenames like OpenCode: "[Image 2 · foto.png]".
        self.query_one("#chat", ChatLog).add_user(self._expand_tokens_for_display(prompt))
        self._refresh_attachments_preview()
        self._generation_task = asyncio.create_task(self.generate(prompt))

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    @staticmethod
    def _normalize_pasted_path(raw: str) -> Path | None:
        """Accept plain paths, file:// URIs and GNOME copied-files lines."""
        candidate = raw.strip().strip("'\"")
        if not candidate or candidate == "x-special/gnome-copied-files":
            return None
        if candidate.startswith("file://"):
            # file:///home/user/img.png -> /home/user/img.png (also localhost form)
            candidate = re.sub(r"^file://(localhost)?", "", candidate)
            # URL-decode %20 etc. without pulling urllib cost into hot path
            candidate = candidate.replace("%20", " ")
        path = Path(candidate).expanduser()
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            return None
        return None

    def _register_attachment(self, resolved: Path) -> str:
        # Always tokenize: the prompt must never show a raw path, even when
        # the model has no vision (the image is then noted as text on send).
        is_image = resolved.suffix.lower() in self.IMAGE_EXTENSIONS
        if is_image and (not self.agent or not self.agent.model.supports_vision):
            self.notify(f"{resolved.name}: the selected model does not support images", severity="warning")
        self._attachment_counter += 1
        kind = "Image" if is_image else "File"
        token = f"[{kind} {self._attachment_counter}]"
        self._attachment_refs[token] = resolved
        return token

    def prepare_prompt_paste(self, text: str) -> str | None:
        """Turn pasted/dropped file paths into OpenCode-style [Image N] pills.

        Unlike the previous strict version (all-or-nothing), this handles:
        - ``file://`` URIs, single/double quotes, ``%20``
        - GNOME ``x-special/gnome-copied-files`` payloads
        - mixed content like ``'/tmp/a.png' que ves?`` -> ``[Image 1] que ves?``
        Returns None when no file was found so the raw paste is preserved.
        """
        if not text or not text.strip():
            return None
        try:
            values = shlex.split(text.strip())
        except ValueError:
            # Unbalanced quotes (typical drag-drop): fall back to whitespace split
            # and strip quotes manually per chunk.
            values = text.strip().split()
        if not values:
            return None
        converted: list[str] = []
        found_any = False
        for value in values:
            resolved = self._normalize_pasted_path(value)
            if resolved is None:
                converted.append(value)
                continue
            converted.append(self._register_attachment(resolved))
            found_any = True
        if not found_any:
            return None
        # Collapse accidental double spaces, keep single spaces like OpenCode.
        result = re.sub(r"\s+", " ", " ".join(converted)).strip()
        self.call_after_refresh(self._refresh_attachments_preview) if self.is_mounted else None
        return result

    def _attachment_label(self, token: str) -> str:
        path = self._attachment_refs.get(token)
        name = path.name if path else "?"
        icon = "◉" if token.startswith("[Image") else "▤"
        short = name if len(name) <= 28 else name[:25] + "…"
        return f"{icon} {token[1:-1]} · {short}"

    def _refresh_attachments_preview(self) -> None:
        """Show an OpenCode-like attachment strip above the prompt."""
        try:
            preview = self.query_one("#attachments-preview", Static)
            shell = self.query_one("#composer-shell")
        except Exception:
            return
        try:
            current = self.query_one("#prompt", Input).value
        except Exception:
            current = ""
        # Prune refs the user deleted from the composer (pill removed -> detached).
        orphan = [tok for tok in list(self._attachment_refs) if tok not in current]
        # Only prune when composer is visible (avoid wiping queued/history restores).
        # Queued prompts keep their own copy of the token string, refs stay.
        for tok in orphan:
            if tok not in " ".join(self._queued_prompts):
                # Keep the bytes mapping if a generation is in flight with that token.
                generating = bool(self._generation_task and not self._generation_task.done())
                if not generating:
                    del self._attachment_refs[tok]
        active = [(tok, self._attachment_refs[tok]) for tok in self._attachment_refs if tok in current]
        # Order by token number for stable display.
        def _num(tok: str) -> int:
            try:
                return int(tok.split()[-1].rstrip("]"))
            except ValueError:
                return 0
        active.sort(key=lambda item: _num(item[0]))
        if not active:
            preview.update("")
            preview.remove_class("-visible")
        else:
            pills = []
            for tok, path in active:
                icon = "◉" if tok.startswith("[Image") else "▤"
                name = path.name if len(path.name) <= 28 else path.name[:25] + "…"
                pills.append(f"[#0b0b0b on #2e2e2e] {icon} {tok[1:-1]} · {name} [/]")
            preview.update(" ".join(pills))
            preview.add_class("-visible")
        self._layout_composer()

    def _layout_composer(self) -> None:
        try:
            shell = self.query_one("#composer-shell", Vertical)
            queue = self.query_one("#queue-preview", Static)
            attachments = self.query_one("#attachments-preview", Static)
        except Exception:
            return
        extra = 0
        if "-visible" in queue.classes:
            try:
                lines = str(queue.renderable).splitlines() or [""]
            except Exception:
                lines = [""]
            # Keep previous behaviour (7 + lines) as 6 + (lines + 1).
            extra += len(lines) + 1
        if "-visible" in attachments.classes:
            extra += 1
        shell.styles.height = 6 + extra

    def _expand_tokens_for_display(self, prompt: str) -> str:
        """Render '[Image 2]' as '[Image 2 · name.png]' in the transcript."""
        def _repl(match: re.Match[str]) -> str:
            token = match.group(0)
            path = self._attachment_refs.get(token)
            if not path:
                return token
            return f"[{token[1:-1]} · {path.name}]"
        return re.sub(r"\[(?:Image|File) \d+\]", _repl, prompt)

    def _prepare_attachments(self, prompt: str) -> tuple[str, list[str]]:
        content = prompt
        images: list[str] = []
        additions: list[str] = []
        vision = bool(self.agent and self.agent.model.supports_vision)
        for token, path in self._attachment_refs.items():
            if token not in prompt:
                continue
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                if not vision:
                    additions.append(
                        f"{token} {path.name}: (image attached but not sent — "
                        "the selected model has no vision support)"
                    )
                    continue
                try:
                    images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
                except OSError as exc:
                    additions.append(f"{token} could not be read: {exc}")
            else:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")[:40_000]
                    additions.append(f"{token} {path.name}:\n{text}")
                except OSError as exc:
                    additions.append(f"{token} could not be read: {exc}")
        if additions:
            content += "\n\nAttached files:\n" + "\n\n".join(additions)
        return content, images

    async def generate(self, prompt: str) -> None:
        chat = self.query_one("#chat", ChatLog)
        input_widget = self.query_one("#prompt", Input)
        input_widget.disabled = False
        input_widget.focus()
        self._set_composer_hint(working=True)
        self._tool_active = False
        started_at = time.monotonic()
        starting_output_tokens = self.agent.stats.output_tokens if self.agent else 0
        model_used = self.agent.model.display_name if self.agent else "model"
        chat.show_activity("Thinking…")
        try:
            assert self.agent is not None
            model_prompt, images = self._prepare_attachments(prompt)
            async for event in self.agent.run(model_prompt, images=images):
                if event.kind == "compacted":
                    count = event.data.get("messages_condensed")
                    before = event.data.get("before_tokens")
                    after = event.data.get("after_tokens")
                    detail = (
                        f" {count} older messages condensed ({before:,} → {after:,} estimated tokens)."
                        if isinstance(count, int) and isinstance(before, int) and isinstance(after, int)
                        else ""
                    )
                    chat.write("Context compacted automatically to preserve this chat." + detail, tone="warning")
                    self._autosave_active_session(force=True)
                elif event.kind == "assistant_start":
                    chat.prepare_assistant()
                    self._autosave_active_session(force=True)
                elif event.kind == "thinking_delta":
                    chat.add_thinking(event.data["text"])
                elif event.kind == "thinking_as_response":
                    await chat.promote_thinking_to_response(event.data["text"])
                elif event.kind == "delta":
                    await chat.add_delta(event.data["text"])
                    self._autosave_active_session()
                elif event.kind == "tool_start":
                    self._tool_active = True
                    chat.show_activity("Working…")
                    self._autosave_active_session(force=True)
                elif event.kind == "tool_end":
                    self._tool_active = False
                    chat.add_tool(event.data["name"], event.data["arguments"])
                    chat.finish_tool(
                        event.data["name"],
                        event.data["arguments"],
                        event.data["result"],
                        event.data["error"],
                        self.show_tool_details,
                    )
                    self._reload_workspace_tree()
                    chat.show_activity("Thinking…")
                    self._autosave_active_session(force=True)
                elif event.kind == "limit":
                    chat.write(event.data["text"], tone="warning")
                    self._autosave_active_session(force=True)
                elif event.kind == "done":
                    chat.finish_thinking()
                    elapsed = max(0.001, time.monotonic() - started_at)
                    output_tokens = max(0, self.agent.stats.output_tokens - starting_output_tokens)
                    rate = output_tokens / elapsed
                    chat.write(
                        f"{model_used}  ·  {self._format_duration(elapsed)}  ·  {rate:.1f} tok/s",
                        tone="stats",
                    )
                if event.kind == "stats":
                    self.update_topbar(event.data)
                    self._autosave_active_session(force=True)
        except asyncio.CancelledError:
            if self.agent:
                self.agent.mark_interrupted()
            chat.clear_activity()
            chat.stop_pending_tool()
            chat.write("Task interrupted.", tone="warning")
            raise
        except (OllamaError, OllamaUnavailable) as exc:
            chat.clear_activity()
            chat.write(explain(exc, area="ollama"), tone="error")
        except Exception as exc:
            chat.clear_activity()
            chat.write(f"Error: {exc}", tone="error")
        finally:
            self._generation_task = None
            self._stop_armed = False
            self._save_active_session()
            if self.is_mounted and not self._shutting_down:
                input_widget.disabled = False
                input_widget.focus()
                self.update_topbar()
                if self._pending_model is not None:
                    model = self._pending_model
                    self._pending_model = None
                    self._activate_model(model)
                    chat.write(f"Model changed to {model.display_name}.")
                queued = self._queued_prompts.pop(0) if self._queued_prompts else None
                if queued is not None:
                    self._update_queue_ui()
                    self._start_prompt(queued)
                else:
                    self._schedule_session_title(self._title_seed or prompt)
                    self._title_seed = None
                    chat.clear_activity()
                    self._set_composer_hint(working=False)

    async def confirm_command(self, request: ConfirmationRequest) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def answer(value: bool | None) -> None:
            if not future.done():
                future.set_result(bool(value))

        self.push_screen(ConfirmScreen(request), answer)
        return await future

    async def authorize_tool(self, request: PermissionRequest) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def answer(value: str | None) -> None:
            # Ask is deliberately per invocation. Persistent rules belong in
            # the Permissions screen, never in a one-off confirmation.
            decision = value if value in {"allow_once", "deny"} else "deny"
            if not future.done():
                future.set_result(decision)

        self.push_screen(PermissionPromptScreen(request), answer)
        return await future

    async def ask_question(self, request: QuestionRequest) -> str | None:
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

        def answer(value: str | None) -> None:
            if not future.done():
                future.set_result(value)

        self.push_screen(QuestionScreen(request), answer)
        return await future

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, round(seconds))
        minutes, remaining = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {remaining}s"
        if minutes:
            return f"{minutes}m {remaining}s"
        return f"{remaining}s"

    def update_topbar(self, stats: dict[str, Any] | None = None) -> None:
        if not self.agent:
            return
        # Ollama reports prompt_eval_count for each completed request. It is the
        # actual tokenized input, unlike a character-count approximation.
        used = int((stats or {}).get("context_tokens", self.agent.context_tokens))
        limit = int((stats or {}).get("context_limit", self.agent.context.max_tokens))
        calls = int((stats or {}).get("tool_calls", self.agent.stats.tool_calls))
        session_title = self.active_session.title if self.active_session else "new session"
        context_percent = min(100, round(used / limit * 100)) if limit else 0
        self._main_static("#session-info").update(
            f"[bold]{session_title[:44]}[/]\n\n"
            f"[dim]CONTEXT[/]  {context_percent}%\n"
            f"[dim]TOKENS[/]   {used:,} / {limit:,}\n"
            f"[dim]TOOLS[/]    {calls}"
            + _system_lines()
        )

    def _main_static(self, selector: str) -> Static:
        """Find persistent UI behind any open modal."""
        for screen in self.screen_stack:
            try:
                return screen.query_one(selector, Static)
            except NoMatches:
                continue
        raise NoMatches(f"No main-screen node matches {selector!r}")

    def _main_chat(self) -> ChatLog:
        for screen in self.screen_stack:
            try:
                return screen.query_one("#chat", ChatLog)
            except NoMatches:
                continue
        raise NoMatches("No main-screen chat exists")

    def _update_model_status(self) -> None:
        if not self.agent:
            return
        if not self.agent.model.supports_reasoning:
            reasoning = "reasoning off"
        elif self.model_settings.reasoning == "enabled":
            reasoning = "reasoning on"
        elif self.model_settings.reasoning == "disabled":
            reasoning = "reasoning off"
        else:
            reasoning = "reasoning auto"
        if self._pending_model is not None:
            self._main_static("#model-status").update(
                f"{self.agent.model.display_name} → {self._pending_model.display_name}  ·  after task"
            )
        else:
            self._main_static("#model-status").update(f"{self.agent.model.display_name}  ·  {reasoning}")

    def _warn_if_no_tools(self) -> None:
        if self.agent and not self.agent.model.supports_tools:
            self._main_chat().write(
                f"{self.agent.model.display_name} does not support tools. It can chat, but cannot inspect files, edit, run commands, or search the web.",
                tone="warning",
            )

    def action_interrupt_or_exit(self) -> None:
        focused = self.focused
        if isinstance(focused, (Input, TextArea)) and focused.selected_text:
            focused.action_copy()
            return
        if self._generation_task and not self._generation_task.done():
            self._request_stop()
        else:
            self.action_request_exit()

    def action_escape(self) -> None:
        """Esc only stops active work; an idle main screen must never exit."""
        if not self._generation_task or self._generation_task.done():
            return
        if self._stop_armed:
            self._stop_armed = False
            self._generation_task.cancel()
            return
        self._stop_armed = True
        self._main_chat().show_activity("Press Esc again to interrupt", active=False, tone="confirm")
        self._main_static("#send-hint").update("Esc again interrupt  ·  wait to continue")
        self.set_timer(3.0, self._clear_stop_arm)

    def _clear_stop_arm(self) -> None:
        if not self._stop_armed:
            return
        self._stop_armed = False
        if self._generation_task and not self._generation_task.done():
            self._main_chat().show_activity("Working…" if self._tool_active else "Thinking…")
            self._set_composer_hint(working=True)

    def _set_composer_hint(self, *, working: bool) -> None:
        if working and self._queued_prompts:
            hint = "Enter add queue  ·  Ctrl+E manage queue  ·  Esc interrupt"
        elif working:
            hint = "Enter queue message  ·  Esc interrupt task"
        else:
            hint = "Enter send  ·  Ctrl+P menu  ·  Ctrl+C exit"
        self._main_static("#send-hint").update(hint)

    def _update_queue_ui(self) -> None:
        preview = self.query_one("#queue-preview", Static)
        if self._queued_prompts:
            visible = self._queued_prompts[:3]
            lines = [f"Queued  ·  {len(self._queued_prompts)} message{'s' if len(self._queued_prompts) != 1 else ''}"]
            # Show friendly filenames in the queue too, like the transcript.
            lines.extend(f"{index}.  {self._expand_tokens_for_display(value)}" for index, value in enumerate(visible, 1))
            remaining = len(self._queued_prompts) - len(visible)
            if remaining:
                lines.append(f"+ {remaining} more")
            lines.append("")  # one quiet separator before the editable prompt
            preview.update("\n".join(lines))
            preview.add_class("-visible")
        else:
            preview.update("")
            preview.remove_class("-visible")
        self._layout_composer()
        self._set_composer_hint(
            working=bool(self._generation_task and not self._generation_task.done())
        )

    def action_edit_queue(self) -> None:
        if not self._queued_prompts:
            return
        self.push_screen(QueueScreen(self._queued_prompts), self._queue_action)

    def action_delete_queue(self) -> None:
        """Keep Ctrl+D useful, but select the target instead of assuming the first."""
        if not self._queued_prompts:
            return
        self.push_screen(QueueScreen(self._queued_prompts), self._queue_action)

    def _queue_action(self, action: str | None) -> None:
        if not action:
            return
        kind, _, raw_index = action.partition(":")
        try:
            index = int(raw_index)
        except ValueError:
            return
        if not 0 <= index < len(self._queued_prompts):
            return
        if kind == "delete":
            self._queued_prompts.pop(index)
            self._update_queue_ui()
            self._save_active_session()
            self.query_one("#prompt", Input).focus()
            return
        if kind == "edit":
            original = self._queued_prompts[index]
            self.push_screen(
                EditQueueScreen(original),
                lambda replacement: self._replace_queued_prompt(index, original, replacement),
            )

    def _replace_queued_prompt(self, index: int, original: str, replacement: str | None) -> None:
        """Avoid overwriting a queue item if it changed while the editor was open."""
        if replacement and 0 <= index < len(self._queued_prompts) and self._queued_prompts[index] == original:
            self._queued_prompts[index] = replacement
        self._update_queue_ui()
        self._save_active_session()
        self.query_one("#prompt", Input).focus()

    def _request_stop(self) -> None:
        if self._stop_confirmation_open:
            return
        self._stop_confirmation_open = True
        self.push_screen(StopScreen(), self._stop_confirmed)

    def _stop_confirmed(self, confirmed: bool | None) -> None:
        self._stop_confirmation_open = False
        self._stop_armed = False
        if confirmed and self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

    def action_request_exit(self) -> None:
        if not self._exit_armed:
            self._exit_armed = True
            self._main_chat().show_activity("Press Ctrl+C again to exit", active=False, tone="confirm")
            self._main_static("#send-hint").update("Ctrl+C again exits  ·  wait to cancel")
            self.set_timer(3.0, self._clear_exit_arm)
            return
        if self._generation_task and not self._generation_task.done():
            self._shutting_down = True
            self._generation_task.cancel()
        self.exit()

    def _clear_exit_arm(self) -> None:
        if not self._exit_armed:
            return
        self._exit_armed = False
        if self._generation_task and not self._generation_task.done():
            self._set_composer_hint(working=True)
            self._main_chat().show_activity("Working…" if self._tool_active else "Thinking…")
        else:
            self._set_composer_hint(working=False)
            self._main_chat().clear_activity()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_compact_context(self) -> None:
        if not self.agent:
            return
        if self._generation_task and not self._generation_task.done():
            self._main_chat().write("Compact after the current task finishes.", tone="warning")
            return
        if self.agent.compact_context():
            details = self.agent.last_compaction or {}
            count = details.get("messages_condensed")
            before = details.get("before_tokens")
            after = details.get("after_tokens")
            suffix = (
                f" {count} older messages condensed ({before:,} → {after:,} estimated tokens)."
                if isinstance(count, int) and isinstance(before, int) and isinstance(after, int)
                else " Older messages were summarized."
            )
            self._main_chat().write("Context compacted." + suffix, tone="warning")
            self._save_active_session()
        else:
            self._main_chat().write("Context is already compact; there is not enough older history to reduce.")

    def action_context_summary(self) -> None:
        if not self.agent:
            return
        summary = next(
            (
                str(message.get("content") or "")
                for message in reversed(self.agent.history)
                if message.get("role") == "system"
                and str(message.get("content") or "").startswith("Compacted conversation memory:")
            ),
            "",
        )
        if summary:
            self._main_chat().write(summary, tone="warning")
        else:
            self._main_chat().write("This session has not been compacted yet.")

    def action_auto_compact(self, *, return_to_context: bool = False, return_to_menu: bool = False) -> None:
        if self._generation_task and not self._generation_task.done():
            return
        self.push_screen(
            AutoCompactScreen(self.model_settings.auto_compact_threshold),
            lambda value: self._auto_compact_selected(value, return_to_context, return_to_menu),
        )

    def _auto_compact_selected(
        self, value: int | str | None, return_to_context: bool = False, return_to_menu: bool = False
    ) -> None:
        if value is not None:
            self._apply_auto_compact(None if value == "off" else int(value))
        if return_to_context:
            self.action_context_settings(return_to_menu=return_to_menu)
            return
        if return_to_menu:
            self.action_menu()

    def _apply_auto_compact(self, threshold: int | None) -> None:
        self.model_settings.auto_compact_threshold = threshold
        self.model_settings_store.save(self.model_settings)
        if self.agent:
            self.agent.set_settings(self.model_settings)

    def action_toggle_tools(self) -> None:
        self.show_tool_details = not self.show_tool_details
        state = "expanded" if self.show_tool_details else "compact"
        self.query_one("#chat", ChatLog).set_tool_details(self.show_tool_details)
        self.query_one("#chat", ChatLog).write(f"Tool details {state}.")

    def action_menu(self) -> None:
        # A modal already owns interaction; never stack command menus on top of it.
        if len(self.screen_stack) > 1:
            return
        working = bool(self._generation_task and not self._generation_task.done())
        self.push_screen(CommandMenu(working=working), self._menu_selected)

    def action_context_settings(self, *, return_to_menu: bool = False, return_to_model_hub: bool = False) -> None:
        if self._generation_task and not self._generation_task.done():
            return
        if not self.agent:
            return
        self.push_screen(
            ContextSettingsScreen(self.model_settings, self.agent.model, self._apply_model_settings, self.action_compact_context),
            lambda _: self._return_from_model_section(return_to_menu, return_to_model_hub),
        )

    def _return_from_model_section(self, return_to_menu: bool, return_to_model_hub: bool) -> None:
        if return_to_model_hub:
            self.action_model_hub(return_to_menu=return_to_menu)
        elif return_to_menu:
            self.action_menu()

    def _menu_selected(self, command: str | None) -> None:
        if command == "model":
            self.action_model_hub(return_to_menu=True)
        elif command == "model-change":
            self.action_models(return_to_menu=False, return_to_hub=False)
        elif command == "model-settings":
            self.action_model_settings(return_to_menu=False, return_to_hub=False)
        elif command in {"model-context", "context-settings", "compact", "autocompact"}:
            if command == "compact":
                self.action_compact_context()
            elif command == "autocompact":
                self.action_auto_compact()
            else:
                self.action_context_settings(return_to_menu=False, return_to_model_hub=False)
        elif command == "sessions":
            self.action_sessions(return_to_menu=True)
        elif command == "sessions-new":
            self._menu_new_session()
        elif command == "permissions":
            self.action_permissions(return_to_menu=True)
        elif command == "workspace":
            self.action_workspace(return_to_menu=True)
        elif command == "files":
            self.action_toggle_sidebar()
        elif command == "commands":
            self.action_commands(return_to_menu=True)
        elif command == "artinium":
            self.action_artinium(return_to_menu=True)
        elif command in {"themes", "updates"}:
            if command == "themes":
                self.action_themes()
            else:
                self.action_updates()
        elif command == "undo":
            self.action_undo()
        elif command == "changes":
            self.action_changes()

    def _menu_new_session(self) -> None:
        if self._generation_task or not self.active_session:
            return
        self._save_active_session()
        self.active_session = SessionRecord.new()
        self._title_seed = None
        if self.agent:
            self.agent.clear()
        try:
            self.query_one("#chat", ChatLog).clear()
        except Exception:
            pass
        self.update_topbar()

    def action_commands(self, *, return_to_menu: bool = False) -> None:
        self.push_screen(
            CommandsScreen(),
            lambda command: self._command_selected(command, return_to_menu),
        )

    def _command_selected(self, command: str | None, return_to_menu: bool = False) -> None:
        if command:
            self._run_slash_command(command)
        elif return_to_menu:
            self.action_menu()

    def action_artinium(self, *, return_to_menu: bool = False) -> None:
        self.push_screen(
            ArtiniumScreen(),
            lambda choice: self._artinium_selected(choice, return_to_menu),
        )

    def _artinium_selected(self, choice: str | None, return_to_menu: bool = False) -> None:
        if choice == "info":
            self.push_screen(InfoScreen(), lambda _: self.action_artinium(return_to_menu=return_to_menu))
        elif choice == "updates":
            self.action_updates(return_to_artinium=True, return_to_menu=return_to_menu)
        elif choice == "themes":
            self.action_themes(return_to_artinium=True, return_to_menu=return_to_menu)
        elif return_to_menu:
            self.action_menu()

    def _session_permission(self, name: str) -> str:
        if self.active_session is None:
            return "ask"
        return self.active_session.tool_permissions.get(name, "ask")

    def action_permissions(self, *, return_to_menu: bool = False) -> None:
        if self._generation_task and not self._generation_task.done():
            self._main_chat().write(
                "Permissions are available after the current task finishes.", tone="warning"
            )
            return
        if self.active_session is None:
            return
        self.push_screen(
            PermissionsScreen(self.active_session, self._save_active_session),
            lambda _: self.action_menu() if return_to_menu else None,
        )

    def action_themes(self, *, return_to_artinium: bool = False, return_to_menu: bool = False) -> None:
        themes = sorted(name for name in self.available_themes if name != "textual-ansi")
        self.push_screen(
            ThemeScreen(themes, self.preferences.theme),
            lambda theme: self._theme_selected(theme, return_to_artinium, return_to_menu),
        )

    def _theme_selected(self, theme: str | None, return_to_artinium: bool, return_to_menu: bool) -> None:
        if theme:
            self._apply_theme(theme)
        if return_to_artinium:
            self.action_artinium(return_to_menu=return_to_menu)

    def _apply_theme(self, theme: str, *, save: bool = True) -> None:
        selected = theme
        if selected == "automatic":
            desktop_theme = " ".join(
                filter(None, (os.environ.get("OMARCHY_THEME"), os.environ.get("GTK_THEME"), os.environ.get("COLORFGBG")))
            ).lower()
            if "light" in desktop_theme or desktop_theme == "15" or desktop_theme.endswith(";15"):
                selected = "textual-light"
            elif "catppuccin" in desktop_theme:
                selected = "catppuccin-mocha"
            elif "gruvbox" in desktop_theme:
                selected = "gruvbox"
            else:
                selected = "textual-dark"
        if selected in self.available_themes:
            self.theme = selected
        if save:
            self.preferences.theme = theme
            self.preferences_store.save(self.preferences)

    def action_updates(self, *, return_to_artinium: bool = False, return_to_menu: bool = False) -> None:
        async def open_after_check() -> None:
            await self._check_updates(announce=False)
            self.push_screen(
                UpdateScreen(self.update_info),
                lambda choice: self._update_selected(choice, return_to_artinium, return_to_menu),
            )
        asyncio.create_task(open_after_check())

    async def _check_updates(self, *, announce: bool) -> None:
        self.update_info = await check_for_update(__version__)
        if (
            announce
            and self.update_info
            and self.update_info.available
            and self.update_info.identity != self.preferences.ignored_version
            and self.is_mounted
        ):
            if len(self.screen_stack) == 1:
                self.push_screen(
                    UpdateScreen(self.update_info),
                    lambda choice: self._update_selected(choice, False, False),
                )
            else:
                self._main_chat().write(
                    f"Artinium {self.update_info.latest} is available. Open Ctrl+P → Artinium → Updates."
                )

    def _update_selected(self, choice: str | None, return_to_artinium: bool, return_to_menu: bool) -> None:
        if choice == "ignore" and self.update_info:
            self.preferences.ignored_version = self.update_info.identity
            self.preferences_store.save(self.preferences)
        elif choice == "install":
            asyncio.create_task(self._install_update())
        if return_to_artinium and choice != "install":
            self.action_artinium(return_to_menu=return_to_menu)

    async def _install_update(self) -> None:
        self._main_chat().write("Installing Artinium update…")
        commit = self.update_info.latest_commit if self.update_info else None
        success, output = await install_update(commit)
        if success:
            self._main_chat().write("Update installed. Restart Artinium to use it.")
        else:
            self._main_chat().write(f"Update failed.\n{output}", tone="error")

    def action_undo(self) -> None:
        if self._generation_task and not self._generation_task.done():
            self._main_chat().write("Undo is available after the current task finishes.", tone="warning")
            return
        result = self.undo_manager.undo()
        if result is None:
            self._main_chat().write("Nothing to undo.")
            return
        message = f"Undid {result['tool']} on {result['path']} ({result['action']})."
        self._main_chat().write(message, tone="warning")
        if self.agent:
            self.agent.history.append({"role": "system", "content": message})
        self._reload_workspace_tree()
        self._save_active_session()

    def action_changes(self) -> None:
        if self._generation_task and not self._generation_task.done():
            self._main_chat().write("Changes are available after the current task finishes.", tone="warning")
            return
        self.push_screen(ChangeHistoryScreen(self.undo_manager.records), self._change_selected)

    def _change_selected(self, record_id: str | None) -> None:
        if not record_id:
            return
        result = self.undo_manager.undo_record(record_id)
        if result is None:
            self._main_chat().write("Only the newest change can be reverted. Undo newer changes first.", tone="warning")
            return
        message = f"Undid {result['tool']} on {result['path']} ({result['action']})."
        self._main_chat().write(message, tone="warning")
        if self.agent:
            self.agent.history.append({"role": "system", "content": message})
        self._reload_workspace_tree()
        self._save_active_session()

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.toggle_class("-hidden")

    def _reload_workspace_tree(self) -> None:
        """Refresh the file explorer only when this screen has one mounted.

        A completed tool must never turn into a chat error merely because the
        optional sidebar is hidden or temporarily absent during a screen swap.
        """
        try:
            self.query_one("#tree", DirectoryTree).reload()
        except NoMatches:
            return

    def action_sessions(self, *, return_to_menu: bool = False) -> None:
        if self._generation_task or not self.active_session:
            return
        self.push_screen(
            SessionScreen(self.session_store.sessions, self.active_session.id),
            lambda action: self._session_selected(action, return_to_menu),
        )

    def _session_selected(self, action: str | None, return_to_menu: bool = False) -> None:
        if not action:
            if return_to_menu:
                self.action_menu()
            return
        if action == "new":
            self._save_active_session()
            self.active_session = SessionRecord.new()
            self._title_seed = None
            if self.agent:
                self.agent.clear()
            self.query_one("#chat", ChatLog).clear()
            self.update_topbar()
            return
        kind, _, session_id = action.partition(":")
        session = next((item for item in self.session_store.sessions if item.id == session_id), None)
        if not session:
            return
        if kind == "select":
            self._save_active_session()
            self.active_session = session
            self._title_seed = None
            if self.agent:
                self.agent.restore(session.history, session.stats)
            self.call_later(self._render_active_session)
            self.update_topbar()
        elif kind == "pin":
            session.pinned = not session.pinned
            self.session_store.save()
            self.action_sessions(return_to_menu=return_to_menu)
        elif kind == "rename":
            self.push_screen(RenameSessionScreen(session.title), lambda title: self._rename_session(session, title, return_to_menu))
        elif kind == "delete":
            self.push_screen(DeleteSessionScreen(), lambda confirmed: self._delete_session(session, bool(confirmed), return_to_menu))

    def _rename_session(self, session: SessionRecord, title: str | None, return_to_menu: bool = False) -> None:
        if not title:
            return
        session.title = title[:80]
        self.session_store.save()
        if session is self.active_session:
            self.update_topbar()
        self.action_sessions(return_to_menu=return_to_menu)

    def _delete_session(self, session: SessionRecord, confirmed: bool, return_to_menu: bool = False) -> None:
        if not confirmed:
            self.action_sessions(return_to_menu=return_to_menu)
            return
        self.session_store.sessions = [item for item in self.session_store.sessions if item.id != session.id]
        self.session_store.save()
        if session is self.active_session:
            self.active_session = SessionRecord.new()
            self._title_seed = None
            if self.agent:
                self.agent.clear()
            self.query_one("#chat", ChatLog).clear()
            self.update_topbar()
        self.action_sessions(return_to_menu=return_to_menu)

    async def _render_active_session(self) -> None:
        if not self.agent:
            return
        chat = self.query_one("#chat", ChatLog)
        chat.clear()
        pending_tools: dict[str, tuple[str, dict[str, Any]]] = {}
        for message in self.agent.history:
            content = str(message.get("content") or "")
            if message.get("role") == "user":
                chat.add_user(content)
            elif message.get("role") == "assistant":
                if content and content != INTERRUPTION_NOTE:
                    await chat.begin_assistant()
                    await chat.add_delta(content)
                for index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function") or {}
                    call_id = str(call.get("id") or f"restored_{index}")
                    pending_tools[call_id] = (
                        str(function.get("name") or ""),
                        Agent._arguments(function.get("arguments")),
                    )
            elif message.get("role") == "tool":
                tool_name, arguments = pending_tools.pop(
                    str(message.get("tool_call_id") or ""),
                    (str(message.get("tool_name") or ""), {}),
                )
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    result = {"error": "Stored tool result could not be read."}
                chat.add_tool(tool_name, arguments)
                chat.finish_tool(tool_name, arguments, result, bool(result.get("error")), self.show_tool_details)
        for tool_name, arguments in pending_tools.values():
            chat.add_tool(tool_name, arguments)
            chat.stop_pending_tool()

    def _save_active_session(self) -> None:
        if not self.agent or not self.active_session:
            return
        if (
            not any(message.get("role") == "user" for message in self.agent.history)
            and not self.active_session.draft
            and not self._queued_prompts
            and not self.undo_manager.records
        ):
            return
        self.session_store.touch(
            self.active_session,
            self.agent.history,
            self.agent.stats,
            queued_prompts=self._queued_prompts,
            draft=self.active_session.draft,
            undo_records=self.undo_manager.dump(),
            model_name=self.agent.model.name,
        )

    def _autosave_active_session(self, *, force: bool = False) -> None:
        """Persist meaningful progress, while avoiding one disk write per token."""
        now = time.monotonic()
        if force or now - self._last_autosave >= 0.5:
            self._save_active_session()
            self._last_autosave = now

    def _schedule_session_title(self, prompt: str) -> None:
        """Generate one local title for a fresh session, outside the chat stream."""
        session = self.active_session
        if not session or session.title != "New session" or not self.agent:
            return
        provisional_title = self._provisional_session_title(prompt)
        session.title = provisional_title
        self.session_store.save()
        self.update_topbar()
        if not isinstance(self.client, OllamaClient):
            return
        task = asyncio.create_task(self._generate_session_title(session, prompt, self.agent.model.name, provisional_title))
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

    @staticmethod
    def _provisional_session_title(prompt: str) -> str:
        words = " ".join(prompt.replace("\n", " ").split()).strip(" .!?¿¡")
        if not words:
            return "New session"
        return words[:60]

    async def _generate_session_title(
        self, session: SessionRecord, prompt: str, model: str, provisional_title: str
    ) -> None:
        try:
            title = await self.client.suggest_title(model, prompt)
        except (OllamaError, OllamaUnavailable):
            return
        title = " ".join(title.replace("\n", " ").strip(" \t\"'`.-–—").split())[:60]
        if not title or session.title != provisional_title:
            return
        session.title = title
        self.session_store.save()
        if session is self.active_session:
            self.update_topbar()

    def action_model_hub(self, *, return_to_menu: bool = False) -> None:
        if not self.agent:
            return
        working = bool(self._generation_task and not self._generation_task.done())
        self.push_screen(ModelHubScreen(working=working), lambda choice: self._model_hub_selected(choice, return_to_menu))

    def _model_hub_selected(self, choice: str | None, return_to_menu: bool = False) -> None:
        if choice == "change":
            self.action_models(return_to_menu=return_to_menu, return_to_hub=True)
        elif choice == "settings":
            self.action_model_settings(return_to_menu=return_to_menu, return_to_hub=True)
        elif choice == "context":
            self.action_context_settings(return_to_menu=return_to_menu, return_to_model_hub=True)
        elif return_to_menu:
            self.action_menu()

    def action_models(self, *, return_to_menu: bool = False, return_to_hub: bool = False) -> None:
        if not self.agent:
            return
        pending = self._pending_model.name if self._pending_model else None
        self.push_screen(
            ModelScreen(self.models, self.agent.model.name, pending),
            lambda name: self._model_selected(name, return_to_menu, return_to_hub),
        )

    def action_model_settings(self, *, return_to_menu: bool = False, return_to_hub: bool = False) -> None:
        if not self.agent or self._generation_task:
            return
        self.push_screen(
            ModelSettingsScreen(self.model_settings, self.agent.model, self._apply_model_settings),
            lambda _: self._return_from_model_section(return_to_menu, return_to_hub),
        )

    def _apply_model_settings(self, settings: ModelSettings) -> None:
        if not self.agent:
            return
        self.model_settings = settings
        self.model_settings_store.save(settings)
        self.agent.set_settings(settings)
        self.update_topbar()
        self._update_model_status()

    def _activate_model(self, model: ModelInfo) -> None:
        """Apply a model only between agent turns, never into an active stream."""
        if not self.agent:
            return
        self.agent.set_model(model)
        self._loaded_models.add(model.name)
        self.update_topbar()
        self._update_model_status()
        self._warn_if_no_tools()

    def _model_selected(self, name: str | None, return_to_menu: bool = False, return_to_hub: bool = False) -> None:
        if name and self.agent:
            model = next((item for item in self.models if item.name == name), None)
            if model:
                if self._generation_task and not self._generation_task.done():
                    self._pending_model = model
                    self._update_model_status()
                    self._main_chat().write(f"{model.display_name} will be used after the current task.")
                else:
                    self._activate_model(model)
        if return_to_hub:
            self.action_model_hub(return_to_menu=return_to_menu)
        elif return_to_menu:
            self.action_menu()

    def action_workspace(self, *, return_to_menu: bool = False) -> None:
        if self._generation_task:
            return
        self.push_screen(
            WorkspaceScreen(self.workspace.root),
            lambda value: self._workspace_selected(value, return_to_menu),
        )

    def _workspace_selected(self, value: str | None, return_to_menu: bool = False) -> None:
        if not value:
            if return_to_menu:
                self.action_menu()
            return
        try:
            new_root = Path(value).expanduser().resolve()
            self.workspace.replace_root(new_root)
            self.undo_manager.records.clear()
            self.session_store = SessionStore(self.workspace.root)
            if self.agent:
                self.agent.tools.workspace = self.workspace
                self.agent.clear()
                self.session_store.load()
                self.active_session = SessionRecord.new()
                self._title_seed = None
            try:
                old_tree = self.query_one("#tree", DirectoryTree)
            except NoMatches:
                old_tree = None
            if old_tree is not None:
                old_tree.path = new_root
                old_tree.reload()
            self.update_topbar()
            self.call_later(self._render_active_session)
        except (WorkspaceError, OSError) as exc:
            self.notify(explain(exc), severity="error")

    async def on_unmount(self) -> None:
        self._shutting_down = True
        self._save_active_session()
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._update_task
        for task in self._title_tasks:
            task.cancel()
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            if self._generation_task is not asyncio.current_task():
                with suppress(asyncio.CancelledError, OllamaError, OllamaUnavailable):
                    await self._generation_task
        # Only release models Artinium selected during this run. Ollama itself
        # remains available for the user's other applications and commands.
        for model in self._loaded_models:
            with suppress(OllamaError, OllamaUnavailable):
                await self.client.unload_model(model)
        await self.client.close()
