from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .workspace import Workspace


@dataclass(slots=True)
class UndoRecord:
    id: str
    tool: str
    path: str
    existed: bool
    content: str


class UndoManager:
    """Recoverable snapshots for successful file writes and exact edits."""

    def __init__(self, workspace: Workspace, limit: int = 50):
        self.workspace = workspace
        self.limit = limit
        self.records: list[UndoRecord] = []

    def snapshot(self, tool: str, arguments: dict[str, Any]) -> UndoRecord | None:
        if tool not in {"write_file", "edit_file"}:
            return None
        path = str(arguments.get("path") or "")
        target = self.workspace.resolve(path)
        existed = target.is_file()
        content = target.read_text(encoding="utf-8") if existed else ""
        return UndoRecord(uuid4().hex, tool, self.workspace.relative(target), existed, content)

    def commit(self, record: UndoRecord | None) -> None:
        if record is None:
            return
        self.records.append(record)
        del self.records[:-self.limit]

    def dump(self) -> list[dict[str, object]]:
        return [
            {"id": item.id, "tool": item.tool, "path": item.path, "existed": item.existed, "content": item.content}
            for item in self.records
        ]

    def restore(self, values: list[dict[str, object]]) -> None:
        restored: list[UndoRecord] = []
        for value in values[-self.limit:]:
            if not isinstance(value, dict):
                continue
            path = value.get("path")
            tool = value.get("tool")
            if not isinstance(path, str) or not isinstance(tool, str):
                continue
            restored.append(UndoRecord(
                str(value.get("id") or uuid4().hex), tool, path,
                bool(value.get("existed")), str(value.get("content") or ""),
            ))
        self.records = restored

    def undo(self) -> dict[str, Any] | None:
        if not self.records:
            return None
        record = self.records.pop()
        return self._restore(record)

    def undo_record(self, record_id: str) -> dict[str, Any] | None:
        """Undo only the latest snapshot, avoiding unsafe out-of-order restores."""
        if not self.records or self.records[-1].id != record_id:
            return None
        return self._restore(self.records.pop())

    def _restore(self, record: UndoRecord) -> dict[str, Any]:
        target = self.workspace.resolve(record.path)
        if record.existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(record.content, encoding="utf-8")
            action = "restored"
        else:
            if target.exists() and target.is_file():
                target.unlink()
            action = "removed"
        return {"path": record.path, "action": action, "tool": record.tool}
