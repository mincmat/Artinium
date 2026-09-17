from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .agent import SessionStats


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class SessionRecord:
    id: str
    title: str = "New session"
    pinned: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    history: list[dict[str, object]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    queued_prompts: list[str] = field(default_factory=list)
    draft: str = ""
    undo_records: list[dict[str, object]] = field(default_factory=list)
    model_name: str = ""

    @classmethod
    def new(cls) -> "SessionRecord":
        return cls(id=uuid4().hex)

    @classmethod
    def from_data(cls, data: object) -> "SessionRecord | None":
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            return None
        history = data.get("history")
        stats = data.get("stats")
        return cls(
            id=data["id"],
            title=str(data.get("title") or "New session"),
            pinned=bool(data.get("pinned")),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            history=history if isinstance(history, list) else [],
            stats=stats if isinstance(stats, dict) else {},
            queued_prompts=[str(item) for item in data.get("queued_prompts", []) if str(item).strip()]
            if isinstance(data.get("queued_prompts"), list) else [],
            draft=str(data.get("draft") or ""),
            undo_records=data.get("undo_records") if isinstance(data.get("undo_records"), list) else [],
            model_name=str(data.get("model_name") or ""),
        )


class SessionStore:
    """Small workspace-local session store, deliberately separate from source files."""

    def __init__(self, workspace: Path):
        self.path = workspace / ".artinium" / "sessions.json"
        self.sessions: list[SessionRecord] = []
        self.active_session_id: str | None = None

    def load(self) -> list[SessionRecord]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
        values = raw.get("sessions", []) if isinstance(raw, dict) else raw
        self.active_session_id = (
            str(raw.get("active_session_id") or "") or None
            if isinstance(raw, dict)
            else None
        )
        self.sessions = [record for item in values if (record := SessionRecord.from_data(item))]
        self._sort()
        return self.sessions

    def save(self) -> None:
        self._sort()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "active_session_id": self.active_session_id,
            "sessions": [asdict(session) for session in self.sessions],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def create(self) -> SessionRecord:
        session = SessionRecord.new()
        self.sessions.append(session)
        return session

    def touch(
        self,
        session: SessionRecord,
        history: list[dict[str, object]],
        stats: SessionStats,
        *,
        queued_prompts: list[str] | None = None,
        draft: str | None = None,
        undo_records: list[dict[str, object]] | None = None,
        model_name: str | None = None,
    ) -> None:
        if not any(item.id == session.id for item in self.sessions):
            self.sessions.append(session)
        session.history = history
        session.stats = {
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "tool_calls": stats.tool_calls,
            "last_prompt_tokens": stats.last_prompt_tokens,
        }
        if queued_prompts is not None:
            session.queued_prompts = list(queued_prompts)
        if draft is not None:
            session.draft = draft
        if undo_records is not None:
            session.undo_records = undo_records
        if model_name is not None:
            session.model_name = model_name
        self.active_session_id = session.id
        session.updated_at = _now()
        self._sort()
        self.save()

    def _sort(self) -> None:
        self.sessions.sort(key=lambda item: item.updated_at, reverse=True)
        self.sessions.sort(key=lambda item: not item.pinned)
