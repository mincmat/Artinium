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
        valid_history = [
            item for item in history
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        ] if isinstance(history, list) else []
        valid_stats: dict[str, int] = {}
        if isinstance(stats, dict):
            for key, value in stats.items():
                try:
                    valid_stats[str(key)] = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        queued = data.get("queued_prompts")
        return cls(
            id=data["id"],
            title=str(data.get("title") or "New session"),
            pinned=bool(data.get("pinned")),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            history=valid_history,
            stats=valid_stats,
            queued_prompts=[item for item in queued if isinstance(item, str) and item.strip()]
            if isinstance(queued, list) else [],
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
        except json.JSONDecodeError:
            # Never silently discard history: quarantine the corrupt file.
            backup = self.path.with_suffix(f".corrupt.{_now().replace(':', '-')}.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            raw = []
        except OSError:
            raw = []
        values = raw.get("sessions", []) if isinstance(raw, dict) else raw
        self.active_session_id = (
            str(raw.get("active_session_id") or "") or None
            if isinstance(raw, dict)
            else None
        )
        self.sessions = [record for item in values if (record := SessionRecord.from_data(item))]
        if self.active_session_id is not None and all(
            session.id != self.active_session_id for session in self.sessions
        ):
            self.active_session_id = self.sessions[0].id if self.sessions else None
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
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            handle.flush()
            try:
                import os
                os.fsync(handle.fileno())
            except OSError:
                pass
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
        # Copy: the caller (agent.history) keeps mutating after touch().
        session.history = list(history)
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
