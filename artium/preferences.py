from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


MUTATING_TOOLS = ("write_file", "edit_file", "run_command")
PERMISSION_VALUES = ("ask", "allow", "deny")


@dataclass(slots=True)
class AppPreferences:
    """Application-wide preferences unrelated to one Ollama model."""

    theme: str = "automatic"
    check_updates: bool = True
    ignored_version: str = ""
    tool_permissions: dict[str, str] = field(
        default_factory=lambda: {name: "ask" for name in MUTATING_TOOLS}
    )

    @classmethod
    def from_data(cls, data: object) -> "AppPreferences":
        if not isinstance(data, dict):
            return cls()
        permissions = {name: "ask" for name in MUTATING_TOOLS}
        raw = data.get("tool_permissions")
        if isinstance(raw, dict):
            for name in permissions:
                value = str(raw.get(name) or "ask")
                permissions[name] = value if value in PERMISSION_VALUES else "ask"
        return cls(
            theme=str(data.get("theme") or "automatic"),
            check_updates=bool(data.get("check_updates", True)),
            ignored_version=str(data.get("ignored_version") or ""),
            tool_permissions=permissions,
        )


class PreferencesStore:
    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".config" / "artinium" / "preferences.json"

    def load(self) -> AppPreferences:
        try:
            return AppPreferences.from_data(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return AppPreferences()

    def save(self, preferences: AppPreferences) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(preferences), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
