from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelSettings:
    """Optional Ollama overrides; empty values preserve the working defaults."""

    reasoning: str = "auto"
    temperature: float | None = None
    max_output_tokens: int | None = None
    context_window: int | None = None
    auto_compact_threshold: int | None = 80

    def options_for(self, model_context: int) -> dict[str, int | float]:
        # Artinium historically used at most 32K automatically. Keep that safe
        # default, but honour a user-selected value up to the model's own limit.
        context = self.context_window or min(model_context, 32_768)
        options: dict[str, int | float] = {"num_ctx": min(context, model_context)}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            options["num_predict"] = self.max_output_tokens
        return options

    def think_value(self, supports_reasoning: bool) -> bool | None:
        if not supports_reasoning or self.reasoning == "auto":
            return None
        return self.reasoning == "enabled"

    @classmethod
    def from_data(cls, value: object) -> "ModelSettings":
        if not isinstance(value, dict):
            return cls()
        reasoning = str(value.get("reasoning") or "auto")
        if reasoning not in {"auto", "enabled", "disabled"}:
            reasoning = "auto"
        temperature = value.get("temperature")
        max_output = value.get("max_output_tokens")
        context = value.get("context_window")
        auto_compact = value.get("auto_compact_threshold", 80)
        return cls(
            reasoning=reasoning,
            temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
            max_output_tokens=int(max_output) if isinstance(max_output, int) and max_output > 0 else None,
            context_window=int(context) if isinstance(context, int) and context >= 4096 else None,
            auto_compact_threshold=(
                int(auto_compact) if isinstance(auto_compact, int) and 50 <= auto_compact <= 95 else None
            ),
        )


class ModelSettingsStore:
    """User-wide preferences, separate from workspace conversations."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".config" / "artinium" / "settings.json"

    def load(self) -> ModelSettings:
        try:
            return ModelSettings.from_data(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return ModelSettings()

    def save(self, settings: ModelSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
