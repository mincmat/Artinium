from __future__ import annotations

from typing import Any


class ContextManager:
    """Cheap deterministic context control, intentionally requiring no extra model call."""

    def __init__(self, max_tokens: int = 16_000, reserve_tokens: int = 3_000):
        self.max_tokens = max(4_096, max_tokens)
        self.reserve_tokens = min(reserve_tokens, self.max_tokens // 3)
        self.last_compaction: dict[str, Any] | None = None

    @staticmethod
    def estimate_tokens(value: Any) -> int:
        # A conservative approximation for mixed prose/code and tool JSON.
        return max(1, len(str(value)) // 3)

    def usage(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.estimate_tokens(message) for message in messages)

    def compact(self, messages: list[dict[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
        self.last_compaction = None
        budget = self.max_tokens - self.reserve_tokens
        if not force and self.usage(messages) <= budget:
            return messages

        # Preserve the newest coherent slice; turn older dialogue into compact memory.
        recent: list[dict[str, Any]] = []
        used = 0
        cutoff = max(2_000, budget * 2 // 3)
        for message in reversed(messages):
            cost = self.estimate_tokens(message)
            if recent and used + cost > cutoff:
                break
            recent.append(message)
            used += cost
        recent.reverse()
        # An orphan tool response is invalid in Ollama's chat protocol.
        while recent and recent[0].get("role") == "tool":
            recent.pop(0)
        old = messages[: len(messages) - len(recent)]
        if not old:
            return messages

        notes: list[str] = []
        for message in old:
            role = message.get("role", "?")
            if role == "tool":
                continue
            content = " ".join(str(message.get("content") or "").split())
            if content:
                notes.append(f"{role}: {content[:280]}")
        memory = "Compacted conversation memory:\n" + "\n".join(notes[-12:])
        compacted = [{"role": "system", "content": memory[:2_800]}, *recent]
        self.last_compaction = {
            "before_tokens": self.usage(messages),
            "after_tokens": self.usage(compacted),
            "messages_condensed": len(old),
            "summary": memory[:2_800],
        }
        return compacted
