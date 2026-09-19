from __future__ import annotations

import os
import re
from difflib import unified_diff
from pathlib import Path
from typing import Any

from ..workspace import IGNORED_DIRS, Workspace, is_probably_binary, walk_files

MAX_READ_CHARS = 40_000
MAX_WRITE_CHARS = 500_000
MAX_EDIT_BYTES = 2_000_000
MAX_SEARCH_BYTES = 10_000_000
MAX_DIFF_CHARS = 24_000


def _contained(workspace: Workspace, candidate: Path) -> Path | None:
    """Resolved candidate if it stays inside the workspace, else None."""
    return workspace.contained(candidate)


def glob_files(
    workspace: Workspace, pattern: str, path: str = ".", limit: int = 200
) -> dict[str, Any]:
    """Match workspace paths using a standard glob, including ``**``."""
    if not pattern.strip():
        raise ValueError("pattern cannot be empty")
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("pattern must stay inside the workspace")
    base = workspace.resolve(path, must_exist=True)
    if not base.is_dir():
        raise ValueError(f"not a directory: {path}")
    limit = max(1, min(int(limit), 500))
    matches: list[str] = []
    for candidate in base.glob(pattern):
        try:
            relative = workspace.relative(candidate)
        except ValueError:
            continue
        if any(part in IGNORED_DIRS for part in Path(relative).parts):
            continue
        matches.append(relative + ("/" if candidate.is_dir() else ""))
        if len(matches) >= limit:
            break
    matches.sort()
    return {"pattern": pattern, "path": workspace.relative(base), "matches": matches, "truncated": len(matches) >= limit}


def grep_files(
    workspace: Workspace,
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    case_sensitive: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Search file contents with a regular expression and optional file glob."""
    if not pattern:
        raise ValueError("pattern cannot be empty")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        expression = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
    base = workspace.resolve(path, must_exist=True)
    candidates = [base] if base.is_file() else base.glob(glob or "**/*")
    limit = max(1, min(int(limit), 300))
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        inside = _contained(workspace, candidate)
        if inside is None or not inside.is_file() or is_probably_binary(inside):
            continue
        try:
            if inside.stat().st_size > MAX_SEARCH_BYTES:
                continue
            with inside.open("r", encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    if expression.search(line):
                        matches.append({"path": workspace.relative(inside), "line": number, "text": line.rstrip()[:300]})
                        if len(matches) >= limit:
                            return {"pattern": pattern, "matches": matches, "truncated": True}
        except OSError:
            continue
    return {"pattern": pattern, "matches": matches, "truncated": False}


def list_files(workspace: Workspace, path: str = ".", depth: int = 2, limit: int = 200) -> dict[str, Any]:
    base = workspace.resolve(path, must_exist=True)
    if not base.is_dir():
        raise ValueError(f"not a directory: {path}")
    depth = max(0, min(int(depth), 4))
    limit = max(1, min(int(limit), 500))
    entries: list[str] = []
    base_parts = len(base.parts)
    for current, dirs, files in os.walk(base):
        level = len(Path(current).parts) - base_parts
        dirs[:] = sorted(
            d for d in dirs
            if d not in IGNORED_DIRS and not d.startswith(".artinium")
        )
        if level >= depth:
            dirs[:] = []
        for directory in dirs:
            inside = _contained(workspace, Path(current) / directory)
            if inside is None:
                continue
            entries.append(workspace.relative(inside) + "/")
        for filename in sorted(files):
            inside = _contained(workspace, Path(current) / filename)
            if inside is None:
                continue
            entries.append(workspace.relative(inside))
        if len(entries) >= limit:
            break
    return {"path": workspace.relative(base), "entries": entries[:limit], "truncated": len(entries) > limit}


def read_file(
    workspace: Workspace, path: str, start_line: int = 1, end_line: int | None = None
) -> dict[str, Any]:
    target = workspace.resolve(path, must_exist=True)
    inside = _contained(workspace, target)
    if inside is None:
        raise ValueError(f"path outside the workspace rejected: {path}")
    target = inside
    if not target.is_file():
        raise ValueError(f"not a file: {path}")
    if is_probably_binary(target):
        raise ValueError(f"binary file rejected: {path}")
    start = max(1, int(start_line))
    requested_end = int(end_line) if end_line is not None else start + 399
    requested_end = max(start, min(requested_end, start + 999))
    selected: list[tuple[int, str]] = []
    last_seen = 0
    has_more = False
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            last_seen = number
            if number < start:
                continue
            if number > requested_end:
                has_more = True
                break
            selected.append((number, line.rstrip("\n\r")))
    content = "\n".join(f"{number}: {line}" for number, line in selected)
    if len(content) > MAX_READ_CHARS:
        content = content[:MAX_READ_CHARS] + "\n… output truncated"
    return {
        "path": workspace.relative(target),
        "start_line": start,
        "end_line": selected[-1][0] if selected else start,
        "total_lines": None if has_more else last_seen,
        "content": content,
        "truncated": has_more or len(content) >= MAX_READ_CHARS,
    }


def write_file(workspace: Workspace, path: str, content: str) -> dict[str, Any]:
    if len(content) > MAX_WRITE_CHARS:
        raise ValueError(f"content is too large ({len(content)} characters)")
    target = workspace.resolve(path)
    if target.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path}")
    inside = _contained(workspace, target)
    if inside is None:
        raise ValueError(f"path outside the workspace rejected: {path}")
    target = inside
    created = not target.exists()
    if target.exists() and target.is_dir():
        raise ValueError(f"path is a directory: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".artinium-tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    return {"path": workspace.relative(target), "chars": len(content), "created": created}


def edit_file(
    workspace: Workspace, path: str, old_text: str, new_text: str, replace_all: bool = False
) -> dict[str, Any]:
    target = workspace.resolve(path, must_exist=True)
    if target.is_symlink():
        raise ValueError(f"refusing to edit through a symlink: {path}")
    inside = _contained(workspace, target)
    if inside is None:
        raise ValueError(f"path outside the workspace rejected: {path}")
    target = inside
    if not target.is_file() or is_probably_binary(target):
        raise ValueError(f"not a text file: {path}")
    if target.stat().st_size > MAX_EDIT_BYTES:
        raise ValueError("file is too large for edit_file; use a more specific tool")
    if not old_text:
        raise ValueError("old_text cannot be empty")
    content = target.read_text(encoding="utf-8")
    occurrences = content.count(old_text)
    if occurrences == 0:
        raise ValueError("old_text was not found; read the file and use exact text")
    if occurrences > 1 and not replace_all:
        raise ValueError(f"old_text appears {occurrences} times; add context or use replace_all")
    updated = content.replace(old_text, new_text, -1 if replace_all else 1)
    temporary = target.with_name(target.name + ".artinium-tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(target)
    relative = workspace.relative(target)
    diff = "\n".join(unified_diff(
        content.splitlines(),
        updated.splitlines(),
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
        n=3,
        lineterm="",
    ))
    truncated = len(diff) > MAX_DIFF_CHARS
    if truncated:
        diff = diff[:MAX_DIFF_CHARS] + "\n… diff truncated"
    return {
        "path": relative,
        "replacements": occurrences if replace_all else 1,
        "diff": diff,
        "diff_truncated": truncated,
    }


def search_files(
    workspace: Workspace, query: str, path: str = ".", case_sensitive: bool = False, limit: int = 50
) -> dict[str, Any]:
    if not query:
        raise ValueError("query cannot be empty")
    base = workspace.resolve(path, must_exist=True)
    limit = max(1, min(int(limit), 100))
    needle = query if case_sensitive else query.casefold()
    matches: list[dict[str, Any]] = []
    candidates = [base] if base.is_file() else walk_files(base)
    for candidate in candidates:
        inside = _contained(workspace, candidate)
        if inside is None or is_probably_binary(inside):
            continue
        try:
            if inside.stat().st_size > MAX_SEARCH_BYTES:
                continue
        except OSError:
            continue
        try:
            with inside.open("r", encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle in haystack:
                        matches.append({
                            "path": workspace.relative(inside),
                            "line": number,
                            "text": line.rstrip()[:300],
                        })
                        if len(matches) >= limit:
                            return {"query": query, "matches": matches, "truncated": True}
        except OSError:
            continue
    return {"query": query, "matches": matches, "truncated": False}
