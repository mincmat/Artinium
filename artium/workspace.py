from __future__ import annotations

import os
from pathlib import Path


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: Path):
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"not a directory: {resolved}")
        self.root = resolved

    def resolve(self, path: str | Path = ".", *, must_exist: bool = False) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve(strict=False)
        else:
            candidate = (self.root / raw).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"path outside the workspace rejected: {path}") from exc
        if must_exist and not candidate.exists():
            raise WorkspaceError(f"does not exist: {path}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def contained(self, path: Path) -> Path | None:
        """Resolve symlinks and return the path only if it stays inside.

        Returns None when the path escapes the workspace (e.g. a planted
        ``ln -s /etc/passwd``) or cannot be resolved. Use it right before
        any open/read/write, not just at argument-validation time.
        """
        try:
            resolved = path.resolve()
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return None
        return resolved

    def replace_root(self, root: Path) -> None:
        replacement = Workspace(root)
        self.root = replacement.root


IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
        return b"\0" in chunk
    except OSError:
        return True


def walk_files(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".artinium"))
        for name in sorted(files):
            yield Path(current) / name
