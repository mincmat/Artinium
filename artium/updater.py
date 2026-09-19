from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sys
from dataclasses import dataclass

import httpx


REPOSITORY = "mincmat/Artinium"
COMMIT_URL = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
PYPROJECT_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{{commit}}/pyproject.toml"
INSTALLER_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/main/install"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


@dataclass(slots=True)
class UpdateInfo:
    current: str
    latest: str
    current_commit: str = ""
    latest_commit: str = ""

    @property
    def available(self) -> bool:
        if self.current_commit and self.latest_commit:
            return self.current_commit != self.latest_commit
        return _version_tuple(self.latest) > _version_tuple(self.current)

    @property
    def identity(self) -> str:
        """The revision to remember when a user dismisses this update."""
        return self.latest_commit or self.latest


def installed_commit() -> str:
    """Read the source revision recorded by Artinium's GitHub installer."""
    state_path = Path(sys.prefix).parent / "install.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    commit = state.get("commit") if isinstance(state, dict) else ""
    return str(commit) if isinstance(commit, str) else ""


async def check_for_update(current: str) -> UpdateInfo | None:
    """Compare the installed GitHub revision with Artinium's main branch."""
    try:
        async with httpx.AsyncClient(timeout=4, follow_redirects=True) as client:
            commit_response = await client.get(COMMIT_URL)
            commit_response.raise_for_status()
        latest_commit = str(commit_response.json().get("sha") or "")
        if not latest_commit:
            return None
        async with httpx.AsyncClient(timeout=4, follow_redirects=True) as client:
            project_response = await client.get(PYPROJECT_URL.format(commit=latest_commit))
            project_response.raise_for_status()
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', project_response.text, re.MULTILINE)
        latest = version_match.group(1) if version_match else ""
        return UpdateInfo(current, latest, installed_commit(), latest_commit) if latest else None
    except (httpx.HTTPError, ValueError):
        return None


def installer_url(commit: str | None) -> str:
    """Installer pinned to a verified commit instead of floating ``main``."""
    if commit:
        return f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/install"
    return INSTALLER_URL


async def install_update(commit: str | None = None) -> tuple[bool, str]:
    # Download fully first (never `curl | bash`): a cut connection must not
    # execute a truncated script, and the code matches the reviewed commit.
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(installer_url(commit))
            response.raise_for_status()
        script = response.text
    except (httpx.HTTPError, ValueError) as exc:
        return False, f"Could not download the installer: {exc}"
    if not script.startswith("#!/usr/bin/env bash"):
        return False, "Downloaded installer failed a sanity check; refusing to run it."
    import tempfile

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
            handle.write(script)
            path = handle.name
        process = await asyncio.create_subprocess_exec(
            "bash",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
    finally:
        try:
            Path(path).unlink()
        except (OSError, NameError):
            pass
    text = output.decode(errors="replace").strip()
    return process.returncode == 0, text[-2000:]
