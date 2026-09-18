from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from dataclasses import dataclass

import httpx


REPOSITORY = "mincmat/Artinium"
COMMIT_URL = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
PYPROJECT_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/main/pyproject.toml"
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
            project_response = await client.get(PYPROJECT_URL)
            project_response.raise_for_status()
        latest_commit = str(commit_response.json().get("sha") or "")
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', project_response.text, re.MULTILINE)
        latest = version_match.group(1) if version_match else ""
        return UpdateInfo(current, latest, installed_commit(), latest_commit) if latest else None
    except (httpx.HTTPError, ValueError):
        return None


async def install_update() -> tuple[bool, str]:
    process = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        f"curl -fsSL {INSTALLER_URL} | bash",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    text = output.decode(errors="replace").strip()
    return process.returncode == 0, text[-2000:]
