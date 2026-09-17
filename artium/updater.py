from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

import httpx


PYPI_URL = "https://pypi.org/pypi/artinium-local/json"


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

    @property
    def available(self) -> bool:
        return _version_tuple(self.latest) > _version_tuple(self.current)


async def check_for_update(current: str) -> UpdateInfo | None:
    """Check PyPI quickly and stay silent when offline or not published."""
    try:
        async with httpx.AsyncClient(timeout=4, follow_redirects=True) as client:
            response = await client.get(PYPI_URL)
            response.raise_for_status()
        latest = str((response.json().get("info") or {}).get("version") or "")
        return UpdateInfo(current, latest) if latest else None
    except (httpx.HTTPError, ValueError):
        return None


async def install_update() -> tuple[bool, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "artinium-local",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    text = output.decode(errors="replace").strip()
    return process.returncode == 0, text[-2000:]
