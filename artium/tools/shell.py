from __future__ import annotations

import asyncio
import os
import re
import signal
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..workspace import Workspace

MAX_OUTPUT = 16_000
RECURSIVE_REMOVE = re.compile(r"\brm\b[^\n]*(?:--recursive\b|-[^\s]*[rR])", re.IGNORECASE)
MASS_REMOVE = re.compile(r"\brm\b[^\n]*(?:\*|(?:^|\s)(?:\.|\.\.|/|~)(?:\s|$))", re.IGNORECASE)
FIND_DELETE = re.compile(r"\bfind\b[^\n]*\s-delete\b", re.IGNORECASE)
GIT_DESTRUCTIVE = re.compile(r"\bgit\s+(?:reset\s+--hard|clean\b[^\n]*-[^\s]*f)", re.IGNORECASE)
DISK_DESTRUCTIVE = re.compile(r"\b(?:mkfs(?:\.|\s)|wipefs\b|dd\b[^\n]*\bof=|shred\b)", re.IGNORECASE)
RECURSIVE_PERMISSIONS = re.compile(r"\b(?:chmod|chown)\b[^\n]*-[^\s]*[Rr]", re.IGNORECASE)
DATABASE_DESTRUCTIVE = re.compile(r"\bDROP\s+(?:DATABASE|TABLE|SCHEMA)\b", re.IGNORECASE)


@dataclass(slots=True)
class ConfirmationRequest:
    command: str
    reason: str


ConfirmCallback = Callable[[ConfirmationRequest], Awaitable[bool]]


def command_risk(command: str) -> str | None:
    """Return a human reason only for commands with broad or irreversible impact."""
    if RECURSIVE_REMOVE.search(command):
        return "recursively deletes a directory tree"
    if MASS_REMOVE.search(command):
        return "may delete many files at once"
    if FIND_DELETE.search(command):
        return "deletes every file matched by find"
    if GIT_DESTRUCTIVE.search(command):
        return "discards Git changes or untracked files"
    if DISK_DESTRUCTIVE.search(command):
        return "can irreversibly modify storage"
    if RECURSIVE_PERMISSIONS.search(command):
        return "recursively changes permissions or ownership"
    if DATABASE_DESTRUCTIVE.search(command):
        return "destroys database data"
    return None


def is_dangerous(command: str) -> bool:
    """Compatibility helper for callers and tests."""
    return command_risk(command) is not None


async def run_command(
    workspace: Workspace,
    command: str,
    timeout: int = 60,
    confirm: ConfirmCallback | None = None,
) -> dict[str, Any]:
    if not command.strip():
        raise ValueError("command cannot be empty")
    try:
        timeout = max(1, min(int(timeout), 300))
    except (TypeError, ValueError):
        timeout = 60
    risk = command_risk(command)
    if risk:
        request = ConfirmationRequest(command, risk)
        if confirm is None or not await confirm(request):
            return {"command": command, "cancelled": True, "reason": "not confirmed by the user"}

    process = await asyncio.create_subprocess_shell(
        command,
        cwd=workspace.root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def stop_process_tree() -> None:
        if process.returncode is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        await process.wait()

    async def drain(stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
        kept = bytearray()
        truncated = False
        drained = 0
        if stream is None:
            return b"", False
        # DRAIN_BUDGET bounds total bytes pulled off the pipe: without it a
        # chatty process spins this loop until timeout even though output is
        # already capped at MAX_OUTPUT.
        while chunk := await stream.read(8192):
            drained += len(chunk)
            remaining = MAX_OUTPUT - len(kept)
            if remaining > 0:
                kept.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
            if drained >= 512_000:
                truncated = True
                break
        return bytes(kept), truncated

    stdout_task = asyncio.create_task(drain(process.stdout))
    stderr_task = asyncio.create_task(drain(process.stderr))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        await stop_process_tree()
        stdout, stdout_cut = await stdout_task
        stderr, stderr_cut = await stderr_task
        timeout_note = f"timed out after {timeout}s"
        decoded_error = stderr.decode(errors="replace")
        return {
            "command": command,
            "stdout": stdout.decode(errors="replace"),
            "stderr": f"{decoded_error}\n{timeout_note}".strip(),
            "exit_code": None,
            "timed_out": True,
            "truncated": stdout_cut or stderr_cut,
        }
    except asyncio.CancelledError:
        await stop_process_tree()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    stdout, stdout_cut = await stdout_task
    stderr, stderr_cut = await stderr_task
    return {
        "command": command,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "exit_code": process.returncode,
        "truncated": stdout_cut or stderr_cut,
    }
