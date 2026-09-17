"""Human-facing explanations for failures Artinium can recognize safely."""

from __future__ import annotations

from typing import Any


def explain(error: Exception | str, *, area: str = "tool") -> str:
    """Return an actionable explanation without exposing a Python traceback."""
    raw = str(error).strip()
    text = raw.lower()

    if "timed out" in text:
        if area == "ollama":
            return "Ollama took too long to respond. It may still be processing a large context; reduce the context window or use a faster model."
        if area == "command":
            return "The command exceeded its time limit and was stopped. Increase its timeout only if it is expected to take longer."
        return "The operation took too long and was stopped. Check the connection, then try again."
    if "not installed" in text or "not in path" in text:
        return "Ollama is not installed or cannot be found. Install Ollama, then restart Artinium."
    if "connection to ollama was lost" in text or "could not connect" in text:
        return "Artinium lost its connection to Ollama. Make sure Ollama is running, then try again."
    if "ollama returned 400" in text:
        return "Ollama rejected this request. The selected model may not support the requested feature or context size."
    if "ollama returned 500" in text:
        return "Ollama could not complete this request. Try a smaller context window or another model."
    if "invalid json" in text:
        return "Ollama returned an invalid response. Retry the request; if it repeats, switch models or update Ollama."
    if "outside the workspace" in text:
        return "This path is outside the current workspace, so Artinium blocked it for safety."
    if "does not exist" in text or "no such file" in text:
        return "The requested file or folder does not exist in this workspace."
    if "permission denied" in text:
        return "Permission was denied by the operating system. Check the file permissions or choose another location."
    if "query cannot be empty" in text or "command cannot be empty" in text:
        return "The tool received an empty request. Ask the model to provide the missing value and try again."
    if "local addresses are not allowed" in text or "private or local network" in text:
        return "For safety, web tools cannot access local or private network addresses."
    if "url must use http or https" in text:
        return "This is not a valid web address. Use a complete http:// or https:// URL."
    if "could not resolve host" in text:
        return "The website address could not be found. Check the URL or your internet connection."
    if "too many redirects" in text:
        return "The website redirected too many times, so Artinium stopped following it."
    if "unsupported content type" in text:
        return "That link does not point to a readable web page or text document."
    if "unknown tool" in text:
        return "The selected model requested a tool Artinium does not provide. Try another model or ask it to continue without that tool."

    if area == "ollama":
        return "Ollama could not complete the request. Try again; if it repeats, check the model and Ollama logs."
    return "This action could not be completed. Review the tool details and try again."


def technical_detail(error: Exception | str) -> str:
    """Keep the original diagnostic available in structured tool data."""
    return str(error).strip() or type(error).__name__
