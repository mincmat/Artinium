from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .ollama_client import OllamaClient, OllamaUnavailable


async def _check() -> int:
    client = OllamaClient()
    try:
        models = await client.list_models()
    except OllamaUnavailable as exc:
        print(f"Ollama unavailable: {exc}")
        print("Start it with: ollama serve")
        return 1
    finally:
        await client.close()

    if not models:
        print("Ollama is available, but no models are installed.")
        print("Download one with: ollama pull qwen3:8b")
        return 1
    selected = OllamaClient.choose_model(models)
    print(f"Ollama available. {len(models)} model(s). Selected: {selected.display_name}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="artinium", description="Local agent for Ollama")
    parser.add_argument("--check", action="store_true", help="check Ollama and exit")
    parser.add_argument("--workspace", type=Path, help="working directory (default: current)")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(asyncio.run(_check()))

    from .ui.app import ArtiumApp

    workspace = (args.workspace or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {workspace}")
    ArtiumApp(workspace).run()


if __name__ == "__main__":
    main()
