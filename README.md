# Artinium

Artinium is a local coding agent for the terminal. It runs through [Ollama](https://ollama.com/), works with the models already installed on the computer, and helps inspect, edit and run code inside a chosen workspace.

Prompts, source code and model inference stay on the computer. Web search and URL fetches only leave the computer when the user asks Artinium to use them.

## Main features

- Fullscreen terminal interface with persistent local sessions.
- Detects installed Ollama models and shows tool, reasoning and vision capabilities.
- Reads, writes and edits files; runs shell commands; searches and fetches the web.
- Includes `glob`, regular-expression `grep` and an interactive `question` tool for missing choices.
- Permission policy for actions that can modify files or execute commands: ask, allow or deny.
- Shows tool activity, file diffs, output, model response time and tokens per second inline in the chat.
- Context controls: selectable context window, manual `/compact` and configurable automatic compaction.
- Change history with `/undo` for edits made during the session.
- Starts Ollama when possible and releases models it loaded when Artinium exits.
- Supports drag-and-drop paths and image paste when the selected model accepts vision input.

## Quick install

On Linux or macOS, run this single command:

```bash
curl -fsSL https://raw.githubusercontent.com/mincmat/Artinium/main/install | bash
```

The installer downloads the latest Artinium source, shows its progress, creates an isolated environment and installs the `artinium` command without requiring `sudo`.

## Requirements and installation

Artinium supports Python 3.11 or newer and requires Ollama.

1. Install [Ollama](https://ollama.com/) for the operating system.
2. Download a model with tool support. For example:

   ```bash
   ollama pull qwen3:8b
   ```

3. Install Artinium with the one-line installer. It detects Linux or macOS, shows download progress, creates an isolated environment and installs the `artinium` command without `sudo`:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/mincmat/Artinium/main/install | bash
   ```

   If `~/.local/bin` is not already on your `PATH`, the installer prints the exact `export` command to add it.

For source development, clone and install the project manually:

   ```bash
   git clone https://github.com/mincmat/Artinium.git
   cd Artinium
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   ```

4. Open the directory you want Artinium to work in and start it:

   ```bash
   artinium
   ```

Use `artinium --check` to verify the Ollama connection without opening the interface. Use `artinium --workspace /path/to/project` to select a workspace explicitly.

## Using Artinium

Artinium selects a suitable local model when it starts. Change it at any time from `Ctrl+P` → **Model**; a change requested while a task is active is applied to the next message.

The agent operates inside the current workspace by default. A request that targets a path outside it is blocked unless the user explicitly asks for that location. Commands and file changes use the current user's permissions, so review the permission prompt before allowing an action.

| Key or command | Action |
|---|---|
| `Ctrl+P` or `/menu` | Open the main menu |
| `/model` | Choose a model or open model settings |
| `/context` | Configure context window and compaction |
| `/compact` | Summarize older conversation context now |
| `/autocompact` | Choose when automatic compaction runs |
| `/permissions` | Change the tool permission policy |
| `/undo` | Revert a recorded file change |
| `/sessions` | Create, switch, rename, pin or delete sessions |
| `Ctrl+O` or `/workspace` | Choose another workspace |
| `Ctrl+B` or `/sidebar` | Show or hide the sidebar |
| `Esc`, then `Esc` | Interrupt the current task after confirmation |
| `Ctrl+C` | Stop a task, or press twice while idle to exit |

While Artinium is working, `Enter` adds a message to the queue. `Ctrl+E` opens the queue to edit or remove messages before they run. In the prompt, `Up` and `Down` navigate submitted messages and restore an unfinished draft.

## Models and privacy

Artinium sends inference requests only to the local Ollama service. It does not require an API key or a cloud model provider.

Tool quality depends on the selected model. Models that Ollama marks with the `tools` capability are recommended for coding tasks. Vision input is available only for models that advertise vision support. A model that does not support tools can still chat, but Artinium explains that it cannot execute actions.

## Local data

Settings, sessions and undo history are stored locally in `.artinium` under the selected workspace. This directory can contain conversation history and should normally stay out of version control.

The application never overwrites a file silently: writing and editing actions are shown in the chat, follow the active permission policy, and can be inspected before the user allows them.

## Source development

```bash
git clone https://github.com/mincmat/Artinium.git
cd Artinium
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
artinium --check
```

Run the interface from a project directory with `artinium`, or use `python -m artium` while developing the package.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers agent behavior, workspace tools and the terminal interface.

## License

Artinium is released under the [MIT License](LICENSE).
