# Artinium

Artinium is a fullscreen local coding agent for the terminal. It communicates
only with [Ollama](https://ollama.com/), discovers installed models, uses real
workspace tools, and keeps context small enough to remain useful with 7B–14B
models.

## Requirements and installation

1. Install Ollama for your system and start it:

   ```bash
   ollama serve
   ```

2. In another terminal, download a model with tool support. For example:

   ```bash
   ollama pull qwen3:8b
   ```

3. Install Artinium (Python 3.11 or later):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

4. Enter the project you want to work on and run:

   ```bash
   artinium
   # o
   python -m artium
   ```

Use `artinium --check` to diagnose the connection without opening the TUI, or
select the starting directory with `artinium --workspace /path/to/project`.

## Usage

Artinium automatically selects a suitable model; if only one model exists, it
uses that model. The side panel shows the active session, the exact input-token
count reported by Ollama for the latest request alongside its configured context
limit, and the number of tools run.

| Key | Action |
|---|---|
| `Ctrl+P` | open the actions menu |
| `Model settings` menu | configure reasoning, temperature, output limit, and context window |
| `Sessions` menu | create, switch, rename, pin, or delete local sessions |
| `Ctrl+O` | change workspace |
| `Ctrl+T` | toggle detail for upcoming tool calls |
| `Ctrl+B` | show or hide the sidebar |
| `Esc`, then `Esc` | interrupt the current task with one inline confirmation |
| `Ctrl+C` | stop the current task, or press twice to exit when idle |

The prompt stays writable while Artinium is working. Each press of `Enter` adds a
message to a FIFO queue. `Ctrl+E` opens the queue: use the arrow keys to select
any message, `Enter` to edit it, or `D`/`Delete` to remove it. Queued messages
run automatically, in order, when the current task ends.

With focus in the prompt, `Up` recalls the most recently submitted message and
continues backwards through earlier prompts; `Down` moves forward and restores
an unfinished draft after the newest entry.

Text selected in any input field can be copied with `Ctrl+C` and pasted with
`Ctrl+V`. In the main prompt, `Ctrl+C` without a selection still stops the
current task or starts the exit confirmation.

The prompt also accepts `/help`, `/menu`, `/model`, `/settings`, `/context`,
`/sessions`, `/workspace`, `/sidebar`, `/compact`, `/undo`, `/permissions`,
`/theme`, and `/update`. Use
`/autocompact` to open its picker, or `/autocompact off`, `/autocompact 70`,
`/autocompact 80`, or `/autocompact 90` to set it directly.

`Ctrl+P` → `Model` groups model operations in one place: choose a detected
model, configure reasoning / temperature / output, or open context settings.
The model list shows its supported tools, reasoning, and vision capabilities on
the second line. During a task, a newly selected model becomes pending and is
applied after that task, before the next queued message. Every setting starts as
**Automatic**, preserving Artinium's established defaults. `Context settings`
contains the context window, manual compact action, and automatic compaction
policy. The context window has the largest RAM impact: Automatic uses up to
32,768 tokens, while a manual value lets the user choose a smaller or larger
window within the model's limit.

The available tools are `list_files`, `read_file`, `write_file`, `edit_file`,
`search_files`, `glob`, `grep`, `question`, `run_command`, `web_search`, and
`fetch_url`. `glob` finds paths, `grep` performs regular-expression content
searches, and `question` pauses the agent with a focused option/free-text UI.
Web search uses
DuckDuckGo's non-JavaScript results, while `fetch_url` reads one specific public
page. Every web result records the source URL and retrieval time; Artinium also
shows the sources used after a web-based answer. File paths are resolved and
validated inside the workspace. Expanding an `edit_file` tool call shows a
unified diff. Commands with broad or irreversible impact (for example recursive
deletion, `git reset --hard`, `dd`, or database drops) require visible
confirmation with the exact command; `sudo` alone does not.

File writes, edits, and shell commands have independent `Ask`, `Allow`, and
`Deny` policies under `Ctrl+P` → `Artinium` → `Tool permissions`. `Ask` shows
the exact target and offers allow once, always allow, or deny. Dangerous shell
commands still require their additional high-risk confirmation. `/undo`
restores the latest `write_file` or `edit_file` snapshot and records the
reversal in model context.

Dragging files into a terminal, or pasting their paths, creates compact
`[File n]` / `[Image n]` prompt attachments. Text is included with a bounded
size; images are sent through Ollama only when the selected model advertises
vision support. Native bitmap clipboard transfer is terminal-dependent, so the
portable route is pasting or dropping the image file path.

Completed turns end with model, elapsed time, and output speed. The model picker
uses monochrome capability marks for thinking, tools, and vision. `Ctrl+P` →
`Artinium` contains application info, tool permissions, update checks, and the
built-in Textual theme collection (including Nord, Gruvbox, Catppuccin,
Solarized, Dracula, Tokyo Night, and light themes). Automatic theme mode checks
common desktop/Omarchy environment hints and otherwise follows a safe terminal
default. When online, startup checks PyPI for a newer `artinium-local` release
and offers install now, later, or ignore this version.

The interface adapts its panels to terminal width. Messages use short
transitions, the status animates while Ollama is working, replies support
Markdown, and every tool call can be expanded to inspect its result.

## Context design

- concise system prompt and compact tool schemas;
- no file tree or file content is injected in advance: the model discovers it;
- reads, searches, and commands have output limits;
- as context fills, older tool results are removed and old dialogue becomes a
  short deterministic memory;
- `/compact` performs that reduction on demand; auto compact is enabled at 80%
  by default and uses Ollama's latest exact prompt count plus local recent
  message accounting to compact before the context window is exhausted;
- at most 12 rounds per request to prevent loops.

## Development and v0.2 limits

Run the suite without external services:

```bash
python -m unittest discover -s tests -v
```

This version does not sandbox shell processes: `run_command` runs with the
workspace as its current directory and protects dangerous patterns through
confirmation, but an allowed command retains the user's permissions. Tool-call
quality depends on the model; models that Ollama marks with the `tools`
capability are recommended. Sessions are persisted locally per workspace; cloud
model providers are not supported.
