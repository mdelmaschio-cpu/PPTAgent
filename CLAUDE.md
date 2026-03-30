# CLAUDE.md — PPTAgent Codebase Guide

This document explains the codebase structure, development workflows, and conventions for AI assistants working in this repository.

## Project Overview

**PPTAgent** is an agentic framework for reflective PowerPoint presentation generation (EMNLP 2025). It contains two cooperating packages:

- **`deeppresenter/`** — The primary, production-ready agent system (v1.1.x). Multi-agent pipeline: Research → Design → PPTX output. Exposed as the `pptagent` CLI.
- **`pptagent/`** — The original module (v1.0.x). Template-based presentation generation; also provides the `pptagent-mcp` MCP server.

**Supported platforms**: Linux and macOS only (POSIX).
**Python**: >= 3.11

---

## Repository Structure

```
PPTAgent/
├── deeppresenter/          # Primary agent framework
│   ├── agents/             # Agent implementations (Research, Design, PPTAgent wrapper)
│   ├── cli/                # Typer-based CLI (commands.py, dependency.py, model.py)
│   ├── docker/             # Dockerfiles: Host.Dockerfile, SandBox.Dockerfile
│   ├── html2pptx/          # Node.js script: HTML → PPTX conversion
│   ├── roles/              # YAML system-prompt configs for each agent role
│   ├── test/               # pytest test suite
│   ├── tools/              # Tool implementations (research, web, image, etc.)
│   ├── tui/                # Textual-based terminal UI
│   ├── utils/              # config.py, constants.py, typings.py, log.py, mcp_client.py, webview.py
│   ├── main.py             # AgentLoop orchestrator (top-level entry for generation)
│   ├── __init__.py         # Exports __version__
│   ├── __main__.py
│   ├── config.yaml.example # Copy to ~/.config/deeppresenter/config.yaml
│   ├── mcp.json.example    # Copy to ~/.config/deeppresenter/mcp.json
│   └── serve.sh            # Local model server startup script
├── pptagent/               # Original template-based module
│   ├── document/           # Document parsing (document.py, doc_utils.py, element.py)
│   ├── presentation/       # PPTX manipulation (presentation.py, shapes.py, layout.py)
│   ├── prompts/            # Jinja2/txt prompt templates
│   ├── roles/              # YAML role configs
│   ├── response/           # Pydantic response models (outline.py, induct.py, pptgen.py)
│   ├── scripts/            # Helper scripts
│   ├── templates/          # Built-in PPTX templates (default, beamer, cip, hit, thu, ucas)
│   ├── test/               # pytest test suite
│   ├── agent.py            # Turn-based agent (LLM prompting + tool dispatch)
│   ├── apis.py             # External API clients (web search, code exec, image gen)
│   ├── llms.py             # LLM wrapper (OpenAI-compatible)
│   ├── mcp_server.py       # FastMCP server (entry: pptagent-mcp)
│   ├── pptgen.py           # PPTGen abstract base + PPTAgent implementation
│   ├── induct.py           # Template induction from existing PPTX files
│   ├── ppteval.py          # Presentation evaluation (content, design, coherence)
│   ├── model_utils.py      # Model utilities
│   ├── multimodal.py       # Image labeling / multimodal processing
│   └── utils.py            # Logging, image processing, language detection
├── resource/               # Example inputs and case studies
├── webui.py                # Gradio web UI (SessionManager + ChatDemo)
├── docker-compose.yml      # Compose: deeppresenter-host on port 7861
├── pyproject.toml          # Single source of truth for metadata and deps
└── .pre-commit-config.yaml # Code quality hooks
```

---

## Development Setup

### Installation

```bash
# Recommended: use uv
pip install uv
uv pip install -e ".[full]"

# Or standard pip
pip install -e ".[full]"
```

The `[full]` extras add: `transformers`, `fasttext`, `einops`, `peft`, `huggingface_hub`, `timm`, `unoserver`.

### External Dependencies

Install via the onboarding wizard or manually:

```bash
pptagent onboard   # Interactive setup (installs Node.js, Docker, Playwright, poppler, llama.cpp if needed)
playwright install chromium
```

### Configuration

1. Copy `deeppresenter/config.yaml.example` → `~/.config/deeppresenter/config.yaml`
2. Copy `deeppresenter/mcp.json.example` → `~/.config/deeppresenter/mcp.json`
3. Set LLM endpoints and API keys in `config.yaml`

**Key environment variables** (override constants in `deeppresenter/utils/constants.py`):

| Variable | Default | Description |
|---|---|---|
| `DEEPPRESENTER_WORKSPACE_BASE` | `~/.cache/deeppresenter` | Session workspace root |
| `DEEPPRESENTER_LOG_LEVEL` | `20` (INFO) | Logging level |
| `DEEPPRESENTER_MAX_LOGGING_LENGTH` | `1024` | Max chars logged per entry |
| `CONTEXT_LENGTH_LIMIT` | `200000` | Token context limit |
| `MAX_TOOLCALL_PER_TURN` | `7` | Max parallel tool calls per turn |
| `RETRY_TIMES` | `10` | Agent retry count |
| `MCP_CALL_TIMEOUT` | `1800` | MCP tool call timeout (seconds) |
| `TOOL_CUTOFF_LEN` | `4096` | Tool output truncation length |

---

## CLI Entry Points

```
pptagent          → deeppresenter.cli:main
pptagent-mcp      → pptagent.mcp_server:main
```

**CLI commands** (all via `pptagent <command>`):

| Command | Purpose |
|---|---|
| `onboard` | Interactive setup wizard; installs system dependencies |
| `generate` | Generate a presentation (accepts file/URL attachments) |
| `config` | View current configuration |
| `reset` | Reset configuration to defaults |
| `serve` | Start local LLM inference service (llama.cpp) |
| `clean` | Clean workspace |

**Web UI**:
```bash
python webui.py   # Gradio at http://localhost:7861
```

**Docker Compose**:
```bash
docker compose up   # Host service on port 7861
```

---

## Key Classes and Modules

### `deeppresenter/main.py` — `AgentLoop`

Top-level orchestrator. Coordinates the full pipeline:
1. Research agent gathers information → produces `manuscript.md`
2. Design agent or PPTAgent converts manuscript to slides
3. HTML → PPTX conversion via Playwright

```python
loop = AgentLoop(config, agent_env, workspace, language)
async for msg in loop.run(request):   # yields ChatMessage or str
    ...
```

### `deeppresenter/agents/agent.py` — `Agent`

Base class for all agents. Handles:
- LLM interaction (OpenAI-compatible API)
- MCP tool loading and dispatch (up to `MAX_TOOLCALL_PER_TURN` parallel calls)
- Context folding when approaching `CONTEXT_LENGTH_LIMIT`
- Message history and token tracking

### `deeppresenter/utils/typings.py` — Core Data Models

```python
class ChatMessage(BaseModel):
    role: Role           # system | user | assistant | tool
    content: str | list[dict]
    tool_calls: list[ToolCall]
    reasoning: str
    cost: CompletionUsage

class InputRequest(BaseModel):
    instruction: str
    attachments: list[Path]
    convert_type: ConvertType   # DEEPPRESENTER | PPTAGENT
    pages: str | int

class ConvertType(str, Enum):
    DEEPPRESENTER = "deeppresenter"   # Free-form HTML-based generation
    PPTAGENT = "pptagent"             # Template-based generation
```

### `deeppresenter/utils/config.py` — `DeepPresenterConfig`

Loads and validates `config.yaml`. Exposes LLM endpoint management and `get_json_from_response()` for extracting JSON from LLM outputs (uses `json_repair` for robustness).

### `pptagent/pptgen.py` — `PPTGen` / `PPTAgent`

Template-based generation pipeline:
1. `set_reference(slide_induction, presentation)` — load template
2. `generate_outline(manuscript)` — produce `Outline`
3. `generate_slides(outline)` — yield `SlidePage` objects

### `pptagent/presentation/presentation.py` — `Presentation`

Wraps `python-pptx`. Primary interface for reading and writing PPTX files.

### `pptagent/mcp_server.py`

FastMCP server exposing PPTAgent tools for external MCP clients (e.g., Claude Desktop).

---

## Session / Workspace Layout

Each generation run creates a session directory:

```
~/.cache/deeppresenter/<session_id>/
├── .history/
│   ├── deeppresenter-loop.log
│   └── tool_history.jsonl
├── .input_request.json
├── .tools.json              # Tool cache
├── manuscript.md            # Research agent output
├── slides/                  # HTML slide files
└── output.pptx              # Final presentation
```

---

## Testing

**Framework**: pytest with `asyncio_mode = "auto"`

```bash
# Run all tests
pytest

# Run only fast (non-LLM) tests
pytest -m "not llm and not parse"

# Run LLM tests (requires OPENAI_API_KEY)
pytest -m llm

# Parallel execution
pytest -n auto
```

**Test markers**:
- `llm` — requires `OPENAI_API_KEY` and LLM access
- `parse` — requires MinerU API
- `asyncio` — asyncio-based tests

Test directories:
- `deeppresenter/test/` — Playwright, MinerU, document, image, mermaid, data science
- `pptagent/test/` — pptgen, APIs, presentation manipulation, LLM, multimodal, induction

---

## Code Quality

Pre-commit hooks (`.pre-commit-config.yaml`):

```bash
pre-commit install   # Install hooks
pre-commit run --all-files   # Run manually
```

Hooks:
- **ruff** (v0.15.6) — linting (`--select I` for import sort) + formatting
- **pyupgrade** (v3.21.2) — enforce Python 3.11+ syntax
- **validate-pyproject** — validate `pyproject.toml`
- **pre-commit-hooks** — YAML check, trailing whitespace, symlink detection, debug statement detection

Ruff ignores (in `pyproject.toml`): `F403, F405, E741, E722`

---

## CI/CD

**GitHub Actions** (`.github/workflows/docker-publish.yml`):
- Triggers: push to `main`, pull requests, tags matching `v*.*.*`
- Builds and pushes two Docker images to GHCR:
  - `Host.Dockerfile` — main service image
  - `SandBox.Dockerfile` — sandboxed code execution environment

---

## Architecture Patterns

### Agent Loop Pattern

```
InputRequest
    └── Research Agent (web search, PDF parse, arxiv, etc.)
            └── manuscript.md
                    └── Design Agent (HTML slide generation)  ← ConvertType.DEEPPRESENTER
                    └── PPTAgent (template-based)             ← ConvertType.PPTAGENT
                            └── output.pptx
```

### Tool Calling

All tool calls go through MCP servers defined in `mcp.json`. Tools execute in parallel (up to `MAX_TOOLCALL_PER_TURN = 7`). Outputs exceeding `TOOL_CUTOFF_LEN` characters are truncated.

### Context Management

When token count approaches `CONTEXT_LENGTH_LIMIT` (200k), the agent compacts history: key information is summarized and saved to disk, then the conversation is pruned. See `MEMORY_COMPACT_MSG` and `CONTEXT_MODE_PROMPT` in `deeppresenter/utils/constants.py`.

### LLM Integration

Uses OpenAI-compatible API. Configure endpoints and models in `config.yaml`. The `oaib` package provides Anthropic model support through the same interface.

### MCP Tool Ecosystem

Default tools (configure in `mcp.json`):
- **Tavily** — web search
- **MinerU** — advanced PDF parsing
- **Firecrawl** — web crawling
- **Custom tools** in `deeppresenter/tools/` — research, code execution, image generation

---

## Common Gotchas

1. **Platform restriction**: The package only runs on Linux/macOS (POSIX). Windows is not supported.
2. **numpy < 2.0.0**: Hard pinned. Do not upgrade numpy beyond 1.x without testing.
3. **fastmcp version**: Pinned to `>=2.10.0,<2.14.0`. Changes to MCP API surface break tool integration.
4. **textual version**: Pinned to `>=0.89.1,<0.90`. The TUI depends on a specific API.
5. **Docker socket**: The Host container mounts `/var/run/docker.sock` to spawn the SandBox container for code execution.
6. **Playwright chromium**: Required for HTML-to-PPTX conversion via `deeppresenter/utils/webview.py`. Must be installed separately with `playwright install chromium`.
7. **Language detection**: Uses `fasttext` models. The `Language` class in `pptagent/utils.py` detects CJK vs Latin text to adjust slide content length.
8. **Version source**: `deeppresenter.__version__` is the dynamic version source for the whole package.

---

## Adding New Features

### New Agent Tool

1. Implement the tool in `deeppresenter/tools/`
2. Expose it via a FastMCP server or add to an existing MCP server
3. Register it in `mcp.json`

### New Agent Role

1. Add a YAML file to `deeppresenter/roles/` with the system prompt
2. Create or subclass `Agent` in `deeppresenter/agents/`
3. Wire into `AgentLoop.run()` in `deeppresenter/main.py`

### New PPT Template

1. Create a PPTX file in `pptagent/templates/<name>/`
2. Add accompanying `.json` layout descriptor and `.txt` prompt fragments
3. Run induction: `pptagent/induct.py` to generate `slide_induction`

---

## Package Metadata

| Field | Value |
|---|---|
| Package name | `pptagent` |
| Version source | `deeppresenter.__version__` |
| Build system | setuptools |
| License | MIT |
| Authors | Hao Zheng (wszh712811@gmail.com) |
| Homepage | https://github.com/icip-cas/PPTAgent |
