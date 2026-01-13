# claudette-agent Development Guidelines

## Overview

**claudette-agent** is a superset of [claudette](https://claudette.answer.ai/), providing a Claudette-compatible API wrapper for the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/python.md). It maintains API compatibility with claudette while adding additional functionality specific to the Agent SDK.

## Reference Documentation

When implementing, modifying, or refactoring claudette-agent, refer to:

### Claude Agent SDK (Primary Backend)
- **Python SDK Reference**: https://platform.claude.com/docs/en/agent-sdk/python.md

### Claudette (API Compatibility Target)
- **Core API (Client, Chat, etc.)**: https://claudette.answer.ai/core.html.md
- **Tool Loop**: https://claudette.answer.ai/toolloop.html.md
- **Async Support**: https://claudette.answer.ai/async.html.md
- **Text Editor**: https://claudette.answer.ai/text_editor.html.md

## Development Environment

- Use `uv` for environment management: `uv venv` to create the virtual environment and `uv run` to execute commands within it.
- Run tests with: `uv run python -m pytest tests/ -v`

## Before Opening a PR

1. Update the **changelog** (`CHANGELOG.md`) with all changes made during the current session.
2. Keep the **documentation** in sync with any new features, changes, or refactors.
3. Maintain and expand **guides** that explain how to navigate the codebase and use new or updated functionality.
4. Always use **Mermaid diagrams** within markdown files for architecture, flows, and visual explanations.
5. Update `docs/how_it_works.md` to reflect the project's internal mechanics.
6. Run the full test suite to ensure nothing is broken.

## Architecture Notes

- claudette-agent wraps the Claude Agent SDK to provide claudette-compatible APIs
- Extra functionalities beyond claudette are welcome if needed to support SDK features
- Consider using [fastcore](https://fastcore.fast.ai/) utilities when they improve implementation clarity
- The `ClaudeAgentOptions` from the SDK is the configuration object that receives parameters from Client/Chat
