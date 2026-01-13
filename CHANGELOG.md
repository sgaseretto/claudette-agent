# Changelog

All notable changes to claudette-agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Extra CLI Arguments (`extra_args`)**: New `extra_args` parameter on `Client`, `Chat`, and `AsyncChat` allows passing additional CLI arguments to ClaudeAgentOptions:
  - Use `{'no-session-persistence': None}` to disable session persistence for truly stateless queries
  - Keys should NOT include `--` prefix (SDK adds it internally)
  - Format: `{'flag-name': None}` for flags, `{'option-name': 'value'}` for options with values
  - Merges with SDK's `extra_args` for CLI argument passthrough
  - Example: `Chat(model=..., setting_sources=[], extra_args={'no-session-persistence': None})`

- **Environment Variables (`env`)**: New `env` parameter on `Client`, `Chat`, and `AsyncChat` allows passing custom environment variables to the Claude CLI process:
  - Pass any environment variable to the underlying Claude CLI
  - Enables true stateless queries by setting `HOME` to a temp directory (prevents session matching)
  - Example: `Chat(model=..., env={'HOME': '/tmp/unique-dir'})` for fully isolated queries
  - Merged with internal env vars (like `MAX_THINKING_TOKENS` for extended thinking)

- **Stateless Queries (`setting_sources`)**: New `setting_sources` parameter on `Client`, `Chat`, and `AsyncChat` allows control over Claude Code settings loading:
  - Default `[]` = stateless mode (no settings loaded, each query is independent)
  - `['user', 'project', 'local']` = session persistence (loads Claude Code settings)
  - Useful for API-style independent queries or parallel processing

- **Extended Thinking Support**: `maxthinktok` parameter now properly enables extended thinking via SDK's `MAX_THINKING_TOKENS` environment variable. Works in `Client`, `Chat`, and `stream()` methods.

- **ToolLoopResult Class**: `toolloop()` now returns a `ToolLoopResult` object that:
  - Can be iterated over like a list
  - Provides a `.value` property to access the final meaningful result
  - Maintains backward compatibility with existing code

- **Image Support**:
  - `mk_msg()` now accepts `bytes` for image data with automatic MIME type detection
  - New `img_msg()` helper function for creating image messages
  - Support for PNG, JPEG, GIF, and WebP formats
  - `mk_msg()` with lists can now contain mixed bytes (images) and strings (text)

- **Text Editor Tool** (`text_editor.py`): Full claudette-compatible implementation with:
  - `view(path, view_range, nums)` - View file or directory contents
  - `create(path, file_text, overwrite)` - Create a new file
  - `insert(path, insert_line, new_str)` - Insert text at line
  - `str_replace(path, old_str, new_str)` - Replace first occurrence (must be unique)
  - `undo_edit(path)` - Undo the last edit
  - `str_replace_editor()` / `str_replace_based_edit_tool()` - Command dispatcher

- **Model Capability Checks**:
  - `can_stream(model)` - Check streaming support
  - `can_set_system_prompt(model)` - Check system prompt support
  - `can_set_temperature(model)` - Check temperature support
  - `can_use_extended_thinking(model)` - Check extended thinking support
  - `can_use_vision(model)` - Check vision/image input support
  - `has_extended_thinking_models` - Set of models with thinking support
  - `text_only_models` - Set of models without vision support

- **Response Prefilling**: `prefill` parameter now instructs Claude to start its response with the provided text by adding it to the prompt

- **Test Notebook**: `test_new_features.ipynb` for testing all new functionality

### Fixed

- `maxthinktok` parameter was being ignored; now correctly passed to SDK via `env={'MAX_THINKING_TOKENS': ...}`
- `prefill` parameter now works with SDK by including it in the prompt instruction

### Changed

- `toolloop()` return type changed from `List` to `ToolLoopResult` (backward compatible)
- `_build_options()` methods now accept and process `maxthinktok` parameter

## [0.1.0] - 2025-01-08

### Added

- Initial release with Claudette-compatible API for Claude Agent SDK
- `Client` and `AsyncClient` classes
- `Chat` and `AsyncChat` classes with conversation history
- Tool support via MCP servers
- Structured outputs via Pydantic models
- Streaming responses
- Usage tracking and cost calculation
- MCP server integration
