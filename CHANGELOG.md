# Changelog

All notable changes to claudette-agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Native Streaming via `include_partial_messages`**: Streaming now uses the SDK's `StreamEvent` with `text_delta` for real character-by-character streaming (matching claudette behavior), instead of yielding complete message blocks. Enable with `stream=True` on Client/Chat.

- **Native Structured Outputs via `output_format`**: New `struct_native()` method on Client uses the SDK's built-in `output_format` with JSON Schema for validated structured outputs via `ResultMessage.structured_output`. The existing `struct()` (prompt engineering approach) is preserved. Chat's `struct()` gains a `native=True` parameter to choose between approaches.

- **Native Extended Thinking via `max_thinking_tokens`/`ThinkingConfig`**: `maxthinktok` now uses the SDK's native `max_thinking_tokens` option instead of the `env['MAX_THINKING_TOKENS']` workaround. Forward-compatible with the SDK's upcoming `thinking` config field.

- **Streaming + Thinking Validation**: Using `stream=True` with `maxthinktok > 0` now raises a clear `ValueError` since the SDK does not support streaming with extended thinking.

- **System Prompt Presets**: The `sp` parameter on Client/Chat now accepts dict format for preset system prompts: `sp={"type": "preset", "preset": "claude_code", "append": "Be concise"}`.

- **New SDK Feature Parameters** on Client/Chat:
  - `max_turns: int` — limit agentic turns (tool-use round trips)
  - `max_budget_usd: float` — maximum budget in USD
  - `fallback_model: str` — fallback model if primary fails
  - `can_use_tool: Callable` — custom permission callback
  - `hooks: dict` — hook configurations for intercepting events
  - `agents: dict` — subagent definitions
  - `enable_file_checkpointing: bool` — enable file change tracking
  - `thinking: dict` — extended thinking config (forward-compatible)
  - `effort: str` — effort level ("low"/"medium"/"high"/"max", forward-compatible)

- **Session Management**: Re-exports `list_sessions()` and `get_session_messages()` from the SDK (when available).

- **SDK Tool Pass-through**: `_convert_to_sdk_tool()` now accepts pre-created `SdkMcpTool` instances directly (pass-through), not just plain callables.

### Fixed

- **`Chat.__call__(stream=True)` now returns `StreamingResponse`**: Previously returned a `Message` object, causing `TypeError` on async iteration. Now properly returns a `StreamingResponse` with character-level streaming and history updates via callback.
- **`stream_text()` no longer duplicates output**: Was yielding text from both `StreamEvent` deltas and the final `AssistantMessage`. Now tracks whether stream events were received and skips `AssistantMessage` content accordingly.
- **`Client.struct_native` no longer fails with "No structured output received"**: The SDK's structured output uses a `StructuredOutput` tool internally, which requires multiple turns. Default `max_turns` increased from 1 to 3.
- **Prefill text no longer duplicated in responses**: `Client._r()` was prepending the prefill string to the response, but since prefill is sent as a prompt instruction, Claude's response already includes it.
- **`all_models` notebook usage**: Fixed notebook cell that called `.keys()` on `all_models` (which is a list, not a dict).

### Changed

- **Shared Parsing Functions**: `_parse_usage()`, `_parse_sdk_message()`, and `_simple_text_message()` moved from `client.py`/`chat.py` to `core.py` to eliminate duplication.
- `StreamingResponse` rewritten to process `StreamEvent` objects for character-level streaming.
- `_build_options()` in both Client and Chat now handles `stream`, `thinking`, `effort`, `max_budget_usd`, `fallback_model`, `can_use_tool`, `hooks`, `agents`, `enable_file_checkpointing` params.
- Extended thinking no longer uses the `env['MAX_THINKING_TOKENS']` workaround.

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
- `_build_options()` now explicitly sets `continue_conversation=False` and `resume=None` to ensure stateless queries (SDK defaults were already these values, but now explicit for clarity)

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
