"""
Text Editor Tool - File manipulation operations for Claude agents.

This module provides claudette-compatible text editing functions that can be
used as tools for Claude to view, create, and modify files.

The main function `str_replace_based_edit_tool` (aliased as `str_replace_editor`)
dispatches to individual operations based on the `command` parameter.

Supported commands:
- view: View file or directory contents
- create: Create a new file
- insert: Insert text at a specific line
- str_replace: Replace first occurrence of text
- undo_edit: Undo the last edit

Example usage with Chat:
    >>> from claudette_agent import Chat
    >>> from claudette_agent.text_editor import str_replace_based_edit_tool
    >>> chat = Chat(tools=[str_replace_based_edit_tool])
    >>> response = await chat("Create a file called test.txt with 'Hello World'")
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict

# Store undo history per file (list of previous contents)
_undo_history: Dict[str, List[str]] = defaultdict(list)

# Maximum lines to show without range
MAX_LINES_DEFAULT = 500


def view(path: str, view_range: Optional[List[int]] = None, nums: bool = False) -> str:
    """
    View file or directory contents.

    Args:
        path: Path to file or directory
        view_range: Optional [start_line, end_line] (1-indexed, inclusive)
        nums: Whether to show line numbers

    Returns:
        File contents or directory listing

    Example:
        >>> view("/tmp/test.txt")
        'Hello World'
        >>> view("/tmp/test.txt", [1, 10], nums=True)
        '1: Hello World\\n2: ...'
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: Path does not exist: {path}"

    if p.is_dir():
        # Return directory listing
        try:
            entries = sorted(p.iterdir())
            result = []
            for entry in entries:
                suffix = "/" if entry.is_dir() else ""
                result.append(f"{entry.name}{suffix}")
            return "\n".join(result) if result else "(empty directory)"
        except PermissionError:
            return f"Error: Permission denied: {path}"

    # Read file
    try:
        content = p.read_text()
        lines = content.splitlines()
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except UnicodeDecodeError:
        return f"Error: Cannot read binary file: {path}"

    # Apply range if specified
    if view_range:
        if len(view_range) != 2:
            return "Error: view_range must be [start_line, end_line]"

        start, end = view_range
        if start < 1:
            return "Error: start_line must be >= 1"
        if end < start:
            return "Error: end_line must be >= start_line"

        # Convert to 0-indexed
        start_idx = start - 1
        end_idx = end  # Inclusive, so don't subtract 1

        if start_idx >= len(lines):
            return f"Error: start_line {start} exceeds file length ({len(lines)} lines)"

        lines = lines[start_idx:end_idx]
        line_offset = start
    else:
        line_offset = 1
        # Truncate if too long
        if len(lines) > MAX_LINES_DEFAULT:
            lines = lines[:MAX_LINES_DEFAULT]
            lines.append(f"... (truncated, showing first {MAX_LINES_DEFAULT} lines)")

    if nums:
        result = []
        for i, line in enumerate(lines):
            if line.startswith("... (truncated"):
                result.append(line)
            else:
                result.append(f"{line_offset + i}: {line}")
        return "\n".join(result)

    return "\n".join(lines)


def create(path: str, file_text: str, overwrite: bool = False) -> str:
    """
    Create a new file with given content.

    Args:
        path: Path for the new file
        file_text: Content to write
        overwrite: Whether to overwrite existing file

    Returns:
        Success or error message

    Example:
        >>> create("/tmp/test.txt", "Hello World")
        'File created: /tmp/test.txt'
    """
    p = Path(path).expanduser().resolve()

    if p.exists() and not overwrite:
        return f"Error: File already exists: {path}. Use overwrite=True to replace."

    try:
        # Create parent directories if needed
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save to undo history if overwriting
        if p.exists():
            _undo_history[str(p)].append(p.read_text())

        p.write_text(file_text)
        return f"File created: {p}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error creating file: {e}"


def insert(path: str, insert_line: int, new_str: str) -> str:
    """
    Insert text after a specific line number.

    Args:
        path: Path to file
        insert_line: Line number after which to insert (0 = before first line)
        new_str: Text to insert

    Returns:
        Success or error message

    Example:
        >>> insert("/tmp/test.txt", 0, "# Header")
        'Text inserted at line 1 in /tmp/test.txt'
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: File does not exist: {path}"

    try:
        content = p.read_text()
        lines = content.splitlines(keepends=True)

        # Validate line number
        if insert_line < 0:
            return "Error: insert_line must be >= 0"
        if insert_line > len(lines):
            return f"Error: insert_line {insert_line} exceeds file length ({len(lines)} lines)"

        # Save to undo history
        _undo_history[str(p)].append(content)

        # Insert the new text
        new_lines = new_str.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'

        lines[insert_line:insert_line] = new_lines

        p.write_text(''.join(lines))
        return f"Text inserted at line {insert_line + 1} in {p}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error inserting text: {e}"


def str_replace(path: str, old_str: str, new_str: str) -> str:
    """
    Replace the first occurrence of old_str with new_str in the file.

    Args:
        path: Path to file
        old_str: Text to find (must appear exactly once)
        new_str: Replacement text

    Returns:
        Success or error message

    Example:
        >>> str_replace("/tmp/test.txt", "Hello", "Hi")
        'Replacement successful in /tmp/test.txt'
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: File does not exist: {path}"

    try:
        content = p.read_text()

        # Check occurrence count
        count = content.count(old_str)
        if count == 0:
            return f"Error: '{old_str}' not found in {path}"
        if count > 1:
            return f"Error: '{old_str}' appears {count} times in {path}. Must appear exactly once for unambiguous replacement."

        # Save to undo history
        _undo_history[str(p)].append(content)

        # Perform replacement
        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content)

        return f"Replacement successful in {p}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error performing replacement: {e}"


def undo_edit(path: str) -> str:
    """
    Undo the last edit to a file.

    Args:
        path: Path to file

    Returns:
        Success or error message

    Example:
        >>> undo_edit("/tmp/test.txt")
        'Undo successful for /tmp/test.txt'
    """
    p = Path(path).expanduser().resolve()
    key = str(p)

    if key not in _undo_history or not _undo_history[key]:
        return f"Error: No edit history for {path}"

    try:
        # Get the previous content
        previous_content = _undo_history[key].pop()
        p.write_text(previous_content)
        return f"Undo successful for {p}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error undoing edit: {e}"


def str_replace_editor(**kwargs) -> str:
    """
    Dispatcher for text editor commands.

    This is the main entry point that routes to individual operations
    based on the 'command' parameter.

    Args:
        command: One of 'view', 'create', 'insert', 'str_replace', 'undo_edit'
        path: Path to file or directory
        **kwargs: Additional arguments for the specific command

    Commands and their parameters:
        - view: path, view_range (optional), nums (optional)
        - create: path, file_text, overwrite (optional)
        - insert: path, insert_line, new_str
        - str_replace: path, old_str, new_str
        - undo_edit: path

    Returns:
        Result of the operation

    Example:
        >>> str_replace_editor(command='view', path='/tmp/test.txt')
        'Hello World'
        >>> str_replace_editor(command='create', path='/tmp/new.txt', file_text='Content')
        'File created: /tmp/new.txt'
    """
    command = kwargs.get('command')
    path = kwargs.get('path')

    if not command:
        return "Error: 'command' parameter is required"
    if not path and command != 'help':
        return "Error: 'path' parameter is required"

    if command == 'view':
        return view(
            path=path,
            view_range=kwargs.get('view_range'),
            nums=kwargs.get('nums', False)
        )
    elif command == 'create':
        file_text = kwargs.get('file_text')
        if file_text is None:
            return "Error: 'file_text' parameter is required for create"
        return create(
            path=path,
            file_text=file_text,
            overwrite=kwargs.get('overwrite', False)
        )
    elif command == 'insert':
        insert_line = kwargs.get('insert_line')
        new_str = kwargs.get('new_str')
        if insert_line is None:
            return "Error: 'insert_line' parameter is required for insert"
        if new_str is None:
            return "Error: 'new_str' parameter is required for insert"
        return insert(path=path, insert_line=insert_line, new_str=new_str)
    elif command == 'str_replace':
        old_str = kwargs.get('old_str')
        new_str = kwargs.get('new_str')
        if old_str is None:
            return "Error: 'old_str' parameter is required for str_replace"
        if new_str is None:
            return "Error: 'new_str' parameter is required for str_replace"
        return str_replace(path=path, old_str=old_str, new_str=new_str)
    elif command == 'undo_edit':
        return undo_edit(path=path)
    else:
        return f"Error: Unknown command '{command}'. Valid commands: view, create, insert, str_replace, undo_edit"


# Alias for claudette compatibility
str_replace_based_edit_tool = str_replace_editor

# Tool configuration for different models (for reference)
text_editor_conf = {
    'sonnet': {'type': 'text_editor_20250728', 'name': 'str_replace_based_edit_tool'},
    'sonnet37': {'type': 'text_editor_20250124', 'name': 'str_replace_editor'},
}


def get_text_editor_schema() -> Dict[str, Any]:
    """
    Get the tool schema for str_replace_editor.

    Returns:
        Dict containing the tool schema for Claude
    """
    return {
        "name": "str_replace_editor",
        "description": "Text editor tool for viewing and modifying files. Supports: view (read file/directory), create (new file), insert (add text at line), str_replace (replace text), undo_edit (revert last change).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "insert", "str_replace", "undo_edit"],
                    "description": "The operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory"
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[start_line, end_line] for view command (1-indexed, inclusive)"
                },
                "nums": {
                    "type": "boolean",
                    "description": "Show line numbers in view output"
                },
                "file_text": {
                    "type": "string",
                    "description": "Content for create command"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Allow overwriting existing file in create"
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Line number after which to insert (0 = before first line)"
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to find for str_replace (must appear exactly once)"
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text for insert or str_replace"
                }
            },
            "required": ["command", "path"]
        }
    }
