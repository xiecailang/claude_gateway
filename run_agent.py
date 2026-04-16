#!/usr/bin/env python3
"""
Multi-turn SWE-bench agent that executes tool calls from the model.

The agent:
1. Sends the Anthropic /v1/messages request to the gateway with tools
2. Parses tool_use blocks from the model response
3. Executes tools locally (Glob, Read, Grep, Bash, Edit, Write, etc.)
4. Sends tool results back
5. Repeats until the model stops requesting tools

Usage:
    python3 run_agent.py --task-dir /path/to/repo --prompt "Fix the bug..."
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9000")
MAX_TURNS = 40
MAX_TOTAL_TIME = 3600

# Claude Code tool definitions - these are sent to the model
TOOLS = [
    {
        "name": "Glob",
        "description": "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths sorted by modification time",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The glob pattern to match files against"},
                "path": {"type": "string", "description": "The directory to search in (optional, defaults to task dir)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search for a pattern in file contents within the repository",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search in (defaults to task dir)"},
                "glob": {"type": "string", "description": "Glob pattern to filter files (e.g. \"*.py\")"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the local filesystem",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The absolute or relative path to the file"},
                "limit": {"type": "integer", "description": "Number of lines to read (optional)"},
                "offset": {"type": "integer", "description": "Line number to start from (optional)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace text in a file. Must provide exact old_string and new_string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "old_string": {"type": "string", "description": "The exact text to replace"},
                "new_string": {"type": "string", "description": "The new text to replace with"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file (creates or overwrites)",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a bash command in the task directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
            },
            "required": ["command"],
        },
    },
]


def log(msg="", **kwargs):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True, **kwargs)


def call_api(messages: list, system: str, tools: list = None, max_tokens: int = 8192, stream: bool = False) -> dict:
    """Call the gateway /v1/messages endpoint."""
    import httpx
    body = {
        "model": "vllm-local",
        "max_tokens": max_tokens,
        "system": system,
        "stream": stream,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    with httpx.Client(timeout=300) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/messages", json=body)
        resp.raise_for_status()
        return resp.json()


def call_api_stream(messages: list, system: str, tools: list = None, max_tokens: int = 8192):
    """Call the gateway /v1/messages endpoint with streaming."""
    import httpx
    body = {
        "model": "vllm-local",
        "max_tokens": max_tokens,
        "system": system,
        "stream": True,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    client = httpx.Client(timeout=300)
    return client.stream("POST", f"{GATEWAY_URL}/v1/messages", json=body)


def extract_text_from_content(content) -> str:
    """Extract text from Anthropic message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    pass  # handled separately
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def extract_tool_uses(content) -> list[dict]:
    """Extract tool_use blocks from content."""
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    return []


def execute_tool(tool_name: str, tool_input: dict, task_dir: str) -> str:
    """Execute a tool call and return the result as a string."""
    # Normalize tool name to match our definitions
    tool_name_normalized = tool_name[0].upper() + tool_name[1:] if tool_name else ""

    if tool_name_normalized == "Glob":
        pattern = tool_input.get("pattern", "*")
        path = tool_input.get("path", "")
        full_path = os.path.join(task_dir, path) if path else task_dir
        try:
            import glob
            matches = glob.glob(os.path.join(full_path, pattern), recursive=True)
            matches = [os.path.relpath(m, task_dir) for m in matches]
            matches.sort()
            if len(matches) > 200:
                return "\n".join(matches[:200]) + f"\n... and {len(matches) - 200} more"
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as e:
            return f"Error: {e}"

    elif tool_name_normalized == "Read":
        file_path = tool_input.get("file_path", "")
        limit = tool_input.get("limit")
        offset = tool_input.get("offset")
        full_path = os.path.join(task_dir, file_path) if not os.path.isabs(file_path) else file_path
        try:
            with open(full_path) as f:
                lines = f.readlines()
            if offset:
                lines = lines[offset - 1:] if offset <= len(lines) else []
            if limit:
                lines = lines[:limit]
            content = "".join(lines)
            if len(content) > 50000:
                content = content[:50000] + f"\n... (truncated, {len(content)} total chars)"
            # Add line numbers for reference
            numbered = ""
            start_line = offset or 1
            for i, line in enumerate(content.split("\n"), start_line):
                numbered += f"{i}: {line}\n"
            return numbered.rstrip()
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    elif tool_name_normalized == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", task_dir)
        glob_pattern = tool_input.get("glob", "*.py")
        full_path = os.path.join(task_dir, path) if not os.path.isabs(path) else path
        try:
            cmd = ["grep", "-rn", "--include=" + glob_pattern, pattern, full_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=task_dir)
            output = result.stdout.strip()
            if len(output) > 50000:
                output = output[:50000] + "\n... (truncated)"
            return output if output else "(no matches)"
        except Exception as e:
            return f"Error: {e}"

    elif tool_name_normalized == "Bash":
        command = tool_input.get("command", "")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=120, cwd=task_dir
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
            if len(output) > 50000:
                output = output[:50000] + "\n... (truncated)"
            return output.strip() if output.strip() else "(empty)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out (120s limit)"
        except Exception as e:
            return f"Error: {e}"

    elif tool_name_normalized == "Edit":
        file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")
        full_path = os.path.join(task_dir, file_path) if not os.path.isabs(file_path) else file_path
        try:
            with open(full_path) as f:
                content = f.read()
            if old_string not in content:
                # Try to find approximate match
                old_lines = old_string.strip().split("\n")
                content_lines = content.split("\n")
                for i in range(len(content_lines) - len(old_lines) + 1):
                    match = True
                    for j in range(len(old_lines)):
                        if content_lines[i + j].strip() != old_lines[j].strip():
                            match = False
                            break
                    if match:
                        # Found approximate match
                        start = sum(len(content_lines[k]) + 1 for k in range(i))
                        end = start + sum(len(old_lines[k]) + 1 for k in range(len(old_lines) - 1))
                        actual_old = content[start:end]
                        new_content = content.replace(actual_old, new_string, 1)
                        with open(full_path, "w") as f:
                            f.write(new_content)
                        return f"Successfully edited {file_path} (approximate match)"
                return f"Error: old_string not found in {file_path}"
            new_content = content.replace(old_string, new_string, 1)
            with open(full_path, "w") as f:
                f.write(new_content)
            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing {file_path}: {e}"

    elif tool_name_normalized == "Write":
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        full_path = os.path.join(task_dir, file_path) if not os.path.isabs(file_path) else file_path
        try:
            os.makedirs(os.path.dirname(full_path) or task_dir, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return f"Successfully wrote {file_path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing {file_path}: {e}"

    else:
        return f"Unknown tool: {tool_name_normalized}"


def run_agent(task_dir: str, prompt: str, system_prompt: str = "") -> dict:
    """Run the multi-turn tool-use agent."""
    start_time = time.time()
    task_dir = os.path.abspath(task_dir)

    if not system_prompt:
        system_prompt = (
            "You are a software engineer working on a bug fix in a git repository. "
            "Your task is to explore the codebase, understand the issue, implement a fix, "
            "and create a patch file. "
            "Always use tools to explore the codebase before making changes. "
            "When you are done, create a patch using Bash: git diff > /tmp/patch.patch"
        )

    # Build initial system prompt with task info
    full_system = system_prompt + "\n\nWorking directory: " + task_dir

    messages = [{"role": "user", "content": prompt}]

    turn = 0
    while turn < MAX_TURNS and (time.time() - start_time) < MAX_TOTAL_TIME:
        turn += 1
        elapsed = time.time() - start_time
        log(f"Turn {turn}/{MAX_TURNS} ({elapsed:.0f}s)")

        # Call API with tools and streaming
        accumulated_text = ""
        accumulated_tool_uses = []
        current_tool_use_index = -1
        current_tool_input_json = ""

        try:
            with call_api_stream(messages, full_system, TOOLS) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Parse SSE event
                    if data.get("type") == "content_block_start":
                        cb = data.get("content_block", {})
                        if cb.get("type") == "text":
                            pass  # text block starting
                        elif cb.get("type") == "tool_use":
                            current_tool_use_index = data.get("index", -1)
                            accumulated_tool_uses.append({
                                "type": "tool_use",
                                "id": cb.get("id", ""),
                                "name": cb.get("name", ""),
                                "input": {},
                                "_input_json": "",
                            })

                    elif data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                accumulated_text += text
                        elif delta.get("type") == "input_json_delta":
                            partial = delta.get("partial_json", "")
                            if partial and accumulated_tool_uses:
                                accumulated_tool_uses[-1]["_input_json"] += partial

                    elif data.get("type") == "message_delta":
                        pass  # end of message

            # Parse tool inputs from JSON strings
            for tool_use in accumulated_tool_uses:
                json_str = tool_use.pop("_input_json", "{}")
                try:
                    tool_use["input"] = json.loads(json_str) if json_str else {}
                except json.JSONDecodeError:
                    tool_use["input"] = {}

            # Show assistant text
            text_preview = accumulated_text.strip()[:200]
            if text_preview:
                log(f"  Assistant: {text_preview}")

            # Show tool calls
            if accumulated_tool_uses:
                for tu in accumulated_tool_uses:
                    log(f"  Tool: {tu['name']}({json.dumps(tu['input'], ensure_ascii=False)[:150]})")

            if not accumulated_tool_uses:
                # No tool calls - model is done
                log(f"  No more tool calls, finished.")
                break

            # Execute tools
            tool_results = []
            for tu in accumulated_tool_uses:
                result = execute_tool(tu["name"], tu["input"], task_dir)
                log(f"  Result: {result[:200]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            # Build assistant message content
            assistant_content = []
            if accumulated_text.strip():
                assistant_content.append({"type": "text", "text": accumulated_text.strip()})
            assistant_content.extend(accumulated_tool_uses)

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        except Exception as e:
            log(f"  API error: {e}")
            return {"status": "api_error", "error": str(e), "turns": turn}

    elapsed = time.time() - start_time
    log(f"Agent finished in {turn} turns ({elapsed:.0f}s)")

    return {
        "status": "completed",
        "turns": turn,
        "elapsed": elapsed,
    }


def create_patch(task_dir: str) -> str:
    """Create a git diff patch file."""
    try:
        r = subprocess.run(["git", "diff"], capture_output=True, text=True, timeout=30, cwd=task_dir)
        patch = r.stdout
        if patch.strip():
            patch_path = f"/tmp/agent_patch_{os.path.basename(task_dir)}.patch"
            with open(patch_path, "w") as f:
                f.write(patch)
            return patch_path
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="", help="System prompt")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    log(f"Task dir: {task_dir}")
    log(f"Gateway: {GATEWAY_URL}")
    log()

    result = run_agent(task_dir, args.prompt, args.system)

    log()
    log(f"Agent finished: {result['status']} in {result.get('turns', '?')} turns ({result.get('elapsed', 0):.0f}s)")

    # Create patch
    patch_path = create_patch(task_dir)
    if patch_path:
        log(f"Patch: {patch_path}")
        result["patch_path"] = patch_path
        with open(patch_path) as f:
            result["patch"] = f.read()
            result["patch_size"] = len(result["patch"])
    else:
        log("No patch")

    # Save result
    result_file = os.path.join(task_dir, "agent_result.json")
    with open(result_file, "w") as f:
        json.dump(result, indent=2, default=str, fp=f)


if __name__ == "__main__":
    main()
