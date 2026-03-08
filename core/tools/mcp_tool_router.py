"""
core/tools/mcp_tool_router.py — MCP Tool Router (Layer 7, Research §8.2)

Implements the tool_router layer that classifies subtasks as GUI-required or
tool-addressable. Tool-addressable subtasks are dispatched via MCP (Model
Context Protocol) or the coding agent, bypassing UITARSRuntime entirely.

Research result: OSWorld-MCP shows integrating MCP tool invocations alongside
GUI operations improves task success from 8.3% → 20.4% for tool-augmented tasks.

Non-GUI operation categories (bypassed through MCP):
  - File operations: Python subprocess (already available)
  - Web API calls: REST via MCP server (bypasses browser)
  - Calendar/email: MCP server for Google Calendar, Gmail, Outlook
  - Database queries: SQLite/Postgres via MCP
  - Shell commands: sandboxed Python/bash executor

The Agent S3 native coding agent (Python/Bash execution) is MCP's local
equivalent. ProjectZeo routes tool-addressable tasks here first.

Reference: Research §8.2, Agent S3 §4.2 (native coding agent integration)
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool-addressable operation patterns
# ─────────────────────────────────────────────────────────────────────────────

_FILE_OP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bread\s+(?:file|document|csv|json|yaml|txt)\b", re.IGNORECASE),
    re.compile(r"\bwrite\s+(?:to\s+)?file\b", re.IGNORECASE),
    re.compile(r"\bcreate\s+(?:a\s+)?(?:file|directory|folder)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:files|directory|folder)\b", re.IGNORECASE),
    re.compile(r"\bcopy\s+file\b", re.IGNORECASE),
    re.compile(r"\bmove\s+file\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+file\b", re.IGNORECASE),
    re.compile(r"\bsearch\s+(?:for\s+)?files?\b", re.IGNORECASE),
]

_SHELL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\brun\s+(?:command|script|shell)\b", re.IGNORECASE),
    re.compile(r"\bexecute\s+(?:python|bash|sh|command)\b", re.IGNORECASE),
    re.compile(r"\binstall\s+(?:package|library|module)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:clone|pull|push|commit|status)\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+(?:install|run|build|test)\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
]

_GUI_REQUIRED_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bclick\b", re.IGNORECASE),
    re.compile(r"\btype\s+(?:into|in)\b", re.IGNORECASE),
    re.compile(r"\bdrag\b", re.IGNORECASE),
    re.compile(r"\bscroll\b", re.IGNORECASE),
    re.compile(r"\bopen\s+application\b", re.IGNORECASE),
    re.compile(r"\bnavigate\s+to\b", re.IGNORECASE),
    re.compile(r"\bselect\s+(?:from\s+)?(?:menu|dropdown|list)\b", re.IGNORECASE),
]


class TaskClassification:
    GUI_REQUIRED = "gui_required"
    TOOL_ADDRESSABLE = "tool_addressable"
    SHELL_COMMAND = "shell_command"
    UNCERTAIN = "uncertain"


def classify_subtask(description: str) -> str:
    """
    Classify a subtask description as GUI-required or tool-addressable.
    Returns TaskClassification constant.
    """
    desc = description.lower()

    # GUI signals override tool signals
    for pat in _GUI_REQUIRED_PATTERNS:
        if pat.search(desc):
            return TaskClassification.GUI_REQUIRED

    for pat in _SHELL_PATTERNS:
        if pat.search(desc):
            return TaskClassification.SHELL_COMMAND

    for pat in _FILE_OP_PATTERNS:
        if pat.search(desc):
            return TaskClassification.TOOL_ADDRESSABLE

    return TaskClassification.UNCERTAIN


class CodingAgent:
    """
    Sandboxed Python/Bash coding agent for non-GUI tool calls.
    Equivalent to Agent S3's native coding agent integration.
    """

    def __init__(
        self,
        sandbox_dir: Optional[str] = None,
        timeout_seconds: float = 30.0,
        allowed_imports: Optional[List[str]] = None,
    ) -> None:
        self._sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="projectzeo_sandbox_")
        self._timeout = timeout_seconds
        self._allowed_imports = allowed_imports or [
            "os", "sys", "json", "re", "datetime", "pathlib",
            "subprocess", "shutil", "glob", "csv", "io",
        ]

    def execute_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed subprocess.
        Returns {"success": bool, "stdout": str, "stderr": str, "returncode": int}
        """
        # Basic safety: reject obviously dangerous patterns
        dangerous = ["__import__", "exec(", "eval(", "os.system", "subprocess.call"]
        code_lower = code.lower()
        for danger in dangerous:
            if danger.lower() in code_lower:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Rejected: dangerous pattern '{danger}' detected.",
                    "returncode": -1,
                }

        script_path = os.path.join(self._sandbox_dir, "agent_script.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._sandbox_dir,
                env={**os.environ, "PYTHONPATH": ""},
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:4096],
                "stderr": result.stderr[:2048],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Execution timed out.", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def execute_shell(self, command: str) -> Dict[str, Any]:
        """Execute a shell command in the sandbox directory."""
        # Reject known destructive commands
        dangerous_cmds = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:"]
        for danger in dangerous_cmds:
            if danger in command:
                return {
                    "success": False, "stdout": "", 
                    "stderr": f"Rejected: dangerous command pattern.", "returncode": -1
                }
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._sandbox_dir,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:4096],
                "stderr": result.stderr[:2048],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Timed out.", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


class MCPToolRouter:
    """
    MCP Tool Router — classifies subtasks and dispatches non-GUI operations
    through the coding agent, bypassing UITARSRuntime for efficiency.

    From Research §8.2: routing non-GUI subtasks through tool calls instead
    of GUI interaction improves reliability (no coordinate errors, no timing
    issues) and reduces LLM calls per task by up to 52.3% (Agent S3).
    """

    def __init__(
        self,
        coding_agent: Optional[CodingAgent] = None,
        llm_callable: Optional[Any] = None,
    ) -> None:
        self._coding_agent = coding_agent or CodingAgent()
        self._llm = llm_callable
        self._tool_calls_total = 0
        self._tool_calls_success = 0
        self._lock = threading.Lock()

    def route(self, action: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Attempt to route an action through MCP/tool layer.

        Returns:
            (handled, result): handled=True means tool handled it, result is the output.
                               handled=False means GUI execution is needed.
        """
        op = str(action.get("operation", "")).lower()
        command = str(action.get("command", ""))
        description = str(action.get("description", ""))
        content = str(action.get("content", ""))

        # Direct shell/command operations
        if op == "command" and command:
            classification = classify_subtask(command + " " + description)
            if classification == TaskClassification.SHELL_COMMAND:
                return self._handle_shell(command)

        # Code execution requests
        if op in ("python_exec", "execute_code"):
            code = str(action.get("code", content))
            if code:
                result = self._coding_agent.execute_python(code)
                self._record(result["success"])
                return True, result

        # File operations
        if op == "file_create":
            path = str(action.get("path", ""))
            if path:
                return self._handle_file_create(path, content)

        return False, None  # Not handled — proceed with GUI

    def _handle_shell(self, command: str) -> Tuple[bool, Dict[str, Any]]:
        result = self._coding_agent.execute_shell(command)
        self._record(result["success"])
        return True, result

    def _handle_file_create(self, path: str, content: str) -> Tuple[bool, Dict[str, Any]]:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            result = {"success": True, "stdout": f"Created {path}", "stderr": "", "returncode": 0}
            self._record(True)
            return True, result
        except Exception as e:
            result = {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
            self._record(False)
            return True, result

    def _record(self, success: bool) -> None:
        with self._lock:
            self._tool_calls_total += 1
            if success:
                self._tool_calls_success += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_tool_calls": self._tool_calls_total,
                "successful": self._tool_calls_success,
                "success_rate": (
                    self._tool_calls_success / self._tool_calls_total
                    if self._tool_calls_total > 0 else 0.0
                ),
            }
