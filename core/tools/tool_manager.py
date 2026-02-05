"""
Tool Manager

Purpose:
Authoritative analysis and verification of system tools.

This module:
- DOES analyze required tools from an objective
- DOES verify tool presence and basic functionality
- DOES NOT install tools
- DOES NOT modify the system
- DOES NOT perform UI actions

It provides READ-ONLY system intelligence for planning and validation.
"""

from typing import Dict, List, Optional, Callable
import shutil
import subprocess
import json
import re


class ToolAnalysisError(RuntimeError):
    pass


class ToolStatus:
    """
    Immutable snapshot of a tool's state.
    """

    def __init__(
        self,
        *,
        name: str,
        installed: bool,
        version: Optional[str],
        functional: bool,
        path: Optional[str],
    ):
        self.name = name
        self.installed = installed
        self.version = version
        self.functional = functional
        self.path = path

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "functional": self.functional,
            "path": self.path,
        }


class ToolManager:
    """
    Deterministic, read-only tool intelligence layer.

    HARD CONTRACT:
    - No installation
    - No mutation
    - No OS configuration changes
    """

    MAX_LLM_CHARS = 12_000

    TOOL_ANALYSIS_PROMPT = """
You are a software environment analysis engine.

Your task:
- Analyze the objective
- Identify REQUIRED tools only (not optional)
- Be conservative
- Prefer common CLI tools

Return STRICT JSON only.

Schema:
{
  "tools": [
    {
      "name": "node",
      "purpose": "run JavaScript backend",
      "min_version": "18"
    }
  ]
}

Rules:
- No commentary
- No markdown
- No assumptions about installation
"""

    def __init__(self, llm_call: Callable[[str], str]):
        """
        llm_call(prompt: str) -> str
        """
        self._llm_call = llm_call

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def analyze_required_tools(self, objective: str) -> List[Dict[str, Optional[str]]]:
        """
        Uses LLM strictly for semantic extraction of required tools.
        No system access.
        """
        if not isinstance(objective, str) or not objective.strip():
            raise ToolAnalysisError("Objective must be non-empty string")

        prompt = (
            self.TOOL_ANALYSIS_PROMPT.strip()
            + "\n\nOBJECTIVE:\n"
            + objective.strip()
        )

        raw = self._llm_call(prompt)

        if not isinstance(raw, str):
            raise ToolAnalysisError("Tool analysis returned non-string")

        raw = raw.strip()
        if len(raw) > self.MAX_LLM_CHARS:
            raise ToolAnalysisError("Tool analysis output too large")

        try:
            parsed = self._extract_json(raw)
            tools = parsed.get("tools")
        except Exception as e:
            raise ToolAnalysisError(f"Invalid tool analysis JSON: {e}")

        if not isinstance(tools, list):
            raise ToolAnalysisError("tools must be a list")

        normalized: List[Dict[str, Optional[str]]] = []

        for t in tools:
            if not isinstance(t, dict):
                raise ToolAnalysisError("tool entry must be object")

            name = t.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ToolAnalysisError("tool.name must be non-empty string")

            normalized.append(
                {
                    "name": name.strip(),
                    "purpose": t.get("purpose"),
                    "min_version": t.get("min_version"),
                }
            )

        return normalized

    def check_tool_status(self, tool_name: str) -> ToolStatus:
        """
        Verifies presence and basic executability of a tool.
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ToolAnalysisError("tool_name must be non-empty string")

        tool_name = tool_name.strip()
        path = shutil.which(tool_name)

        if not path:
            return ToolStatus(
                name=tool_name,
                installed=False,
                version=None,
                functional=False,
                path=None,
            )

        version = self._get_version(tool_name)
        functional = self._basic_functionality_check(tool_name)

        return ToolStatus(
            name=tool_name,
            installed=True,
            version=version,
            functional=functional,
            path=path,
        )

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------

    def _basic_functionality_check(self, tool: str) -> bool:
        """
        Executes a minimal, non-destructive invocation.
        """
        try:
            subprocess.run(
                [tool, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            return True
        except Exception:
            return False

    def _get_version(self, tool: str) -> Optional[str]:
        """
        Best-effort version extraction.
        """
        candidates = [
            [tool, "--version"],
            [tool, "-v"],
            [tool, "version"],
        ]

        for cmd in candidates:
            try:
                out = subprocess.check_output(
                    cmd,
                    stderr=subprocess.STDOUT,
                    timeout=3,
                ).decode(errors="ignore")

                version = self._parse_version(out)
                if version:
                    return version
            except Exception:
                continue

        return None

    def _parse_version(self, text: str) -> Optional[str]:
        """
        Extracts first semantic-looking version string.
        """
        match = re.search(r"\d+\.\d+(\.\d+)?", text)
        if match:
            return match.group(0)
        return None

    def _extract_json(self, text: str) -> Dict:
        """
        Extracts first JSON object defensively.
        """
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ToolAnalysisError("No JSON object found")

        return json.loads(match.group(0))
