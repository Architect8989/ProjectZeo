from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Server types
# ---------------------------------------------------------------------------

class MCPServerType:
    FILESYSTEM = "filesystem"
    SHELL      = "shell"
    MEMORY     = "memory"


@dataclass
class MCPTool:
    """Describes one tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_type: str


@dataclass
class MCPCallResult:
    """Result from an MCP tool call."""
    success: bool
    output: Any
    error: Optional[str] = None
    latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# MCP Client (JSON-RPC over HTTP)
# ---------------------------------------------------------------------------

class MCPClient:
    
    def __init__(self, base_url: str, server_type: str) -> None:
        self._url = base_url.rstrip("/")
        self._server_type = server_type
        self._session_id: Optional[str] = None
        self._tools: List[MCPTool] = []
        self._connected = False
        self._lock = threading.Lock()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import httpx  # noqa: PLC0415
            self._client = httpx.Client(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
                headers={"Content-Type": "application/json"},
            )
        except ImportError:
            pass
        return self._client

    def connect(self) -> bool:
        """Initialize the MCP session and discover tools."""
        try:
            client = self._get_client()
            if client is None:
                return False

            # MCP initialize call
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "projectzeo", "version": "1.0.0"},
                },
            }
            resp = client.post(
                f"{self._url}/mcp",
                content=json.dumps(init_payload),
                timeout=10.0,
            )
            if resp.status_code != 200:
                return False

            # Discover available tools
            tools_payload = {
                "jsonrpc": "2.0", "id": 2,
                "method": "tools/list", "params": {},
            }
            resp2 = client.post(
                f"{self._url}/mcp",
                content=json.dumps(tools_payload),
                timeout=10.0,
            )
            if resp2.status_code == 200:
                data = resp2.json()
                tools_raw = data.get("result", {}).get("tools", [])
                self._tools = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                        server_type=self._server_type,
                    )
                    for t in tools_raw
                ]

            self._connected = True
            _logger.info(
                "[MCPClient] Connected to %s server at %s. Tools: %s",
                self._server_type, self._url,
                [t.name for t in self._tools],
            )
            return True

        except Exception as exc:
            _logger.debug("[MCPClient] Connect failed (%s): %s", self._server_type, exc)
            return False

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        """Call a named tool on this MCP server."""
        start = time.monotonic()
        try:
            client = self._get_client()
            if client is None:
                return MCPCallResult(False, None, "httpx not available")

            payload = {
                "jsonrpc": "2.0", "id": int(time.time() * 1000) % 100000,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            resp = client.post(
                f"{self._url}/mcp",
                content=json.dumps(payload),
                timeout=60.0,
            )
            latency = time.monotonic() - start

            if resp.status_code != 200:
                return MCPCallResult(
                    False, None,
                    f"MCP server returned {resp.status_code}",
                    latency,
                )

            data = resp.json()
            result = data.get("result", {})
            error_obj = data.get("error")

            if error_obj:
                return MCPCallResult(
                    False, None,
                    f"MCP error: {error_obj.get('message', str(error_obj))}",
                    latency,
                )

            # Unwrap content array
            content = result.get("content", [])
            if isinstance(content, list):
                output = "\n".join(
                    str(c.get("text") or c.get("data") or "")
                    for c in content
                    if isinstance(c, dict)
                )
            else:
                output = str(result)

            return MCPCallResult(True, output, None, latency)

        except Exception as exc:
            return MCPCallResult(False, None, str(exc), time.monotonic() - start)

    def is_connected(self) -> bool:
        return self._connected

    def get_tools(self) -> List[MCPTool]:
        return list(self._tools)


# ---------------------------------------------------------------------------
# MCPToolRegistry — process-global singleton
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    

    _instance: Optional["MCPToolRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._enabled = (
            os.environ.get("PROJECTZEO_MCP_ENABLED", "0").strip() in ("1", "true", "yes")
        )
        self._clients: Dict[str, MCPClient] = {}
        self._tools_by_name: Dict[str, MCPTool] = {}
        self._lock = threading.Lock()
        self._policy_engine = None

        if self._enabled:
            self._init_servers()
        else:
            _logger.info(
                "[MCPToolRegistry] MCP disabled. Set PROJECTZEO_MCP_ENABLED=1 to enable."
            )

    @classmethod
    def get_instance(cls) -> "MCPToolRegistry":
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _init_servers(self) -> None:
        """Connect to configured MCP servers."""
        server_configs = {
            MCPServerType.FILESYSTEM: os.environ.get("PROJECTZEO_MCP_FILESYSTEM_URL", ""),
            MCPServerType.SHELL:      os.environ.get("PROJECTZEO_MCP_SHELL_URL", ""),
            MCPServerType.MEMORY:     os.environ.get("PROJECTZEO_MCP_MEMORY_URL", ""),
        }

        for server_type, url in server_configs.items():
            if not url:
                continue
            client = MCPClient(base_url=url, server_type=server_type)
            if client.connect():
                self._clients[server_type] = client
                for tool in client.get_tools():
                    self._tools_by_name[tool.name] = tool

    def set_policy_engine(self, policy_engine) -> None:
        """Wire the PolicyEngine for pre-call validation."""
        self._policy_engine = policy_engine

    def call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        focused_app: str = "__unknown_app__",
    ) -> MCPCallResult:
        """
        Call a named MCP tool with pre-validation.

        Args:
            tool_name:   Name of the tool (e.g. "read_file", "execute_command")
            arguments:   Tool arguments dict
            focused_app: Currently focused app (for policy validation)

        Returns:
            MCPCallResult with success/output/error.
        """
        if not self._enabled:
            return MCPCallResult(False, None, "MCP not enabled")

        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            return MCPCallResult(
                False, None,
                f"Tool '{tool_name}' not found. Available: {sorted(self._tools_by_name.keys())}",
            )

        # PolicyEngine pre-validation
        if self._policy_engine is not None:
            action_dict = {
                "operation": "mcp_tool",
                "tool_name": tool_name,
                **arguments,
            }
            # Map MCP shell tool → command operation for policy checks
            if tool.server_type == MCPServerType.SHELL:
                action_dict["operation"] = "command"
                action_dict["command"] = arguments.get("command", "")

            try:
                decision, reason = self._policy_engine.validate_action_dict(
                    action_dict, focused_app=focused_app
                )
                if decision == "DENY":
                    return MCPCallResult(
                        False, None,
                        f"PolicyEngine denied MCP tool call '{tool_name}': {reason}",
                    )
            except Exception as policy_exc:
                _logger.warning("[MCPToolRegistry] Policy check failed: %s", policy_exc)

        client = self._clients.get(tool.server_type)
        if client is None:
            return MCPCallResult(False, None, f"No client for server type {tool.server_type!r}")

        return client.call_tool(tool_name, arguments)

    def list_tools(self) -> List[str]:
        """Return names of all discovered tools."""
        return sorted(self._tools_by_name.keys())

    def is_enabled(self) -> bool:
        return self._enabled

    def is_connected(self) -> bool:
        return bool(self._clients)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "connected_servers": list(self._clients.keys()),
            "total_tools": len(self._tools_by_name),
            "tools": self.list_tools(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_registry() -> MCPToolRegistry:
    """Return the process-global MCPToolRegistry."""
    return MCPToolRegistry.get_instance()


def call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    focused_app: str = "__unknown_app__",
) -> MCPCallResult:
    """Convenience wrapper for direct tool calls from operate.py."""
    return MCPToolRegistry.get_instance().call(tool_name, arguments, focused_app)
