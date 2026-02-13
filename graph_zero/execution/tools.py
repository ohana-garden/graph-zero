"""
Graph Zero Execution Layer

Tool Graph: Agents can only execute tools they have CAN_EXECUTE edges to.
Sandbox Profiles: Trust-gated capability levels.
Output Sanitization: Strip PII, tainted references, and unsafe content.
MCP Bridge: Stub for Model Context Protocol integration.
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET


# ============================================================
# Sandbox Profiles — trust-gated capability tiers
# ============================================================

class SandboxTier(Enum):
    """Capability tiers, gated by trust ceiling."""
    OBSERVER = "OBSERVER"       # trust < 0.2: read terrain, no writes
    PARTICIPANT = "PARTICIPANT"  # trust 0.2-0.5: Ua-class mutations, basic tools
    CONTRIBUTOR = "CONTRIBUTOR"  # trust 0.5-0.8: terrain additions, advanced tools
    STEWARD = "STEWARD"         # trust > 0.8: governance, all tools


@dataclass
class SandboxProfile:
    """What an agent is allowed to do."""
    tier: SandboxTier
    can_read_terrain: bool = True
    can_write_ua: bool = False
    can_write_protected: bool = False
    can_add_terrain: bool = False
    can_execute_tools: bool = False
    can_access_network: bool = False
    max_tool_calls_per_hour: int = 0
    allowed_tool_domains: list[str] = field(default_factory=list)


SANDBOX_PROFILES = {
    SandboxTier.OBSERVER: SandboxProfile(
        tier=SandboxTier.OBSERVER,
        can_read_terrain=True,
        max_tool_calls_per_hour=0,
    ),
    SandboxTier.PARTICIPANT: SandboxProfile(
        tier=SandboxTier.PARTICIPANT,
        can_write_ua=True,
        can_execute_tools=True,
        max_tool_calls_per_hour=50,
        allowed_tool_domains=["query", "message"],
    ),
    SandboxTier.CONTRIBUTOR: SandboxProfile(
        tier=SandboxTier.CONTRIBUTOR,
        can_write_ua=True,
        can_add_terrain=True,
        can_execute_tools=True,
        can_access_network=True,
        max_tool_calls_per_hour=200,
        allowed_tool_domains=["query", "message", "garden", "weather", "analysis"],
    ),
    SandboxTier.STEWARD: SandboxProfile(
        tier=SandboxTier.STEWARD,
        can_write_ua=True,
        can_write_protected=True,
        can_add_terrain=True,
        can_execute_tools=True,
        can_access_network=True,
        max_tool_calls_per_hour=1000,
        allowed_tool_domains=["*"],  # all domains
    ),
}


def get_sandbox_for_trust(trust_ceiling: float) -> SandboxProfile:
    """Determine sandbox tier from trust ceiling."""
    if trust_ceiling >= 0.8:
        return SANDBOX_PROFILES[SandboxTier.STEWARD]
    elif trust_ceiling >= 0.5:
        return SANDBOX_PROFILES[SandboxTier.CONTRIBUTOR]
    elif trust_ceiling >= 0.2:
        return SANDBOX_PROFILES[SandboxTier.PARTICIPANT]
    else:
        return SANDBOX_PROFILES[SandboxTier.OBSERVER]


# ============================================================
# Tool Graph — permission-checked tool execution
# ============================================================

@dataclass
class ToolDefinition:
    """A tool that agents can execute."""
    tool_id: str
    name: str
    domain: str
    description: str
    required_trust: float = 0.0
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    is_network: bool = False
    is_destructive: bool = False


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_ms: int = 0
    sanitized: bool = False


class ToolExecutionError(Enum):
    NO_PERMISSION = "NO_PERMISSION"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TRUST_TOO_LOW = "TRUST_TOO_LOW"
    DOMAIN_BLOCKED = "DOMAIN_BLOCKED"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_BLOCKED = "NETWORK_BLOCKED"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class ToolGraph:
    """Manages tool permissions and execution via graph edges."""

    def __init__(self, graph: PropertyGraph):
        self.graph = graph
        self._rate_tracker: dict[str, list[int]] = {}  # agent_id -> [timestamps]
        self._tool_handlers: dict[str, callable] = {}   # tool_id -> handler

    def register_tool(self, tool: ToolDefinition) -> Node:
        """Register a tool in the graph."""
        return self.graph.add_node(tool.tool_id, NT.TOOL, {
            "name": tool.name,
            "domain": tool.domain,
            "description": tool.description,
            "required_trust": tool.required_trust,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "is_network": tool.is_network,
            "is_destructive": tool.is_destructive,
        })

    def grant_tool(self, agent_id: str, tool_id: str,
                   granted_by: str = "system") -> Optional[Edge]:
        """Grant an agent permission to execute a tool."""
        g = self.graph
        if not g.has_node(agent_id) or not g.has_node(tool_id):
            return None
        return g.add_edge(agent_id, tool_id, ET.CAN_EXECUTE, {
            "granted_by": granted_by,
            "granted_at": int(time.time() * 1000),
        })

    def revoke_tool(self, agent_id: str, tool_id: str) -> bool:
        """Revoke tool permission."""
        edges = self.graph.get_edges_between(agent_id, tool_id, ET.CAN_EXECUTE)
        for edge in edges:
            self.graph.remove_edge(edge.id)
        return len(edges) > 0

    def register_handler(self, tool_id: str, handler: callable) -> None:
        """Register a callable handler for a tool."""
        self._tool_handlers[tool_id] = handler

    def can_execute(self, agent_id: str, tool_id: str) -> tuple[bool, Optional[ToolExecutionError]]:
        """Check if an agent can execute a tool (without executing it)."""
        g = self.graph

        # Tool exists?
        tool_node = g.get_node(tool_id)
        if not tool_node or tool_node.node_type != NT.TOOL:
            return False, ToolExecutionError.TOOL_NOT_FOUND

        # CAN_EXECUTE edge exists?
        edges = g.get_edges_between(agent_id, tool_id, ET.CAN_EXECUTE)
        if not edges:
            return False, ToolExecutionError.NO_PERMISSION

        # Trust ceiling check
        agent = g.get_node(agent_id)
        if not agent:
            return False, ToolExecutionError.NO_PERMISSION

        trust = agent.get("trust_ceiling", 0.0)
        required = tool_node.get("required_trust", 0.0)
        if trust < required:
            return False, ToolExecutionError.TRUST_TOO_LOW

        # Sandbox domain check
        sandbox = get_sandbox_for_trust(trust)
        domain = tool_node.get("domain", "")
        if sandbox.allowed_tool_domains != ["*"] and domain not in sandbox.allowed_tool_domains:
            return False, ToolExecutionError.DOMAIN_BLOCKED

        # Network check
        if tool_node.get("is_network") and not sandbox.can_access_network:
            return False, ToolExecutionError.NETWORK_BLOCKED

        # Rate limit check
        if not self._check_rate(agent_id, sandbox.max_tool_calls_per_hour):
            return False, ToolExecutionError.RATE_LIMITED

        return True, None

    def execute(self, agent_id: str, tool_id: str,
                inputs: Optional[dict] = None) -> ToolResult:
        """Execute a tool on behalf of an agent.

        Checks permissions, executes handler, sanitizes output.
        """
        allowed, error = self.can_execute(agent_id, tool_id)
        if not allowed:
            return ToolResult(success=False, error=error.value if error else "UNKNOWN")

        handler = self._tool_handlers.get(tool_id)
        if not handler:
            return ToolResult(success=False, error="NO_HANDLER")

        # Track rate
        self._track_call(agent_id)

        start = time.time()
        try:
            output = handler(inputs or {})
            elapsed = int((time.time() - start) * 1000)
            sanitized_output = sanitize_output(output)
            return ToolResult(
                success=True,
                output=sanitized_output,
                execution_ms=elapsed,
                sanitized=True,
            )
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            return ToolResult(
                success=False,
                error=f"EXECUTION_FAILED: {str(e)}",
                execution_ms=elapsed,
            )

    def get_agent_tools(self, agent_id: str) -> list[Node]:
        """Get all tools an agent has permission to execute."""
        return self.graph.get_neighbors(agent_id, ET.CAN_EXECUTE, direction="out")

    def _check_rate(self, agent_id: str, max_per_hour: int) -> bool:
        if max_per_hour <= 0:
            return False
        ts = int(time.time() * 1000)
        cutoff = ts - 3600000
        calls = self._rate_tracker.get(agent_id, [])
        recent = [t for t in calls if t > cutoff]
        return len(recent) < max_per_hour

    def _track_call(self, agent_id: str) -> None:
        ts = int(time.time() * 1000)
        if agent_id not in self._rate_tracker:
            self._rate_tracker[agent_id] = []
        self._rate_tracker[agent_id].append(ts)
        # Prune old entries
        cutoff = ts - 3600000
        self._rate_tracker[agent_id] = [
            t for t in self._rate_tracker[agent_id] if t > cutoff
        ]


# ============================================================
# Output Sanitization
# ============================================================

# PII patterns
_PII_PATTERNS = [
    (re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'), '[SSN_REDACTED]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE_REDACTED]'),
    (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), '[CC_REDACTED]'),
]


def sanitize_output(output: Any) -> Any:
    """Sanitize tool output: strip PII, normalize."""
    if isinstance(output, str):
        return _sanitize_string(output)
    elif isinstance(output, dict):
        return {k: sanitize_output(v) for k, v in output.items()}
    elif isinstance(output, list):
        return [sanitize_output(item) for item in output]
    return output


def _sanitize_string(s: str) -> str:
    """Apply PII redaction patterns."""
    result = s
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ============================================================
# MCP Bridge Stub
# ============================================================

@dataclass
class MCPRequest:
    """Model Context Protocol request."""
    method: str
    params: dict
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.sha256(
                f"{self.method}:{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class MCPResponse:
    """Model Context Protocol response."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class MCPBridge:
    """Stub for MCP server integration.

    In production, this connects to MCP servers (FalkorDB, Hume, etc.)
    via JSON-RPC over stdio/SSE. Here it provides the interface shape
    for tool registration and invocation.
    """

    def __init__(self, tool_graph: ToolGraph):
        self.tool_graph = tool_graph
        self._servers: dict[str, dict] = {}  # server_name -> config

    def register_server(self, name: str, uri: str,
                        capabilities: list[str]) -> None:
        """Register an MCP server."""
        self._servers[name] = {
            "uri": uri,
            "capabilities": capabilities,
            "connected": False,
        }

    def list_servers(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self._servers.items()]

    def invoke(self, server_name: str, request: MCPRequest,
               agent_id: str) -> MCPResponse:
        """Invoke a method on an MCP server (stub).

        In production: JSON-RPC call with auth.
        Here: returns stub response for testing the interface.
        """
        if server_name not in self._servers:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Server '{server_name}' not registered"
            )

        server = self._servers[server_name]
        method = request.method

        if method not in server["capabilities"]:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                error=f"Method '{method}' not in server capabilities"
            )

        # Stub: echo back params as result
        return MCPResponse(
            request_id=request.request_id,
            success=True,
            result={"method": method, "params": request.params, "stub": True}
        )
