"""
Graph Zero Execution Layer

Tools are graph citizens — nodes with trust requirements, capabilities,
sandbox profiles, and audit trails. No tool runs unaccountable.

Architecture:
  - ToolNode: registered tool with metadata, trust floor, sandbox profile
  - SandboxProfile: what a tool can access (network, filesystem, graph domains)
  - ExecutionRequest: proposed tool invocation, gated by trust
  - ExecutionResult: captured output with provenance
  - MCPBridge: interface to MCP servers for tool discovery and invocation
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET


# ============================================================
# Tool Registry
# ============================================================

class ToolDomain(Enum):
    """Categories of tool capability."""
    INFORMATION = "information"       # read-only data retrieval
    COMMUNICATION = "communication"   # messaging, notifications
    COMPUTATION = "computation"       # math, data processing
    PHYSICAL = "physical"             # IoT, sensors, actuators
    FINANCIAL = "financial"           # Kala transactions, payments
    GOVERNANCE = "governance"         # voting, policy changes
    GARDENING = "gardening"           # plant ID, soil, weather
    HEALTH = "health"                 # health queries, med info


class SandboxLevel(Enum):
    """Sandbox restriction levels."""
    STRICT = "strict"       # no network, no filesystem, pure computation
    STANDARD = "standard"   # limited network (allowlist), read-only fs
    ELEVATED = "elevated"   # broader network, write fs in sandbox dir
    TRUSTED = "trusted"     # full network, requires Class B authorization


@dataclass
class SandboxProfile:
    """What a tool is allowed to access."""
    level: SandboxLevel
    network_allowlist: list[str] = field(default_factory=list)   # domains
    network_denylist: list[str] = field(default_factory=list)
    filesystem_read: list[str] = field(default_factory=list)     # paths
    filesystem_write: list[str] = field(default_factory=list)
    graph_domains_read: list[str] = field(default_factory=list)  # node types
    graph_domains_write: list[str] = field(default_factory=list)
    max_execution_ms: int = 30000       # 30 second default timeout
    max_output_bytes: int = 1048576     # 1MB default
    can_invoke_tools: bool = False      # can this tool call other tools?

    def allows_network(self, domain: str) -> bool:
        if domain in self.network_denylist:
            return False
        if self.level == SandboxLevel.STRICT:
            return False
        if self.network_allowlist:
            return domain in self.network_allowlist
        return self.level in (SandboxLevel.ELEVATED, SandboxLevel.TRUSTED)


# Predefined sandbox profiles
SANDBOX_STRICT = SandboxProfile(level=SandboxLevel.STRICT)
SANDBOX_STANDARD = SandboxProfile(
    level=SandboxLevel.STANDARD,
    max_execution_ms=15000,
)
SANDBOX_ELEVATED = SandboxProfile(
    level=SandboxLevel.ELEVATED,
    max_execution_ms=60000,
    max_output_bytes=5242880,  # 5MB
)


@dataclass
class ToolSpec:
    """Specification for registering a tool."""
    tool_id: str
    name: str
    description: str
    domain: ToolDomain
    sandbox: SandboxProfile
    trust_floor: float              # minimum agent trust_ceiling to use
    input_schema: dict = field(default_factory=dict)   # JSON Schema
    output_schema: dict = field(default_factory=dict)
    mcp_server_url: Optional[str] = None  # MCP endpoint
    mcp_tool_name: Optional[str] = None   # tool name within MCP server
    version: str = "1.0.0"
    requires_class_b: bool = False  # needs endorsement to execute?


def register_tool(graph: PropertyGraph, spec: ToolSpec) -> Node:
    """Register a tool in the graph."""
    return graph.add_node(spec.tool_id, NT.TOOL, {
        "name": spec.name,
        "description": spec.description,
        "domain": spec.domain.value,
        "sandbox_level": spec.sandbox.level.value,
        "sandbox_network_allowlist": spec.sandbox.network_allowlist,
        "sandbox_max_execution_ms": spec.sandbox.max_execution_ms,
        "sandbox_max_output_bytes": spec.sandbox.max_output_bytes,
        "sandbox_can_invoke_tools": spec.sandbox.can_invoke_tools,
        "trust_floor": spec.trust_floor,
        "input_schema": spec.input_schema,
        "output_schema": spec.output_schema,
        "mcp_server_url": spec.mcp_server_url,
        "mcp_tool_name": spec.mcp_tool_name,
        "version": spec.version,
        "requires_class_b": spec.requires_class_b,
        "registered_at": int(time.time() * 1000),
        "invocation_count": 0,
        "last_invoked": None,
    })


def grant_tool_access(graph: PropertyGraph, vessel_id: str,
                      tool_id: str, granted_by: str = "") -> Optional[Edge]:
    """Grant an agent access to a tool (CAN_EXECUTE edge)."""
    if not graph.has_node(vessel_id) or not graph.has_node(tool_id):
        return None
    return graph.add_edge(vessel_id, tool_id, ET.CAN_EXECUTE, {
        "granted_at": int(time.time() * 1000),
        "granted_by": granted_by,
        "revoked": False,
    })


def revoke_tool_access(graph: PropertyGraph, vessel_id: str,
                       tool_id: str) -> bool:
    """Revoke an agent's access to a tool."""
    edges = graph.get_edges_between(vessel_id, tool_id, ET.CAN_EXECUTE)
    for edge in edges:
        edge.set("revoked", True)
        edge.set("revoked_at", int(time.time() * 1000))
    return bool(edges)


# ============================================================
# Execution Request / Result
# ============================================================

class ExecutionStatus(Enum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class ExecutionRequest:
    """A proposed tool invocation."""
    request_id: str
    vessel_id: str
    tool_id: str
    input_data: dict
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    status: ExecutionStatus = ExecutionStatus.PENDING
    context: str = ""   # why is this tool being invoked?

    @property
    def request_hash(self) -> str:
        """Deterministic hash for audit trail."""
        h = hashlib.sha256()
        h.update(self.request_id.encode())
        h.update(self.vessel_id.encode())
        h.update(self.tool_id.encode())
        for k in sorted(self.input_data.keys()):
            h.update(k.encode())
            h.update(str(self.input_data[k]).encode())
        return h.hexdigest()


@dataclass
class ExecutionResult:
    """Captured output from a tool invocation."""
    request_id: str
    tool_id: str
    status: ExecutionStatus
    output_data: Any = None
    error_message: str = ""
    execution_ms: int = 0
    output_bytes: int = 0
    sanitized: bool = False
    provenance_type: str = "INFERENCE"  # tool outputs start as inference
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


# ============================================================
# Execution Engine
# ============================================================

class ExecutionEngine:
    """Manages tool invocations with trust gating and audit trails.

    Every tool invocation goes through:
    1. Authorization (trust check)
    2. Sandbox validation
    3. Execution (or MCP bridge)
    4. Output sanitization
    5. Audit recording
    """

    def __init__(self, graph: PropertyGraph):
        self.graph = graph
        self._request_counter = 0
        self._handlers: dict[str, callable] = {}  # tool_id -> handler function
        self._audit_log: list[dict] = []

    def register_handler(self, tool_id: str, handler: callable) -> None:
        """Register a local execution handler for a tool."""
        self._handlers[tool_id] = handler

    def authorize(self, request: ExecutionRequest,
                  agent_trust_ceiling: float) -> tuple[bool, str]:
        """Check if agent is authorized to use this tool.

        Returns (authorized, reason).
        """
        tool_node = self.graph.get_node(request.tool_id)
        if not tool_node:
            return False, "Tool not found"

        # Check CAN_EXECUTE edge exists and is not revoked
        edges = self.graph.get_edges_between(request.vessel_id, request.tool_id,
                                              ET.CAN_EXECUTE)
        active_grants = [e for e in edges if not e.get("revoked", False)]
        if not active_grants:
            return False, "No active tool grant"

        # Check trust floor
        trust_floor = tool_node.get("trust_floor", 0.0)
        if agent_trust_ceiling < trust_floor:
            return False, f"Trust {agent_trust_ceiling:.2f} < floor {trust_floor:.2f}"

        # Class B check
        if tool_node.get("requires_class_b", False):
            return False, "Requires Class B authorization (not implemented in this call)"

        return True, "Authorized"

    def execute(self, request: ExecutionRequest,
                agent_trust_ceiling: float) -> ExecutionResult:
        """Execute a tool invocation through the full pipeline."""
        # 1. Authorization
        authorized, reason = self.authorize(request, agent_trust_ceiling)
        if not authorized:
            result = ExecutionResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ExecutionStatus.DENIED,
                error_message=reason,
            )
            self._record_audit(request, result)
            return result

        request.status = ExecutionStatus.AUTHORIZED

        # 2. Get tool and sandbox
        tool_node = self.graph.get_node(request.tool_id)
        max_ms = tool_node.get("sandbox_max_execution_ms", 30000)
        max_bytes = tool_node.get("sandbox_max_output_bytes", 1048576)

        # 3. Execute
        request.status = ExecutionStatus.RUNNING
        start_time = time.time()

        try:
            handler = self._handlers.get(request.tool_id)
            if handler:
                output = handler(request.input_data)
            elif tool_node.get("mcp_server_url"):
                output = self._mcp_invoke(tool_node, request.input_data)
            else:
                raise RuntimeError(f"No handler or MCP endpoint for {request.tool_id}")

            elapsed_ms = int((time.time() - start_time) * 1000)

            # 4. Size check
            output_str = str(output)
            output_bytes = len(output_str.encode('utf-8'))
            if output_bytes > max_bytes:
                output = self._truncate_output(output, max_bytes)
                output_bytes = max_bytes

            # 5. Sanitize
            sanitized_output = self._sanitize_output(output)

            result = ExecutionResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ExecutionStatus.COMPLETED,
                output_data=sanitized_output,
                execution_ms=elapsed_ms,
                output_bytes=output_bytes,
                sanitized=True,
                provenance_type="INFERENCE",  # tool output is always inference
            )

        except TimeoutError:
            result = ExecutionResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Exceeded {max_ms}ms timeout",
                execution_ms=max_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            result = ExecutionResult(
                request_id=request.request_id,
                tool_id=request.tool_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_ms=elapsed_ms,
            )

        # 6. Update tool stats
        tool_node.set("invocation_count", tool_node.get("invocation_count", 0) + 1)
        tool_node.set("last_invoked", int(time.time() * 1000))

        # 7. Audit
        self._record_audit(request, result)
        return result

    def create_request(self, vessel_id: str, tool_id: str,
                       input_data: dict, context: str = "") -> ExecutionRequest:
        """Create a new execution request."""
        self._request_counter += 1
        return ExecutionRequest(
            request_id=f"exec_{self._request_counter}",
            vessel_id=vessel_id,
            tool_id=tool_id,
            input_data=input_data,
            context=context,
        )

    def get_available_tools(self, vessel_id: str,
                            agent_trust_ceiling: float) -> list[Node]:
        """Get tools available to an agent given their trust ceiling."""
        available = []
        for edge in self.graph.get_outgoing(vessel_id, ET.CAN_EXECUTE):
            if edge.get("revoked", False):
                continue
            tool_node = self.graph.get_node(edge.target_id)
            if tool_node and tool_node.get("trust_floor", 0.0) <= agent_trust_ceiling:
                available.append(tool_node)
        return available

    @property
    def audit_log(self) -> list[dict]:
        return list(self._audit_log)

    # --------------------------------------------------------
    # MCP Bridge
    # --------------------------------------------------------

    def _mcp_invoke(self, tool_node: Node, input_data: dict) -> Any:
        """Invoke a tool via MCP protocol.

        In production: HTTP POST to MCP server with JSON-RPC.
        Here: stub that records the intent.
        """
        return {
            "_mcp_stub": True,
            "server_url": tool_node.get("mcp_server_url"),
            "tool_name": tool_node.get("mcp_tool_name"),
            "input": input_data,
            "note": "MCP invocation would happen here in production",
        }

    # --------------------------------------------------------
    # Output Sanitization
    # --------------------------------------------------------

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize tool output.

        - Strip potential PII patterns (basic)
        - Enforce output schema if defined
        - Mark provenance as INFERENCE
        """
        if isinstance(output, str):
            return self._scrub_patterns(output)
        if isinstance(output, dict):
            return {k: self._sanitize_output(v) for k, v in output.items()}
        if isinstance(output, list):
            return [self._sanitize_output(item) for item in output]
        return output

    def _scrub_patterns(self, text: str) -> str:
        """Basic PII scrubbing. Production would use a proper NER model."""
        import re
        # SSN pattern
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        # Credit card pattern (basic)
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
                       '[CC_REDACTED]', text)
        # Email (basic)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                       '[EMAIL_REDACTED]', text)
        return text

    def _truncate_output(self, output: Any, max_bytes: int) -> Any:
        """Truncate output to max bytes."""
        s = str(output)
        if len(s.encode('utf-8')) > max_bytes:
            return s[:max_bytes] + "...[TRUNCATED]"
        return output

    def _record_audit(self, request: ExecutionRequest,
                      result: ExecutionResult) -> None:
        """Record execution in audit log."""
        self._audit_log.append({
            "request_id": request.request_id,
            "request_hash": request.request_hash,
            "vessel_id": request.vessel_id,
            "tool_id": request.tool_id,
            "status": result.status.value,
            "execution_ms": result.execution_ms,
            "error": result.error_message,
            "timestamp": result.timestamp,
        })
