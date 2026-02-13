"""Tests for Graph Zero execution layer: tool registry, trust gating, sandbox, audit."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import NT, ET, bootstrap_community, create_agent
from graph_zero.execution.execution import (
    ExecutionEngine, ExecutionRequest, ExecutionResult, ExecutionStatus,
    ToolSpec, ToolDomain, SandboxProfile, SandboxLevel,
    register_tool, grant_tool_access, revoke_tool_access,
    SANDBOX_STRICT, SANDBOX_STANDARD, SANDBOX_ELEVATED,
)


def make_env():
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Test")
    create_agent(g, "comm1", "alice", "Alice", "human")
    create_agent(g, "comm1", "bob", "Bob", "agent")
    engine = ExecutionEngine(g)
    return g, engine


# ============================================================
# Tool Registry Tests
# ============================================================

def test_register_tool():
    g, engine = make_env()
    tool = register_tool(g, ToolSpec(
        tool_id="tool_plant_id", name="Plant Identifier",
        description="Identify plants from images",
        domain=ToolDomain.GARDENING, sandbox=SANDBOX_STANDARD,
        trust_floor=0.3,
    ))
    assert tool is not None
    assert tool.get("name") == "Plant Identifier"
    assert tool.get("trust_floor") == 0.3
    assert tool.get("invocation_count") == 0
    print("  ✓ register_tool")

def test_grant_revoke_access():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool1", name="Tool One", description="Test",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    edge = grant_tool_access(g, "alice", "tool1", "admin")
    assert edge is not None
    assert not edge.get("revoked")

    ok = revoke_tool_access(g, "alice", "tool1")
    assert ok
    edges = g.get_edges_between("alice", "tool1", ET.CAN_EXECUTE)
    assert all(e.get("revoked") for e in edges)
    print("  ✓ grant_revoke_access")


# ============================================================
# Authorization Tests
# ============================================================

def test_authorize_success():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool1", name="Safe Tool", description="Safe",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.2,
    ))
    grant_tool_access(g, "alice", "tool1")
    req = engine.create_request("alice", "tool1", {"query": "test"})
    ok, reason = engine.authorize(req, agent_trust_ceiling=0.5)
    assert ok
    assert reason == "Authorized"
    print("  ✓ authorize_success")

def test_authorize_no_grant():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool1", name="Tool", description="Test",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    # No grant given to bob
    req = engine.create_request("bob", "tool1", {})
    ok, reason = engine.authorize(req, agent_trust_ceiling=0.5)
    assert not ok
    assert "No active" in reason
    print("  ✓ authorize_no_grant")

def test_authorize_revoked():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool1", name="Tool", description="Test",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    grant_tool_access(g, "alice", "tool1")
    revoke_tool_access(g, "alice", "tool1")
    req = engine.create_request("alice", "tool1", {})
    ok, reason = engine.authorize(req, agent_trust_ceiling=0.5)
    assert not ok
    assert "No active" in reason
    print("  ✓ authorize_revoked")

def test_authorize_trust_too_low():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_gov", name="Governance Tool", description="Sensitive",
        domain=ToolDomain.GOVERNANCE, sandbox=SANDBOX_ELEVATED,
        trust_floor=0.8,  # high trust required
    ))
    grant_tool_access(g, "bob", "tool_gov")
    req = engine.create_request("bob", "tool_gov", {})
    ok, reason = engine.authorize(req, agent_trust_ceiling=0.3)
    assert not ok
    assert "Trust" in reason
    print("  ✓ authorize_trust_too_low")


# ============================================================
# Execution Tests
# ============================================================

def test_execute_success():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_add", name="Adder", description="Adds numbers",
        domain=ToolDomain.COMPUTATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    grant_tool_access(g, "alice", "tool_add")
    engine.register_handler("tool_add", lambda d: d.get("a", 0) + d.get("b", 0))

    req = engine.create_request("alice", "tool_add", {"a": 17, "b": 25})
    result = engine.execute(req, agent_trust_ceiling=0.5)
    assert result.status == ExecutionStatus.COMPLETED
    assert result.output_data == 42
    assert result.execution_ms >= 0
    assert result.sanitized

    # Tool stats updated
    tool = g.get_node("tool_add")
    assert tool.get("invocation_count") == 1
    print("  ✓ execute_success")

def test_execute_denied():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_secret", name="Secret", description="Restricted",
        domain=ToolDomain.GOVERNANCE, sandbox=SANDBOX_ELEVATED,
        trust_floor=0.9,
    ))
    grant_tool_access(g, "bob", "tool_secret")

    req = engine.create_request("bob", "tool_secret", {})
    result = engine.execute(req, agent_trust_ceiling=0.1)
    assert result.status == ExecutionStatus.DENIED
    assert "Trust" in result.error_message
    print("  ✓ execute_denied")

def test_execute_handler_failure():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_crash", name="Crasher", description="Always fails",
        domain=ToolDomain.COMPUTATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    grant_tool_access(g, "alice", "tool_crash")
    engine.register_handler("tool_crash", lambda d: 1/0)

    req = engine.create_request("alice", "tool_crash", {})
    result = engine.execute(req, agent_trust_ceiling=0.5)
    assert result.status == ExecutionStatus.FAILED
    assert "division by zero" in result.error_message
    print("  ✓ execute_handler_failure")

def test_execute_mcp_stub():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_mcp", name="MCP Weather", description="Weather via MCP",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STANDARD,
        trust_floor=0.1,
        mcp_server_url="https://weather.mcp.example.com",
        mcp_tool_name="get_weather",
    ))
    grant_tool_access(g, "alice", "tool_mcp")
    req = engine.create_request("alice", "tool_mcp", {"location": "Pahoa"})
    result = engine.execute(req, agent_trust_ceiling=0.5)
    assert result.status == ExecutionStatus.COMPLETED
    assert result.output_data["_mcp_stub"] is True
    assert result.provenance_type == "INFERENCE"
    print("  ✓ execute_mcp_stub")


# ============================================================
# Sandbox Tests
# ============================================================

def test_sandbox_network():
    profile = SandboxProfile(
        level=SandboxLevel.STANDARD,
        network_allowlist=["api.weather.gov", "plants.usda.gov"],
    )
    assert profile.allows_network("api.weather.gov")
    assert profile.allows_network("plants.usda.gov")
    assert not profile.allows_network("evil.com")

    strict = SandboxProfile(level=SandboxLevel.STRICT)
    assert not strict.allows_network("anything.com")
    print("  ✓ sandbox_network")


# ============================================================
# Sanitization Tests
# ============================================================

def test_sanitize_ssn():
    g, engine = make_env()
    output = engine._scrub_patterns("My SSN is 123-45-6789 and yours is 987-65-4321")
    assert "123-45-6789" not in output
    assert "[SSN_REDACTED]" in output
    print("  ✓ sanitize_ssn")

def test_sanitize_email():
    g, engine = make_env()
    output = engine._scrub_patterns("Email me at test@example.com for details")
    assert "test@example.com" not in output
    assert "[EMAIL_REDACTED]" in output
    print("  ✓ sanitize_email")

def test_sanitize_credit_card():
    g, engine = make_env()
    output = engine._scrub_patterns("Card: 4111-1111-1111-1111")
    assert "4111" not in output
    assert "[CC_REDACTED]" in output
    print("  ✓ sanitize_credit_card")


# ============================================================
# Audit Tests
# ============================================================

def test_audit_trail():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="tool_audit", name="Audit Test", description="Test",
        domain=ToolDomain.COMPUTATION, sandbox=SANDBOX_STRICT,
        trust_floor=0.0,
    ))
    grant_tool_access(g, "alice", "tool_audit")
    engine.register_handler("tool_audit", lambda d: "result")

    # 3 executions
    for i in range(3):
        req = engine.create_request("alice", "tool_audit", {"i": i})
        engine.execute(req, agent_trust_ceiling=0.5)

    assert len(engine.audit_log) == 3
    assert all(entry["status"] == "completed" for entry in engine.audit_log)
    assert all(entry["vessel_id"] == "alice" for entry in engine.audit_log)
    print("  ✓ audit_trail (3 entries)")

def test_available_tools():
    g, engine = make_env()
    register_tool(g, ToolSpec(
        tool_id="t1", name="Low Trust", description="Anyone",
        domain=ToolDomain.INFORMATION, sandbox=SANDBOX_STRICT, trust_floor=0.0,
    ))
    register_tool(g, ToolSpec(
        tool_id="t2", name="Mid Trust", description="Some",
        domain=ToolDomain.COMPUTATION, sandbox=SANDBOX_STANDARD, trust_floor=0.5,
    ))
    register_tool(g, ToolSpec(
        tool_id="t3", name="High Trust", description="Few",
        domain=ToolDomain.GOVERNANCE, sandbox=SANDBOX_ELEVATED, trust_floor=0.9,
    ))
    for tid in ["t1", "t2", "t3"]:
        grant_tool_access(g, "alice", tid)

    # Trust 0.6 → can use t1 and t2 but not t3
    available = engine.get_available_tools("alice", 0.6)
    names = {t.get("name") for t in available}
    assert "Low Trust" in names
    assert "Mid Trust" in names
    assert "High Trust" not in names
    print("  ✓ available_tools (trust-filtered)")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing execution layer...\n")

    print("Tool Registry:")
    test_register_tool()
    test_grant_revoke_access()

    print("\nAuthorization:")
    test_authorize_success()
    test_authorize_no_grant()
    test_authorize_revoked()
    test_authorize_trust_too_low()

    print("\nExecution:")
    test_execute_success()
    test_execute_denied()
    test_execute_handler_failure()
    test_execute_mcp_stub()

    print("\nSandbox:")
    test_sandbox_network()

    print("\nSanitization:")
    test_sanitize_ssn()
    test_sanitize_email()
    test_sanitize_credit_card()

    print("\nAudit & Discovery:")
    test_audit_trail()
    test_available_tools()

    print("\n" + "=" * 50)
    print("ALL EXECUTION TESTS PASSED ✓")
    print("=" * 50)
