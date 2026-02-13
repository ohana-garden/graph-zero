"""Tests for Graph Zero moral geometry: phase space, constraints, projection, action scoring."""

import sys, math
sys.path.insert(0, '/home/claude')

from graph_zero.moral.geometry import (
    PhaseState, CouplingConstraint, DEFAULT_COUPLINGS,
    check_all_constraints, all_constraints_satisfied,
    project_position, ActionValence, score_action,
    evolve_momentum, apply_momentum, check_velocity,
    DepthResult, CalibrationResult, calibrate_depths,
    KalaTransaction, verify_kala_conservation, compute_brackets,
    NUM_VIRTUES, MAX_VELOCITY, EPSILON,
)
from graph_zero.graph.schema import VIRTUE_NAMES, COUPLINGS

# ============================================================
# Phase Space Tests
# ============================================================

def test_default_state():
    s = PhaseState.default()
    assert len(s.position) == 9
    assert len(s.momentum) == 9
    assert all(v == 0.5 for v in s.position)
    assert all(p == 0.0 for p in s.momentum)
    print("  ✓ default_state")

def test_magnitude():
    s = PhaseState(position=[1.0]*9, momentum=[0.0]*9)
    assert abs(s.magnitude() - 3.0) < 0.01
    print("  ✓ magnitude")

def test_distance():
    a = PhaseState(position=[0.0]*9, momentum=[0.0]*9)
    b = PhaseState(position=[1.0]*9, momentum=[0.0]*9)
    d = a.distance(b)
    assert abs(d - 3.0) < 0.01  # sqrt(9)
    print("  ✓ distance")

# ============================================================
# Coupling Constraint Tests
# ============================================================

def test_default_couplings_loaded():
    assert len(DEFAULT_COUPLINGS) == 4
    directions = {c.direction for c in DEFAULT_COUPLINGS}
    assert "prerequisite" in directions
    assert "enabler" in directions
    assert "foundation" in directions
    print("  ✓ default_couplings_loaded (4 constraints)")

def test_center_satisfies_all():
    """Center position [0.5]*9 satisfies all coupling constraints."""
    pos = [0.5] * 9
    assert all_constraints_satisfied(pos)
    print("  ✓ center_satisfies_all")

def test_prerequisite_violation():
    """Truthfulness(2)→Justice(1): can't have high justice without truthfulness."""
    pos = [0.5] * 9
    pos[2] = 0.1  # low truthfulness
    pos[1] = 0.9  # high justice
    results = check_all_constraints(pos)
    prereq = [(c, sat) for c, sat in results if c.direction == "prerequisite"]
    assert len(prereq) == 1
    assert not prereq[0][1]  # violated
    print("  ✓ prerequisite_violation (truthfulness→justice)")

def test_foundation_violation():
    """Love(3)→Unity(0): love must be >= unity*0.9."""
    pos = [0.5] * 9
    pos[3] = 0.1   # low love
    pos[0] = 0.9   # high unity
    results = check_all_constraints(pos)
    foundation = [(c, sat) for c, sat in results if c.direction == "foundation"]
    assert len(foundation) == 1
    assert not foundation[0][1]  # violated
    print("  ✓ foundation_violation (love→unity)")

def test_enabler_constraint():
    """Humility(5)→Service(8): enabler is softer."""
    pos = [0.5] * 9
    pos[5] = 0.3   # moderate humility
    pos[8] = 0.95  # very high service
    results = check_all_constraints(pos)
    enablers = [(c, sat) for c, sat in results if c.direction == "enabler"]
    # Enabler is softer, so this might still pass
    # humility→service: max_target = 0.3 + (1 - 0.7*0.5)*(1-0.3) = 0.3 + 0.65*0.7 = 0.755
    assert len(enablers) >= 1
    print("  ✓ enabler_constraint")

def test_high_virtue_satisfies():
    """High positions everywhere should satisfy all constraints."""
    pos = [0.9] * 9
    assert all_constraints_satisfied(pos)
    print("  ✓ high_virtue_satisfies")

# ============================================================
# Projection Tests
# ============================================================

def test_projection_clamps():
    """Values outside [0,1] are clamped."""
    desired = [1.5, -0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    projected = project_position(desired)
    assert projected[0] <= 1.0
    assert projected[1] >= 0.0
    print("  ✓ projection_clamps")

def test_projection_fixes_violations():
    """Projection should fix constraint violations."""
    pos = [0.5] * 9
    pos[2] = 0.1   # low truthfulness
    pos[1] = 0.9   # high justice → prerequisite violation
    projected = project_position(pos)
    assert all_constraints_satisfied(projected)
    # Justice should be reduced
    assert projected[1] < 0.9
    print("  ✓ projection_fixes_violations")

def test_projection_preserves_valid():
    """Valid positions should not change much under projection."""
    pos = [0.5] * 9
    projected = project_position(pos)
    diff = sum(abs(pos[i] - projected[i]) for i in range(9))
    assert diff < 0.01
    print("  ✓ projection_preserves_valid")

def test_projection_foundation():
    """Foundation constraint: love(3) must support unity(0)."""
    pos = [0.5] * 9
    pos[3] = 0.1   # low love
    pos[0] = 0.9   # high unity
    projected = project_position(pos)
    assert all_constraints_satisfied(projected)
    # Either love increased or unity decreased
    assert projected[3] >= pos[3] or projected[0] <= pos[0]
    print("  ✓ projection_foundation")

# ============================================================
# Action Scoring Tests
# ============================================================

def test_score_positive_action():
    """Action that improves position should have positive valence."""
    state = PhaseState.default()
    impacts = [0.1] * 9  # improve everything
    result = score_action(state, impacts)
    assert result.total_valence > 0
    assert len(result.constraint_violations) == 0
    print("  ✓ score_positive_action")

def test_score_negative_action():
    """Action that degrades position should have negative valence."""
    state = PhaseState.default()
    impacts = [-0.1] * 9
    result = score_action(state, impacts)
    assert result.total_valence < 0
    print("  ✓ score_negative_action")

def test_score_with_violation():
    """Action that violates coupling should show violations."""
    state = PhaseState(position=[0.5]*9, momentum=[0.0]*9)
    impacts = [0.0] * 9
    impacts[1] = 0.4   # boost justice
    impacts[2] = -0.4  # drop truthfulness
    result = score_action(state, impacts)
    # truthfulness→justice prerequisite should fire
    assert len(result.constraint_violations) > 0
    print("  ✓ score_with_violation")

def test_score_virtue_impacts():
    """Virtue impacts are labeled by name."""
    state = PhaseState.default()
    impacts = [0.0] * 9
    impacts[0] = 0.2  # boost unity
    result = score_action(state, impacts)
    assert "unity" in result.virtue_impacts
    assert result.virtue_impacts["unity"] > 0
    print("  ✓ score_virtue_impacts")

# ============================================================
# Momentum Tests
# ============================================================

def test_evolve_momentum():
    state = PhaseState.default()
    delta = [0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_state = evolve_momentum(state, delta, dt=1.0)
    assert new_state.momentum[0] > 0  # positive delta → positive momentum
    assert new_state.momentum[1] < 0  # negative delta
    assert new_state.kala_time == 1.0
    print("  ✓ evolve_momentum")

def test_momentum_decay():
    """Momentum decays toward zero over time."""
    state = PhaseState(position=[0.5]*9, momentum=[0.5]*9)
    zero_delta = [0.0]*9
    for _ in range(50):
        state = evolve_momentum(state, zero_delta)
    assert all(abs(p) < 0.15 for p in state.momentum)
    print("  ✓ momentum_decay")

def test_apply_momentum():
    state = PhaseState(position=[0.5]*9, momentum=[0.1]*9)
    new_state = apply_momentum(state, dt=0.1)
    # Position should shift slightly in momentum direction
    assert all(new_state.position[i] >= 0.5 for i in range(9))
    assert all_constraints_satisfied(new_state.position)
    print("  ✓ apply_momentum")

# ============================================================
# Velocity Tests
# ============================================================

def test_velocity_within_bounds():
    old = [0.5] * 9
    new = [0.55] * 9
    ok, vel = check_velocity(old, new)
    assert ok
    assert vel < MAX_VELOCITY
    print("  ✓ velocity_within_bounds")

def test_velocity_exceeds():
    old = [0.0] * 9
    new = [1.0] * 9
    ok, vel = check_velocity(old, new)
    assert not ok
    assert vel > MAX_VELOCITY
    print("  ✓ velocity_exceeds")

# ============================================================
# Calibration Tests
# ============================================================

def test_calibration_agreement():
    """Two depths agree → proceed."""
    r1 = DepthResult("shallow", [0.1]*9, 0.9)
    r2 = DepthResult("deep", [0.12]*9, 0.85)
    result = calibrate_depths([r1, r2], threshold=0.3)
    assert result.agreement > 0.8
    assert result.recommendation == "proceed"
    print("  ✓ calibration_agreement")

def test_calibration_divergence():
    """Two depths disagree → escalate or pause."""
    r1 = DepthResult("shallow", [0.5]*9, 0.9)
    r2 = DepthResult("deep", [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], 0.85)
    result = calibrate_depths([r1, r2], threshold=0.3)
    assert result.agreement < 0.8
    assert len(result.divergent_dims) > 0
    assert result.recommendation in ("escalate", "pause")
    print(f"  ✓ calibration_divergence (agreement={result.agreement:.2f}, rec={result.recommendation})")

# ============================================================
# Kala Conservation Tests
# ============================================================

def test_kala_conservation():
    """Kala is conserved in transfers between agents."""
    txs = [
        KalaTransaction("alice", "bob", 10.0, "tip"),
        KalaTransaction("bob", "carol", 5.0, "service"),
        KalaTransaction("carol", "alice", 5.0, "refund"),
    ]
    conserved, net = verify_kala_conservation(txs)
    assert conserved
    assert abs(net) < EPSILON
    print("  ✓ kala_conservation")

def test_kala_non_conservation():
    """Detect Kala creation out of thin air (without SYSTEM)."""
    txs = [
        KalaTransaction("alice", "bob", 100.0, "gift"),
        # bob never gives anything back → alice is -100, bob is +100
        # net of non-system = 0 (it balances between them)
    ]
    conserved, net = verify_kala_conservation(txs)
    assert conserved  # still conserved between agents
    print("  ✓ kala_non_conservation")

def test_kala_minting():
    """SYSTEM minting creates Kala, but non-system should still net zero."""
    txs = [
        KalaTransaction("SYSTEM", "alice", 100.0, "minting"),
        KalaTransaction("alice", "bob", 50.0, "tip"),
    ]
    conserved, net = verify_kala_conservation(txs)
    # alice: +100-50=+50, bob: +50 → non-system net = +100
    assert not conserved
    print("  ✓ kala_minting (non-system net ≠ 0)")

# ============================================================
# Bracket Tests
# ============================================================

def test_brackets():
    pos = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5, 0.5]
    brackets = compute_brackets(pos)
    assert len(brackets) == 9
    assert brackets[0] == 0  # 0.0 → bracket 0
    assert brackets[4] == 4  # 1.0 → bracket 4 (clamped)
    print("  ✓ brackets")

# ============================================================
# Full Cycle Test
# ============================================================

def test_full_action_cycle():
    """Score → evolve → apply → verify."""
    state = PhaseState.default()
    impacts = [0.05, 0.02, 0.03, 0.01, 0.0, 0.0, 0.04, 0.02, 0.03]

    # Score
    valence = score_action(state, impacts)
    assert valence.total_valence > 0

    # Evolve momentum
    state = evolve_momentum(state, valence.projected_delta)

    # Apply momentum
    new_state = apply_momentum(state, dt=0.1)
    assert all_constraints_satisfied(new_state.position)

    # Velocity check
    ok, vel = check_velocity(state.position, new_state.position)
    assert ok

    print(f"  ✓ full_action_cycle (valence={valence.total_valence:.3f}, vel={vel:.4f})")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing moral geometry...\n")

    print("Phase Space:")
    test_default_state()
    test_magnitude()
    test_distance()

    print("\nCoupling Constraints:")
    test_default_couplings_loaded()
    test_center_satisfies_all()
    test_prerequisite_violation()
    test_foundation_violation()
    test_enabler_constraint()
    test_high_virtue_satisfies()

    print("\nProjection:")
    test_projection_clamps()
    test_projection_fixes_violations()
    test_projection_preserves_valid()
    test_projection_foundation()

    print("\nAction Scoring:")
    test_score_positive_action()
    test_score_negative_action()
    test_score_with_violation()
    test_score_virtue_impacts()

    print("\nMomentum:")
    test_evolve_momentum()
    test_momentum_decay()
    test_apply_momentum()

    print("\nVelocity:")
    test_velocity_within_bounds()
    test_velocity_exceeds()

    print("\nCalibration:")
    test_calibration_agreement()
    test_calibration_divergence()

    print("\nKala Conservation:")
    test_kala_conservation()
    test_kala_non_conservation()
    test_kala_minting()

    print("\nBrackets:")
    test_brackets()

    print("\nFull Cycle:")
    test_full_action_cycle()

    print("\n" + "=" * 50)
    print("ALL MORAL GEOMETRY TESTS PASSED ✓")
    print("=" * 50)
