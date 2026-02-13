"""
Graph Zero Moral Geometry

The 19-dimensional phase space:
  9 position dimensions (virtues v0-v8)
  9 momentum dimensions (rates of change p0-p8)
  1 time dimension (kala)

Trustworthiness IS the metric tensor — it defines what "close" means.
Kala IS the scalar field — the reward landscape over the manifold.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from graph_zero.graph.schema import COUPLINGS, VIRTUE_NAMES

NUM_VIRTUES = 9
MAX_VELOCITY = 0.5
MAX_MOMENTUM = 1.0
MOMENTUM_DECAY = 0.95
EPSILON = 1e-6


# ============================================================
# Phase Space State
# ============================================================

@dataclass
class PhaseState:
    """Full 19D state of an agent's moral position."""
    position: list[float]     # 9 floats in [0, 1]
    momentum: list[float]     # 9 floats
    kala_time: float = 0.0

    def __post_init__(self):
        assert len(self.position) == NUM_VIRTUES
        assert len(self.momentum) == NUM_VIRTUES

    @staticmethod
    def default() -> 'PhaseState':
        return PhaseState(position=[0.5] * NUM_VIRTUES, momentum=[0.0] * NUM_VIRTUES)

    def magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.position))

    def momentum_magnitude(self) -> float:
        return math.sqrt(sum(p * p for p in self.momentum))

    def distance(self, other: 'PhaseState') -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))


# ============================================================
# Coupling Constraints
# ============================================================

@dataclass
class CouplingConstraint:
    from_dim: int
    to_dim: int
    coefficient: float
    direction: str  # prerequisite, enabler, foundation

    def max_target(self, position: list[float]) -> float:
        v_from = position[self.from_dim]
        if self.direction == 'prerequisite':
            return v_from + (1.0 - self.coefficient) * (1.0 - v_from)
        elif self.direction == 'enabler':
            return v_from + (1.0 - self.coefficient * 0.5) * (1.0 - v_from)
        elif self.direction == 'foundation':
            if self.coefficient > 0:
                return min(1.0, v_from / self.coefficient)
            return 1.0
        return 1.0

    def is_satisfied(self, position: list[float]) -> bool:
        if self.direction == 'foundation':
            return position[self.from_dim] >= position[self.to_dim] * self.coefficient - EPSILON
        else:
            return position[self.to_dim] <= self.max_target(position) + EPSILON


DEFAULT_COUPLINGS = [
    CouplingConstraint(from_dim=f, to_dim=t, coefficient=c, direction=d)
    for f, t, c, d in COUPLINGS
]


def check_all_constraints(position: list[float],
                          couplings: list[CouplingConstraint] = None) -> list[tuple[CouplingConstraint, bool]]:
    if couplings is None:
        couplings = DEFAULT_COUPLINGS
    return [(c, c.is_satisfied(position)) for c in couplings]


def all_constraints_satisfied(position: list[float],
                              couplings: list[CouplingConstraint] = None) -> bool:
    return all(sat for _, sat in check_all_constraints(position, couplings))


# ============================================================
# Position Projection (iterative clamping)
# ============================================================

def project_position(desired: list[float],
                     couplings: list[CouplingConstraint] = None,
                     max_iterations: int = 50) -> list[float]:
    """Project a desired position onto the constraint manifold."""
    if couplings is None:
        couplings = DEFAULT_COUPLINGS
    pos = [max(0.0, min(1.0, v)) for v in desired]

    for _ in range(max_iterations):
        changed = False
        for c in couplings:
            max_val = c.max_target(pos)
            if pos[c.to_dim] > max_val + EPSILON:
                pos[c.to_dim] = max_val
                changed = True
            if c.direction == 'foundation':
                min_from = pos[c.to_dim] * c.coefficient
                if pos[c.from_dim] < min_from - EPSILON:
                    pos[c.from_dim] = min_from
                    changed = True
        if not changed:
            break

    return [max(0.0, min(1.0, v)) for v in pos]


# ============================================================
# ActionValence — scoring an action's moral impact
# ============================================================

@dataclass
class ActionValence:
    raw_delta: list[float]
    projected_delta: list[float]
    total_valence: float
    virtue_impacts: dict[str, float]
    constraint_violations: list[str]
    depth_agreement: float


def score_action(current: PhaseState, action_impacts: list[float],
                 couplings: list[CouplingConstraint] = None,
                 weights: Optional[list[float]] = None) -> ActionValence:
    """Score how an action shifts moral position."""
    if couplings is None:
        couplings = DEFAULT_COUPLINGS
    if weights is None:
        weights = [1.0] * NUM_VIRTUES
    assert len(action_impacts) == NUM_VIRTUES

    desired = [current.position[i] + action_impacts[i] for i in range(NUM_VIRTUES)]
    projected = project_position(desired, couplings)
    projected_delta = [projected[i] - current.position[i] for i in range(NUM_VIRTUES)]

    violations = []
    for c in couplings:
        if not c.is_satisfied(desired):
            from_name = VIRTUE_NAMES[c.from_dim][0]
            to_name = VIRTUE_NAMES[c.to_dim][0]
            violations.append(f"{from_name}→{to_name} ({c.direction})")

    weighted_sum = sum(projected_delta[i] * weights[i] for i in range(NUM_VIRTUES))
    total_valence = max(-1.0, min(1.0, weighted_sum / max(sum(weights), EPSILON)))

    virtue_impacts = {VIRTUE_NAMES[i][0]: projected_delta[i] for i in range(NUM_VIRTUES)}

    return ActionValence(
        raw_delta=action_impacts,
        projected_delta=projected_delta,
        total_valence=total_valence,
        virtue_impacts=virtue_impacts,
        constraint_violations=violations,
        depth_agreement=1.0,
    )


# ============================================================
# Momentum Dynamics
# ============================================================

def evolve_momentum(state: PhaseState, delta_position: list[float],
                    dt: float = 1.0) -> PhaseState:
    new_momentum = []
    for i in range(NUM_VIRTUES):
        p_new = MOMENTUM_DECAY * state.momentum[i] + delta_position[i] / max(dt, EPSILON)
        p_new = max(-MAX_MOMENTUM, min(MAX_MOMENTUM, p_new))
        new_momentum.append(p_new)
    return PhaseState(position=state.position[:], momentum=new_momentum,
                      kala_time=state.kala_time + dt)


def apply_momentum(state: PhaseState, dt: float = 0.1,
                   couplings: list[CouplingConstraint] = None) -> PhaseState:
    desired = [state.position[i] + state.momentum[i] * dt for i in range(NUM_VIRTUES)]
    projected = project_position(desired, couplings)
    return PhaseState(position=projected, momentum=state.momentum[:],
                      kala_time=state.kala_time)


# ============================================================
# Velocity Check (for Gate E_VELOCITY)
# ============================================================

def check_velocity(old_position: list[float], new_position: list[float],
                   max_velocity: float = MAX_VELOCITY) -> tuple[bool, float]:
    delta_sq = sum((new_position[i] - old_position[i]) ** 2 for i in range(NUM_VIRTUES))
    velocity = math.sqrt(delta_sq)
    return velocity <= max_velocity + EPSILON, velocity


# ============================================================
# Calibration — detecting depth divergence
# ============================================================

@dataclass
class DepthResult:
    depth_name: str
    action_scores: list[float]
    confidence: float


@dataclass
class CalibrationResult:
    agreement: float
    divergent_dims: list[int]
    recommendation: str  # "proceed", "escalate", "pause"


def calibrate_depths(results: list[DepthResult],
                     threshold: float = 0.3) -> CalibrationResult:
    if len(results) < 2:
        return CalibrationResult(agreement=1.0, divergent_dims=[], recommendation="proceed")

    shallow = results[0].action_scores
    divergent = []
    for deeper in results[1:]:
        for i in range(NUM_VIRTUES):
            if abs(shallow[i] - deeper.action_scores[i]) > threshold and i not in divergent:
                divergent.append(i)

    agreement = 1.0 - len(divergent) / NUM_VIRTUES
    if agreement >= 0.8:
        recommendation = "proceed"
    elif agreement >= 0.5:
        recommendation = "escalate"
    else:
        recommendation = "pause"

    return CalibrationResult(agreement=agreement, divergent_dims=divergent,
                             recommendation=recommendation)


# ============================================================
# Kala Conservation
# ============================================================

@dataclass
class KalaTransaction:
    sender: str
    receiver: str
    amount: float
    reason: str


def verify_kala_conservation(transactions: list[KalaTransaction]) -> tuple[bool, float]:
    balances: dict[str, float] = {}
    for tx in transactions:
        balances[tx.sender] = balances.get(tx.sender, 0) - tx.amount
        balances[tx.receiver] = balances.get(tx.receiver, 0) + tx.amount
    non_system = {k: v for k, v in balances.items() if k != "SYSTEM"}
    net = sum(non_system.values())
    return abs(net) < EPSILON, net


def compute_brackets(position: list[float], num_brackets: int = 5) -> list[int]:
    brackets = []
    for v in position:
        b = int(v * num_brackets)
        b = max(0, min(num_brackets - 1, b))
        brackets.append(b)
    return brackets
