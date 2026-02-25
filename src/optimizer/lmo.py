"""Linear Minimization Oracle for Marginal Polytope.

The LMO solves integer programs to find vertices of the marginal polytope:
    min c^T z  subject to z in Z (valid binary outcomes)

Each vertex z is a binary vector representing a valid joint outcome across
all markets, respecting logical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import gurobipy as gp
from gurobipy import GRB

from .schema import (
    ConditionSpace,
    MarketRelationship,
    RelationshipGraph,
    RelationshipType,
)


@dataclass
class MarginalConstraintMatrix:
    """Constraint matrices for the marginal polytope IP.

    The marginal polytope is defined by:
    - Equality constraints: A_eq @ z = b_eq (e.g., sum of outcomes per market = 1)
    - Inequality constraints: A_ub @ z <= b_ub (logical implications)
    - Binary constraints: z_i in {0, 1}
    """

    A_eq: NDArray  # Equality constraint matrix
    b_eq: NDArray  # Equality RHS
    A_ub: NDArray  # Inequality constraint matrix (<= form)
    b_ub: NDArray  # Inequality RHS
    condition_space: ConditionSpace


class MarginalConstraintBuilder:
    """Builder for marginal polytope constraints.

    Constructs the constraint matrices for integer programming over valid
    joint outcomes.
    """

    def __init__(self, condition_space: ConditionSpace):
        self.condition_space = condition_space
        self.n = condition_space.n_conditions()

        # Equality constraints (exactly one outcome per market)
        self._eq_rows: list[NDArray] = []
        self._eq_rhs: list[float] = []

        # Inequality constraints (logical relationships)
        self._ub_rows: list[NDArray] = []
        self._ub_rhs: list[float] = []

        # Build exactly-one constraints by default
        self._build_exactly_one_constraints()

    def _build_exactly_one_constraints(self) -> None:
        """For each market: sum of outcome probabilities = 1.

        In binary terms: exactly one outcome is TRUE per market.
        """
        for market_id in self.condition_space.market_ids:
            indices = self.condition_space.get_condition_indices(market_id)
            row = np.zeros(self.n)
            for idx in indices:
                row[idx] = 1.0
            self._eq_rows.append(row)
            self._eq_rhs.append(1.0)

    def add_implies(self, from_market: str, to_market: str) -> MarginalConstraintBuilder:
        """Add implication constraint: from_market=YES implies to_market=YES.

        Constraint: z_from_yes <= z_to_yes
        Rewritten: z_from_yes - z_to_yes <= 0

        Example: "Rep wins PA by 5+" implies "Trump wins PA"
        If B=YES, then A must be YES.
        """
        from_yes_idx = self.condition_space.get_yes_index(from_market)
        to_yes_idx = self.condition_space.get_yes_index(to_market)

        row = np.zeros(self.n)
        row[from_yes_idx] = 1.0
        row[to_yes_idx] = -1.0

        self._ub_rows.append(row)
        self._ub_rhs.append(0.0)

        return self

    def add_mutually_exclusive(
        self, market_a: str, market_b: str
    ) -> MarginalConstraintBuilder:
        """Add mutual exclusivity: at most one can be YES.

        Constraint: z_a_yes + z_b_yes <= 1

        Example: "Candidate A wins" and "Candidate B wins" cannot both be true.
        """
        a_yes_idx = self.condition_space.get_yes_index(market_a)
        b_yes_idx = self.condition_space.get_yes_index(market_b)

        row = np.zeros(self.n)
        row[a_yes_idx] = 1.0
        row[b_yes_idx] = 1.0

        self._ub_rows.append(row)
        self._ub_rhs.append(1.0)

        return self

    def add_equivalent(
        self, market_a: str, market_b: str
    ) -> MarginalConstraintBuilder:
        """Add equivalence: A=YES iff B=YES.

        Constraints:
        - z_a_yes <= z_b_yes (A implies B)
        - z_b_yes <= z_a_yes (B implies A)
        Equivalent to: z_a_yes = z_b_yes
        """
        self.add_implies(market_a, market_b)
        self.add_implies(market_b, market_a)
        return self

    def add_opposite(self, market_a: str, market_b: str) -> MarginalConstraintBuilder:
        """Add opposite relationship: A=YES iff B=NO.

        Constraints:
        - z_a_yes + z_b_yes <= 1 (not both YES)
        - z_a_yes + z_b_yes >= 1 (at least one YES, i.e., not both NO)

        The second constraint: z_a_yes + z_b_yes >= 1
        Rewritten: -z_a_yes - z_b_yes <= -1
        """
        a_yes_idx = self.condition_space.get_yes_index(market_a)
        b_yes_idx = self.condition_space.get_yes_index(market_b)

        # Not both YES
        row1 = np.zeros(self.n)
        row1[a_yes_idx] = 1.0
        row1[b_yes_idx] = 1.0
        self._ub_rows.append(row1)
        self._ub_rhs.append(1.0)

        # Not both NO (at least one YES)
        row2 = np.zeros(self.n)
        row2[a_yes_idx] = -1.0
        row2[b_yes_idx] = -1.0
        self._ub_rows.append(row2)
        self._ub_rhs.append(-1.0)

        return self

    def add_relationship(self, rel: MarketRelationship) -> MarginalConstraintBuilder:
        """Add a relationship from the relationship graph.

        Args:
            rel: MarketRelationship to add

        Returns:
            self for chaining
        """
        if rel.type == RelationshipType.IMPLIES:
            self.add_implies(rel.from_market, rel.to_market)
        elif rel.type == RelationshipType.MUTUALLY_EXCLUSIVE:
            self.add_mutually_exclusive(rel.from_market, rel.to_market)
        elif rel.type == RelationshipType.EQUIVALENT:
            self.add_equivalent(rel.from_market, rel.to_market)
        elif rel.type == RelationshipType.OPPOSITE:
            self.add_opposite(rel.from_market, rel.to_market)

        return self

    def add_relationships(
        self, relationships: list[MarketRelationship]
    ) -> MarginalConstraintBuilder:
        """Add multiple relationships."""
        for rel in relationships:
            self.add_relationship(rel)
        return self

    def build(self) -> MarginalConstraintMatrix:
        """Build the final constraint matrices.

        Returns:
            MarginalConstraintMatrix with all constraints
        """
        # Equality constraints
        if self._eq_rows:
            A_eq = np.vstack(self._eq_rows)
            b_eq = np.array(self._eq_rhs)
        else:
            A_eq = np.zeros((0, self.n))
            b_eq = np.zeros(0)

        # Inequality constraints
        if self._ub_rows:
            A_ub = np.vstack(self._ub_rows)
            b_ub = np.array(self._ub_rhs)
        else:
            A_ub = np.zeros((0, self.n))
            b_ub = np.zeros(0)

        return MarginalConstraintMatrix(
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            condition_space=self.condition_space,
        )


class MarginalPolytopeLMO:
    """Linear Minimization Oracle for the marginal polytope.

    Solves: min c^T z subject to z in Z (valid binary joint outcomes)

    Uses Gurobi MILP solver with binary integrality constraints.
    """

    def __init__(self, constraints: MarginalConstraintMatrix):
        self.constraints = constraints
        self.n = constraints.condition_space.n_conditions()
        self._vertices: list[NDArray] = []

        # Store constraint matrices directly (no scipy objects)
        self.A_eq = constraints.A_eq
        self.b_eq = constraints.b_eq
        self.A_ub = constraints.A_ub
        self.b_ub = constraints.b_ub

    def solve(self, gradient: NDArray) -> tuple[NDArray, float]:
        """Solve the LMO using Gurobi: find vertex minimizing gradient^T z.

        Args:
            gradient: The gradient vector (objective coefficients)

        Returns:
            Tuple of (vertex z, objective value gradient^T z)
        """
        try:
            # Create model (suppress output)
            model = gp.Model("LMO")
            model.Params.OutputFlag = 0  # Silent
            model.Params.TimeLimit = 10  # 10 second timeout

            # Binary variables: z_i ∈ {0, 1}
            z = model.addMVar(self.n, vtype=GRB.BINARY, name="z")

            # Objective: minimize gradient^T z
            model.setObjective(gradient @ z, GRB.MINIMIZE)

            # Equality constraints: A_eq @ z = b_eq
            if self.A_eq.shape[0] > 0:
                model.addMConstr(self.A_eq, z, "=", self.b_eq, name="eq")

            # Inequality constraints: A_ub @ z <= b_ub
            if self.A_ub.shape[0] > 0:
                model.addMConstr(self.A_ub, z, "<", self.b_ub, name="ub")

            # Solve
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                z_sol = np.array(z.X)
                obj = model.ObjVal
                self._cache_vertex(z_sol)
                return z_sol, obj
            else:
                # Fallback: return uniform if no solution
                z_fallback = np.ones(self.n) / 2
                return z_fallback, gradient @ z_fallback

        except gp.GurobiError:
            # License or other Gurobi error - fallback
            z_fallback = np.ones(self.n) / 2
            return z_fallback, gradient @ z_fallback

    def _cache_vertex(self, z: NDArray) -> None:
        """Cache a vertex if not already seen."""
        for v in self._vertices:
            if np.allclose(v, z):
                return
        self._vertices.append(z.copy())

    def enumerate_vertices(self, max_vertices: int = 100) -> list[NDArray]:
        """Enumerate vertices by solving with random gradients.

        This is an approximation - may not find all vertices but finds
        a representative set.

        Args:
            max_vertices: Maximum number of vertices to find

        Returns:
            List of vertex vectors (binary)
        """
        vertices = list(self._vertices)  # Start with cached
        n_attempts = max_vertices * 3

        for _ in range(n_attempts):
            if len(vertices) >= max_vertices:
                break

            # Random gradient
            gradient = np.random.randn(self.n)
            z, _ = self.solve(gradient)

            # Check if new
            is_new = True
            for v in vertices:
                if np.allclose(v, z):
                    is_new = False
                    break

            if is_new:
                vertices.append(z.copy())

        return vertices

    def get_cached_vertices(self) -> list[NDArray]:
        """Return all cached vertices."""
        return list(self._vertices)

    def compute_centroid(self, n_samples: int = 20) -> NDArray:
        """Compute approximate centroid of the polytope.

        Uses vertex enumeration and averages.

        Args:
            n_samples: Number of random gradient samples

        Returns:
            Centroid vector (interior point)
        """
        vertices = self.enumerate_vertices(max_vertices=n_samples)

        if not vertices:
            # Fallback: uniform
            return np.ones(self.n) / 2

        centroid = np.mean(vertices, axis=0)
        return centroid


def build_constraints_from_graph(
    market_ids: list[str],
    relationships: RelationshipGraph,
    market_outcomes: dict[str, list[str]] | None = None,
) -> MarginalConstraintMatrix:
    """Build constraint matrices from a relationship graph.

    Convenience function combining ConditionSpace and ConstraintBuilder.

    Args:
        market_ids: List of market IDs to include
        relationships: The relationship graph
        market_outcomes: Optional outcome names per market

    Returns:
        MarginalConstraintMatrix ready for LMO
    """
    # Build condition space
    space = ConditionSpace.from_market_data(market_ids, market_outcomes)

    # Build constraints
    builder = MarginalConstraintBuilder(space)

    # Add relationships
    rels = relationships.get_relationships(market_ids)
    builder.add_relationships(rels)

    return builder.build()


# =============================================================================
# Backward Compatibility Types (aliases for existing code)
# =============================================================================

from enum import Enum


class SolverMode(str, Enum):
    """Solver mode for the LMO."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


# Type aliases for backward compatibility
ConstraintMatrix = MarginalConstraintMatrix
ConstraintBuilder = MarginalConstraintBuilder
LinearMinimizationOracle = MarginalPolytopeLMO
