"""Linear Minimization Oracle (LMO) for Frank-Wolfe.

OPT-02: LMO with Integer Programming Support
- Solves: min <g, x> subject to Ax <= b
- Supports both continuous (LP) and integer (MILP) optimization
- Integer mode ensures discrete, executable trades
- Constraints come from logical relationships between markets
"""

import logging
import numpy as np
from numpy.typing import NDArray
import highspy
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from dataclasses import dataclass
from typing import Literal

from .schema import MarketRelationship, MarketCluster, RelationshipGraph

logger = logging.getLogger(__name__)


# Solver mode types
SolverMode = Literal["continuous", "integer", "binary"]


@dataclass
class ConstraintMatrix:
    """Linear constraint representation Ax <= b."""
    A: NDArray[np.float64]
    b: NDArray[np.float64]
    lb: NDArray[np.float64]
    ub: NDArray[np.float64]
    market_ids: list[str]
    constraint_names: list[str]


class ConstraintBuilder:
    """Build constraint matrices from relationship graphs."""

    def __init__(self, market_ids: list[str]):
        self.market_ids = list(market_ids)
        self.n = len(market_ids)
        self.id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
        self._constraints: list[tuple[NDArray[np.float64], float, str]] = []
        logger.debug("[LMO] ConstraintBuilder initialized with %d markets", self.n)

    def _get_idx(self, market_id: str) -> int:
        if market_id not in self.id_to_idx:
            raise ValueError("Unknown market ID: %s" % market_id)
        return self.id_to_idx[market_id]

    def add_implies(self, from_market: str, to_market: str) -> None:
        """Add implication constraint: P(A) <= P(B)."""
        i = self._get_idx(from_market)
        j = self._get_idx(to_market)

        coef = np.zeros(self.n)
        coef[i] = 1.0
        coef[j] = -1.0

        self._constraints.append((coef, 0.0, "implies(%s->%s)" % (from_market, to_market)))
        logger.debug("[LMO] Added implies constraint: %s -> %s", from_market, to_market)

    def add_mutually_exclusive(self, market_a: str, market_b: str) -> None:
        """Add mutual exclusivity: P(A) + P(B) <= 1."""
        i = self._get_idx(market_a)
        j = self._get_idx(market_b)

        coef = np.zeros(self.n)
        coef[i] = 1.0
        coef[j] = 1.0

        self._constraints.append((coef, 1.0, "mutex(%s,%s)" % (market_a, market_b)))
        logger.debug("[LMO] Added mutex constraint: %s, %s", market_a, market_b)

    def add_incompatible(self, market_a: str, market_b: str) -> None:
        """Add incompatibility constraint: P(A) + P(B) <= 1.
        
        Semantically same as mutually_exclusive but represents structural
        impossibility rather than direct competition.
        """
        i = self._get_idx(market_a)
        j = self._get_idx(market_b)

        coef = np.zeros(self.n)
        coef[i] = 1.0
        coef[j] = 1.0

        self._constraints.append((coef, 1.0, "incompatible(%s,%s)" % (market_a, market_b)))
        logger.debug("[LMO] Added incompatible constraint: %s, %s", market_a, market_b)

    def add_exhaustive(self, market_ids: list[str]) -> None:
        """Add exhaustive constraint: sum(P(markets)) >= 1.
        
        At least one outcome must occur.
        Represented as -sum(P) <= -1.
        """
        if len(market_ids) < 2:
            logger.debug("[LMO] Skipping exhaustive with < 2 markets")
            return
            
        coef = np.zeros(self.n)
        for mid in market_ids:
            if mid in self.id_to_idx:
                coef[self._get_idx(mid)] = -1.0

        self._constraints.append((coef, -1.0, "exhaustive(%s)" % ",".join(market_ids[:3])))
        logger.debug("[LMO] Added exhaustive constraint for %d markets", len(market_ids))

    def add_binary_market(self, yes_market: str, no_market: str) -> None:
        """Add binary market constraint: P(YES) + P(NO) = 1."""
        i = self._get_idx(yes_market)
        j = self._get_idx(no_market)

        coef1 = np.zeros(self.n)
        coef1[i] = 1.0
        coef1[j] = 1.0
        self._constraints.append((coef1, 1.0, "binary_ub(%s,%s)" % (yes_market, no_market)))

        coef2 = np.zeros(self.n)
        coef2[i] = -1.0
        coef2[j] = -1.0
        self._constraints.append((coef2, -1.0, "binary_lb(%s,%s)" % (yes_market, no_market)))
        logger.debug("[LMO] Added binary constraint: %s, %s", yes_market, no_market)

    def add_and_constraint(self, market_a: str, market_b: str, result: str | None = None) -> None:
        """Add AND constraint approximation."""
        if result:
            r = self._get_idx(result)
            i = self._get_idx(market_a)
            j = self._get_idx(market_b)

            coef1 = np.zeros(self.n)
            coef1[r] = 1.0
            coef1[i] = -1.0
            self._constraints.append((coef1, 0.0, "and_a(%s<=%s)" % (result, market_a)))

            coef2 = np.zeros(self.n)
            coef2[r] = 1.0
            coef2[j] = -1.0
            self._constraints.append((coef2, 0.0, "and_b(%s<=%s)" % (result, market_b)))
            logger.debug("[LMO] Added AND constraint: %s AND %s = %s", market_a, market_b, result)

    def add_or_constraint(self, market_a: str, market_b: str, result: str | None = None) -> None:
        """Add OR constraint approximation."""
        if result:
            r = self._get_idx(result)
            i = self._get_idx(market_a)
            j = self._get_idx(market_b)

            coef1 = np.zeros(self.n)
            coef1[r] = -1.0
            coef1[i] = 1.0
            self._constraints.append((coef1, 0.0, "or_a(%s>=%s)" % (result, market_a)))

            coef2 = np.zeros(self.n)
            coef2[r] = -1.0
            coef2[j] = 1.0
            self._constraints.append((coef2, 0.0, "or_b(%s>=%s)" % (result, market_b)))

            coef3 = np.zeros(self.n)
            coef3[r] = 1.0
            coef3[i] = -1.0
            coef3[j] = -1.0
            self._constraints.append((coef3, 0.0, "or_ub(%s<=%s+%s)" % (result, market_a, market_b)))
            logger.debug("[LMO] Added OR constraint: %s OR %s = %s", market_a, market_b, result)

    def add_prerequisite(self, prerequisite: str, dependent: str) -> None:
        """Add prerequisite constraint: P(dependent) <= P(prerequisite)."""
        self.add_implies(dependent, prerequisite)

    def add_relationship(self, rel: MarketRelationship, cluster_market_ids: list[str] | None = None) -> None:
        """Add constraint from a MarketRelationship.
        
        Args:
            rel: The relationship to add
            cluster_market_ids: Market IDs in the cluster (for exhaustive constraints)
        """
        if rel.type == "implies":
            if rel.to_market:
                self.add_implies(rel.from_market, rel.to_market)
        elif rel.type == "mutually_exclusive":
            if rel.to_market:
                self.add_mutually_exclusive(rel.from_market, rel.to_market)
        elif rel.type == "incompatible":
            # Handle incompatible same as mutually_exclusive
            if rel.to_market:
                self.add_incompatible(rel.from_market, rel.to_market)
        elif rel.type == "exhaustive":
            # Exhaustive is a unary constraint on a SET of markets
            # The cluster_market_ids provides the full set
            if cluster_market_ids:
                self.add_exhaustive(cluster_market_ids)
            else:
                # Fallback: just log warning, can't build constraint without market list
                logger.warning("[LMO] Exhaustive constraint without cluster_market_ids, skipping")
        elif rel.type == "and":
            if rel.to_market:
                self.add_and_constraint(rel.from_market, rel.to_market)
        elif rel.type == "or":
            if rel.to_market:
                self.add_or_constraint(rel.from_market, rel.to_market)
        elif rel.type == "prerequisite":
            if rel.to_market:
                self.add_prerequisite(rel.from_market, rel.to_market)
        else:
            logger.warning("[LMO] Unknown relationship type: %s", rel.type)

    def build(self) -> ConstraintMatrix:
        """Build the constraint matrix."""
        logger.info("[LMO] Building constraint matrix: %d constraints for %d markets",
                    len(self._constraints), self.n)

        if not self._constraints:
            return ConstraintMatrix(
                A=np.zeros((0, self.n)),
                b=np.zeros(0),
                lb=np.zeros(self.n),
                ub=np.ones(self.n),
                market_ids=self.market_ids,
                constraint_names=[],
            )

        A = np.vstack([c[0] for c in self._constraints])
        b = np.array([c[1] for c in self._constraints])
        names = [c[2] for c in self._constraints]

        return ConstraintMatrix(
            A=A,
            b=b,
            lb=np.zeros(self.n),
            ub=np.ones(self.n),
            market_ids=self.market_ids,
            constraint_names=names,
        )


def build_constraints_from_graph(graph: RelationshipGraph) -> ConstraintMatrix:
    """Build constraint matrix from a RelationshipGraph."""
    market_ids = list(graph.get_all_market_ids())
    logger.info("[LMO] Building constraints from graph: %d markets", len(market_ids))

    builder = ConstraintBuilder(market_ids)

    # Process each cluster to handle exhaustive constraints properly
    for cluster in graph.clusters:
        for rel in cluster.relationships:
            # Pass cluster market_ids for exhaustive constraints
            builder.add_relationship(rel, cluster_market_ids=cluster.market_ids)

    return builder.build()


class LinearMinimizationOracle:
    """Linear Minimization Oracle with Integer Programming support.
    
    Supports three solver modes:
    - "continuous": Standard LP (HiGHS) - for iterative Frank-Wolfe optimization
    - "integer": Integer LP (scipy MILP) - for discrete position quantities
    - "binary": Binary IP (scipy MILP) - for 0/1 decisions
    
    The default mode is "continuous" for backward compatibility with Frank-Wolfe.
    Use "integer" or "binary" mode for final trade execution decisions.
    """

    def __init__(
        self, 
        constraints: ConstraintMatrix,
        mode: SolverMode = "continuous",
        discrete_unit: float = 0.01,
    ):
        """Initialize the LMO.
        
        Args:
            constraints: The constraint matrix defining the feasible region
            mode: Solver mode - "continuous", "integer", or "binary"
            discrete_unit: For "integer" mode, the minimum discrete unit (e.g., 0.01 = 1 cent)
        """
        self.constraints = constraints
        self.n = len(constraints.market_ids)
        self.mode = mode
        self.discrete_unit = discrete_unit
        logger.debug("[LMO] Oracle initialized: %d vars, %d constraints, mode=%s",
                     self.n, constraints.A.shape[0], mode)

    def solve(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve the linear minimization problem.
        
        Args:
            g: Gradient/cost vector to minimize
            epsilon: Contraction factor for barrier method (0 = no contraction)
            center: Center point for contraction (required if epsilon > 0)
            
        Returns:
            Tuple of (optimal solution x, objective value g.x)
        """
        if self.mode == "continuous":
            return self._solve_lp(g, epsilon, center)
        elif self.mode == "binary":
            return self._solve_binary_ip(g, epsilon, center)
        else:  # "integer"
            return self._solve_integer_ip(g, epsilon, center)

    def _solve_lp(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve using continuous LP (HiGHS) - original implementation."""
        logger.debug("[LMO] Solving LP: epsilon=%.4f", epsilon)

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)

        h.addVars(self.n, self.constraints.lb, self.constraints.ub)

        for i in range(self.n):
            h.changeColCost(i, float(g[i]))

        if self.constraints.A.shape[0] > 0:
            if epsilon > 0:
                if center is None:
                    center = np.full(self.n, 0.5)

                lb_contracted = (1 - epsilon) * self.constraints.lb + epsilon * center
                ub_contracted = (1 - epsilon) * self.constraints.ub + epsilon * center

                for i in range(self.n):
                    h.changeColBounds(
                        i,
                        max(float(lb_contracted[i]), 0.0),
                        min(float(ub_contracted[i]), 1.0)
                    )

                b_contracted = self.constraints.b.copy()
                if epsilon < 1.0:
                    Ac = self.constraints.A @ center
                    b_contracted = self.constraints.b + epsilon / (1 - epsilon) * (Ac - self.constraints.b)

                for i in range(self.constraints.A.shape[0]):
                    row = self.constraints.A[i]
                    nonzero_idx = np.nonzero(row)[0]
                    h.addRow(
                        -highspy.kHighsInf,
                        float(b_contracted[i]),
                        len(nonzero_idx),
                        nonzero_idx.tolist(),
                        row[nonzero_idx].tolist(),
                    )
            else:
                for i in range(self.constraints.A.shape[0]):
                    row = self.constraints.A[i]
                    nonzero_idx = np.nonzero(row)[0]
                    h.addRow(
                        -highspy.kHighsInf,
                        float(self.constraints.b[i]),
                        len(nonzero_idx),
                        nonzero_idx.tolist(),
                        row[nonzero_idx].tolist(),
                    )

        h.run()

        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            logger.warning("[LMO] LP not optimal, status=%s, using fallback", status)
            if center is not None:
                return center.copy(), float(np.dot(g, center))
            else:
                x_fallback = np.full(self.n, 0.5)
                return x_fallback, float(np.dot(g, x_fallback))

        solution = h.getSolution()
        x = np.array(solution.col_value[:self.n])
        obj = float(np.dot(g, x))

        logger.debug("[LMO] LP solved: obj=%.6f", obj)
        return x, obj

    def _solve_binary_ip(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve using binary integer programming (scipy MILP).
        
        Variables are constrained to {0, 1}.
        """
        logger.debug("[LMO] Solving Binary IP: epsilon=%.4f", epsilon)

        # Apply epsilon contraction if specified
        lb = self.constraints.lb.copy()
        ub = self.constraints.ub.copy()
        b = self.constraints.b.copy()

        if epsilon > 0:
            if center is None:
                center = np.full(self.n, 0.5)
            lb = (1 - epsilon) * lb + epsilon * center
            ub = (1 - epsilon) * ub + epsilon * center
            if epsilon < 1.0:
                Ac = self.constraints.A @ center
                b = self.constraints.b + epsilon / (1 - epsilon) * (Ac - self.constraints.b)

        # For binary variables, bounds are 0 and 1
        lb = np.maximum(lb, 0.0)
        ub = np.minimum(ub, 1.0)

        # Build scipy MILP problem
        # integrality: 1 = integer variable (with bounds 0-1, effectively binary)
        integrality = np.ones(self.n, dtype=int)

        bounds = Bounds(lb=lb, ub=ub)

        constraints_list = []
        if self.constraints.A.shape[0] > 0:
            # Ax <= b  =>  -inf <= Ax <= b
            A_sparse = csc_matrix(self.constraints.A)
            constraints_list.append(
                LinearConstraint(A_sparse, -np.inf, b)
            )

        try:
            result = milp(
                c=g,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints_list if constraints_list else None,
                options={"disp": False, "time_limit": 60}
            )

            if result.success:
                x = np.array(result.x)
                # Round to ensure truly binary (solver might return 0.9999...)
                x = np.round(x).astype(float)
                x = np.clip(x, 0.0, 1.0)
                obj = float(np.dot(g, x))
                logger.debug("[LMO] Binary IP solved: obj=%.6f", obj)
                return x, obj
            else:
                logger.warning("[LMO] Binary IP failed: %s, using LP fallback", result.message)
                return self._solve_lp(g, epsilon, center)

        except Exception as e:
            logger.warning("[LMO] Binary IP exception: %s, using LP fallback", str(e))
            return self._solve_lp(g, epsilon, center)

    def _solve_integer_ip(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve using integer programming with discrete units (scipy MILP).
        
        Variables are constrained to multiples of discrete_unit (e.g., 0.01).
        This is achieved by scaling: z = x / discrete_unit, z is integer.
        """
        logger.debug("[LMO] Solving Integer IP: epsilon=%.4f, unit=%.4f", epsilon, self.discrete_unit)

        # Scale factor for discretization
        scale = self.discrete_unit

        # Apply epsilon contraction if specified
        lb = self.constraints.lb.copy()
        ub = self.constraints.ub.copy()
        b = self.constraints.b.copy()

        if epsilon > 0:
            if center is None:
                center = np.full(self.n, 0.5)
            lb = (1 - epsilon) * lb + epsilon * center
            ub = (1 - epsilon) * ub + epsilon * center
            if epsilon < 1.0:
                Ac = self.constraints.A @ center
                b = self.constraints.b + epsilon / (1 - epsilon) * (Ac - self.constraints.b)

        # Scale bounds to integer domain
        # x in [lb, ub] => z in [lb/scale, ub/scale] where z is integer
        lb_scaled = np.ceil(lb / scale).astype(float)
        ub_scaled = np.floor(ub / scale).astype(float)

        # Ensure feasible bounds
        lb_scaled = np.maximum(lb_scaled, 0.0)
        ub_scaled = np.maximum(ub_scaled, lb_scaled)

        # Scale constraint matrix
        # Ax <= b => A(z*scale) <= b => (A*scale)z <= b
        A_scaled = self.constraints.A * scale
        
        # Scale the cost vector
        # min g'x = min g'(z*scale) = min (g*scale)'z
        g_scaled = g * scale

        # All variables are integers
        integrality = np.ones(self.n, dtype=int)

        bounds = Bounds(lb=lb_scaled, ub=ub_scaled)

        constraints_list = []
        if self.constraints.A.shape[0] > 0:
            A_sparse = csc_matrix(A_scaled)
            constraints_list.append(
                LinearConstraint(A_sparse, -np.inf, b)
            )

        try:
            result = milp(
                c=g_scaled,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints_list if constraints_list else None,
                options={"disp": False, "time_limit": 60}
            )

            if result.success:
                z = np.array(result.x)
                # Convert back to original scale
                x = z * scale
                # Ensure within original bounds
                x = np.clip(x, self.constraints.lb, self.constraints.ub)
                obj = float(np.dot(g, x))
                logger.debug("[LMO] Integer IP solved: obj=%.6f", obj)
                return x, obj
            else:
                logger.warning("[LMO] Integer IP failed: %s, using LP fallback", result.message)
                return self._solve_lp(g, epsilon, center)

        except Exception as e:
            logger.warning("[LMO] Integer IP exception: %s, using LP fallback", str(e))
            return self._solve_lp(g, epsilon, center)

    def solve_discrete(
        self,
        g: NDArray[np.float64],
        mode: SolverMode = "integer",
        discrete_unit: float | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Convenience method to solve with discrete constraints.
        
        This temporarily overrides the instance mode for a single solve.
        Useful when the LMO is used in continuous mode for Frank-Wolfe
        but you need a discrete solution for execution.
        
        Args:
            g: Gradient/cost vector
            mode: "integer" or "binary"
            discrete_unit: Override discrete unit (for "integer" mode)
            
        Returns:
            Tuple of (optimal solution x, objective value g.x)
        """
        original_mode = self.mode
        original_unit = self.discrete_unit
        
        try:
            self.mode = mode
            if discrete_unit is not None:
                self.discrete_unit = discrete_unit
            return self.solve(g)
        finally:
            self.mode = original_mode
            self.discrete_unit = original_unit

    def find_violated_constraints(
        self,
        x: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> list[tuple[str, float]]:
        """Find which constraints are violated by point x."""
        violations = []

        if self.constraints.A.shape[0] > 0:
            lhs = self.constraints.A @ x
            for i, (val, rhs, name) in enumerate(zip(
                lhs, self.constraints.b, self.constraints.constraint_names
            )):
                if val > rhs + tol:
                    violations.append((name, float(val - rhs)))

        for i, (xi, lb, ub, mid) in enumerate(zip(
            x, self.constraints.lb, self.constraints.ub, self.constraints.market_ids
        )):
            if xi < lb - tol:
                violations.append(("lb(%s)" % mid, float(lb - xi)))
            if xi > ub + tol:
                violations.append(("ub(%s)" % mid, float(xi - ub)))

        if violations:
            logger.debug("[LMO] Found %d constraint violations", len(violations))

        return violations

    def get_extreme_points(self, num_points: int = 10) -> list[NDArray[np.float64]]:
        """Find extreme points of the polytope.
        
        For continuous mode, uses random directions.
        For binary mode, extreme points are exactly the binary vertices.
        """
        logger.debug("[LMO] Finding %d extreme points (mode=%s)", num_points, self.mode)

        rng = np.random.default_rng(42)
        extreme_points = []
        seen = set()

        for _ in range(num_points * 3):
            g = rng.standard_normal(self.n)
            x, _ = self.solve(g)

            # Round for comparison (especially important for integer modes)
            key = tuple(np.round(x, 6))
            if key not in seen:
                seen.add(key)
                extreme_points.append(x)

                if len(extreme_points) >= num_points:
                    break

        logger.debug("[LMO] Found %d unique extreme points", len(extreme_points))
        return extreme_points
