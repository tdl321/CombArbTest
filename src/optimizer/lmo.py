"""Linear Minimization Oracle (LMO) for Frank-Wolfe.

OPT-02: LMO with HiGHS Solver
- Solves: min <g, x> subject to Ax <= b
- Constraints come from logical relationships between markets
"""

import logging
import numpy as np
from numpy.typing import NDArray
import highspy
from dataclasses import dataclass
from typing import Literal

from .schema import MarketRelationship, MarketCluster, RelationshipGraph

logger = logging.getLogger(__name__)


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

    def add_relationship(self, rel: MarketRelationship) -> None:
        """Add constraint from a MarketRelationship."""
        if rel.type == "implies":
            if rel.to_market:
                self.add_implies(rel.from_market, rel.to_market)
        elif rel.type == "mutually_exclusive":
            if rel.to_market:
                self.add_mutually_exclusive(rel.from_market, rel.to_market)
        elif rel.type == "and":
            if rel.to_market:
                self.add_and_constraint(rel.from_market, rel.to_market)
        elif rel.type == "or":
            if rel.to_market:
                self.add_or_constraint(rel.from_market, rel.to_market)
        elif rel.type == "prerequisite":
            if rel.to_market:
                self.add_prerequisite(rel.from_market, rel.to_market)

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

    for rel in graph.get_all_relationships():
        builder.add_relationship(rel)

    return builder.build()


class LinearMinimizationOracle:
    """Linear Minimization Oracle using HiGHS solver."""

    def __init__(self, constraints: ConstraintMatrix):
        self.constraints = constraints
        self.n = len(constraints.market_ids)
        logger.debug("[LMO] Oracle initialized: %d vars, %d constraints",
                     self.n, constraints.A.shape[0])

    def solve(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve the linear minimization problem."""
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
        """Find extreme points of the polytope."""
        logger.debug("[LMO] Finding %d extreme points", num_points)

        rng = np.random.default_rng(42)
        extreme_points = []
        seen = set()

        for _ in range(num_points * 3):
            g = rng.standard_normal(self.n)
            x, _ = self.solve(g)

            key = tuple(np.round(x, 6))
            if key not in seen:
                seen.add(key)
                extreme_points.append(x)

                if len(extreme_points) >= num_points:
                    break

        logger.debug("[LMO] Found %d unique extreme points", len(extreme_points))
        return extreme_points
