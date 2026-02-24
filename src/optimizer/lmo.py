"""Linear Minimization Oracle (LMO) for Frank-Wolfe.

OPT-02: LMO with HiGHS Solver
- Solves: min <g, x> subject to Ax <= b
- Constraints come from logical relationships between markets
"""

import numpy as np
from numpy.typing import NDArray
import highspy
from dataclasses import dataclass
from typing import Literal

from .schema import MarketRelationship, MarketCluster, RelationshipGraph


@dataclass
class ConstraintMatrix:
    """Linear constraint representation Ax <= b."""
    A: NDArray[np.float64]  # Constraint matrix (m x n)
    b: NDArray[np.float64]  # Right-hand side (m,)
    lb: NDArray[np.float64]  # Lower bounds on x (n,)
    ub: NDArray[np.float64]  # Upper bounds on x (n,)
    market_ids: list[str]  # Map from index to market ID
    constraint_names: list[str]  # Description of each constraint


class ConstraintBuilder:
    """Build constraint matrices from relationship graphs."""
    
    def __init__(self, market_ids: list[str]):
        """Initialize with list of market IDs.
        
        Args:
            market_ids: List of market IDs. Each binary market should
                       have both YES and NO outcomes (e.g., "A_YES", "A_NO").
        """
        self.market_ids = list(market_ids)
        self.n = len(market_ids)
        self.id_to_idx = {mid: i for i, mid in enumerate(market_ids)}
        
        # Constraints: (coefficients, rhs, name)
        self._constraints: list[tuple[NDArray[np.float64], float, str]] = []
    
    def _get_idx(self, market_id: str) -> int:
        """Get index for market ID."""
        if market_id not in self.id_to_idx:
            raise ValueError(f"Unknown market ID: {market_id}")
        return self.id_to_idx[market_id]
    
    def add_implies(self, from_market: str, to_market: str) -> None:
        """Add implication constraint: P(A) <= P(B).
        
        If A implies B, then A can only be true when B is true,
        so P(A) cannot exceed P(B).
        """
        i = self._get_idx(from_market)
        j = self._get_idx(to_market)
        
        # P(A) - P(B) <= 0  =>  x[i] - x[j] <= 0
        coef = np.zeros(self.n)
        coef[i] = 1.0
        coef[j] = -1.0
        
        self._constraints.append((coef, 0.0, f"implies({from_market}->{to_market})"))
    
    def add_mutually_exclusive(self, market_a: str, market_b: str) -> None:
        """Add mutual exclusivity: P(A) + P(B) <= 1.
        
        If A and B cannot both be true, their combined probability
        cannot exceed 1.
        """
        i = self._get_idx(market_a)
        j = self._get_idx(market_b)
        
        # P(A) + P(B) <= 1  =>  x[i] + x[j] <= 1
        coef = np.zeros(self.n)
        coef[i] = 1.0
        coef[j] = 1.0
        
        self._constraints.append((coef, 1.0, f"mutex({market_a},{market_b})"))
    
    def add_binary_market(self, yes_market: str, no_market: str) -> None:
        """Add binary market constraint: P(YES) + P(NO) = 1.
        
        For a binary market, the YES and NO outcomes must sum to 1.
        We implement this as two inequalities:
        P(YES) + P(NO) <= 1 and P(YES) + P(NO) >= 1
        """
        i = self._get_idx(yes_market)
        j = self._get_idx(no_market)
        
        # P(YES) + P(NO) <= 1
        coef1 = np.zeros(self.n)
        coef1[i] = 1.0
        coef1[j] = 1.0
        self._constraints.append((coef1, 1.0, f"binary_ub({yes_market},{no_market})"))
        
        # P(YES) + P(NO) >= 1  =>  -P(YES) - P(NO) <= -1
        coef2 = np.zeros(self.n)
        coef2[i] = -1.0
        coef2[j] = -1.0
        self._constraints.append((coef2, -1.0, f"binary_lb({yes_market},{no_market})"))
    
    def add_and_constraint(self, market_a: str, market_b: str, result: str | None = None) -> None:
        """Add AND constraint approximation.
        
        For P(A AND B), we have:
        - P(A AND B) <= P(A)
        - P(A AND B) <= P(B)
        
        If result is provided, it represents a market for the conjunction.
        Otherwise, we just ensure individual market constraints are consistent.
        """
        if result:
            r = self._get_idx(result)
            i = self._get_idx(market_a)
            j = self._get_idx(market_b)
            
            # P(A AND B) <= P(A)
            coef1 = np.zeros(self.n)
            coef1[r] = 1.0
            coef1[i] = -1.0
            self._constraints.append((coef1, 0.0, f"and_a({result}<={market_a})"))
            
            # P(A AND B) <= P(B)
            coef2 = np.zeros(self.n)
            coef2[r] = 1.0
            coef2[j] = -1.0
            self._constraints.append((coef2, 0.0, f"and_b({result}<={market_b})"))
    
    def add_or_constraint(self, market_a: str, market_b: str, result: str | None = None) -> None:
        """Add OR constraint approximation.
        
        For P(A OR B), we have:
        - P(A OR B) >= P(A)
        - P(A OR B) >= P(B)
        - P(A OR B) <= P(A) + P(B)  (inclusion-exclusion upper bound)
        """
        if result:
            r = self._get_idx(result)
            i = self._get_idx(market_a)
            j = self._get_idx(market_b)
            
            # P(A OR B) >= P(A)  =>  -P(A OR B) + P(A) <= 0
            coef1 = np.zeros(self.n)
            coef1[r] = -1.0
            coef1[i] = 1.0
            self._constraints.append((coef1, 0.0, f"or_a({result}>={market_a})"))
            
            # P(A OR B) >= P(B)  =>  -P(A OR B) + P(B) <= 0
            coef2 = np.zeros(self.n)
            coef2[r] = -1.0
            coef2[j] = 1.0
            self._constraints.append((coef2, 0.0, f"or_b({result}>={market_b})"))
            
            # P(A OR B) <= P(A) + P(B)
            coef3 = np.zeros(self.n)
            coef3[r] = 1.0
            coef3[i] = -1.0
            coef3[j] = -1.0
            self._constraints.append((coef3, 0.0, f"or_ub({result}<={market_a}+{market_b})"))
    
    def add_prerequisite(self, prerequisite: str, dependent: str) -> None:
        """Add prerequisite constraint: P(dependent) <= P(prerequisite).
        
        Same as implies but with clearer semantics for temporal dependencies.
        """
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
        if not self._constraints:
            # No constraints, just bounds
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
    """Build constraint matrix from a RelationshipGraph.
    
    Args:
        graph: RelationshipGraph from LLM
        
    Returns:
        ConstraintMatrix for use with LMO
    """
    market_ids = list(graph.get_all_market_ids())
    builder = ConstraintBuilder(market_ids)
    
    for rel in graph.get_all_relationships():
        builder.add_relationship(rel)
    
    return builder.build()


class LinearMinimizationOracle:
    """Linear Minimization Oracle using HiGHS solver.
    
    Solves: min <g, x> subject to:
        - Ax <= b (relationship constraints)
        - lb <= x <= ub (box constraints, typically [0, 1])
        - Optional: contracted polytope for barrier method
    """
    
    def __init__(self, constraints: ConstraintMatrix):
        """Initialize LMO with constraints.
        
        Args:
            constraints: Constraint matrix from ConstraintBuilder
        """
        self.constraints = constraints
        self.n = len(constraints.market_ids)
        
    def solve(
        self,
        g: NDArray[np.float64],
        epsilon: float = 0.0,
        center: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Solve the linear minimization problem.
        
        min <g, x> subject to:
            Ax <= b (if epsilon = 0)
            OR
            (1-epsilon)*(Ax <= b) + epsilon*(x = center)  (barrier variant)
        
        The barrier variant contracts the polytope toward center,
        ensuring we stay in the interior for numerical stability.
        
        Args:
            g: Gradient vector (the linear objective)
            epsilon: Barrier parameter (0 = full polytope, >0 = contracted)
            center: Center point for contraction (default: uniform)
            
        Returns:
            Tuple of (solution x, objective value <g, x>)
        """
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)  # Suppress output
        
        # Set up the LP: min g^T x
        h.addVars(self.n, self.constraints.lb, self.constraints.ub)
        
        # Set objective coefficients
        for i in range(self.n):
            h.changeColCost(i, float(g[i]))
        
        # Add constraints
        if self.constraints.A.shape[0] > 0:
            if epsilon > 0:
                # Barrier variant: contract polytope toward center
                if center is None:
                    center = np.full(self.n, 0.5)  # Default to uniform
                
                # Contract: x_contracted = (1-eps)*x + eps*center
                # Original: Ax <= b
                # Contracted: A[(1-eps)*x + eps*center] <= b
                #           => A*(1-eps)*x <= b - A*eps*center
                #           => Ax <= (b - A*eps*center) / (1-eps)
                # But we also need to contract bounds
                
                # Simpler approach: modify bounds
                lb_contracted = (1 - epsilon) * self.constraints.lb + epsilon * center
                ub_contracted = (1 - epsilon) * self.constraints.ub + epsilon * center
                
                # Update bounds
                for i in range(self.n):
                    h.changeColBounds(
                        i,
                        max(float(lb_contracted[i]), 0.0),
                        min(float(ub_contracted[i]), 1.0)
                    )
                
                # Contract the RHS of Ax <= b
                # (1-eps)(Ax) <= (1-eps)b + eps*A*center
                # Ax <= b + eps/(1-eps) * (A*center - b)
                b_contracted = self.constraints.b.copy()
                if epsilon < 1.0:
                    Ac = self.constraints.A @ center
                    b_contracted = self.constraints.b + epsilon / (1 - epsilon) * (Ac - self.constraints.b)
                
                for i in range(self.constraints.A.shape[0]):
                    row = self.constraints.A[i]
                    nonzero_idx = np.nonzero(row)[0]
                    h.addRow(
                        -highspy.kHighsInf,  # Lower bound
                        float(b_contracted[i]),  # Upper bound
                        len(nonzero_idx),
                        nonzero_idx.tolist(),
                        row[nonzero_idx].tolist(),
                    )
            else:
                # Standard constraints
                for i in range(self.constraints.A.shape[0]):
                    row = self.constraints.A[i]
                    nonzero_idx = np.nonzero(row)[0]
                    h.addRow(
                        -highspy.kHighsInf,  # Lower bound (no lower bound)
                        float(self.constraints.b[i]),  # Upper bound
                        len(nonzero_idx),
                        nonzero_idx.tolist(),
                        row[nonzero_idx].tolist(),
                    )
        
        # Solve
        h.run()
        
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            # If infeasible, return a point in the interior
            if center is not None:
                return center.copy(), float(np.dot(g, center))
            else:
                x_fallback = np.full(self.n, 0.5)
                return x_fallback, float(np.dot(g, x_fallback))
        
        # Get solution
        solution = h.getSolution()
        x = np.array(solution.col_value[:self.n])
        obj = float(np.dot(g, x))
        
        return x, obj
    
    def find_violated_constraints(
        self,
        x: NDArray[np.float64],
        tol: float = 1e-6,
    ) -> list[tuple[str, float]]:
        """Find which constraints are violated by point x.
        
        Args:
            x: Point to check
            tol: Tolerance for violation
            
        Returns:
            List of (constraint_name, violation_amount) for violated constraints
        """
        violations = []
        
        # Check Ax <= b
        if self.constraints.A.shape[0] > 0:
            lhs = self.constraints.A @ x
            for i, (val, rhs, name) in enumerate(zip(
                lhs, self.constraints.b, self.constraints.constraint_names
            )):
                if val > rhs + tol:
                    violations.append((name, float(val - rhs)))
        
        # Check bounds
        for i, (xi, lb, ub, mid) in enumerate(zip(
            x, self.constraints.lb, self.constraints.ub, self.constraints.market_ids
        )):
            if xi < lb - tol:
                violations.append((f"lb({mid})", float(lb - xi)))
            if xi > ub + tol:
                violations.append((f"ub({mid})", float(xi - ub)))
        
        return violations
    
    def get_extreme_points(self, num_points: int = 10) -> list[NDArray[np.float64]]:
        """Find extreme points of the polytope.
        
        Uses random objective vectors to find vertices.
        Useful for InitFW to find interior starting point.
        
        Args:
            num_points: Number of extreme points to find
            
        Returns:
            List of extreme point arrays
        """
        rng = np.random.default_rng(42)
        extreme_points = []
        seen = set()
        
        for _ in range(num_points * 3):  # Try more to get unique points
            g = rng.standard_normal(self.n)
            x, _ = self.solve(g)
            
            # Round for hashing
            key = tuple(np.round(x, 6))
            if key not in seen:
                seen.add(key)
                extreme_points.append(x)
                
                if len(extreme_points) >= num_points:
                    break
        
        return extreme_points
