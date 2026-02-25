# Combinatorial Arbitrage Detection System v2: Marginal Polytope Update

**Date**: 2026-02-25
**Phase**: 06-marginal-polytope-lmo (COMPLETE)

---

## Executive Summary

This system detects **arbitrage opportunities** in prediction markets (like Polymarket) where multiple markets have logical relationships. For example, "Republicans win Pennsylvania by 5+" **implies** "Trump wins Pennsylvania". If market prices violate these logical constraints, there's free money to be extracted.

The core idea: **Find the closest "coherent" (logically consistent) price vector to current market prices. The distance between them (measured by KL divergence) IS the arbitrage profit.**

---

## Part 1: The Mathematical Foundation

### The Problem We're Solving

Given:
- **N markets**, each with binary outcomes (YES/NO)
- **Logical constraints** between markets (implications, mutex, etc.)
- **Current market prices** `p_i` for each market's YES outcome

Find:
- **Coherent prices** `μ_i` that satisfy all logical constraints
- **KL divergence** = profit opportunity

### Why KL Divergence?

KL divergence measures "informational inconsistency" between two probability distributions:

```
KL(p || q) = Σ p_i * log(p_i / q_i)
```

For binary markets (Bernoulli):
```
KL(p || q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
```

**Key insight**: KL divergence is the maximum risk-free profit extractable from mispriced markets (measured in nats, convertible to dollars).

---

## Part 2: The Marginal Polytope

### What Is It?

The **marginal polytope M** is the set of all valid joint probability distributions over market outcomes. Each vertex of this polytope represents a **deterministic world state** (exactly one outcome per market is TRUE).

For 2 binary markets A and B:
- 4 vertices: (A=YES, B=YES), (A=YES, B=NO), (A=NO, B=YES), (A=NO, B=NO)
- The polytope is a 4D hypercube where each market sums to 1

### Adding Constraints Removes Vertices

If we add **B→A** (B implies A):
- (A=NO, B=YES) becomes **impossible** - removed from polytope
- Only 3 vertices remain
- The coherent price space is now smaller

**Code location**: `src/optimizer/lmo.py:100-120`

```python
def add_implies(self, from_market: str, to_market: str):
    """B=YES implies A=YES.
    Constraint: z_B_yes - z_A_yes <= 0
    """
    row = np.zeros(self.n)
    row[from_yes_idx] = 1.0   # +z_B
    row[to_yes_idx] = -1.0    # -z_A
    self._ub_rows.append(row)
    self._ub_rhs.append(0.0)  # <= 0
```

This constraint says: `z_B_yes ≤ z_A_yes`

If B=YES (z_B_yes=1), then A must be YES (z_A_yes=1) for the constraint to hold.

---

## Part 3: The Condition Space Model

### schema.py - Data Structures

**`Condition`** - A single outcome for a single market:
```python
@dataclass
class Condition:
    condition_id: str   # "market_A::YES"
    market_id: str      # "market_A"
    outcome_index: int  # 0=YES, 1=NO
    outcome_name: str   # "YES" or "NO"
```

**`ConditionSpace`** - All conditions across all markets:
```python
# For markets A, B with YES/NO outcomes:
# Index 0: A::YES
# Index 1: A::NO
# Index 2: B::YES
# Index 3: B::NO
```

The condition space is a **flattened vector representation** where position i corresponds to condition i being TRUE.

**Example**: 2 markets A, B
- Vector `[1, 0, 0, 1]` means: A=YES, B=NO
- Vector `[0, 1, 1, 0]` means: A=NO, B=YES

**Code location**: `src/optimizer/schema.py:33-116`

---

## Part 4: The Constraint System

### MarginalConstraintBuilder

Builds two types of constraints:

#### 1. Equality Constraints (Always Present)
**Each market has exactly one outcome TRUE:**

```python
# For market A with indices [0, 1]:
# z_0 + z_1 = 1
```

This means: P(A=YES) + P(A=NO) = 1

**Code**: `src/optimizer/lmo.py:66-77`
```python
def _build_exactly_one_constraints(self):
    for market_id in self.condition_space.market_ids:
        indices = self.get_condition_indices(market_id)
        row = np.zeros(self.n)
        for idx in indices:
            row[idx] = 1.0
        self._eq_rows.append(row)
        self._eq_rhs.append(1.0)
```

#### 2. Inequality Constraints (From Relationships)

**Implication (B→A)**:
```
z_B_yes ≤ z_A_yes
Rewritten: z_B_yes - z_A_yes ≤ 0
```

**Mutual Exclusivity (A ⊕ B)**:
```
z_A_yes + z_B_yes ≤ 1
```
(At most one can be TRUE)

**Opposite (A = ¬B)**:
```
z_A_yes + z_B_yes ≤ 1  (not both TRUE)
z_A_yes + z_B_yes ≥ 1  (not both FALSE)
```

**Code**: `src/optimizer/lmo.py:79-162`

---

## Part 5: The Linear Minimization Oracle (LMO)

### What Does It Do?

The LMO solves:
```
min gradient^T z
s.t. z is a vertex of the marginal polytope
```

Given a gradient vector (direction of steepest descent), it finds the vertex of the polytope that minimizes the dot product with that gradient.

### How?

It's a **Mixed Integer Linear Program (MILP)** solved by scipy:

```python
def solve(self, gradient: NDArray) -> tuple[NDArray, float]:
    result = milp(
        c=gradient,                    # Minimize gradient^T z
        constraints=constraints,        # A_eq, A_ub
        integrality=np.ones(n),        # All variables binary
        bounds=Bounds(0, 1),           # z_i ∈ {0, 1}
    )
    return np.round(result.x), gradient @ result.x
```

**Code**: `src/optimizer/lmo.py:262-298`

### Vertex Enumeration

For initialization and debugging, we can enumerate all vertices by solving with random gradients:

```python
def enumerate_vertices(self, max_vertices: int = 100):
    for _ in range(n_attempts):
        gradient = np.random.randn(self.n)  # Random direction
        z, _ = self.solve(gradient)
        if z not in vertices:
            vertices.append(z)
    return vertices
```

**Example Results**:
- 2 independent markets: 4 vertices
- 2 markets with B→A: 3 vertices (A=NO,B=YES is invalid)
- 3 markets with C→B→A: 4 vertices

---

## Part 6: KL Divergence Computation

### divergence.py - The Distance Functions

**Categorical KL** (for condition vectors):
```python
def categorical_kl(theta, mu, condition_space):
    total_kl = 0.0
    for market_id in condition_space.market_ids:
        indices = condition_space.get_condition_indices(market_id)

        theta_m = theta[indices]  # Market prices [p_yes, p_no]
        mu_m = mu[indices]        # Coherent prices [q_yes, q_no]

        # Normalize to sum to 1
        theta_m = theta_m / theta_m.sum()
        mu_m = mu_m / mu_m.sum()

        # KL for this market
        kl_m = np.sum(theta_m * np.log(theta_m / mu_m))
        total_kl += kl_m

    return total_kl
```

**Gradient of KL** (for optimization):
```python
def categorical_kl_gradient(theta, mu, condition_space):
    # d/d(mu_i) KL(theta || mu) = -theta_i / mu_i
    gradient[indices] = -theta_m / mu_m_safe
    return gradient
```

**Code**: `src/optimizer/divergence.py:15-96`

---

## Part 7: The Frank-Wolfe Algorithm

### Why Frank-Wolfe?

Standard **projected gradient descent** requires projecting onto the marginal polytope at each step. But the polytope has **exponentially many vertices** (2^N for N binary markets), making projection intractable.

Frank-Wolfe avoids projection by using the **LMO** instead:

```
1. Compute gradient at current point
2. Find vertex minimizing gradient (via LMO)
3. Move toward that vertex
4. Repeat
```

### The Main Loop

**Code**: `src/optimizer/frank_wolfe.py:35-134`

```python
def marginal_frank_wolfe(market_prices, constraints, config):
    # 1. Build theta vector (market prices in condition space)
    theta = build_theta_from_prices(market_prices, space)

    # 2. Initialize LMO
    lmo = MarginalPolytopeLMO(constraints)

    # 3. Initialize mu from centroid (interior point)
    mu = lmo.compute_centroid(n_samples=20)
    mu_interior = _contract_toward_centroid(mu, epsilon)

    # 4. Frank-Wolfe loop
    for t in range(max_iterations):
        # Gradient of KL(theta || mu) w.r.t. mu
        gradient = categorical_kl_gradient(theta, mu_interior, space)

        # LMO: find vertex minimizing gradient^T z
        z_new, _ = lmo.solve(gradient)

        # Duality gap: gradient^T (mu - z_new)
        gap = gradient @ (mu_interior - z_new)

        if gap < tolerance:
            converged = True
            break

        # Direction: vertex - current
        direction = z_new - mu_interior

        # Line search for optimal step size
        gamma = line_search_exact(theta, mu_interior, direction, space)

        # Update
        mu_interior = mu_interior + gamma * direction

        # Decay epsilon (barrier contraction)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        mu_interior = _contract_toward_centroid(mu_interior, epsilon, mu)

    # 5. Compute final KL divergence
    kl_divergence = categorical_kl(theta, mu_interior, space)

    return MarginalArbitrageResult(...)
```

### Step-by-Step Walkthrough

**Step 1: Initialize**
- Convert market prices `{'A': [0.45, 0.55], 'B': [0.60, 0.40]}` to theta vector `[0.45, 0.55, 0.60, 0.40]`
- Find interior point by averaging random vertices (centroid)

**Step 2: Compute Gradient**
- `gradient_i = -theta_i / mu_i`
- Points toward where we should move to minimize KL

**Step 3: Solve LMO**
- Find vertex z that minimizes `gradient^T z`
- This is the "most beneficial" corner of the polytope

**Step 4: Compute Duality Gap**
- `gap = gradient^T (mu - z)`
- This bounds how far we are from optimal
- When gap < tolerance, we've converged

**Step 5: Line Search**
- Find optimal step size γ ∈ [0, 1]
- We're moving along the line from `mu` to `z`

**Step 6: Update**
- `mu_new = mu + γ * (z - mu)`
- This is a convex combination, so stays in polytope

**Step 7: Barrier Contraction**
- Pull slightly toward interior to avoid boundary issues
- `mu_contracted = (1 - ε) * mu + ε * centroid`

---

## Part 8: Barrier Frank-Wolfe

### The Problem at Boundaries

KL divergence has terms like `log(1/mu_i)`. As `mu_i → 0`, this explodes to infinity.

### The Solution

Contract the polytope toward an interior point:

```python
def _contract_toward_centroid(mu, epsilon, centroid):
    return (1 - epsilon) * mu + epsilon * centroid
```

We optimize over `M_ε = (1 - ε)M + ε*centroid` instead of M.

As we converge (gap shrinks), we decay ε:
```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

This lets us get arbitrarily close to the true optimum.

**Code**: `src/optimizer/frank_wolfe.py:137-157`

---

## Part 9: Line Search

### Finding Optimal Step Size

Given direction `d = z - mu`, find `γ` minimizing `KL(theta || mu + γ*d)`:

```python
def line_search_exact(theta, mu, direction, condition_space):
    # Golden section search
    phi = (1 + np.sqrt(5)) / 2

    def objective(gamma):
        mu_new = mu + gamma * direction
        if np.any(mu_new < 0) or np.any(mu_new > 1):
            return float("inf")
        return categorical_kl(theta, mu_new, condition_space)

    # Binary search variant
    while abs(b - a) > tol:
        if objective(c) < objective(d):
            b = d
        else:
            a = c

    return (a + b) / 2
```

**Code**: `src/optimizer/divergence.py:141-193`

---

## Part 10: Result Interpretation

### MarginalArbitrageResult

```python
class MarginalArbitrageResult:
    # Input prices per condition
    condition_prices: dict[str, float]  # {'A::YES': 0.45, 'A::NO': 0.55, ...}

    # Projected coherent prices
    coherent_condition_prices: dict[str, float]

    # By market
    market_prices: dict[str, list[float]]       # {'A': [0.45, 0.55], ...}
    coherent_market_prices: dict[str, list[float]]

    # THE KEY METRIC
    kl_divergence: float  # Arbitrage profit in nats

    # Convergence info
    duality_gap: float
    converged: bool
    iterations: int

    def has_arbitrage(self, threshold=0.01):
        return self.kl_divergence > threshold
```

### Example Interpretation

```python
result = detect_arbitrage_simple(
    market_prices={'A': [0.45, 0.55], 'B': [0.60, 0.40]},
    implications=[('B', 'A')]  # B → A
)
```

**Violation**: B=YES at 60%, but A=YES only at 45%. Impossible!
- If B=YES (60% likely), A must be YES (should be ≥60%)
- Market is mispriced

**Result**:
- `kl_divergence = 0.0324` → Arbitrage exists
- `coherent_market_prices = {'A': [0.52, 0.48], 'B': [0.52, 0.48]}`
- Both prices converge to ~52% (the logically consistent value)

---

## Part 11: API Layers

### New API (Full Power)

```python
from src.optimizer import find_marginal_arbitrage, detect_arbitrage_simple

# With relationship graph
result = find_marginal_arbitrage(
    market_prices={'A': [0.45, 0.55], 'B': [0.60, 0.40]},
    relationships=graph,
)

# Simplified
result = detect_arbitrage_simple(
    market_prices={'A': [0.45, 0.55], 'B': [0.60, 0.40]},
    implications=[('B', 'A')],  # B → A
    mutex_pairs=[('C', 'D')],   # C ⊕ D
)
```

### Backward-Compatible API

```python
from src.optimizer import find_arbitrage

# Single float = P(YES)
result = find_arbitrage(
    market_prices={'A': 0.45, 'B': 0.60},
    relationships=graph,
)
```

Internally converts `0.45` → `[0.45, 0.55]` and back.

**Code**: `src/optimizer/frank_wolfe.py:269-385`

---

## Part 12: Constraint Types

### Supported Relationships

| Type | Meaning | Constraint |
|------|---------|------------|
| `IMPLIES` | B→A | `z_B ≤ z_A` |
| `MUTUALLY_EXCLUSIVE` | A⊕B | `z_A + z_B ≤ 1` |
| `EQUIVALENT` | A↔B | `z_A = z_B` |
| `OPPOSITE` | A = ¬B | `z_A + z_B = 1` |

### Chained Implications

For C→B→A (e.g., "Win by 10+ → Win by 5+ → Win"):
```python
builder.add_implies('C', 'B')  # C → B
builder.add_implies('B', 'A')  # B → A
```

Valid vertices:
1. (A=Y, B=Y, C=Y) - All win
2. (A=Y, B=Y, C=N) - Win by 5+ but not 10+
3. (A=Y, B=N, C=N) - Just win
4. (A=N, B=N, C=N) - Lose

Invalid: Any state where C=Y but B=N, or B=Y but A=N.

---

## Part 13: Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
├─────────────────────────────────────────────────────────────────┤
│ market_prices = {'trump_pa': [0.45, 0.55], 'rep_pa': [0.60, 0.40]} │
│ implications = [('rep_pa', 'trump_pa')]                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUILD CONDITION SPACE                         │
├─────────────────────────────────────────────────────────────────┤
│ Conditions:                                                      │
│   [0] trump_pa::YES                                             │
│   [1] trump_pa::NO                                              │
│   [2] rep_pa::YES                                               │
│   [3] rep_pa::NO                                                │
│                                                                  │
│ theta = [0.45, 0.55, 0.60, 0.40]                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUILD CONSTRAINTS                             │
├─────────────────────────────────────────────────────────────────┤
│ Equality (exactly-one per market):                              │
│   z[0] + z[1] = 1  (trump_pa)                                   │
│   z[2] + z[3] = 1  (rep_pa)                                     │
│                                                                  │
│ Inequality (implication rep_pa → trump_pa):                     │
│   z[2] - z[0] ≤ 0  (rep_pa_yes ≤ trump_pa_yes)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENUMERATE VERTICES                            │
├─────────────────────────────────────────────────────────────────┤
│ Valid vertices (binary vectors):                                │
│   [1, 0, 1, 0] - trump=YES, rep=YES  ✓                          │
│   [1, 0, 0, 1] - trump=YES, rep=NO   ✓                          │
│   [0, 1, 0, 1] - trump=NO,  rep=NO   ✓                          │
│   [0, 1, 1, 0] - trump=NO,  rep=YES  ✗ INVALID (violates B→A)  │
│                                                                  │
│ Centroid = average of valid vertices                            │
│          = [0.67, 0.33, 0.33, 0.67]                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRANK-WOLFE LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│ Iteration 0:                                                    │
│   mu = [0.67, 0.33, 0.33, 0.67] (centroid)                     │
│   gradient = -theta/mu = [-0.67, -1.67, -1.82, -0.60]          │
│   LMO finds vertex minimizing gradient^T z                      │
│   gap = 0.8, update mu                                          │
│                                                                  │
│ Iteration 1-N:                                                  │
│   Repeat until gap < tolerance                                  │
│                                                                  │
│ Final mu ≈ [0.52, 0.48, 0.52, 0.48]                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE KL DIVERGENCE                         │
├─────────────────────────────────────────────────────────────────┤
│ KL = Σ theta_i * log(theta_i / mu_i)                            │
│                                                                  │
│ Market trump_pa:                                                │
│   0.45*log(0.45/0.52) + 0.55*log(0.55/0.48) = 0.016            │
│                                                                  │
│ Market rep_pa:                                                  │
│   0.60*log(0.60/0.52) + 0.40*log(0.40/0.48) = 0.016            │
│                                                                  │
│ Total KL = 0.032 (ARBITRAGE DETECTED!)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                    │
├─────────────────────────────────────────────────────────────────┤
│ MarginalArbitrageResult:                                        │
│   market_prices = {'trump_pa': [0.45, 0.55], 'rep_pa': [0.60]}  │
│   coherent_prices = {'trump_pa': [0.52, 0.48], 'rep_pa': [0.52]}│
│   kl_divergence = 0.032                                         │
│   has_arbitrage() = True                                        │
│                                                                  │
│ INTERPRETATION:                                                  │
│   - Market says trump=45%, rep=60%                              │
│   - But rep→trump, so trump should be ≥60%                      │
│   - Coherent: both ~52%                                          │
│   - Profit opportunity: ~3.2% edge                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 14: Real-World Example

### Pennsylvania Election Markets

```python
# Market A: Trump wins 2024 election (ID: 253591)
# Market B: Republican wins Pennsylvania (ID: 255152)
# Relationship: B → A (if Rep wins PA, Trump wins national)

result = detect_arbitrage_simple(
    market_prices={
        'trump_national': [0.45, 0.55],  # 45% Trump wins
        'rep_pa': [0.60, 0.40],          # 60% Rep wins PA
    },
    implications=[('rep_pa', 'trump_national')]
)

# Result: KL = 0.032, arbitrage exists
# Trade: Buy Trump (underpriced), sell Rep PA (overpriced)
```

### Three-Way Chain

```python
# C: Rep wins PA by 10+
# B: Rep wins PA by 5+
# A: Trump wins PA
# Chain: C → B → A

result = detect_arbitrage_simple(
    market_prices={
        'A': [0.50, 0.50],  # Trump PA 50%
        'B': [0.45, 0.55],  # Rep 5+ is 45%
        'C': [0.55, 0.45],  # Rep 10+ is 55% (VIOLATION!)
    },
    implications=[('C', 'B'), ('B', 'A')]
)

# C > B is impossible (10+ can't be more likely than 5+)
# Result: Arbitrage detected
```

---

## Part 15: Guaranteed Profit Calculation

```python
# From the result:
guaranteed_profit = result.kl_divergence - result.duality_gap

# Only execute if profit exceeds costs:
if guaranteed_profit > transaction_costs + slippage:
    execute_trade()
```

The **duality gap** bounds our uncertainty. As Frank-Wolfe converges, the gap shrinks and our profit estimate becomes more precise.

---

## Component Summary

| Component | File | Purpose |
|-----------|------|---------|
| `ConditionSpace` | `schema.py` | Maps markets to condition indices |
| `MarginalConstraintBuilder` | `lmo.py` | Creates IP constraints from relationships |
| `MarginalPolytopeLMO` | `lmo.py` | Finds vertices via MILP |
| `categorical_kl` | `divergence.py` | Measures price divergence |
| `marginal_frank_wolfe` | `frank_wolfe.py` | Main optimization loop |
| `MarginalArbitrageResult` | `schema.py` | Output with all metrics |

---

## Key Insight

By representing market constraints as a convex polytope and finding the closest point in that polytope to current prices, we can exactly quantify arbitrage opportunities. The Frank-Wolfe algorithm makes this tractable even for large constraint systems.

---

## Version History

- **v1**: Simple mutex constraints only (Trump vs Harris)
- **v2**: Full marginal polytope with implication chains, MILP-based LMO, Barrier Frank-Wolfe
