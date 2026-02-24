This document outlines the mathematical framework for **combinatorial arbitrage** within prediction markets, a strategy that exploits mispricings across interdependent events.

---

## The Theoretical Math Behind Combinatorial Arbitrage in Prediction Markets

### Market Microstructure and the Logic of Contingent Claims

This thesis provides a mathematical reconstruction of combinatorial arbitrage on prediction markets like Polymarket. It represents a sophisticated methodology tailored to these markets, looking for exploitable mispricings over a **probabilistic forest of dependencies**.

For instance, a team winning a championship is dependent on winning preceding rounds (quarter-finals, semi-finals, etc.). If the implied probability of the championship win is inconsistent with the combined price of these prerequisite wins, an arbitrage opportunity exists. This complexity creates a higher barrier to entry compared to simple singular market rebalancing. Research suggests these strategies have accounted for roughly **$40 million** in extracted profit.

Implementation requires four key mathematical domains:

1. **Convex Optimization:** For efficient search.
2. **Information Geometry:** For distance measurement.
3. **Operations Research:** For constraint mapping.
4. **Algorithmic Game Theory:** To understand market maker pricing rules.

---

## Convex Optimization Problem

We use constrained convex optimization to find the "correct" or arbitrage-free set of probabilities for related events. We seek the closest, most coherent set of prices to current market prices under the constraint that all probabilities must add up to 1. In a convex problem, the local minimum is the global minimum, ensuring a single solution.

The solution exists within a **feasible set**—the marginal polytope ($M$), which is the convex hull of all logically possible terminal payoff vectors.

### Solving with the Frank-Wolfe Algorithm

Standard methods like **projected gradient descent** are computationally intractable here because the marginal polytope has an exponential number of vertices, making projections difficult. Instead, we use the **Frank-Wolfe (FW) algorithm**, a conditional gradient method that avoids explicit projections through a **Linear Minimization Oracle (LMO)**.

#### Initial Position (InitFW)

The system finds a stable interior position in three steps:

1. 
**Integer Programming (IP) Solver:** Checks if an event outcome is fixed to 0 or 1 by searching for valid terminal states where $z_i=1$ and $z_i=0$.


2. 
**Settlement:** If a team has dropped out (fixed to 0) but the market hasn't reflected it, the IP solver finds valid extreme vertices (corners of the polytope).


3. 
**Centroid Calculation:** The solver calculates the average of the valid vertices ($Z_0$) to guarantee a starting point ($u$) in the relative interior for numerical stability.



$$u = \frac{1}{|Z_0|} \sum_{z \in Z_0} z$$



#### Iterative Solver

At each iteration $k$, the algorithm computes a linear approximation of the objective function $f$ around the current price vector $x^{(k)}$ and solves the following:

$$s^{(k)} = \text{argmin}_{s \in M} \langle \nabla f(x^{(k)}), s \rangle$$



This identifies the vertex ($s^{(k)}$) with the steepest gradient leading to the optimal solution. The update rule then outputs a new vertex as a weighted average:

$$x^{(k+1)} = x^{(k)} + \gamma_k (s^{(k)} - x^{(k)})$$



Where $\gamma$ is the step size. Since both points are within the polytope, their weighted average remains within it, removing the need for projection.

---

## Managing Convergence

To prevent the function from crashing as price vectors approach the boundary (0 or 1), we use the **Barrier Frank-Wolfe** variant. This technique contracts the polytope toward the valid interior point $u$:

$$M_{\epsilon} = (1 - \epsilon)M + \epsilon u$$



As the duality gap $g(\mu_k)$ shrinks, $\epsilon$ decreases, allowing the system to converge to the true optimal arbitrage-free price state.

---

## Information Geometry

Information geometry treats probabilistic distributions as points on a surface. This allows us to use a "ruler" to measure the distance between market prices and arbitrage-free prices.

### Bregman Projections and KL Divergence

We use **Bregman projections** to find the coherent price vectors $\theta$ closest to the market state. The divergence between two distributions $p$ and $q$ is defined by the potential function $\phi$:

$$D_{\phi}(p||q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle$$



By choosing the potential function as the negative entropy $\phi(x) = \sum x_i \ln x_i$, we transition to the **Kullback-Leibler (KL) Divergence**. This measures the informational inconsistency in the market.

---

## Core Equation: Maximizing Profit

The maximum risk-free profit ($\Pi$) available in a market state is determined by the minimum KL divergence:

$$\Pi = \min_{\mu \in \mathcal{M}} D_{KL}(\mu || p(\theta))$$



* 
**$\mu$**: A coherent price vector within the marginal polytope $M$.


* 
**$p(\theta)$**: Current market prices determined by the cost function $\nabla C(\theta)$.



### Implementation Logic

1. 
**Minimization:** The algorithm searches for a distribution $\mu$ that satisfies logical constraints (e.g., sums to 1) while minimizing the distance to actual market prices. Larger distances indicate greater mispricing.


2. 
**Guaranteed Profit:** Real profit is calculated as the current divergence minus the potential error (the gap):



$$\text{Guaranteed Profit} = D_{KL}(\mu_t || p(\theta)) - g(\mu_t)$$





3. 
**Execution:** A trade signal is generated only if the guaranteed profit exceeds transaction costs and liquidity slippage:



$$\text{Guaranteed Profit} > (\text{Transaction Costs} + \text{Liquidity Slippage})$$






