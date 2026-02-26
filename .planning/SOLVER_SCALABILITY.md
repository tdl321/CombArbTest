# Solver Scalability: Vertex Enumeration vs MILP

## The Problem

The Rust solver uses **combinatorial vertex enumeration** — enumerate all 2^N joint outcomes, filter by constraints, then the LMO is a simple linear scan. This gives 3,500x+ speedup over Python for small problems.

**But it only works for small clusters (2-10 markets).** For 20+ markets, 2^N becomes infeasible.

## Current State

| Approach | Speed | Scales to |
|----------|-------|-----------|
| Vertex enumeration (Rust) | ~0.03ms per solve | ~15 markets max (2^15 = 32K vertices) |
| HiGHS MILP (Python) | ~1ms per solve | Hundreds of markets |
| HiGHS MILP (Rust, tried) | ~8ms per solve | Hundreds, but slower than Python due to build flags |

## Two Possible Use Cases

### Case A: Many small clusters in parallel
- Cluster 408K markets into thousands of small groups (2-10 markets each)
- Solve each cluster independently
- Vertex enumeration is **ideal** here — rayon parallelism across clusters
- This is what the current LLM clustering pipeline produces

### Case B: One large joint optimization
- Find arbitrage across 100+ markets simultaneously
- The joint distribution space is 2^100 — cannot enumerate
- Need MILP, column generation, or constraint generation
- This catches complex multi-hop arbitrage that small clusters miss

## Research Questions

1. **Is Case B actually needed?** The IMDEA 2025 study found 99.76% of Polymarket arbitrage profit comes from simple partition checks. Complex multi-hop arbitrage across many markets yielded only $95K vs $39.5M.

2. **Can smarter clustering reduce the need for large solvers?** If the LLM groups related markets well, most arbitrage is captured within small clusters. The question is: how much are we leaving on the table by not solving jointly?

3. **If we need large-scale solving, what algorithm?**
   - **Column generation**: Start with a small set of vertices, add promising ones via MILP pricing problem. Best of both worlds — works like vertex enumeration but discovers vertices on-demand.
   - **Constraint generation (cutting planes)**: Dual of column generation. Start with LP relaxation, add violated constraints.
   - **ADMM decomposition**: Split large problem into overlapping subproblems, solve in parallel, enforce consistency.
   - **Belief propagation**: Message-passing on the market dependency graph. Approximate but very fast for sparse graphs.

4. **Hybrid approach**: Use vertex enumeration for clusters ≤ 15 markets, fall back to MILP/column generation for larger clusters. The Rust solver already has the constraint builder — just needs a MILP backend for the fallback path.

5. **Does the problem structure help?** Prediction market constraints are sparse — most markets are independent, with only local dependencies. This sparsity could be exploited by decomposition methods that avoid building the full joint distribution.

## Recommended Next Steps

1. Profile the actual cluster sizes from LLM grouping — if 99% are ≤ 10 markets, vertex enumeration covers production use
2. Quantify profit left on the table by not solving large clusters jointly
3. If large-scale needed: implement column generation in Rust (adds vertices on-demand via MILP subproblem, but the master problem stays as a linear scan over discovered vertices)
4. Benchmark HiGHS via system-installed library (not compiled from source) to see if Rust MILP can match Python speed

## References

- METHODOLOGY_REVIEW.md — Section on LP relaxation and column generation
- RUST_SOLVER_MIGRATION.md — Original plan assumed persistent HiGHS model
- MATH_AUDIT.md — Constraint formulations that any solver must support
