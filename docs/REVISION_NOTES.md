# Revision Notes: Original Plan to FidelityMCTS

The original MonteCarloGym plan remains the classical foundation. Revision 2
changes the research center of gravity.

## Preserved

- transactional Gymnasium state snapshot/restore;
- state nodes, action edges, stochastic outcome links, and explicit search
  paths;
- transpositions and subtree reuse;
- pluggable tree, rollout, evaluator, and backup policies;
- UCT, Bayesian posterior sampling, Crazy Stone, AlphaGo/APV, AlphaGo Zero,
  RAVE, and MAST;
- strict handling of termination, truncation, RNG state, and value perspective.

## Changed

| Original emphasis | Revision 2 |
|---|---|
| broad MCTS algorithm library | classical kernel plus adaptive-compute research layer |
| one environment/simulator per search | portfolio of cheap, intermediate, and executable models |
| choose a task action | jointly choose task and compute actions |
| fixed rollout/evaluation budget | branch-level model, token, depth, verifier, and stop choices |
| simulator values enter backup directly | provenance-aware evidence and discrepancy aggregation |
| tree data for reuse | paired verified replay for calibration and self-improvement |
| ordinary experiment examples | preregisterable, matched-budget, Pareto experiment harness |
| predictive logging | propensity logging and optional causal/off-policy evaluation |
| embedded Python agent | Python core plus deployable planner service and thin agent adapters |

## Research claim

The revised claim is not that MonteCarloGym contains more algorithms or is the
fastest Python implementation. Those may become useful engineering properties.
The falsifiable paper claim is:

> Under equal resource and risk budgets, branch-level joint allocation of
> simulator fidelity, model tier, token budget, rollout depth, verification,
> and stopping improves the success-cost-risk Pareto frontier over single-model
> search, query-level routing, and fixed cascades.

## Scope guardrails

- Predictive MCTS may replace or augment policy optimization in an RLHF-style
  loop, but not preference elicitation, reward validation, or safety review.
- Causal correction is required where logged router selection causes bias; a
  universal causal world model is not required.
- Learned simulation is cheap evidence, not verified truth.
- Executable simulation occurs in clones or sandboxes; speculative search does
  not act on production systems.
- The included toy harness validates interfaces and accounting only. It is not
  evidence for the paper's empirical hypothesis.
- The Phase 3 shallow-tree harness integrates learned and executable evidence,
  but remains a controlled engineering diagnostic rather than a paper result.
- Phase 4 supplies persistent verified replay, contextual calibration, learned
  EVC-proxy routing, randomized audit traffic, off-policy estimators, adaptive
  MCTS-frontier evaluation, and guarded preregistration mechanics. Its real
  FrozenLake integration is still an exploratory L1 pilot, not paper evidence.
