# MonteCarloGym / FidelityMCTS

MonteCarloGym is a proposed open-source planning library for Gymnasium-compatible
environments. FidelityMCTS is its research layer: a planner that learns not only
which task action to take, but also how to allocate inference and simulation
compute while searching.

The central research question is:

> Can a planner learn when to think, which model to use, when to simulate,
> when to verify, and when to act?

The repository is deliberately split into two layers:

1. **A reusable MCTS kernel** with transactional Gymnasium state handling,
   state/action trees, transpositions, UCT, PUCT, Thompson sampling, RAVE/MAST,
   neural evaluation, and interchangeable backup operators.
2. **An adaptive-compute layer** with model portfolios, multi-fidelity
   simulation, branch-level routing, token/depth budgets, stopping rules,
   verified replay, discrepancy learning, and optional causal correction.

The repository contains:

- a revised architecture and API specification;
- a complete Phase 1 classical UCT vertical slice with transactional
  Gymnasium-style simulation, stochastic outcome links, random rollout, mean
  backup, hard budgets, and subtree reuse;
- Phase 2 compatibility presets for PUCT policy/value search, direct and mixed
  evaluation, Thompson and root sampling, robust/mix backup, RAVE, and MAST;
- a complete Phase 3 multi-fidelity slice with branch-level compute actions,
  conservative resource reservations, fixed routers and stopping policies,
  provenance-aware evidence, verified replay, and online discrepancy estimates;
- a Phase 4 learned-routing slice with persistent verified replay, calibrated
  contextual discrepancy and EVC-proxy models, randomized audit traffic,
  propensity-aware off-policy estimators, and budget-aware MCTS frontiers;
- a real, optional-Gymnasium FrozenLake exploratory pilot plus fingerprinted
  preregistration and confirmatory-run guards;
- a deterministic learned-linear/executable-tree integration benchmark;
- an experiment protocol suitable for a research implementation;
- a runnable, dependency-free toy benchmark for multi-fidelity routing;
- tests and CI;
- a draft LaTeX paper and bibliography.

The toy harness is a diagnostic for the routing abstraction, not evidence for
the paper's empirical claims. The draft paper intentionally labels all
experiments as planned until measurements are run.

## Quick start

```bash
python -m pip install -e .
python experiments/run.py \
  --config experiments/configs/toy.json \
  --output output/toy
python -m unittest discover -s tests -v
PYTHONPATH=src python examples/classical_mcts.py
PYTHONPATH=src python examples/phase2_puct.py
PYTHONPATH=src python examples/phase3_multifidelity.py
PYTHONPATH=src python examples/phase4_learned_routing.py
python experiments/run.py \
  --config experiments/configs/phase3_tree.json \
  --output output/phase3-tree
python -m pip install -e '.[gym]'
python experiments/run_gymnasium.py \
  --stage exploratory \
  --config experiments/pilots/frozenlake_smoke.json \
  --output output/pilots/frozenlake-smoke
```

The command writes per-run JSONL records and an aggregate `summary.json`.

## Documents

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): revised system design.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): hypotheses, baselines,
  benchmarks, metrics, ablations, and statistical protocol.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): release surfaces and governance.
- [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md): what changed from the
  original architecture and what was retained.
- [`docs/RELEASING.md`](docs/RELEASING.md): TestPyPI and production release
  procedure.
- [`paper/main.tex`](paper/main.tex): draft research paper.

## Status

Phases 1–4 are implemented as research infrastructure. One dependency-injected
classical engine provides UCT, transactional native/deep-copy simulation,
explicit state/action/outcome graph statistics and paths, random rollouts, mean
backup, hard iteration/call/cost budgets, `MCTSAgent.compute_action()`, and
basic subtree reuse. The dependency-free fixture example runs without
Gymnasium; Gymnasium remains an optional extra.

Phase 2 adds PUCT with framework-neutral policy/value predictors, direct and
mixed evaluation, Crazy Stone robust/mix backup, Normal-Gamma/Dirichlet
Bayesian statistics, local Thompson selection, a frozen-belief tabular
root-sampled generative model, and independently composable RAVE/MAST sharing.

Phase 3 adds `AdaptiveComputePlanner`, fixed cheap-only, accurate-only, cascade,
and ambiguity-threshold routers, injected stopping and aggregation policies,
full cost/token/model/environment-call accounting, evidence provenance,
verified cheap/accurate replay pairs, and an online discrepancy model. Its
dependency-free shallow-tree benchmark pairs a fitted cheap value model with an
isolated executable rollout model. Both included adaptive benchmarks are
engineering diagnostics, not evidence for the paper's empirical claims.

Phase 4 adds a dependency-free linear EVC proxy, contextual discrepancy
calibration with empirical intervals, append-only verified JSONL replay,
epsilon-greedy audit routes with exact propensities, IPS/self-normalized
IPS/doubly robust estimators, and adaptive frontier evaluation whose nested use
is absorbed by the outer MCTS budget. The FrozenLake harness executes primary
methods and declared ablations on isolated native clones, while keeping pilot,
candidate-protocol, frozen-registration, and confirmatory-output locations
separate.

The EVC target is currently absolute cheap-versus-verified discrepancy: a
transparent utility proxy, not a causal estimate. BrowserGym, L2/L3 executable
benchmarks, query/token-router baselines, learned stopping, distributed
execution, and the complete paper-scale statistical analysis remain planned.
No included pilot is a paper result.

## Installing from PyPI

After the first release is published:

```bash
python -m pip install montecarlgym
python -m pip install 'montecarlgym[gym]'  # optional Gymnasium integration
```

Maintainer publication instructions, including a TestPyPI rehearsal and GitHub
Trusted Publishing, are in [`docs/RELEASING.md`](docs/RELEASING.md).

## License

MIT. Model weights, datasets, benchmarks, and external environments may have
their own licenses and must be checked separately.
