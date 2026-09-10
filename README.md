# MonteCarloGym / FidelityMCTS

MonteCarloGym is a pre-alpha planning library for Gymnasium-compatible
environments. FidelityMCTS is its experimental research layer: a planner that
can allocate inference and simulation compute while choosing a task action.

Following a September 2026 literature and capability review, the central
research question is deliberately narrower:

> On stateful, sequential tool tasks, can selective predictive and executable
> evidence improve verified success versus cost and risk over strong fixed,
> query-level, and non-tree test-time-scaling policies?

The project is continuing with an empirical pivot, not proceeding to the
existing confirmatory candidate. MCTS is now a method to test against non-tree
controllers rather than a presumed part of the winning result. See the
[canonical research status](docs/RESEARCH_STATUS.md) for the fit assessment,
evidence, revised claim, next-study gates, and redirect criteria.

The repository is deliberately split into two layers:

1. **A reusable MCTS kernel** with transactional Gymnasium state handling,
   state/action trees, transpositions, UCT, PUCT, Thompson sampling, RAVE/MAST,
   neural evaluation, and interchangeable backup operators.
2. **Adaptive-compute research infrastructure** with portfolio-based frontier
   valuation, binary cheap/accurate routing, fixed token/depth settings,
   stopping rules, verified replay, discrepancy learning, and exploratory
   off-policy utilities. Multi-model tree transitions and joint routing are
   target architecture, not current features.

The repository contains:

- a revised architecture and API specification;
- a Phase 1 classical UCT vertical slice with transactional
  Gymnasium-style simulation, stochastic outcome links, random rollout, mean
  backup, per-planning-call hard budgets, and subtree reuse;
- Phase 2 compatibility presets for PUCT policy/value search, direct and mixed
  evaluation, Thompson and root sampling, robust/mix backup, RAVE, and MAST;
- Phase 3 multi-fidelity frontier/action valuation with conservative resource
  reservations, fixed routers and stopping policies, provenance-aware evidence,
  verified replay, and online discrepancy estimates; outer-tree transitions
  still come from one simulation model;
- a Phase 4 learned-routing slice with persistent verified replay, calibrated
  contextual discrepancy and discrepancy-as-EVC-proxy models, randomized audit traffic,
  propensity-aware off-policy estimators, and budget-aware MCTS frontiers;
- an optional-Gymnasium FrozenLake exploratory pilot plus fingerprinting and
  confirmatory-run guard mechanics;
- an offline executable SQLite L2 query construction/repair benchmark with
  immutable split fixtures, disposable verification, ten local comparison
  labels, immutable raw records, and exploratory analysis utilities;
- a deterministic learned-linear/executable-tree integration benchmark;
- a revised experiment protocol with explicit validity gates for the next study;
- a runnable, dependency-free toy benchmark for multi-fidelity routing;
- tests and CI;
- a draft LaTeX paper and bibliography.

The toy, FrozenLake, and SQLite harnesses are diagnostics, not evidence for the
paper's empirical claims. The draft paper distinguishes those completed local
runs from the unrun sequential and confirmatory study.

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
python experiments/run_sqlite.py \
  --stage exploratory \
  --config experiments/pilots/sqlite_l2_smoke.json \
  --output output/pilots/sqlite-l2-smoke
```

The command writes per-run JSONL records and an aggregate `summary.json`.

## Documents

- [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md): canonical implementation
  status, September 2026 fit assessment, revised claim, and next-study gates.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): revised system design.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): hypotheses, baselines,
  benchmarks, metrics, ablations, and statistical protocol.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): release surfaces and governance.
- [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md): what changed from the
  original architecture and what was retained.
- [`docs/RELEASING.md`](docs/RELEASING.md): TestPyPI and production release
  procedure.
- [`docs/PHASE5A_SQLITE.md`](docs/PHASE5A_SQLITE.md): offline L2 benchmark,
  raw artifacts, analysis, power diagnostics, and preregistration boundary.
- [`paper/main.tex`](paper/main.tex): draft research paper.
- [`output/pdf/FidelityMCTS_draft.pdf`](output/pdf/FidelityMCTS_draft.pdf):
  rebuilt September 2026 proposal PDF.

## Status

The source version is `0.2.0a2`. Phases 1-2 are implemented and tested as a
classical MCTS kernel. Phases 3-4 are implemented as adaptive-compute research
infrastructure. Phase 5A is an executable, deterministic SQLite mechanism pilot.

There is no external stateful agent benchmark adapter, counterfactual
decision-utility learner, frozen preregistration, confirmatory run, or paper
result. In particular, the current one-decision SQLite fixture does not test the
revised sequential-agent claim. Detailed status is maintained only in
[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) to prevent roadmap drift.

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
