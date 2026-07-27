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

This scaffold contains:

- a revised architecture and API specification;
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
```

The command writes per-run JSONL records and an aggregate `summary.json`.

## Documents

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): revised system design.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): hypotheses, baselines,
  benchmarks, metrics, ablations, and statistical protocol.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): release surfaces and governance.
- [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md): what changed from the
  original architecture and what was retained.
- [`paper/main.tex`](paper/main.tex): draft research paper.

## Status

This is an architecture and experiment-harness scaffold. The classical MCTS
kernel and production integrations described in the documents are the planned
implementation target. The included Python code establishes the new protocols
and makes the controlled multi-fidelity experiment executable.

## License

MIT. Model weights, datasets, benchmarks, and external environments may have
their own licenses and must be checked separately.
