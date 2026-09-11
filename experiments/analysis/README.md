# Analysis scaffold

**Status (2026-09-10):** exploratory. No confirmatory analysis has been frozen.

Analysis code should consume `runs.jsonl` and never import a live planner.
Planned outputs include:

- paired bootstrap tables;
- success-cost-risk Pareto frontiers;
- hypervolume and budget-performance curves;
- calibration and discrepancy plots;
- route and stopping diagnostics;
- naive, IPS, self-normalized IPS, and doubly robust comparisons;
- failure and safety taxonomies.

Each figure/table script must record its input run identifiers and analysis
configuration.

Phase 5A emits exploratory versions of many of these outputs for SQLite in
`montecarlgym.experiments.analysis`. It does not yet implement the candidate's
task-clustered resampling, hypervolume hypothesis test, Holm correction,
non-inferiority test, or policy-ranking OPE. `experiments/analyze_sqlite.py`
reads only raw JSONL and the resolved protocol and never executes a planner, but
it does not yet enforce every stored record/manifest hash before aggregation.
