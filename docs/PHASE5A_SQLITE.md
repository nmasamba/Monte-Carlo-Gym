# Phase 5A offline SQLite L2 benchmark

Phase 5A adds the first locally executable L2 vertical slice. It is an offline
SQL query construction/repair benchmark backed only by Python's standard
`sqlite3` module. It does not use a network, API credentials, a remote model,
BrowserGym, or a host database.

The current measurements are exploratory diagnostics. They are not paper
results and the candidate protocol has not been preregistered.

## Safety and verification boundary

Each task contains an immutable schema/data setup, a natural-language request,
candidate SQL repairs, and an objective expected result. The executable model:

- copies an immutable in-memory SQLite template into a fresh disposable
  connection for every query;
- enables `query_only`, rejects mutation/DDL/attach/pragma operations with an
  authorizer, and closes the clone in a `finally` path;
- interrupts excessive VM work or wall-clock time with a progress handler;
- preserves completed, terminated, and truncated/timeout outcomes;
- returns `-1`, never an implicit zero, for a verified query failure;
- charges the full conservative executable quote for SQL errors and timeouts.

The selected candidate ID is a task action. Cheap prediction and disposable
execution requests are `ComputeAction` values. The planner never submits the
selected query to a live or external database.

## Partitions

`development_training`, `calibration`, and `exploratory_pilot` contain
immutable fixtures with disjoint task-template families. The
`future_confirmatory` partition is only a reservation: its task IDs and seeds
are `null`, no fixture loader can access it, and Phase 5A's runner refuses a
confirmatory stage.

Candidate ordering is deterministically permuted from the paired pilot seed.
Every method receives the identical task object, task ID, order, and paired
seed. Router training reads development fixtures; interval calibration and
random-baseline quota calibration read calibration fixtures. Neither reads
exploratory outcomes before planning.

## Quick example

No optional dependency is needed:

```bash
PYTHONPATH=src python examples/phase5a_sqlite.py
```

Run the complete smoke matrix—ten baselines, six ablations, and five budgets:

```bash
python experiments/run_sqlite.py \
  --stage exploratory \
  --config experiments/pilots/sqlite_l2_smoke.json \
  --output output/pilots/sqlite-l2-smoke-001
```

Use a new output directory for every run. Exploratory output paths containing
`confirmatory` or `preregistered` are rejected.

## Raw records and reproduction

The runner writes raw records before invoking aggregation:

```text
<run>/
  resolved_protocol.json
  partitions.json
  environment.json
  raw/
    decisions.jsonl
    episodes.jsonl
    failures.jsonl
  artifact_manifest.json
  analysis.json
  per_task_differences.jsonl
  summary.json
```

Each raw line has a content SHA-256, and each raw file is created exclusively,
fsynced, and made read-only. `artifact_manifest.json` records file hashes. A
decision records task/compute action kind, route propensity, calibration
features, model fidelity, provenance, verifier status, costs, calls, tokens,
latency, failures, and terminal flags. Episodes record success, return,
normalized cost, calls by fidelity, execution failures, stopping reason, route
propensities, and provenance.

Recompute analysis without changing the run directory:

```bash
python experiments/analyze_sqlite.py \
  --input output/pilots/sqlite-l2-smoke-001 \
  --output output/pilots/sqlite-l2-smoke-001-reanalysis.json
```

The analysis includes matched-budget summaries, paired effects and bootstrap
intervals, per-task differences, a fixed-reference Pareto frontier and
hypervolume, RMSE/Brier/NLL/coverage/ECE, naive/IPS/SNIPS/doubly robust OPE,
overlap and effective-sample-size warnings, comparison with randomized online
pilot outcomes, ablation comparisons, failures, negative/null findings, and an
explicitly exploratory variance/power diagnostic.

## Baselines and ablations

Every budget executes no-search/direct, cheap-only, accurate-only, fixed
cascade, calibration-matched random escalation, uncertainty threshold,
learned direct, the existing absolute-discrepancy EVC proxy, FidelityMCTS
adaptive frontier planning, and an executable counterfactual oracle restricted
to exploratory fixtures.

The smoke also runs no discrepancy correction, no audit traffic, fixed
stopping, no verification, no tree reuse, and cheap-fidelity-only ablations.
Tree reuse is structurally inapplicable to this one-decision benchmark and is
reported as such; equality for that ablation is an expected negative finding,
not evidence of no tree-reuse effect in sequential tasks.

## Power analysis and preregistration preparation

The candidate is
`experiments/protocols/sqlite_l2_phase5a_candidate.json`. It fixes hypotheses,
outcomes, materialized split definitions, methods, budgets, exclusions,
failure handling, paired pilot seeds, stopping rules, multiplicity policy,
estimands, confidence intervals, Pareto reference, and ablations. It also
states that confirmatory IDs/seeds are unmaterialized.

Do not freeze that file yet. First review the exploratory variance and failure
profile, decide whether this fixture breadth and budget grid are sufficient,
add the remaining benchmark integrations, refresh artifact hashes, choose the
confirmatory sample size, and obtain user approval. Only then materialize and
freeze confirmatory IDs/seeds and externally timestamp the final protocol.

The local candidate is not a preregistration. Phase 5A does not write to
`experiments/confirmatory`, `experiments/preregistered`, or any confirmatory
output directory.
