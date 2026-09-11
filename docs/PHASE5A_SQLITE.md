# Phase 5A offline SQLite L2 benchmark

**As of:** 2026-09-10

**Status:** executable L2 mechanism fixture; instrumentation diagnostic only

**Research scope:** see [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)

Phase 5A adds the first locally executable L2 vertical slice. It is an offline
SQL query construction/repair benchmark backed only by Python's standard
`sqlite3` module. It does not use a network, API credentials, a remote model,
BrowserGym, or a host database.

The current measurements are exploratory diagnostics. They are not paper
results and the candidate protocol has not been preregistered. Because the task
has one enumerated decision and the `fidelity_mcts` method invokes the adaptive
planner directly, this fixture does not exercise an outer MCTS tree, sequential
branch allocation, tree reuse, open tool actions, context management, or an
end-to-end agent budget. Here "L2" describes executable database verification,
not submission-level sequential-agent evidence.

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

Run the local smoke matrix - ten comparison methods, six named diagnostic
variants, and five per-planning-call budgets:

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

The current reanalysis command reads the raw files but does not yet reject every
per-record or artifact-manifest hash mismatch before aggregation. Treat the
hashes as generated audit metadata, not as a fully enforced analysis boundary.

Recompute analysis without changing the run directory:

```bash
python experiments/analyze_sqlite.py \
  --input output/pilots/sqlite-l2-smoke-001 \
  --output output/pilots/sqlite-l2-smoke-001-reanalysis.json
```

The analysis includes per-call-budget summaries, paired effects and generic
bootstrap intervals, per-task differences, a fixed-reference Pareto frontier
and hypervolume, RMSE/Brier/NLL/coverage/ECE, a limited
naive/IPS/SNIPS/doubly robust OPE diagnostic, overlap and effective-sample-size
warnings, named-variant comparisons, failures, negative/null findings, and an
exploratory variance/power diagnostic. It does not implement policy-level
router rankings, the candidate's declared task-stratified resampling, a
hypervolume hypothesis test, Holm correction, or its non-inferiority test.

## Baselines and ablations

Every budget executes no-search/direct, cheap-only, accurate-only, fixed
cascade, calibration-matched random escalation, uncertainty threshold,
learned direct, the existing absolute-discrepancy EVC proxy, the directly called
adaptive planner under the historical `fidelity_mcts` identifier, and a
budget-limited ordered counterfactual policy restricted to exploratory fixtures.
The last policy is not guaranteed to be an oracle upper bound.

The smoke also runs variants named no discrepancy correction, no audit traffic,
fixed stopping, no verification, no tree reuse, and cheap-fidelity-only. These
are not six isolated ablations: the no-discrepancy path still uses an updating
running-discrepancy model, and the no-verification and cheap-fidelity-only paths
reduce to the same cheap-only route. Tree reuse is structurally inapplicable to
this one-decision benchmark and is reported as such. None of these comparisons
supports a component claim until the interventions are separated.

## Power analysis and preregistration preparation

The retained mechanism-study candidate is
`experiments/protocols/sqlite_l2_phase5a_candidate.json`. It fixes hypotheses,
outcomes, materialized split definitions, methods, budgets, exclusions,
failure handling, paired pilot seeds, stopping rules, multiplicity policy,
estimands, confidence intervals, Pareto reference, and ablations. It also
states that confirmatory IDs/seeds are unmaterialized.

Do not freeze or promote that file unchanged. Its eight seeds permute only four
underlying exploratory tasks, so the embedded 32-pair power calculation is not
an independent-unit analysis. Pilot effect/interval/power values also lack
versioned raw inputs in this repository. Retain the file for instrumentation
audit, but create a fresh sequential L2b candidate only after the headroom,
environment, target, baseline, reliability, replication, and analysis gates in
`RESEARCH_STATUS.md` pass.

The local candidate is not a preregistration. Phase 5A does not write to
`experiments/confirmatory`, `experiments/preregistered`, or any confirmatory
output directory.
