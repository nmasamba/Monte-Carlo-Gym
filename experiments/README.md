# Experiment harness

Run the controlled interface diagnostic:

```bash
python experiments/run.py \
  --config experiments/configs/toy.json \
  --output output/toy
```

Outputs:

- `resolved_config.json`
- `environment.json`
- `runs.jsonl`
- `summary.json`

The methods receive paired tasks but independent deterministic sampling streams.
The benchmark is intended to catch accounting, routing, and reproducibility
errors before adding full tree environments.

Run the Phase 3 learned-model/executable-model integration diagnostic:

```bash
python experiments/run.py \
  --config experiments/configs/phase3_tree.json \
  --output output/phase3-tree
```

This exercises fixed cheap-only, executable-only, cascade, and threshold
routing under matched hard envelopes. The learned model is a fitted linear
fixture and the executable model performs isolated deterministic rollouts. The
measurements are engineering diagnostics, not paper results.

To add a benchmark, provide task sampling and scoring functions and register
them in the runner. To add a method, implement the public `Planner` protocol and
register its constructor. Production experiment suites should preserve the same
episode-record fields and add compressed per-search traces.

## Phase 4 real Gymnasium pilots

Install the optional dependency and run the one-seed engineering smoke first:

```bash
python -m pip install -e '.[gym]'
python experiments/run_gymnasium.py \
  --stage exploratory \
  --config experiments/pilots/frozenlake_smoke.json \
  --output output/pilots/frozenlake-smoke-001
```

Then use a new output directory for the larger exploratory L1 pilot:

```bash
python experiments/run_gymnasium.py \
  --stage exploratory \
  --config experiments/pilots/frozenlake_v1.json \
  --output output/pilots/frozenlake-v1-001
```

The runner trains the cheap table and router only from declared training and
router-calibration seeds. It emits the resolved protocol, runtime fingerprint,
state-grouped router-calibration diagnostics, model checkpoint, per-episode
records, failures, and append-only verified replay. Every listed ablation is
executed. Output directories must be fresh, so reruns cannot merge silently.

These are exploratory pilots. Use them for debugging, budget selection,
variance estimates, and power analysis only. Do not copy their endpoint values
into a paper's confirmatory results.

## Phase 5A offline executable SQLite L2

The dependency-free L2 smoke runs every required local baseline and prescribed
ablation at five hard budgets:

```bash
python experiments/run_sqlite.py \
  --stage exploratory \
  --config experiments/pilots/sqlite_l2_smoke.json \
  --output output/pilots/sqlite-l2-smoke-001
```

Raw decisions, episodes, and infrastructure failures are written and
fingerprinted before aggregation. Reproduce the analysis into a new file:

```bash
python experiments/analyze_sqlite.py \
  --input output/pilots/sqlite-l2-smoke-001 \
  --output output/pilots/sqlite-l2-smoke-001-reanalysis.json
```

The future-confirmatory SQLite partition is reserved but unmaterialized. The
runner refuses confirmatory and preregistered output paths. See
`docs/PHASE5A_SQLITE.md` for the execution boundary, record schema, analysis,
ablations, power diagnostic, and approval gates.

## Freeze before confirmatory data

The directories have distinct roles:

| Location | May change? | Contains outcomes? |
|---|---:|---:|
| `experiments/pilots/` | Yes | No; exploratory protocols only |
| `experiments/protocols/` | Yes, before registration | No; confirmatory candidates |
| `experiments/preregistered/` | Never in place | No; frozen manifests |
| `output/pilots/` | New run directories only | Exploratory outcomes |
| `output/confirmatory/<study-id>/<run-id>/` | New empty run only | Untouched confirmatory outcomes |

After pilots, write a complete candidate with `"stage": "confirmatory"`,
including hypotheses, endpoint families, benchmark/data/checkpoint versions and
hashes, splits, methods, ablations, budget grid, confirmatory seeds, exclusions,
failure/retry policy, interval method, correction, and fixed stopping rule.
Validate it without freezing:

```bash
python experiments/preregister.py \
  --protocol experiments/protocols/<study-id>.json \
  --validate-only
```

Commit that candidate and all implementation/analysis code, run the full
validation suite, and make the worktree clean. Freeze it exactly once:

```bash
python experiments/preregister.py \
  --protocol experiments/protocols/<study-id>.json \
  --output experiments/preregistered/<study-id>.json
```

Upload the exact frozen JSON and referenced materials to a timestamped public
preregistration service before inspecting confirmatory outcomes. A subsequent
commit may add only the frozen manifest under `experiments/preregistered/`.
For OSF, follow the official registration workflow at
<https://help.osf.io/article/330-welcome-to-registrations>; submitted
registration files cannot be edited in place, and an embargo may be selected
before submission when blinded review requires it.
Then the guarded command is:

```bash
python experiments/run_gymnasium.py \
  --stage confirmatory \
  --preregistration experiments/preregistered/<study-id>.json \
  --output output/confirmatory/<study-id>/<run-id>
```

It refuses a dirty checkout, changed implementation, protocol/hash mismatch,
mutable config, output outside a `confirmatory` directory, or non-empty output.
Programmatic confirmatory callers must independently attest that the frozen
source revision was verified.
