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

To add a benchmark, provide task sampling and scoring functions and register
them in the runner. To add a method, implement the public `Planner` protocol and
register its constructor. Production experiment suites should preserve the same
episode-record fields and add compressed per-search traces.
