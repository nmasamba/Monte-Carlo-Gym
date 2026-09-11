# Experiment suites

Place immutable suite manifests here. A suite selects benchmark adapters,
task-family splits, methods, budgets, seeds, and analysis endpoints without
embedding credentials or provider-specific secrets.

Candidate manifest sequence after the research-review gates are met:

- `gymnasium_classical.json`
- `sequential_tool_headroom.json`
- `agent_world_model_or_tau_pinned.json`
- `tree_vs_non_tree.json`
- `browser_desktop_replication.json`
- `causal_router_audit.json`

Every suite must pin the harness, context policy, action interface, environment,
grader, verifier, model, and reasoning configuration. It must also distinguish
per-decision limits from the hard episode-level cost, latency, token, and risk
envelope. Final test manifests must be frozen before confirmatory runs; the
current SQLite fixture is not eligible for that sequence. See
[`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md).
