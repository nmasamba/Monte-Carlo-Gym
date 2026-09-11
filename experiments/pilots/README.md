# Exploratory pilots

This directory contains mutable pilot protocols. Pilot runs may be used to find
bugs, estimate variance, select budgets, and perform power analysis. Their
outputs belong under `output/pilots/` and are never paper results.

After pilot decisions are complete, create a confirmatory candidate under
`experiments/protocols/`, replace all pilot-derived choices, lock
checkpoint/data hashes, commit the clean source revision, and freeze it with
`experiments/preregister.py`. An exploratory protocol cannot be frozen.

`sqlite_l2_smoke.json` is the Phase 5A offline executable smoke. It covers the
local ten-method/six-variant matrix at five per-planning-call budgets on a small
exploratory subset. It is a one-decision instrumentation diagnostic, not the
complete revised baseline/ablation set and not a paper result.
