# Exploratory pilots

This directory contains mutable pilot protocols. Pilot runs may be used to find
bugs, estimate variance, select budgets, and perform power analysis. Their
outputs belong under `output/pilots/` and are never paper results.

After pilot decisions are complete, create a confirmatory candidate under
`experiments/protocols/`, replace all pilot-derived choices, lock
checkpoint/data hashes, commit the clean source revision, and freeze it with
`experiments/preregister.py`. An exploratory protocol cannot be frozen.

`sqlite_l2_smoke.json` is the Phase 5A offline executable smoke. It covers the
complete baseline/ablation matrix at five budgets on a small exploratory subset
and remains an engineering diagnostic rather than a paper result.
