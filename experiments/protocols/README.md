# Confirmatory protocol candidates

After exploratory pilots are complete, place the proposed complete protocol in
this directory with `"stage": "confirmatory"`. Candidate protocols are mutable
until preregistration, so they are not evidence that any analysis was fixed in
advance.

Validate a candidate without freezing it:

```bash
python experiments/preregister.py \
  --protocol experiments/protocols/<study-id>.json \
  --validate-only
```

Commit the candidate and its implementation, finish every validation, and make
the worktree clean. Then freeze it into `experiments/preregistered/`. The frozen
manifest records the candidate's clean source revision and SHA-256 fingerprint.
Only manifest-only commits under `experiments/preregistered/` may follow that
revision before the confirmatory runner refuses execution.

`sqlite_l2_phase5a_candidate.json` is intentionally still exploratory and has
an empty `confirmatory_seeds` list backed by an explicit
`reserved_unmaterialized` seed policy. It is a preparation artifact, not a
confirmatory protocol ready to freeze. Do not change its stage or materialize
the reservation until the user approves the final benchmark, methods, budgets,
analysis, power decision, and refreshed artifact hashes.
