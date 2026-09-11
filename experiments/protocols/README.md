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
confirmatory protocol ready to freeze. The September 2026 review retains it as
an auditable mechanism-study record: its validity notes now document the
one-decision scope, non-independent repetitions, non-isolated ablations, and
invalidated power calculation. It is not the revised sequential-study
candidate. Do not freeze or promote it unchanged.

Create a fresh candidate only after the gates in
[`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md) are satisfied,
including demonstrated oracle headroom, an episode-scoped budget, independently
verified terminal state, capacity-matched non-tree baselines, and a clustered
analysis plan.
