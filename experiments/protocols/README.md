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
