# Contributing

MonteCarloGym is pre-alpha. Discuss substantial API changes in an issue or
architectural decision record before implementation.

Before changing a research claim, benchmark, baseline, budget definition, or
method label, read [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) and
update every owning document listed there in the same change. The canonical
status document takes precedence over older roadmap prose.

## Development

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
```

New components should:

- implement a documented protocol rather than add algorithm-name branches;
- include deterministic unit tests;
- preserve exact cost, token, latency, and model-version metadata;
- distinguish planning-call limits from hard episode-level limits;
- pin environment, harness, context policy, action interface, grader, and
  verifier versions in experiment manifests;
- avoid mutating a live environment during search;
- document external licenses and access requirements;
- include an ablation path when they are part of a research claim.
- avoid calling a method MCTS when it bypasses the outer tree, or an oracle when
  budget/order constraints prevent it from being an upper bound.

Do not commit secrets, raw chain-of-thought, proprietary task data, or
personally identifiable replay records.
