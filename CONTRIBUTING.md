# Contributing

MonteCarloGym is pre-alpha. Discuss substantial API changes in an issue or
architectural decision record before implementation.

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
- avoid mutating a live environment during search;
- document external licenses and access requirements;
- include an ablation path when they are part of a research claim.

Do not commit secrets, raw chain-of-thought, proprietary task data, or
personally identifiable replay records.
