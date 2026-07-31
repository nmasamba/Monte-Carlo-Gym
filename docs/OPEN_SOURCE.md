# Open-Source and Release Plan

## 1. Product surfaces

MonteCarloGym should be developed in public on GitHub and released through
several complementary channels:

| Surface | Artifact | Audience |
|---|---|---|
| PyPI | `montecarlgym` Python package | Gymnasium and ML researchers |
| GitHub | source, issues, roadmap, examples | contributors and reviewers |
| OCI registry | planner service image | enterprise and agent deployments |
| Hugging Face | router/discrepancy checkpoints, datasets, model cards | research users |
| npm / agent registry | thin MCP or OpenClaw-style adapter | TypeScript agent users |
| Zenodo | versioned research artifact and DOI | paper reproducibility |

“PyTorch repository” is not a normal hosting destination for third-party
packages. PyTorch should be an optional backend integration. The core package
must remain importable without PyTorch, Transformers, a browser, or a remote
model account.

## 2. Package boundaries

Recommended optional extras:

```text
montecarlgym                  # standard-library core
montecarlgym[gym]             # Gymnasium wrapper
montecarlgym[torch]           # neural evaluators and batching
montecarlgym[transformers]    # local foundation-model adapters
montecarlgym[browser]         # BrowserGym/WorkArena adapters
montecarlgym[service]         # HTTP/gRPC/MCP planner service
montecarlgym[research]        # analysis and benchmark dependencies
montecarlgym[all]             # convenience, not used in minimal CI
```

External model weights and environments retain their own licenses. Extras must
not imply redistribution rights.

## 3. Repository governance

- MIT for the initial scaffold; reconsider Apache-2.0 before the first public
  release if an explicit patent grant is important to the contributor base.
- Developer Certificate of Origin initially; add a CLA only if a foundation or
  company requires it.
- Public architectural decision records for API-breaking choices.
- Two maintainer reviews for security-sensitive adapters.
- Semantic versioning and deprecation windows.
- `main` protected by tests, lint, type checks, and reproducibility checks.
- A lightweight technical steering group after multiple independent maintainers
  are active.

## 4. Contribution tracks

Contributors should be able to work independently on:

- environment snapshot strategies;
- tree policies and backups;
- Bayesian posterior components;
- neural evaluator adapters;
- simulator/model connectors;
- benchmarks and verifiers;
- routing and stopping algorithms;
- off-policy estimators;
- visualization and trace tooling;
- documentation and reproducibility.

Every plugin-like component should have a protocol conformance test and at least
one deterministic fixture.

## 5. Release gates

### 0.1 alpha

- public protocols;
- correct UCT kernel;
- transactional Gymnasium examples;
- toy adaptive routing harness;
- documentation and CI.

### 0.2 research beta

- six classical presets;
- first learned-model/executable-simulator pair;
- full traces and Pareto reports;
- router baselines and ablations.

Phase 5A satisfies the first local executable-pair and analysis-infrastructure
portion of this gate with SQLite. It does not satisfy independent benchmark
reproduction, sequential L2/L3 breadth, or confirmatory evidence.

### 0.3 service beta

- containerized planner service;
- authentication, quotas, timeouts, and redaction;
- MCP/agent adapter;
- multi-worker simulator execution.

### 1.0

- stable public API;
- independent benchmark reproduction;
- security review;
- migration guide;
- governance and long-term maintenance plan.

## 6. Community positioning

The project should be described as:

> A simulator-portfolio planning system that makes MCTS compatible with
> Gymnasium, learned world models, executable agent environments, and explicit
> inference budgets.

The differentiator is not “fast Python MCTS” or “many algorithms.” Those are
valuable engineering goals. The research identity is the joint, branch-level
optimization of task decisions and compute decisions under fidelity, token,
latency, cost, and risk constraints.

## 7. Security disclosure

Publish:

- `SECURITY.md` with a private reporting channel;
- supported versions;
- threat model for environment serialization and tool execution;
- model-output and prompt-injection policy;
- credential handling requirements;
- rules for externally visible actions;
- disclosure timelines.

The reference agent adapter must use sandbox or dry-run evaluation and must not
grant the search process production credentials by default.
