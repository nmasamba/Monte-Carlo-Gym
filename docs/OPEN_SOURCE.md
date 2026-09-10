# Open-Source and Release Plan

**Status as of 2026-09-10:** pre-alpha planning document. None of the listed
distribution surfaces is asserted to be publicly released. Research positioning
must follow [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

## 1. Product surfaces

MonteCarloGym is intended to be developed in public on GitHub and released through
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

These are maturity gates, not claims implied by the current internal version
number. In particular, local version `0.2.0a2` does not satisfy the research-beta
gate below. Before a public release, either align the version scheme with these
gates or relabel the gates; do not infer readiness from the version alone.

### 0.1 alpha

- public protocols;
- correct UCT kernel;
- transactional Gymnasium examples;
- toy adaptive routing harness;
- documentation and CI.

### 0.2 research beta

- six classical presets;
- a genuinely sequential predictive/executable evidence pair;
- independently verified state and episode-scoped resource accounting;
- full traces, reliability and Pareto reports;
- fixed, query-level, capacity-matched non-tree, and MCTS baselines;
- isolated router and controller ablations.

Phase 5A exercises the local executable-pair and analysis infrastructure with
SQLite, but does not satisfy this gate. It is one-decision, uses normalized
costs, has non-isolated ablations, and supplies neither sequential headroom nor
confirmatory evidence.

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

> A research kernel for selectively acquiring predictive and isolated
> executable evidence during stateful agent planning under measured resource
> and risk constraints.

The differentiator is not “fast Python MCTS” or “many algorithms.” Those are
valuable engineering goals. The proposed research contribution is selective,
branch-local evidence acquisition with independent verification. MCTS is one
controller candidate, not the novelty by itself, and must earn its complexity
against capacity-matched non-tree alternatives.

## 7. Security disclosure

Before exposing executable-environment or agent-service adapters, publish:

- `SECURITY.md` with a private reporting channel;
- supported versions;
- threat model for environment serialization and tool execution;
- model-output and prompt-injection policy;
- credential handling requirements;
- rules for externally visible actions;
- disclosure timelines.

The reference agent adapter must use sandbox or dry-run evaluation and must not
grant the search process production credentials by default.
