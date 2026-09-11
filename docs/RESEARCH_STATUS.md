# Research status and direction

**As of:** 2026-09-10

**Decision:** continue only with a narrowed empirical pivot

**Authority:** this file is the source of truth for implementation status,
research readiness, and the current claim. Architecture and experiment documents
describe the target design; the paper is a proposal until this file says that
confirmatory evidence exists.

## Executive assessment

The project remains useful, but the current broad paper thesis is not ready for
confirmatory study. Recent work strengthens the motivating premise - inference
compute, planning, tool use, verification, and model choice should be allocated
adaptively - while substantially reducing the novelty of claiming that premise
alone. Dynamic planning, per-step agentic test-time scaling, multi-model
orchestration, and explicit value of computation in MCTS all now have direct
prior art.

The repository should therefore retain MonteCarloGym as tested planning
infrastructure and treat FidelityMCTS as one candidate controller, not as the
presumed winner. The next research phase must first establish that selective
access to a predictive model and an executable environment has useful
counterfactual headroom on stateful, sequential tasks. It must then compare the
tree controller with strong non-tree agent and test-time-scaling baselines.

This is a re-think of the empirical claim and benchmark, not a rewrite of the
kernel.

## Current state

| Area | Status | What the status means |
|---|---|---|
| Package | implemented, pre-alpha (`0.2.0a2`) | Local source and tests exist; no release is asserted here. |
| Classical kernel (Phases 1-2) | implemented and tested | Transactional simulation, UCT/PUCT, Bayesian presets, backups, RAVE/MAST, and reuse fixtures exist. |
| Multi-fidelity controller (Phase 3) | implemented as frontier-valuation infrastructure | Hard resource accounting, fixed routing, provenance, paired replay, and a shallow-tree diagnostic exist. Outer MCTS transitions still come from one simulation model; portfolio `next_state` observations are not added to the search graph. |
| Learned routing (Phase 4) | implemented as infrastructure | A linear discrepancy/EVC proxy, calibration, audit traffic, OPE estimators, frontier integration, and an exploratory FrozenLake runner exist. |
| SQLite study (Phase 5A) | executable mechanism pilot | It is a deterministic, one-decision query-repair fixture with exploratory analysis. It is not evidence for sequential branch allocation or agent performance. |
| External agent environments | not implemented | There is no current adapter or result for Agent World Model, tau-bench, BrowserGym, WorkArena, OSWorld, or a native CLI-agent benchmark. |
| Learned utility target | not implemented | The current label is absolute cheap-versus-verified discrepancy, not causal or counterfactual improvement in the final decision. |
| Joint action space | not implemented | The learned router makes a binary cheap/accurate escalation with configured token/depth values; it does not jointly learn model, token, depth, verifier, and stop choices. |
| Risk constraint | not implemented as a hard budget | Risk is recorded and can enter utility, but `SearchBudget` has no risk ceiling. |
| Budget scope | inconsistent across harnesses | `SearchBudget` caps one planning call. The FrozenLake protocol reuses that allowance at each environment step, so its summed episode cost is not a hard episode-level budget. |
| Confirmatory analysis | not implemented | SQLite analysis is exploratory: it does not implement the declared stratified resampling, hypervolume hypothesis interval/test, Holm family, non-inferiority test, or router-ranking OPE claim. |
| Confirmatory evidence | absent | Nothing is preregistered or externally timestamped; no included pilot is a paper result. |
| Draft paper | proposal only | It describes a method and revised protocol, not measured findings. |

## Why the direction still matters

Recent evidence supports four premises behind the project:

1. Agent performance can improve with more inference-time work, but uniform
   scaling is inefficient. Studies of general agents, web agents, coding agents,
   and multi-agent tool mixtures report gains from reflection, diverse rollouts,
   aggregation, or selective per-step allocation.
2. Predictive language world models and large collections of executable agent
   environments now exist. This makes the proposed cheap-prediction versus
   expensive-execution asymmetry more testable than it was when the architecture
   was first drafted.
3. Long-horizon agents remain unreliable. Repeated-run consistency, recovery,
   context management, and state-based verification are therefore first-class
   outcomes, not optional diagnostics.
4. Compute allocation is increasingly an agent-level control problem spanning
   model choice, reasoning effort, tools, parallel rollouts, verification, and
   stopping. Hard accounting and auditable provenance remain valuable system
   contributions.

## Why the broad claim must change

The original claim jointly covered branch choice, model tier, simulator
fidelity, token budget, rollout depth, verification, stopping, self-learning,
and causal correction. That scope now has several problems:

- **Novelty is crowded.** Value-of-computation MCTS predates this project, and
  recent agent systems directly study when to plan, how to allocate per-step
  samples, how to select models and reasoning budgets, and how to aggregate
  parallel long-horizon trajectories.
- **The current L2 label overstates the SQLite fixture.** A one-decision,
  enumerated query-repair task cannot test long-horizon state tracking, branch
  reuse, open-ended action proposal, context pressure, recovery, or user/tool
  interaction. Its method identifier `fidelity_mcts` invokes the adaptive
  planner directly rather than running a multi-step MCTS tree. It is an L2
  mechanism fixture, not submission-level L2 evidence.
- **The learner optimizes the wrong proxy.** Absolute prediction discrepancy
  can prioritize a large error that would not change the selected action, while
  missing a small error near a decision boundary. The primary target must be
  marginal decision utility or regret reduction after acquiring evidence.
- **MCTS is not yet justified for language agents.** Large or open action spaces,
  partial observability, changing external state, expensive cloning, and long
  contexts can make a literal state-action tree a poor abstraction. Hierarchical
  actions, progressive widening, trajectory summaries, or a non-tree controller
  may be better.
- **The implemented composition is narrower than multi-model tree search.** The
  model portfolio adaptively values actions at an already-created MCTS frontier;
  it does not choose which model generates each tree transition. A sequential
  study must either integrate model-indexed transitions explicitly or retain the
  narrower evidence-acquisition claim.
- **Several analysis labels exceed their estimands.** The SQLite "oracle" is a
  budget-limited ordered exhaustive policy, not an unconstrained oracle upper
  bound. Its OPE diagnostic covers selected randomized verification decisions,
  not complete router policies or rankings. Neither result can support the
  corresponding broad hypothesis without redesign.
- **Matched budget is not yet one invariant.** The core budget is per planner
  call, while the sequential FrozenLake harness makes several planner calls and
  accumulates their use. A future protocol must define and enforce both
  per-decision and end-to-end episode envelopes before making matched-budget
  claims.

## Revised research question

Primary question:

> On stateful, sequential, executable tool tasks with heterogeneous evidence
> costs, can a controller that selectively acquires predictive and executable
> evidence improve repeated-run verified success versus cost and risk over
> fixed, query-level, and non-tree test-time-scaling policies?

MCTS-specific secondary question:

> Conditional on measurable oracle headroom, does tree-structured branch
> allocation add value beyond a capacity- and budget-matched non-tree
> controller?

The working name `FidelityMCTS` may remain, but a paper must not make an
MCTS-specific claim unless the secondary question is supported. If a non-tree
controller wins, the honest result is an adaptive evidence-acquisition study
with MonteCarloGym as infrastructure.

## Known validity blockers in the current pilots

These are documentation and study-design blockers, not confirmatory findings:

- SQLite seeds change candidate order but not the underlying four exploratory
  tasks. Treating 4 tasks by 8 permutations as 32 independent task-seed units
  is pseudoreplication; inference must cluster at task or template-family level.
- The SQLite learner is fitted once at the largest configured budget. Its
  remaining-budget features therefore do not train a budget-conditional policy,
  and the current frontier path does not supply varying search depth.
- Router feasibility is not yet a true action mask. The planner exposes all
  portfolio IDs, selects a route, and only then attempts to reserve its quote;
  it can stop on an unaffordable route even when a cheaper route remains.
- The current evidence aggregator treats the first configured accurate-fidelity
  model as the verified source. A portfolio with multiple accurate sources needs
  an explicit identity, trust, and aggregation rule before use.
- The repository implements replay and training components, but no evaluated
  multi-round self-learning and promotion experiment. The self-learning research
  question remains untouched.
- Pilot-derived effect, interval, and power numbers are embedded in the mutable
  candidate, but their raw pilot inputs are not versioned in the repository.
  They are planning notes, not reproducible evidence.
- The reanalysis path reads raw JSONL but does not yet enforce every per-record
  content hash or the artifact manifest before aggregation. Runtime metadata is
  also incomplete relative to the documented reproducibility contract: replay
  does not require every resource, model, prompt, environment, harness, grader,
  and verifier version. The retained candidate's artifact map is explicitly
  partial and omits the raw pilot lineage.
- The two SQLite ablations named `fidelity_no_verification` and
  `fidelity_cheap_fidelity_only` currently reduce to the same cheap-only route;
  `fidelity_no_discrepancy` still uses an updating running-discrepancy model.
  These labels do not identify three isolated interventions.

## Required next study

Complete these gates in order:

1. **Headroom gate.** On a materialized development split, compute a
   counterfactual oracle that can see every feasible cheap and executable
   outcome. Stop if selective evidence cannot improve the direct or fixed
   policies enough to justify its overhead.
2. **Sequential environment gate.** Add one stateful, executable tool benchmark
   with at least two held-out task families and multi-step state transitions.
   Agent World Model or the current tau benchmark family are suitable primary
   candidates; ToolSandbox is a smaller fallback. Keep browser/desktop work as a
   replication until the mechanism works in a deterministic tool environment.
3. **Action/state gate.** Define observable information-state identity,
   candidate or macro-action generation, branching control, legal-action
   validation, context policy, and safe clone/reset semantics. Do not assume an
   enumerable Gym action space.
4. **Target gate.** Train the router on estimated decision improvement, action
   switches, regret reduction, or another preregistered utility delta. Keep
   absolute discrepancy as a baseline feature, not the EVC label.
5. **Baseline gate.** Compare harness-native direct/ReAct, always-plan and
   never-plan, fixed cascade, uncertainty threshold, query-level routing,
   parallel best-of-N plus verifier/aggregator, a dynamic non-tree controller,
   and FidelityMCTS. Match total model, token, tool, environment, latency, and
   monetary budgets.
6. **Analysis gate.** Implement the actual estimands before outcome access:
   cluster/stratified paired resampling at the declared unit, budget-curve and
   hypervolume comparisons, multiplicity handling, non-inferiority tests where
   claimed, and policy-level randomized/OPE comparisons.
7. **Reliability gate.** Measure repeated-run `pass^k`, state-based verified
   success, false-success rate, recovery after injected tool faults, policy
   compliance, and tail cost/latency. Hold the context-management policy fixed
   across the primary comparison and log compaction/retrieval costs.
8. **Replication gate.** Replicate on a second environment family or a
   browser/desktop benchmark before making a general agent claim.
9. **Freeze gate.** Only after the preceding gates pass should a fresh
   confirmatory protocol be frozen and externally registered. The Phase 5A
   SQLite candidate must not be promoted into that protocol unchanged.

## Stop or redirect criteria

Redirect the research layer, while retaining the reusable kernel, if any of the
following survives reasonable pilot iteration:

- the counterfactual oracle has negligible verified-success/cost headroom;
- a fixed cascade or simple uncertainty rule matches the learned controller;
- a non-tree controller matches FidelityMCTS without its search overhead;
- predictive-model error is not calibrated enough to route safely on held-out
  task families;
- executable verification is too slow, noisy, or incomplete to provide a
  defensible reference outcome;
- the claimed gain disappears under repeated trials, harness changes, fault
  injection, or full cost accounting.

Negative outcomes are valid. They would support a narrower package centered on
the classical kernel, resource accounting, and reproducible agent evaluation.

## Evidence reviewed

The assessment uses primary papers and official project sources available by
2026-09-10. Peer-reviewed status varies; arXiv-only findings are evidence for
study design, not settled fact.

| Evidence | Implication for this project |
|---|---|
| [Static and Dynamic Values of Computation in MCTS (UAI 2020)](https://proceedings.mlr.press/v124/sezener20a.html) | Explicit EVC for MCTS is prior art; novelty must come from heterogeneous agent evidence and evaluation, not EVC alone. |
| [Learning When to Plan (2025)](https://arxiv.org/abs/2509.03581) | Dynamic planning in sequential agents is a direct baseline and narrows the novelty claim. |
| [Scaling Test-time Compute for LLM Agents (2025)](https://arxiv.org/abs/2506.12928) | Reflection, diverse rollouts, verification, and aggregation belong in the baseline set. |
| [Agentic Test-Time Scaling for WebAgents (2026)](https://arxiv.org/abs/2602.12276) | Per-step uncertainty-based allocation is especially close prior work and must be compared directly. |
| [ATLAS (2026)](https://arxiv.org/abs/2606.01667) | Stateful orchestration over solver, effort, evidence, and stopping further crowds the broad controller claim. |
| [Scaling Test-Time Compute for Agentic Coding (2026)](https://arxiv.org/abs/2604.16529) | Long-horizon scaling depends on trajectory representation, selection, and reuse; raw tree statistics are not enough. |
| [Qwen-AgentWorld (2026)](https://arxiv.org/abs/2606.24597) | A general language world model makes the cheap predictive tier plausible. |
| [Agent World Model (2026)](https://arxiv.org/abs/2602.10090) | Code- and database-backed executable environments make controlled stateful verification practical. |
| [tau2-bench (2025)](https://arxiv.org/abs/2506.07982) and [current tau benchmark repository](https://github.com/sierra-research/tau2-bench) | Dual-control, stateful tasks and state-based grading are better mechanism tests than one-shot SQL repair; use the maintained version, not a stale benchmark snapshot. |
| [Toolathlon (2025)](https://arxiv.org/abs/2510.25726) and [MCPMark (2025)](https://arxiv.org/abs/2509.24002) | Modern tool tasks expose large, structured action spaces and real state verifiers; action proposal and its cost must be part of the controller contract. |
| [From Confident Closing to Silent Failure (2026)](https://arxiv.org/abs/2606.09863) | Agent self-reports and generic LLM judges cannot replace environment-state verification. |
| [AgentRewardBench (2025)](https://arxiv.org/abs/2504.08942) | No single model judge is uniformly reliable across agent evaluations; verifier identity, evidence, and calibration must be reported. |
| [UltraHorizon (2025)](https://arxiv.org/abs/2509.21766) and [METR time horizons](https://metr.org/time-horizons/) | Long-horizon reliability and adaptation remain unsolved and should be measured separately from single-run task success. |
| [Context as an Environment (2026)](https://arxiv.org/abs/2608.21690) | Context management is part of the agent environment; it must be controlled and accounted for in long runs. |
| [LongHorizon-Harness (2026)](https://arxiv.org/abs/2608.01964), [Harness-Bench (2026)](https://arxiv.org/abs/2605.27922), and [WildClawBench (2026)](https://arxiv.org/abs/2605.10912) | External task state, context organization, and native harness choice can materially affect measured performance, so harness/scaffold versions are experimental factors. |
| [Anthropic infrastructure-noise study (2025)](https://www.anthropic.com/engineering/infrastructure-noise) and [OpenAI coding-evaluation audit (2026)](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) | Resource configuration and broken tasks can be as large as plausible method gains; match infrastructure, audit tasks, and repeat trials. |

## Documentation ownership

To prevent another status drift:

- update this file whenever implementation readiness, the current claim, or a
  freeze decision changes;
- keep `README.md` to a short summary that links here instead of duplicating the
  roadmap;
- use `ARCHITECTURE.md` for conceptual and target design, with every major
  element marked implemented or planned;
- use `EXPERIMENTS.md` for the evaluation contract and candidate hypotheses;
- use phase documents for as-built mechanics only;
- treat `paper/main.tex` as a versioned proposal until confirmatory artifacts
  exist;
- never call a hypothesis "preregistered" before an immutable manifest is
  externally timestamped;
- give time-sensitive external claims an as-of date and a primary-source link.
