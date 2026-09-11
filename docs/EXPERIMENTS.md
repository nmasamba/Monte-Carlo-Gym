# FidelityMCTS Experiment Harness and Evaluation Plan

**As of:** 2026-09-10

**Status:** revised evaluation contract; no hypothesis in this document is
preregistered

**Research readiness:** see [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)

## 1. Purpose

The harness must determine whether selective predictive and executable evidence
is actually better than simpler alternatives, and separately whether MCTS adds
value over a non-tree controller. It is not enough to show that the planner can
call two models or that an executable environment is more accurate.

The primary unit of comparison is an end-to-end agent episode under enforced
per-decision and per-episode resource and safety envelopes. Every method
receives the same task distribution, model portfolio, harness, context policy,
external-action policy, and episode budget. The current `SearchBudget` enforces
one planning call only; the repository does not yet enforce this required
episode envelope.

## 2. Research questions

### RQ1: Selective evidence acquisition (primary)

On stateful, sequential tool tasks, does selectively acquiring predictive and
executable evidence improve repeated-run verified success versus cost and risk
over fixed, query-level, sampling/aggregation, and dynamic non-tree policies?

### RQ2: Tree-controller value (primary conditional)

Conditional on measurable counterfactual-oracle headroom, does tree-structured
branch allocation improve over a capacity- and budget-matched non-tree
controller?

### RQ3: Decision-utility routing

Does a router trained on marginal decision utility or regret reduction
outperform absolute-discrepancy and uncertainty-only routing?

### RQ4: Learned stopping

Can a stop policy reduce tokens, executable calls, and latency without
materially reducing repeated-run success or increasing safety violations?

### RQ5: Reliability and robustness

Do the gains survive repeated trials, task paraphrases, harness changes, context
pressure, and controlled tool/API faults without increasing false-success or
policy-violation rates?

### RQ6: Verified self-learning (future)

Does learning from paired predictive and executable outcomes improve model
calibration, route selection, and downstream planning across frozen rounds?

### RQ7: Selection bias and causal correction (future)

Do propensity logging, audit exploration, and doubly robust evaluation produce
more reliable offline router selection than naive averages over selectively
verified branches?

### RQ8: Classical compatibility (engineering)

Do UCT, PUCT, Thompson sampling, robust/mix backup, RAVE, and MAST reproduce
known qualitative behavior and competitive reference results when adaptive
routing is disabled?

## 3. Candidate hypotheses

These hypotheses are design candidates only. They are not preregistered and
must not be described as confirmatory until a complete immutable protocol is
externally timestamped.

- **H1:** Selective evidence acquisition improves a fixed-reference hypervolume
  over verified success, measured cost, and risk relative to fixed cascade and
  dynamic non-tree policies.
- **H2:** Conditional on demonstrated oracle headroom, FidelityMCTS improves
  that hypervolume over a capacity-matched non-tree controller.
- **H3:** Decision-utility routing improves executable-call precision and
  verified success over absolute-discrepancy and uncertainty-only routing.
- **H4:** A learned stopping policy lowers median resource use while its
  success-rate difference remains within a preregistered non-inferiority margin.
- **H5:** Any primary gain remains positive under a preregistered repeated-run
  reliability metric and controlled tool-fault stratum.
- **H6:** Verified replay improves predictive-model calibration on held-out task
  families, not just training templates.
- **H7:** Causally corrected offline estimates rank candidate routers closer to
  their online randomized ranking than naive logged averages.

Failure to support any hypothesis is a valid research outcome.

## 4. Benchmark ladder

Use a ladder so failures can be localized before expensive agent runs.

| Level | Environment | Predictive evidence | Executable reference | Main purpose |
|---|---|---|---|---|
| L0 | synthetic multi-fidelity bandit/tree | biased stochastic oracle | ground-truth oracle | identifiability, budgets, router tests |
| L1 | Gymnasium control/toy-text/custom POMDP | learned dynamics/value | cloned native environment | classical MCTS and stochastic state handling |
| L2a | local one-decision mechanism fixture | lexical/learned score | disposable SQLite clone | instrumentation only; not sequential-agent evidence |
| L2b | stateful tool-agent tasks | language world model or compact predictor | pinned code/database environment | primary sequential mechanism test |
| L3 | BrowserGym/WorkArena/OSWorld family | LLM world model or DOM predictor | browser/desktop sandbox | later cross-interface replication |
| L4 | held-out native-runtime or enterprise-like tasks | distilled/local model or prior rollouts | real tool harness plus independent verifier | external validity, scale, privacy, latency, risk |

The current SQLite study is L2a. It does not satisfy the L2b requirement. No
agent claim should rely on L0-L2a, and no L3/L4 run should begin before L2b
correctness, oracle headroom, and budget invariants pass.

## 5. Candidate simulator pairs

### 5.1 Language-agent tasks

- Predictive: a pinned Qwen-AgentWorld checkpoint or a compact task-specific
  predictor.
- Executable: a pinned Agent World Model or maintained tau benchmark
  environment; ToolSandbox is a smaller fallback.
- Verification: task unit tests, database assertions, and final-state checks.

### 5.2 Browser workflows

- Predictive: compact LLM/DOM transition predictor or WebDreamer-style model.
- Executable: BrowserGym, WorkArena, or OSWorld in isolated containers/VMs.
- Verification: URL/DOM assertions, workflow-specific validators, and policy
  checks.

### 5.3 Gymnasium

- Predictive: learned ensemble dynamics model plus value head.
- Executable: cloned environment state.
- Verification: exact observed transition and return.

Each pair must publish a model/environment card describing what executable
verification does and does not establish. An executable simulator may still be
incomplete or differ from production.

## 6. Methods and baselines

### 6.1 Required baselines

1. **Harness-native direct/ReAct agent:** no external search controller.
2. **Always-plan and never-plan policies:** isolate dynamic planning value.
3. **Predictive-only and executable-only policies.**
4. **Fixed cascade:** predictive evaluation then execute/verify top-\(k\).
5. **Random escalation:** match the learned method's executable-call rate.
6. **Uncertainty threshold:** tune only on calibration data.
7. **Query-level router:** choose one model/tier per task, then hold fixed.
8. **Model/reasoning-budget router:** jointly choose one model and effort or
   output budget per task.
9. **Sequential revise/reflect policy:** use the same verifier and total budget.
10. **Parallel best-of-N plus listwise verifier/aggregator:** include a
    cost-matched multi-agent variant only where task parallelism is appropriate.
11. **Dynamic non-tree evidence controller:** use the same observations,
    features, learner capacity, and action choices as FidelityMCTS where
    possible.
12. **FidelityMCTS:** tree-structured branch evidence acquisition.
13. **Counterfactual headroom policy:** inspect all feasible outcomes on a small
    development subset. Call it an oracle upper bound only when it is not
    constrained by candidate order or the evaluated method's budget.

The current SQLite runner implements a local ten-method diagnostic, not this
required baseline set. In particular it has no harness-native agent,
model/reasoning-budget router, revise/reflect, parallel aggregation, or matched
non-tree controller. Its method named `oracle_counterfactual` remains
budget-limited and order-sensitive and is not a guaranteed upper bound.

### 6.2 Classical controls

- UCT plus random rollout and mean backup.
- PUCT plus direct value evaluation.
- Thompson sampling with conjugate posterior fixture.
- robust and mix backup.
- RAVE and MAST, independently and jointly.

These controls test implementation validity; they are not the primary novelty
comparison.

## 7. Experimental factors

Vary:

- budget at 5-10 logarithmically spaced levels;
- cheap-model bias, variance, and distribution shift;
- accurate-model cost and latency;
- action branching factor;
- candidate/macro-action proposal policy and progressive-widening schedule;
- horizon;
- reward sparsity;
- verifier coverage;
- irreversible-action risk;
- model portfolio size;
- queue/load conditions;
- token price and context length;
- context retention, retrieval, and compaction policy;
- harness/scaffold and reasoning-effort configuration;
- tool timeouts, rate limits, partial responses, and schema changes;
- benchmark, task, environment, and grader version.

Evaluate both stationary and changing cost regimes. A useful router should not
depend on a single provider-price snapshot.

## 8. Metrics

### 8.1 Task quality

- success rate;
- normalized return;
- regret where ground truth is available;
- constraint satisfaction;
- terminal verifier pass rate;
- human preference win rate for tasks requiring judgment.

Report environment-state success independently from the agent's stated
completion status and from any generic model judge.

### 8.2 Resource use

- input, output, and total tokens;
- normalized monetary cost;
- accurate simulator calls;
- model calls by tier;
- environment steps;
- GPU/CPU seconds;
- median and tail latency;
- search nodes expanded;
- peak memory;
- context tokens retained, retrieved, and compacted;
- proposal, aggregation, and verifier calls;
- per-decision and full-episode totals;
- concurrency-adjusted wall time for parallel methods.

### 8.3 Safety and reliability

- attempted unsafe/forbidden actions;
- approval requests;
- irreversible side effects;
- simulator/real discrepancy on safety-relevant fields;
- timeout and tool-error rate;
- recovery rate;
- repeated-run `pass@k` and `pass^k`;
- false-success and false-failure rates;
- policy and communication violations;
- stale-state and invalid-action errors;
- verifier false-positive/negative rate and assertion coverage;
- tail and conditional-value-at-risk cost, latency, and failure measures.

### 8.4 Model and router diagnostics

- value RMSE and negative log likelihood;
- interval coverage and expected calibration error;
- cheap-versus-accurate discrepancy;
- escalation precision and recall against oracle-useful calls;
- stop-decision regret;
- route propensity entropy;
- offline policy evaluation bias.
- action-switch rate and realized regret reduction after evidence;
- route performance by task family, horizon, harness, and fault stratum.

### 8.5 Primary summary

Report:

- Pareto frontiers, never only one weighted score;
- hypervolume under fixed, preregistered reference points;
- success at matched cost;
- cost at matched success;
- risk at matched success and cost;
- area under the budget-performance curve.

## 9. Ablations

Remove or replace one component at a time:

- no model discrepancy predictor;
- no calibrated uncertainty;
- no high-fidelity verification;
- no token routing;
- fixed rollout depth;
- fixed stopping;
- no tree reuse;
- no verified replay;
- synthetic-only replay;
- no propensity logging;
- naive logged evaluation instead of IPS/doubly robust;
- model routing without branch routing;
- branch routing without model routing;
- one shared value head versus model-specific heads;
- no risk constraint;
- no transposition table.

For every learned component, include a capacity-matched simple baseline.

## 10. Statistical protocol

### 10.1 Units and splits

- The task instance, not an individual tree simulation, is the independent
  experimental unit.
- Repeated seeds that only permute actions or resample the same fixed task are
  repeated measurements, not new independent tasks. Cluster inference at the
  task or task-template family as the data-generating process requires.
- Split by task template/family when possible to prevent paraphrase leakage.
- Keep development, calibration, router-training, and final test sets separate.
- Freeze test environments and verifier versions before final runs.
- Audit benchmark tasks and graders before freezing; a versioned artifact may be
  reproducible yet invalidated by later benchmark corrections.

### 10.2 Seeds

- Minimum 30 independent seeds for cheap synthetic experiments.
- For expensive environments, choose sample size by power analysis from a
  preregistered pilot, not by stopping when significance appears.
- Pair seeds and task instances across methods.
- Record all framework, environment, model, and sampling seeds.

### 10.3 Intervals and tests

- Use paired cluster/stratified bootstrap confidence intervals over independent
  task instances or held-out task families; never flatten task, seed, and budget
  rows into independent observations.
- Report effect sizes and intervals, not only \(p\)-values.
- Correct confirmatory families for multiple comparisons.
- Use a non-inferiority test for stopping-policy quality claims.
- Plot per-task paired differences to expose heterogeneous effects.
- Report all preregistered endpoints, including negative results.

### 10.4 Repeated API calls

Provider non-determinism is part of the system. Repeat an appropriate subset
across times and service-load strata. Preserve provider/model version, region,
sampling settings, reasoning effort, harness version, context policy,
infrastructure allocation, and response usage metadata. Treat small
model-score differences as unresolved when they are comparable to measured
infrastructure or grader noise.

## 11. Causal and off-policy protocol

Selective verification creates missing counterfactuals. The harness therefore
logs:

- router context;
- feasible routes;
- chosen route;
- probability/propensity of that choice;
- predicted value and cost;
- realized outcome for chosen route;
- randomized-audit indicator.

Run a small, safety-bounded randomized audit allocation to identify route
effects. Compare:

- naive verified-only estimate;
- inverse propensity score estimate;
- self-normalized IPS;
- direct outcome model;
- doubly robust estimate;
- true online randomized result.

Primary causal diagnostic: absolute error in the estimated difference between
two routers, plus the fraction of pairwise router rankings recovered.

Do not use causal language for quantities without a defensible intervention,
overlap, and identification argument.

The current SQLite analysis does not implement this policy-level comparison. It
forms records from selected `random_matched` verification decisions and uses
their conditional verified pass rate as the randomized reference. That is an
instrumentation diagnostic, not an estimate or ranking of complete candidate
router policies, and it cannot answer RQ7.

## 12. Future self-learning protocol

The repository has replay, fitting, and promotion-related components, but no
evaluated multi-round update loop. The following is a target protocol, not a
completed Phase 4 result.

One round is:

1. freeze planner, router, models, and reward versions;
2. collect search traces and selective verified outcomes;
3. run data validation and leakage checks;
4. train cheap-model, discrepancy, value, or router candidates;
5. evaluate offline with held-out tasks and off-policy estimators;
6. run randomized canary evaluation;
7. promote only if quality, cost, calibration, and risk gates pass.

Compare at least:

- no update;
- cheap model only;
- router only;
- discrepancy model only;
- joint update;
- joint update without causal correction.

Track whether performance compounds or collapses over multiple rounds.

## 13. Harness contract

Each method implements:

```python
class Planner(Protocol):
    def plan(
        self,
        state: State,
        *,
        models: ModelPortfolio,
        budget: SearchBudget,
        seed: int,
    ) -> PlanResult: ...
```

Each benchmark implements:

```python
class Benchmark(Protocol):
    def sample(self, seed: int) -> Task: ...
    def score(self, task: Task, result: PlanResult) -> EpisodeMetrics: ...
```

The runner:

1. resolves and validates configuration;
2. materializes a run identifier and environment fingerprint;
3. creates both per-decision and episode-level ledgers;
4. runs paired task seeds across methods;
5. writes an append-only record after each decision and episode;
6. verifies record and manifest hashes before analysis;
7. aggregates all declared units, including failures under the frozen policy;
8. emits summary, configuration, runtime metadata, and failures.

Items 3 and 6 are requirements for the next harness revision. The current
FrozenLake runner accumulates several per-call budgets without enforcing a
single episode cap, and the current SQLite reanalysis does not validate every
content/manifest hash before aggregation.

## 14. Artifact schema

```text
output/<suite>/<run-id>/
  resolved_config.json
  environment.json
  runs.jsonl
  summary.json
  failures.jsonl
  traces/
    <episode-id>.jsonl.zst
  checkpoints/
  figures/
  tables/
```

One episode record includes:

```json
{
  "schema_version": 1,
  "method": "adaptive",
  "task_id": "toy-000041",
  "seed": 41,
  "action": "a2",
  "success": true,
  "regret": 0.0,
  "return": 0.83,
  "cost": 23.0,
  "tokens": 112,
  "accurate_calls": 1,
  "latency_s": 0.004,
  "risk": 0.0,
  "stop_reason": "evidence_sufficient",
  "versions": {
    "planner": "0.2.0a2",
    "cheap_model": "fixture-v1",
    "accurate_model": "fixture-v1"
  }
}
```

## 15. Reproducibility checklist

- exact source revision;
- lockfile or container digest;
- resolved configuration;
- all random seeds;
- hardware and operating system;
- environment and dataset version;
- model and prompt/template versions;
- harness, action interface, context policy, reasoning effort, and grader
  versions;
- raw structured traces;
- raw-record and artifact-manifest verification before analysis;
- exclusions and failure handling;
- budget reference prices;
- statistical analysis script;
- license and access instructions.

## 16. Included toy scaffold

The repository includes a dependency-free L0 benchmark. Each task has several
actions with hidden true values. The cheap model has action-dependent bias and
noise; the accurate model has lower noise and higher cost. Four planners run on
identical tasks:

- cheap only;
- accurate only;
- fixed top-\(k\) cascade;
- ambiguity-aware adaptive fidelity.

Run:

```bash
python experiments/run.py \
  --config experiments/configs/toy.json \
  --output output/toy
```

This fixture exists to test interfaces, accounting, reproducibility, and
qualitative routing behavior. It is not a substitute for MCTS tree benchmarks
or executable agent environments.

## 17. Minimum evidence for a submission

Before submission, require:

- positive counterfactual-oracle headroom on development data;
- at least one genuinely sequential, stateful executable L2b suite with two
  held-out task families and one independent replication environment;
- all required baselines at matched budgets;
- a direct FidelityMCTS versus capacity-matched non-tree comparison;
- at least five budget points and Pareto analysis;
- component ablations;
- held-out task-family generalization;
- calibration and discrepancy results;
- state-based verification with measured verifier error/coverage;
- repeated-run reliability, safety, tool-fault, false-success, and failure
  reporting;
- independently sampled tasks with cluster-aware paired intervals;
- compute and monetary accounting;
- enforced per-decision and episode-level envelopes;
- released configurations, harness, and representative traces;
- no empirical statement based only on the included toy, FrozenLake, or SQLite
  mechanism fixtures.

## 18. Operational preregistration boundary

Preregistration means making the complete confirmatory design immutable and
externally timestamped before inspecting any outcome from its held-out
confirmatory seeds or tasks. It is stronger than committing a general plan:
the hypotheses, endpoint families, task unit, benchmark and verifier versions,
training/calibration/test split, artifact hashes, methods, ablations, budgets,
sample size and seeds, randomization, exclusions, failure/retry policy,
stopping rule, interval method, multiple-comparison correction, and Pareto or
hypervolume reference points must be fixed.

The repository enforces this workflow with separate locations:

```text
experiments/pilots/           mutable exploratory protocols
experiments/protocols/        mutable confirmatory candidates
experiments/preregistered/    immutable fingerprinted manifests
output/pilots/                exploratory outcomes
output/confirmatory/          untouched confirmatory outcomes
```

The FrozenLake L1 protocol is a pilot of the Phase 4 machinery. It is not by
itself the complete paper protocol described in this document. Before a paper
preregistration, pass every gate in `RESEARCH_STATUS.md`, including oracle
headroom, a sequential L2b integration, the tree-versus-non-tree comparison,
isolated ablations, an enforced episode budget, task-clustered inference,
locked analysis code, and fixed statistical reference points. Then validate,
freeze, and externally register a fresh complete candidate as described in
`experiments/README.md`.

After registration, do not run ad-hoc checks on confirmatory seeds. The first
access to those outcomes must be through the registered runner. Any amendment
must be timestamped and justified before accessing the affected outcomes; it
uses a new study identifier and never overwrites the original manifest or raw
run directory.

## 19. Phase 5A executable mechanism pilot

Phase 5A implements an offline SQLite query construction/repair fixture. It
uses immutable development, calibration, and exploratory fixtures; disposable
in-memory execution; objective result-set verification; ten local comparison
methods; six named diagnostic variants; five per-planning-call budget points;
paired task IDs and order seeds; immutable decision/episode/failure records; and
exploratory paired, Pareto, calibration, OPE/overlap, and power summaries.

Those labels require important qualifications. The `fidelity_mcts` path calls
the adaptive planner directly and does not run an outer MCTS tree. The
`oracle_counterfactual` path is constrained by candidate order and budget, so it
is not necessarily an upper bound. Two no-verification/cheap-only variants are
the same route, while the no-discrepancy variant still learns a running
discrepancy correction. The OPE output is not a candidate-router ranking study.
The generic bootstrap also does not implement the candidate's declared
task-stratified confirmatory analysis, hypervolume test, Holm procedure, or
non-inferiority test.

The current EVC learner targets absolute verified discrepancy and remains an
explicit proxy baseline. It is fitted at one configured budget and does not
learn varying depth or remaining-budget behavior. The benchmark has one task
decision, so tree reuse is structurally inapplicable. Its seeds permute the four
underlying exploratory tasks rather than create independent task instances;
those repetitions cannot be used as 32 independent units. These limitations
mean the draft protocol must not be frozen or promoted unchanged.

The candidate under `experiments/protocols/` declares the future-confirmatory
partition and seed policy without materializing either task IDs or seeds. It is
a mutable, unregistered mechanism-study candidate retained for audit and
instrumentation work, not the candidate for the revised sequential-agent claim.
No Phase 5A command can run a confirmatory SQLite study.
