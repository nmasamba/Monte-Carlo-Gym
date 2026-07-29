# FidelityMCTS Experiment Harness and Evaluation Plan

## 1. Purpose

The harness must determine whether adaptive branch-level compute allocation is
actually better than simpler alternatives. It is not enough to show that the
planner can call two simulators or that an expensive model is more accurate.

The primary unit of comparison is an end-to-end planning episode under a
resource and safety envelope. Every method receives the same task distribution,
available models, external-action policy, and budget.

## 2. Research questions

### RQ1: Multi-fidelity allocation

Does selectively escalating ambiguous or high-impact branches outperform
cheap-only, accurate-only, random escalation, and fixed cascades on the
success-cost Pareto frontier?

### RQ2: Joint model and token routing

Does branch-level joint selection of model tier and token budget outperform
query-level model routing followed by a fixed search procedure?

### RQ3: Learned stopping

Can a stop policy reduce tokens, simulator calls, and latency without materially
reducing task success or increasing safety violations?

### RQ4: Verified self-learning

Does learning from paired cheap predictions and executable outcomes improve
simulator calibration, route selection, and downstream planning across rounds?

### RQ5: Selection bias and causal correction

Do propensity logging, audit exploration, and doubly robust evaluation produce
more reliable offline router selection than naive averages over selectively
verified branches?

### RQ6: Classical compatibility

Do UCT, PUCT, Thompson sampling, robust/mix backup, RAVE, and MAST reproduce
known qualitative behavior and competitive reference results when adaptive
routing is disabled?

## 3. Preregistered hypotheses

- **H1:** FidelityMCTS dominates cheap-only search and fixed cascades in
  hypervolume over success, normalized cost, and risk.
- **H2:** At matched success, FidelityMCTS uses fewer accurate simulator calls
  than accurate-only search.
- **H3:** At matched budget, branch-level routing exceeds query-level routing
  on tasks whose useful fidelity varies within a search tree.
- **H4:** Discrepancy-aware routing improves high-fidelity call precision over
  uncertainty-only routing.
- **H5:** A learned stopping policy lowers median resource use while its
  success-rate difference remains within a preregistered non-inferiority margin.
- **H6:** Verified replay improves cheap-model calibration on held-out task
  families, not just training templates.
- **H7:** Causally corrected offline estimates rank candidate routers closer to
  their online randomized ranking than naive logged averages.

Failure to support any hypothesis is a valid research outcome.

## 4. Benchmark ladder

Use a ladder so failures can be localized before expensive agent runs.

| Level | Environment | Cheap model | Accurate model | Main purpose |
|---|---|---|---|---|
| L0 | synthetic multi-fidelity bandit/tree | biased stochastic oracle | ground-truth oracle | identifiability, budgets, router tests |
| L1 | Gymnasium control/toy-text/custom POMDP | learned dynamics/value | cloned native environment | classical MCTS and stochastic state handling |
| L2 | AgentWorldModel-1K-style tasks | language world model | code/database executable environment | tool-effect prediction and verified replay |
| L3 | BrowserGym/WorkArena | LLM world model or DOM predictor | browser sandbox | long-horizon enterprise workflows |
| L4 | held-out enterprise-like tasks | distilled/local model | sandbox plus deterministic verifiers | scale, privacy, latency, risk |

No claim should rely only on L0. No L3/L4 run should begin before L0-L2
correctness and budget invariants pass.

## 5. Candidate simulator pairs

### 5.1 Language-agent tasks

- Cheap: Qwen-AgentWorld or Dreamer-7B adapter.
- Accurate: AgentWorldModel-1K executable environment.
- Verification: task unit tests, database assertions, and final-state checks.

### 5.2 Browser workflows

- Cheap: compact LLM/DOM transition predictor or WebDreamer-style model.
- Accurate: BrowserGym and WorkArena in isolated browser containers.
- Verification: URL/DOM assertions, workflow-specific validators, and policy
  checks.

### 5.3 Gymnasium

- Cheap: learned ensemble dynamics model plus value head.
- Accurate: cloned environment state.
- Verification: exact observed transition and return.

Each pair must publish a model card describing the meaning of “accurate.” An
executable simulator may still be incomplete or differ from production.

## 6. Methods and baselines

### 6.1 Required baselines

1. **No search / direct policy.**
2. **Cheap-only MCTS.**
3. **Accurate-only MCTS.**
4. **Fixed cascade:** cheap search then verify top-\(k\).
5. **Random escalation:** match FidelityMCTS's accurate-call rate.
6. **Uncertainty threshold:** hand-tuned escalation.
7. **Query-level router:** choose one model/tier per task, then hold fixed.
8. **Token router:** jointly choose one model and output budget per task.
9. **FidelityMCTS:** branch-level model, token, depth, verifier, and stop choice.
10. **Oracle router:** upper bound using counterfactual outcomes, only where the
    benchmark can compute them.

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
- horizon;
- reward sparsity;
- verifier coverage;
- irreversible-action risk;
- model portfolio size;
- queue/load conditions;
- token price and context length.

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

### 8.2 Resource use

- input, output, and total tokens;
- normalized monetary cost;
- accurate simulator calls;
- model calls by tier;
- environment steps;
- GPU/CPU seconds;
- median and tail latency;
- search nodes expanded;
- peak memory.

### 8.3 Safety and reliability

- attempted unsafe/forbidden actions;
- approval requests;
- irreversible side effects;
- simulator/real discrepancy on safety-relevant fields;
- timeout and tool-error rate;
- recovery rate.

### 8.4 Model and router diagnostics

- value RMSE and negative log likelihood;
- interval coverage and expected calibration error;
- cheap-versus-accurate discrepancy;
- escalation precision and recall against oracle-useful calls;
- stop-decision regret;
- route propensity entropy;
- offline policy evaluation bias.

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
- Split by task template/family when possible to prevent paraphrase leakage.
- Keep development, calibration, router-training, and final test sets separate.
- Freeze test environments and verifier versions before final runs.

### 10.2 Seeds

- Minimum 30 independent seeds for cheap synthetic experiments.
- For expensive environments, choose sample size by power analysis from a
  preregistered pilot, not by stopping when significance appears.
- Pair seeds and task instances across methods.
- Record all framework, environment, model, and sampling seeds.

### 10.3 Intervals and tests

- Use paired stratified bootstrap confidence intervals over task instances.
- Report effect sizes and intervals, not only \(p\)-values.
- Correct confirmatory families for multiple comparisons.
- Use a non-inferiority test for stopping-policy quality claims.
- Plot per-task paired differences to expose heterogeneous effects.
- Report all preregistered endpoints, including negative results.

### 10.4 Repeated API calls

Provider non-determinism is part of the system. Repeat an appropriate subset
across times and service-load strata. Preserve provider/model version, region,
sampling settings, and response usage metadata.

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

## 12. Self-learning protocol

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
3. runs paired task seeds across methods;
4. writes an append-only record after each episode;
5. aggregates only completed, valid records;
6. emits summary, configuration, runtime metadata, and failures.

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
    "planner": "0.1.0a0",
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
- raw structured traces;
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

- at least one L1 environment suite and one executable L2/L3 suite;
- all required baselines at matched budgets;
- at least five budget points and Pareto analysis;
- component ablations;
- held-out task-family generalization;
- calibration and discrepancy results;
- safety and failure reporting;
- multiple seeds with paired intervals;
- compute and monetary accounting;
- released configurations, harness, and representative traces;
- no empirical statement based only on the included toy fixture.

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
preregistration, add the missing L2/L3 benchmark integrations, all required
matched-budget baselines, the final ablation matrix, power analysis, locked
analysis code, and fixed statistical reference points. Then validate, freeze,
and externally register the complete candidate as described in
`experiments/README.md`.

After registration, do not run ad-hoc checks on confirmatory seeds. The first
access to those outcomes must be through the registered runner. Any amendment
must be timestamped and justified before accessing the affected outcomes; it
uses a new study identifier and never overwrites the original manifest or raw
run directory.
