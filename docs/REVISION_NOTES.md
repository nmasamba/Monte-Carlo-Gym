# Revision notes: original plan to the September 2026 research pivot

This file records changes in direction. The current decision, implementation
boundary, validity blockers, and next-study gates are canonical in
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

## Preserved from the original plan

- transactional Gymnasium state snapshot/restore;
- state nodes, action edges, stochastic outcome links, and explicit search
  paths;
- transpositions and subtree reuse;
- pluggable tree, rollout, evaluator, and backup policies;
- UCT, Bayesian posterior sampling, Crazy Stone, AlphaGo/APV, AlphaGo Zero,
  RAVE, and MAST;
- strict handling of termination, truncation, RNG state, and value perspective.

## Revision 2: adaptive-compute research layer

Revision 2 moved the center of gravity from a broad algorithm library toward a
portfolio of predictive and executable evidence sources, explicit resource
accounting, adaptive routing, persistent verified replay, and a
preregisterable experiment harness. It proposed branch-level joint allocation
of model, tokens, depth, verification, and stopping as the main research claim.

That proposal remains useful target architecture, but subsequent implementation
and literature review showed that the claim was broader than the delivered
system and too close to rapidly developing test-time-compute routing work.

## Revision 3: evidence acquisition first (September 2026)

Revision 3 narrows the research program without discarding the classical
kernel or the evidence-routing infrastructure.

| Earlier emphasis | Revision 3 decision |
|---|---|
| MCTS as the defining controller | MCTS is one controller candidate that must beat capacity-matched non-tree policies |
| joint optimization of every compute dimension | isolate selective predictive-versus-executable evidence acquisition first |
| one simulator portfolio spanning tree transitions | current portfolio code is described accurately as frontier/action valuation only |
| query-level success and average cost | independently verified terminal state, repeated-run reliability, episode totals, and tail risk |
| finite legal-action enumeration | add proposal provenance, proposal cost, macro-actions, or progressive widening for open tool spaces |
| context and harness as background details | freeze them as first-class experimental conditions |
| SQLite Phase 5A as readiness evidence | retain it as a one-decision mechanism and instrumentation diagnostic |
| verified replay as self-improvement evidence | treat replay, calibration, and OPE as infrastructure until a multi-round loop is evaluated |

The revised primary question is:

> In a fixed, versioned agent harness with independently verified task state,
> can branch-local selection between predictive evidence and isolated
> executable evidence improve repeated-run verified success versus measured
> cost, latency, and risk over fixed, query-level, and non-tree adaptive
> baselines?

A separate, conditional question asks whether an MCTS controller improves that
frontier over a capacity- and budget-matched non-tree controller once the task
has demonstrated enough oracle headroom to justify search.

## Current implementation boundary

- The classical kernel, transactional native/deep-copy snapshots, cost ledgers,
  frontier evaluator, persistent replay, simple discrepancy calibration,
  binary learned routing, randomized audits, OPE utilities, and
  preregistration mechanics exist and have tests.
- The adaptive evaluator scores already-created frontier actions; outer MCTS
  still obtains transitions from a single environment model.
- The learned router chooses only cheap versus accurate evaluation. Token and
  depth choices are fixed, verification is coupled to accurate evaluation, and
  risk is logged rather than enforced as a hard ceiling.
- The FrozenLake example applies a fresh planning-call budget at every
  environment step. It is not a hard episode-budget experiment.
- The SQLite fixture is one decision, uses normalized costs, lacks the required
  global routers, and cannot test tree reuse or sequential planning.
- No confirmatory protocol or evaluated multi-round self-learning loop exists.

## Scope guardrails

- Predictive search may complement policy optimization in an RLHF-style loop;
  it does not replace preference elicitation, reward validation, or safety
  review.
- Logged-policy correction is useful where selection induces bias; a universal
  causal world model is not required.
- Learned simulation is provisional evidence, not verified truth.
- Executable probes run only in disposable clones or sandboxes; speculative
  search must not act on production systems.
- Verification must identify its implementation and version and retain
  assertion-level evidence; a Boolean success flag alone is insufficient.
- Toy, FrozenLake, and SQLite fixtures establish engineering behavior only.
- No confirmatory claim is made until a fresh sequential protocol passes every
  gate in `RESEARCH_STATUS.md` and is frozen before data collection.
