"""Named classical families assembled from one dependency-injected engine."""

from __future__ import annotations

from .agent import MCTSAgent
from .bayes import BayesianBackup, ThompsonTreePolicy
from .config import MCTSConfig
from .core import MeanBackup, MixBackup, RobustBackup, StateCodec
from .core.backup import MixWeightSchedule
from .evaluators import MixedEvaluator, PolicyValueEvaluator, PolicyValuePredictor
from .policies import (
    MostVisitedActionSelector,
    PUCTTreePolicy,
    RandomRolloutEvaluator,
    UCTTreePolicy,
)
from .sharing import (
    MASTBackup,
    MASTRolloutPolicy,
    MoveStatisticsTable,
    RAVEBackup,
    RAVETreePolicy,
)
from .types import SearchBudget


def uct_preset(
    *,
    budget: SearchBudget,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """UCT plus random rollout and mean backup."""

    return MCTSAgent(
        budget=budget,
        seed=seed,
        config=config,
        state_codec=state_codec,
    )


def alphago_zero_preset(
    *,
    budget: SearchBudget,
    predictor: PolicyValuePredictor,
    seed: int = 0,
    exploration_constant: float = 1.5,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """PUCT plus combined policy/value expansion and no rollout."""

    evaluator = PolicyValueEvaluator(predictor)
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=PUCTTreePolicy(exploration_constant),
        expander=evaluator,
        evaluator=evaluator,
        action_selector=MostVisitedActionSelector(),
        config=config,
        state_codec=state_codec,
    )


def alphago_apv_preset(
    *,
    budget: SearchBudget,
    predictor: PolicyValuePredictor,
    value_weight: float = 0.5,
    seed: int = 0,
    exploration_constant: float = 1.5,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """PUCT plus a policy/value and fast-random-rollout mixture."""

    resolved = config or MCTSConfig()
    policy_value = PolicyValueEvaluator(predictor)
    rollout = RandomRolloutEvaluator(discount=resolved.discount)
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=PUCTTreePolicy(exploration_constant),
        expander=policy_value,
        evaluator=MixedEvaluator(policy_value, rollout, value_weight),
        config=resolved,
        state_codec=state_codec,
    )


def dng_mcts_preset(
    *,
    budget: SearchBudget,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """Local Normal-Gamma Thompson selection and Bayesian tree updates."""

    resolved = config or MCTSConfig()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=ThompsonTreePolicy(),
        backup=BayesianBackup(discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )


def mcbrl_root_sampling_preset(
    *,
    budget: SearchBudget,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """UCT/mean search for a root-sampled posterior generative model.

    Pass a ``TabularRootSamplingModel`` (or compatible model) to
    ``compute_action`` so each iteration receives one sampled MDP while the
    caller-owned root belief remains frozen.
    """

    resolved = config or MCTSConfig()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=UCTTreePolicy(),
        backup=MeanBackup(discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )


def crazy_stone_robust_preset(
    *,
    budget: SearchBudget,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """UCT selection with most-visited-child robust propagation."""

    resolved = config or MCTSConfig()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=UCTTreePolicy(),
        backup=RobustBackup(discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )


def crazy_stone_mix_preset(
    *,
    budget: SearchBudget,
    schedule: MixWeightSchedule,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """UCT selection with injected mean/robust mixing schedule."""

    resolved = config or MCTSConfig()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=UCTTreePolicy(),
        backup=MixBackup(discount=resolved.discount, schedule=schedule),
        config=resolved,
        state_codec=state_codec,
    )


def rave_preset(
    *,
    budget: SearchBudget,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """RAVE/AMAF selection and backup over a random rollout evaluator."""

    resolved = config or MCTSConfig()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=RAVETreePolicy(),
        backup=RAVEBackup(discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )


def mast_preset(
    *,
    budget: SearchBudget,
    table: MoveStatisticsTable | None = None,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """UCT with globally shared MAST rollout action statistics."""

    resolved = config or MCTSConfig()
    resolved_table = table or MoveStatisticsTable()
    return MCTSAgent(
        budget=budget,
        seed=seed,
        evaluator=RandomRolloutEvaluator(
            policy=MASTRolloutPolicy(resolved_table),
            discount=resolved.discount,
        ),
        backup=MASTBackup(resolved_table, discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )


def rave_mast_preset(
    *,
    budget: SearchBudget,
    table: MoveStatisticsTable | None = None,
    seed: int = 0,
    config: MCTSConfig | None = None,
    state_codec: StateCodec | None = None,
) -> MCTSAgent:
    """Compose RAVE tree sharing and MAST rollout sharing."""

    resolved = config or MCTSConfig()
    resolved_table = table or MoveStatisticsTable()
    mast_backup = MASTBackup(resolved_table, discount=resolved.discount)
    return MCTSAgent(
        budget=budget,
        seed=seed,
        tree_policy=RAVETreePolicy(),
        evaluator=RandomRolloutEvaluator(
            policy=MASTRolloutPolicy(resolved_table),
            discount=resolved.discount,
        ),
        backup=RAVEBackup(base=mast_backup, discount=resolved.discount),
        config=resolved,
        state_codec=state_codec,
    )
