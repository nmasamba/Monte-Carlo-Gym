from __future__ import annotations

import unittest
from random import Random

from montecarlgym.core.backup import ConstantMixWeight, MixBackup, RobustBackup
from montecarlgym.core.path import Evaluation, SearchPath
from montecarlgym.core.tree import ActionEdge, OutcomeLink, StateNode
from montecarlgym.sharing import (
    MASTBackup,
    MASTRolloutPolicy,
    MoveStatisticsTable,
    RAVEBackup,
    RAVETreePolicy,
)


def two_level_path() -> tuple[SearchPath, ActionEdge, StateNode]:
    top = StateNode("top", "top")
    middle = StateNode("middle", "middle")
    leaf = StateNode("leaf", "leaf")
    top_edge = ActionEdge("top-action")
    top_outcome = OutcomeLink("top-outcome", 0.0, middle)
    selected = ActionEdge("selected")
    selected_outcome = OutcomeLink("selected-outcome", 0.0, leaf)
    alternative = ActionEdge(
        "robust",
        visits=3,
        total_return=6.0,
        mean_value=2.0,
    )
    top.edges[top_edge.action] = top_edge
    middle.edges = {
        selected.action: selected,
        alternative.action: alternative,
    }
    path = SearchPath(top)
    path.append(top, top_edge, top_outcome)
    path.append(middle, selected, selected_outcome)
    return path, top_edge, middle


class AdvancedBackupTests(unittest.TestCase):
    def test_robust_backup_uses_most_visited_not_largest_sample(self) -> None:
        path, top_edge, _ = two_level_path()

        RobustBackup().update(path, Evaluation(0.0))

        self.assertEqual(top_edge.mean_value, 2.0)

    def test_mix_backup_stays_between_mean_and_robust_components(self) -> None:
        path, top_edge, _ = two_level_path()

        MixBackup(schedule=ConstantMixWeight(0.25)).update(
            path,
            Evaluation(0.0),
        )

        self.assertAlmostEqual(top_edge.mean_value, 1.625)
        self.assertGreaterEqual(top_edge.mean_value, 1.5)
        self.assertLessEqual(top_edge.mean_value, 2.0)

    def test_rave_does_not_amaf_double_count_direct_edge(self) -> None:
        root = StateNode("root", 0)
        leaf = StateNode("leaf", 1)
        direct = ActionEdge("a")
        other = ActionEdge("b")
        root.edges = {"a": direct, "b": other}
        outcome = OutcomeLink("outcome", 1.0, leaf)
        path = SearchPath(root)
        path.append(root, direct, outcome)

        RAVEBackup().update(
            path,
            Evaluation(0.0, rollout_actions=("a", "b")),
        )

        self.assertEqual(direct.visits, 1)
        self.assertEqual(direct.amaf_visits, 0)
        self.assertEqual(other.amaf_visits, 1)
        self.assertEqual(other.amaf_mean_value, 1.0)

    def test_rave_policy_uses_amaf_signal(self) -> None:
        node = StateNode("root", 0, visits=10)
        low = ActionEdge("low", visits=5, amaf_visits=10)
        high = ActionEdge(
            "high",
            visits=5,
            amaf_visits=10,
            amaf_total_return=20.0,
            amaf_mean_value=2.0,
        )
        node.edges = {low.action: low, high.action: high}

        selected = RAVETreePolicy(exploration_constant=0.0).select(
            node,
            Random(3),
        )

        self.assertIs(selected, high)


class MASTTests(unittest.TestCase):
    def test_mast_rollout_prefers_high_value_move(self) -> None:
        table = MoveStatisticsTable()
        for _ in range(4):
            table.update("low", -2.0)
            table.update("high", 2.0)
        policy = MASTRolloutPolicy(table, temperature=0.1, epsilon=0.0)

        choices = [
            policy.choose_action(("low", "high"), Random(seed))
            for seed in range(20)
        ]

        self.assertEqual(choices.count("high"), 20)

    def test_mast_backup_updates_tree_and_rollout_moves(self) -> None:
        table = MoveStatisticsTable()
        root = StateNode("root", 0)
        leaf = StateNode("leaf", 1)
        edge = ActionEdge("tree")
        outcome = OutcomeLink("outcome", 1.0, leaf)
        root.edges[edge.action] = edge
        path = SearchPath(root)
        path.append(root, edge, outcome)

        MASTBackup(table).update(
            path,
            Evaluation(2.0, rollout_actions=("rollout",)),
        )

        self.assertEqual(edge.visits, 1)
        self.assertEqual(table.get("tree").mean_value, 3.0)
        self.assertEqual(table.get("rollout").mean_value, 3.0)


if __name__ == "__main__":
    unittest.main()
