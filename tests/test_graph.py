from __future__ import annotations

import pytest

from pr_split.exceptions import PlanValidationError
from pr_split.graph import PlanDAG
from pr_split.schemas import Group


def _group(gid: str, depends_on: list[str] | None = None) -> Group:
    return Group(id=gid, title=gid, description=gid, depends_on=depends_on or [])


class TestPlanDAG:
    def test_linear_topo_order(self) -> None:
        groups = [_group("a"), _group("b", ["a"]), _group("c", ["b"])]
        dag = PlanDAG(groups)
        order = dag.topological_order()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_topo_order(self) -> None:
        groups = [
            _group("a"),
            _group("b", ["a"]),
            _group("c", ["a"]),
            _group("d", ["b", "c"]),
        ]
        dag = PlanDAG(groups)
        order = dag.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detected(self) -> None:
        groups = [_group("a", ["b"]), _group("b", ["a"])]
        dag = PlanDAG(groups)
        with pytest.raises(PlanValidationError):
            dag.validate_acyclic()

    def test_unknown_dependency_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="'b' depends on unknown group 'zzz'"):
            PlanDAG([_group("a"), _group("b", ["zzz"])])

    def test_acyclic_passes(self) -> None:
        groups = [_group("a"), _group("b", ["a"])]
        dag = PlanDAG(groups)
        dag.validate_acyclic()

    def test_roots_and_leaves(self) -> None:
        groups = [_group("a"), _group("b", ["a"]), _group("c", ["a"])]
        dag = PlanDAG(groups)
        assert set(dag.roots()) == {"a"}
        assert set(dag.leaves()) == {"b", "c"}

    def test_parents_and_children(self) -> None:
        groups = [_group("a"), _group("b", ["a"]), _group("c", ["a"])]
        dag = PlanDAG(groups)
        assert dag.parents("b") == ["a"]
        assert set(dag.children("a")) == {"b", "c"}

    def test_is_merge_node(self) -> None:
        groups = [
            _group("a"),
            _group("b"),
            _group("c", ["a", "b"]),
        ]
        dag = PlanDAG(groups)
        assert not dag.is_merge_node("a")
        assert dag.is_merge_node("c")

    def test_ancestors(self) -> None:
        groups = [_group("a"), _group("b", ["a"]), _group("c", ["b"])]
        dag = PlanDAG(groups)
        assert dag.ancestors("c") == {"a", "b"}
        assert dag.ancestors("a") == set()

    def test_descendants(self) -> None:
        groups = [_group("a"), _group("b", ["a"]), _group("c", ["b"])]
        dag = PlanDAG(groups)
        assert dag.descendants("a") == {"b", "c"}
        assert dag.descendants("c") == set()

    def test_iter_ready_batches(self) -> None:
        groups = [
            _group("a"),
            _group("b"),
            _group("c", ["a", "b"]),
        ]
        dag = PlanDAG(groups)
        batches = list(dag.iter_ready())
        assert len(batches) == 2
        assert set(batches[0]) == {"a", "b"}
        assert batches[1] == ["c"]

    def test_multiple_roots(self) -> None:
        groups = [_group("a"), _group("b"), _group("c")]
        dag = PlanDAG(groups)
        assert set(dag.roots()) == {"a", "b", "c"}
        assert set(dag.leaves()) == {"a", "b", "c"}
        order = dag.topological_order()
        assert set(order) == {"a", "b", "c"}


class TestLinearChains:
    def test_single_node(self) -> None:
        dag = PlanDAG([_group("a")])
        assert dag.linear_chains() == [["a"]]

    def test_forest_of_chains(self) -> None:
        groups = [
            _group("pr-1"),
            _group("pr-2"),
            _group("pr-3", ["pr-2"]),
            _group("pr-4"),
            _group("pr-5", ["pr-4"]),
        ]
        dag = PlanDAG(groups)
        assert sorted(dag.linear_chains()) == [
            ["pr-1"],
            ["pr-2", "pr-3"],
            ["pr-4", "pr-5"],
        ]

    def test_diamond_breaks_chains(self) -> None:
        groups = [
            _group("a"),
            _group("b", ["a"]),
            _group("c", ["a"]),
            _group("d", ["b", "c"]),
        ]
        dag = PlanDAG(groups)
        assert sorted(dag.linear_chains()) == [["a"], ["b"], ["c"], ["d"]]

    def test_fan_out_keeps_descendant_runs(self) -> None:
        groups = [
            _group("a"),
            _group("b", ["a"]),
            _group("c", ["b"]),
            _group("d", ["a"]),
        ]
        dag = PlanDAG(groups)
        assert sorted(dag.linear_chains()) == [["a"], ["b", "c"], ["d"]]


class TestPlanDAGRejectsDuplicateIds:
    def test_duplicate_group_id_is_a_validation_error(self) -> None:
        with pytest.raises(PlanValidationError, match="'pr-1' is used more than once"):
            PlanDAG([_group("pr-1"), _group("pr-1")])

    def test_distinct_ids_are_fine(self) -> None:
        dag = PlanDAG([_group("pr-1"), _group("pr-2", ["pr-1"])])
        assert dag.parents("pr-2") == ["pr-1"]


class TestPlanDAGDedupesDependencies:
    def test_duplicate_dependency_is_a_single_edge(self) -> None:
        dag = PlanDAG([_group("pr-1"), _group("pr-2", ["pr-1", "pr-1"])])
        assert dag.parents("pr-2") == ["pr-1"]
        assert dag.children("pr-1") == ["pr-2"]
        assert dag.is_merge_node("pr-2") is False
        assert dag.linear_chains() == [["pr-1", "pr-2"]]

    def test_duplicate_dependency_does_not_break_batching(self) -> None:
        dag = PlanDAG([_group("pr-1"), _group("pr-2", ["pr-1", "pr-1"])])
        assert list(dag.iter_ready()) == [["pr-1"], ["pr-2"]]
