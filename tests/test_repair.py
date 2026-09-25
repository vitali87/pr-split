from __future__ import annotations

from unittest.mock import MagicMock, patch

from pr_split.constants import AssignmentType, PartitionStrategy, Provider
from pr_split.diff_ops.parser import parse_diff
from pr_split.graph import PlanDAG
from pr_split.planner.client import _plan_split_with_llm
from pr_split.planner.repair import repair_plan
from pr_split.planner.validator import validate_plan
from pr_split.schemas import Group, GroupAssignment

_DIFF = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 a1
+a2
 a3
 a4
@@ -20,3 +21,4 @@
 a20
+a21
 a22
 a23
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,2 @@
+b1
+b2
"""


def _partial(path: str, *indices: int) -> GroupAssignment:
    return GroupAssignment(
        file_path=path, assignment_type=AssignmentType.PARTIAL_HUNKS, hunk_indices=list(indices)
    )


def _whole(path: str) -> GroupAssignment:
    return GroupAssignment(
        file_path=path, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[0]
    )


def _group(gid: str, *assignments: GroupAssignment, deps: list[str] | None = None) -> Group:
    return Group(
        id=gid, title=gid, description="", depends_on=deps or [], assignments=list(assignments)
    )


def _valid(groups: list[Group]) -> None:
    parsed = parse_diff(_DIFF)
    validate_plan(groups, parsed, PlanDAG(groups), max_loc=400)


def _held(groups: list[Group]) -> dict[str, list[tuple[str, int]]]:
    counts = {"a.py": 2, "b.py": 1}
    return {
        g.id: sorted(
            (a.file_path, i) for a in g.assignments for i in a.covered_indices(counts[a.file_path])
        )
        for g in groups
    }


def test_left_out_hunk_joins_the_group_holding_its_file() -> None:
    groups = repair_plan(
        [_group("pr-1", _partial("a.py", 0)), _group("pr-2", _whole("b.py"))], parse_diff(_DIFF)
    )
    assert _held(groups) == {"pr-1": [("a.py", 0), ("a.py", 1)], "pr-2": [("b.py", 0)]}
    _valid(groups)


def test_hunk_claimed_twice_stays_with_the_first_group() -> None:
    groups = repair_plan(
        [
            _group("pr-1", _whole("a.py")),
            _group("pr-2", _partial("a.py", 1), _whole("b.py")),
        ],
        parse_diff(_DIFF),
    )
    assert _held(groups) == {"pr-1": [("a.py", 0), ("a.py", 1)], "pr-2": [("b.py", 0)]}
    _valid(groups)


def test_whole_file_overlapping_a_partial_claim_keeps_the_rest() -> None:
    groups = repair_plan(
        [_group("pr-1", _partial("a.py", 0), _whole("b.py")), _group("pr-2", _whole("a.py"))],
        parse_diff(_DIFF),
    )
    assert _held(groups) == {"pr-1": [("a.py", 0), ("b.py", 0)], "pr-2": [("a.py", 1)]}
    _valid(groups)


def test_invented_files_hunks_and_dependencies_are_dropped() -> None:
    groups = repair_plan(
        [
            _group("pr-1", _whole("a.py"), _whole("ghost.py"), deps=["pr-9", "pr-1"]),
            _group("pr-2", _partial("b.py", 0, 7), deps=["pr-1", "pr-1"]),
        ],
        parse_diff(_DIFF),
    )
    assert _held(groups) == {"pr-1": [("a.py", 0), ("a.py", 1)], "pr-2": [("b.py", 0)]}
    assert groups[0].depends_on == []
    assert groups[1].depends_on == ["pr-1"]
    _valid(groups)


def test_emptied_group_is_dropped_and_its_order_kept() -> None:
    groups = repair_plan(
        [
            _group("pr-1", _whole("a.py")),
            _group("pr-2", _partial("a.py", 0), deps=["pr-1"]),
            _group("pr-3", _whole("b.py"), deps=["pr-2"]),
        ],
        parse_diff(_DIFF),
    )
    assert [g.id for g in groups] == ["pr-1", "pr-3"]
    assert groups[1].depends_on == ["pr-1"]
    _valid(groups)


def test_estimated_loc_matches_the_repaired_plan() -> None:
    groups = repair_plan([_group("pr-1", _partial("a.py", 0))], parse_diff(_DIFF))
    assert sum(g.estimated_loc for g in groups) == parse_diff(_DIFF).stats["total_loc"]


@patch("pr_split.planner.client._call_llm")
@patch("pr_split.planner.client._count_tokens", return_value=1)
def test_single_shot_plan_that_skipped_a_hunk_now_validates(
    mock_count: MagicMock, mock_call: MagicMock
) -> None:
    from tests.test_client import _make_settings

    mock_call.return_value = {
        "groups": [
            {
                "id": "pr-1",
                "title": "a",
                "description": "",
                "depends_on": [],
                "assignments": [
                    {"file_path": "a.py", "assignment_type": "partial_hunks", "hunk_indices": [0]}
                ],
                "estimated_loc": 1,
            },
            {
                "id": "pr-2",
                "title": "b",
                "description": "",
                "depends_on": ["pr-7"],
                "assignments": [
                    {"file_path": "b.py", "assignment_type": "whole_file", "hunk_indices": [0]}
                ],
                "estimated_loc": 2,
            },
        ]
    }
    settings = _make_settings(Provider.OPENAI, partition_strategy=PartitionStrategy.LLM)

    groups = _plan_split_with_llm(parse_diff(_DIFF), settings)

    _valid(groups)
    assert groups[1].depends_on == []
