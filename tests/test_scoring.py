from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.planner.scoring import score_plan
from pr_split.schemas import Group, GroupAssignment


def _group(
    gid: str,
    file_path: str,
    hunk_indices: list[int],
    loc: int,
    depends_on: list[str] | None = None,
) -> Group:
    return Group(
        id=gid,
        title=gid,
        description=gid,
        depends_on=depends_on or [],
        assignments=[
            GroupAssignment(
                file_path=file_path,
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=hunk_indices,
            )
        ],
        estimated_loc=loc,
    )


class TestScorePlan:
    def test_counts_basic_metrics(self) -> None:
        groups = [
            _group("g1", "a.py", [0], 120),
            _group("g2", "b.py", [0], 80, depends_on=["g1"]),
        ]
        metrics = score_plan(groups, 100)
        assert metrics.total_groups == 2
        assert metrics.max_group_loc == 120
        assert metrics.loc_overflow == 20
        assert metrics.dependency_edges == 1
        assert metrics.dag_depth == 2

    def test_file_scatter_detects_same_file_in_multiple_groups(self) -> None:
        groups = [
            _group("g1", "shared.py", [0], 40),
            _group("g2", "shared.py", [1], 30),
        ]
        metrics = score_plan(groups, 100)
        assert metrics.file_scatter == 1

    def test_tiny_groups_are_counted(self) -> None:
        groups = [
            _group("g1", "a.py", [0], 10),
            _group("g2", "b.py", [0], 90),
        ]
        metrics = score_plan(groups, 100)
        assert metrics.tiny_groups == 1
