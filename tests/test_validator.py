from __future__ import annotations

import pytest

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import PlanValidationError
from pr_split.graph import PlanDAG
from pr_split.planner.validator import (
    validate_coverage,
    validate_loc,
    validate_loc_bounds,
    validate_no_conflicts,
)
from pr_split.schemas import Group, GroupAssignment

SAMPLE_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,4 @@
+lineA
+lineB
+lineC
+lineD
"""

WHOLE = AssignmentType.WHOLE_FILE
PARTIAL = AssignmentType.PARTIAL_HUNKS


def _ga(path: str, atype: AssignmentType, indices: list[int]) -> GroupAssignment:
    return GroupAssignment(
        file_path=path,
        assignment_type=atype,
        hunk_indices=indices,
    )


def _make_group(
    gid: str,
    assignments: list[GroupAssignment],
    loc: int,
    depends_on: list[str] | None = None,
) -> Group:
    return Group(
        id=gid,
        title=gid,
        description=gid,
        depends_on=depends_on or [],
        assignments=assignments,
        estimated_loc=loc,
    )


class TestValidateCoverage:
    def test_full_coverage_passes(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
            _make_group("g2", [_ga("b.py", WHOLE, [0])], 4),
        ]
        validate_coverage(groups, parsed)

    def test_missing_hunk_raises(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
        ]
        with pytest.raises(PlanValidationError, match="not assigned"):
            validate_coverage(groups, parsed)

    def test_duplicate_assignment_raises(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
            _make_group(
                "g2",
                [_ga("a.py", WHOLE, [0]), _ga("b.py", WHOLE, [0])],
                7,
            ),
        ]
        with pytest.raises(PlanValidationError, match="multiple groups"):
            validate_coverage(groups, parsed)


class TestValidateLoc:
    def test_matching_loc_passes(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [], 3),
            _make_group("g2", [], 4),
        ]
        validate_loc(groups, parsed)

    def test_mismatched_loc_raises(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [], 5),
            _make_group("g2", [], 5),
        ]
        with pytest.raises(PlanValidationError, match="does not match"):
            validate_loc(groups, parsed)


class TestValidateNoConflicts:
    def test_independent_disjoint_files_passes(self) -> None:
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
            _make_group("g2", [_ga("b.py", WHOLE, [0])], 4),
        ]
        dag = PlanDAG(groups)
        validate_no_conflicts(groups, dag)

    def test_independent_overlapping_hunks_raises(self) -> None:
        groups = [
            _make_group("g1", [_ga("a.py", PARTIAL, [0])], 3),
            _make_group("g2", [_ga("a.py", PARTIAL, [0])], 3),
        ]
        dag = PlanDAG(groups)
        with pytest.raises(PlanValidationError, match="overlapping"):
            validate_no_conflicts(groups, dag)

    def test_dependent_groups_skip_conflict_check(self) -> None:
        groups = [
            _make_group("g1", [_ga("a.py", PARTIAL, [0])], 3),
            _make_group(
                "g2",
                [_ga("a.py", PARTIAL, [0])],
                3,
                depends_on=["g1"],
            ),
        ]
        dag = PlanDAG(groups)
        validate_no_conflicts(groups, dag)


class TestValidateLocBounds:
    def test_within_bounds_no_warnings(self) -> None:
        groups = [_make_group("g1", [], 100)]
        warnings = validate_loc_bounds(groups, 400)
        assert warnings == []

    def test_exceeding_bounds_returns_warnings(self) -> None:
        groups = [_make_group("g1", [], 500)]
        warnings = validate_loc_bounds(groups, 400)
        assert len(warnings) == 1
        assert "g1" in warnings[0]
