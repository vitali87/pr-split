from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.graph import PlanDAG
from pr_split.planner.validator import validate_plan
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


def _ga(path: str, atype: AssignmentType, indices: list[int]) -> GroupAssignment:
    return GroupAssignment(file_path=path, assignment_type=atype, hunk_indices=indices)


def _make_group(
    gid: str, assignments: list[GroupAssignment], loc: int,
    depends_on: list[str] | None = None,
) -> Group:
    return Group(
        id=gid, title=gid, description=gid,
        depends_on=depends_on or [], assignments=assignments, estimated_loc=loc,
    )


class TestValidatePlanIntegration:
    def test_valid_plan_returns_no_warnings(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
            _make_group("g2", [_ga("b.py", WHOLE, [0])], 4),
        ]
        dag = PlanDAG(groups)
        warnings = validate_plan(groups, parsed, dag, 400)
        assert warnings == []

    def test_valid_plan_with_loc_warning(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _make_group("g1", [_ga("a.py", WHOLE, [0])], 3),
            _make_group("g2", [_ga("b.py", WHOLE, [0])], 4),
        ]
        dag = PlanDAG(groups)
        warnings = validate_plan(groups, parsed, dag, 2)
        assert len(warnings) == 2
