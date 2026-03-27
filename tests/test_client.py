from __future__ import annotations

from unittest.mock import patch

import pytest

from pr_split.config import Settings
from pr_split.constants import AssignmentType, PartitionStrategy, Provider
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import LLMError, PRSplitError
from pr_split.planner.client import (
    RawToolOutput,
    _extract_raw_output,
    _groups_to_raw_dicts,
    _merge_chunk_groups,
    _parse_groups,
    _refine_plan_with_llm,
    plan_split,
)
from pr_split.schemas import Group, GroupAssignment


class TestExtractRawOutput:
    def test_valid_groups(self) -> None:
        block_input = {"groups": [{"id": "pr-1"}]}
        result = _extract_raw_output(block_input)
        assert len(result) == 1
        assert result[0]["id"] == "pr-1"

    def test_missing_groups_raises(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"other_key": 123})

    def test_groups_not_list_raises(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"groups": "not_a_list"})


class TestParseGroups:
    def test_basic_parse(self) -> None:
        raw = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "feat: add auth",
                    "description": "Auth module",
                    "depends_on": [],
                    "assignments": [
                        {
                            "file_path": "auth.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0, 1],
                        }
                    ],
                    "estimated_loc": 50,
                }
            ]
        )
        groups = _parse_groups(raw)
        assert len(groups) == 1
        assert groups[0].id == "pr-1"
        assert groups[0].assignments[0].assignment_type == AssignmentType.WHOLE_FILE
        assert groups[0].assignments[0].hunk_indices == [0, 1]

    def test_multiple_groups(self) -> None:
        raw = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "t1",
                    "description": "d1",
                    "depends_on": [],
                    "assignments": [],
                    "estimated_loc": 10,
                },
                {
                    "id": "pr-2",
                    "title": "t2",
                    "description": "d2",
                    "depends_on": ["pr-1"],
                    "assignments": [],
                    "estimated_loc": 20,
                },
            ]
        )
        groups = _parse_groups(raw)
        assert len(groups) == 2
        assert groups[1].depends_on == ["pr-1"]


class TestMergeChunkGroups:
    def test_new_group_appended(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1")
        g2 = Group(id="pr-2", title="t2", description="d2")
        result = _merge_chunk_groups([g1], [g2])
        assert len(result) == 2

    def test_existing_group_assignments_merged(self) -> None:
        g1 = Group(
            id="pr-1",
            title="t1",
            description="d1",
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                )
            ],
        )
        g1_chunk2 = Group(
            id="pr-1",
            title="t1",
            description="d1",
            assignments=[
                GroupAssignment(
                    file_path="b.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0, 1],
                )
            ],
        )
        result = _merge_chunk_groups([g1], [g1_chunk2])
        assert len(result) == 1
        assert len(result[0].assignments) == 2

    def test_existing_group_deps_merged(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1", depends_on=["pr-0"])
        g1_extra = Group(id="pr-1", title="t1", description="d1", depends_on=["pr-0", "pr-2"])
        result = _merge_chunk_groups([g1], [g1_extra])
        assert len(result) == 1
        assert "pr-2" in result[0].depends_on
        assert result[0].depends_on.count("pr-0") == 1

    def test_empty_accumulated(self) -> None:
        g2 = Group(id="pr-2", title="t2", description="d2")
        result = _merge_chunk_groups([], [g2])
        assert len(result) == 1


class TestExtractRawOutputEdgeCases:
    def test_empty_groups_list(self) -> None:
        result = _extract_raw_output({"groups": []})
        assert result == []

    def test_error_message_includes_available_keys(self) -> None:
        with pytest.raises(LLMError, match=r"alpha.*beta|beta.*alpha"):
            _extract_raw_output({"alpha": 1, "beta": 2})


class TestParseGroupsEdgeCases:
    def test_empty_groups_list(self) -> None:
        raw = RawToolOutput(groups=[])
        assert _parse_groups(raw) == []

    def test_preserves_estimated_loc(self) -> None:
        raw = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "t",
                    "description": "d",
                    "depends_on": [],
                    "assignments": [],
                    "estimated_loc": 42,
                }
            ]
        )
        groups = _parse_groups(raw)
        assert groups[0].estimated_loc == 42


class TestMergeChunkGroupsExtended:
    def test_preserves_order(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1")
        g2 = Group(id="pr-2", title="t2", description="d2")
        g3 = Group(id="pr-3", title="t3", description="d3")
        result = _merge_chunk_groups([g1, g2], [g3])
        assert [g.id for g in result] == ["pr-1", "pr-2", "pr-3"]

    def test_assignments_appended_in_order(self) -> None:
        a1 = GroupAssignment(
            file_path="a.py",
            assignment_type=AssignmentType.WHOLE_FILE,
            hunk_indices=[0],
        )
        a2 = GroupAssignment(
            file_path="b.py",
            assignment_type=AssignmentType.WHOLE_FILE,
            hunk_indices=[0],
        )
        g1 = Group(id="pr-1", title="t", description="d", assignments=[a1])
        g1_chunk2 = Group(id="pr-1", title="t", description="d", assignments=[a2])
        result = _merge_chunk_groups([g1], [g1_chunk2])
        assert result[0].assignments[0].file_path == "a.py"
        assert result[0].assignments[1].file_path == "b.py"

    def test_mixed_new_and_existing(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1")
        g1_update = Group(
            id="pr-1",
            title="t1",
            description="d1",
            assignments=[
                GroupAssignment(
                    file_path="x.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        g2 = Group(id="pr-2", title="t2", description="d2")
        result = _merge_chunk_groups([g1], [g1_update, g2])
        assert len(result) == 2
        assert len(result[0].assignments) == 1


class TestPlanSplitBackendSelection:
    @patch("pr_split.planner.client.partition_diff")
    @patch("pr_split.planner.client.score_plan")
    def test_graph_backend_uses_partition_diff(
        self,
        mock_score,
        mock_partition,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        parsed = parse_diff(
            """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1 @@
+x
"""
        )
        mock_partition.return_value = [Group(id="pr-1", title="t", description="d")]
        mock_score.return_value = type(
            "Metrics",
            (),
            {
                "total_groups": 1,
                "max_group_loc": 1,
                "loc_underflow": 0,
                "loc_overflow": 0,
                "dag_width": 1,
                "dag_depth": 1,
                "file_scatter": 0,
                "objective": 1,
            },
        )()
        settings = Settings(
            provider=Provider.ANTHROPIC,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = plan_split(parsed, settings)
        assert len(groups) == 1
        mock_partition.assert_called_once()

    def test_unsupported_partition_strategy_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        parsed = parse_diff(
            """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1 @@
+x
"""
        )
        settings = Settings(provider=Provider.ANTHROPIC)
        settings.partition_strategy = "invalid"

        with pytest.raises(PRSplitError, match="Unsupported partition strategy"):
            plan_split(parsed, settings)


_TWO_FILE_DIFF = """\
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


def _undersized_groups() -> list[Group]:
    return [
        Group(
            id="pr-1",
            title="feat: add a",
            description="File a",
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
            estimated_loc=3,
            estimated_added=3,
            estimated_removed=0,
        ),
        Group(
            id="pr-2",
            title="feat: add b",
            description="File b",
            depends_on=["pr-1"],
            assignments=[
                GroupAssignment(
                    file_path="b.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
            estimated_loc=4,
            estimated_added=4,
            estimated_removed=0,
        ),
    ]


def _merged_group() -> list[Group]:
    return [
        Group(
            id="pr-1",
            title="feat: add a and b",
            description="Both files",
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="b.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
            estimated_loc=7,
            estimated_added=7,
            estimated_removed=0,
        ),
    ]


class TestRefinePlanWithLlm:
    def test_no_refinement_when_zero_iterations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=5,
            max_loc=10,
            max_refinement_iterations=0,
        )
        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="")
        assert len(result) == 2

    @patch("pr_split.planner.client._call_llm")
    def test_refinement_fixes_violations(
        self, mock_call_llm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=5,
            max_loc=10,
            max_refinement_iterations=2,
        )

        mock_call_llm.return_value = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "feat: add a and b",
                    "description": "Both files",
                    "depends_on": [],
                    "assignments": [
                        {
                            "file_path": "a.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0],
                        },
                        {
                            "file_path": "b.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0],
                        },
                    ],
                    "estimated_loc": 7,
                }
            ]
        )

        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="system")
        assert len(result) == 1
        assert result[0].estimated_loc == 7
        mock_call_llm.assert_called_once()

    def test_no_refinement_when_no_violations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=1,
            max_loc=10,
            max_refinement_iterations=2,
        )
        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="")
        assert len(result) == 2

    @patch("pr_split.planner.client._call_llm")
    def test_refinement_stops_on_llm_error(
        self, mock_call_llm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=5,
            max_loc=10,
            max_refinement_iterations=3,
        )
        mock_call_llm.side_effect = LLMError("API failure")

        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="system")
        assert len(result) == 2
        mock_call_llm.assert_called_once()

    def test_no_refinement_when_min_loc_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            max_loc=10,
            max_refinement_iterations=2,
        )
        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="")
        assert len(result) == 2


class TestGroupsToRawDicts:
    def test_round_trip(self) -> None:
        groups = _undersized_groups()
        raw = _groups_to_raw_dicts(groups)
        assert len(raw) == 2
        assert raw[0]["id"] == "pr-1"
        assert raw[0]["assignments"][0]["file_path"] == "a.py"
        assert raw[0]["assignments"][0]["assignment_type"] == "whole_file"
