from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.exceptions import LLMError
from pr_split.planner.client import (
    RawToolOutput,
    _extract_raw_output,
    _merge_chunk_groups,
    _parse_groups,
)
from pr_split.schemas import Group, GroupAssignment

import pytest


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
