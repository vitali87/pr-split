from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pr_split.config import Settings
from pr_split.constants import AssignmentType, PartitionStrategy, Provider
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import LLMError, PRSplitError
from pr_split.planner.client import (
    RawToolOutput,
    _call_anthropic,
    _call_chunk_with_retry,
    _call_llm,
    _call_openai,
    _count_tokens,
    _count_tokens_anthropic,
    _count_tokens_openai,
    _extract_raw_output,
    _groups_to_raw_dicts,
    _merge_chunk_groups,
    _parse_groups,
    _plan_split_chunked,
    _plan_split_with_llm,
    _refine_plan_with_llm,
    plan_split,
)
from pr_split.planner.prompts import SPLIT_TOOL_NAME
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

    def test_unsupported_partition_strategy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_no_refinement_when_zero_iterations(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_no_refinement_when_no_violations(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_no_refinement_when_min_loc_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ---------------------------------------------------------------------------
# Helper to build a Settings object without requiring real API keys
# ---------------------------------------------------------------------------
def _make_settings(
    provider: Provider = Provider.ANTHROPIC,
    partition_strategy: PartitionStrategy = PartitionStrategy.GRAPH,
    **overrides: object,
) -> Settings:
    """Build a Settings bypassing the API-key validator (partition_strategy != LLM)."""
    import os

    env_backup: dict[str, str | None] = {}
    # If LLM strategy is requested, inject a fake key via env so the validator passes
    if partition_strategy == PartitionStrategy.LLM:
        if provider == Provider.ANTHROPIC:
            env_backup["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
            os.environ["ANTHROPIC_API_KEY"] = "sk-fake"
        elif provider == Provider.OPENAI:
            env_backup["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "sk-fake"
    try:
        return Settings(
            provider=provider,
            partition_strategy=partition_strategy,
            **overrides,  # type: ignore[arg-type]
        )
    finally:
        for key, val in env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


_SAMPLE_RAW_GROUPS: list[dict[str, object]] = [
    {
        "id": "pr-1",
        "title": "t",
        "description": "d",
        "depends_on": [],
        "assignments": [],
        "estimated_loc": 1,
    }
]


# ---------------------------------------------------------------------------
# _count_tokens_anthropic
# ---------------------------------------------------------------------------
class TestCountTokensAnthropic:
    @patch("pr_split.planner.client.anthropic.Anthropic")
    def test_returns_input_tokens(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=42)
        settings = _make_settings(Provider.ANTHROPIC)
        result = _count_tokens_anthropic("sys", "usr", settings=settings)
        assert result == 42
        mock_client.messages.count_tokens.assert_called_once()


# ---------------------------------------------------------------------------
# _count_tokens_openai
# ---------------------------------------------------------------------------
class TestCountTokensOpenai:
    @patch("pr_split.planner.client.tiktoken.encoding_for_model")
    def test_known_model(self, mock_enc_for_model: MagicMock) -> None:
        fake_enc = MagicMock()
        fake_enc.encode.side_effect = lambda t: list(range(len(t)))
        mock_enc_for_model.return_value = fake_enc
        result = _count_tokens_openai(["hi", "there"], model="gpt-4o")
        assert result == 7  # 2 + 5

    @patch("pr_split.planner.client.tiktoken.get_encoding")
    @patch(
        "pr_split.planner.client.tiktoken.encoding_for_model",
        side_effect=KeyError("unknown"),
    )
    def test_unknown_model_falls_back(self, _mock_enc: MagicMock, mock_get: MagicMock) -> None:
        fake_enc = MagicMock()
        fake_enc.encode.return_value = [1, 2, 3]
        mock_get.return_value = fake_enc
        result = _count_tokens_openai(["abc"], model="unknown-model")
        assert result == 3
        mock_get.assert_called_once_with("o200k_base")


# ---------------------------------------------------------------------------
# _count_tokens dispatcher
# ---------------------------------------------------------------------------
class TestCountTokensDispatcher:
    @patch("pr_split.planner.client._count_tokens_anthropic", return_value=10)
    def test_anthropic_dispatch(self, mock_fn: MagicMock) -> None:
        settings = _make_settings(Provider.ANTHROPIC)
        result = _count_tokens("s", "u", settings=settings)
        assert result == 10
        mock_fn.assert_called_once()

    @patch("pr_split.planner.client._count_tokens_openai", return_value=20)
    def test_openai_dispatch(self, mock_fn: MagicMock) -> None:
        settings = _make_settings(Provider.OPENAI)
        result = _count_tokens("s", "u", settings=settings)
        assert result == 20
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# _call_anthropic
# ---------------------------------------------------------------------------
class TestCallAnthropic:
    @patch("pr_split.planner.client.anthropic.Anthropic")
    def test_success(self, mock_cls: MagicMock) -> None:
        from anthropic.types.beta import BetaToolUseBlock

        tool_block = BetaToolUseBlock(
            id="tu_1",
            type="tool_use",
            name=SPLIT_TOOL_NAME,
            input={"groups": _SAMPLE_RAW_GROUPS},
        )
        mock_response = SimpleNamespace(
            stop_reason="tool_use",
            content=[tool_block],
        )
        mock_cls.return_value.beta.messages.create.return_value = mock_response
        settings = _make_settings(Provider.ANTHROPIC)
        result = _call_anthropic("sys", "usr", settings=settings)
        assert result["groups"] == _SAMPLE_RAW_GROUPS

    @patch("pr_split.planner.client.anthropic.Anthropic")
    def test_api_error_raises_llm_error(self, mock_cls: MagicMock) -> None:
        import anthropic as _anthropic

        mock_cls.return_value.beta.messages.create.side_effect = _anthropic.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
        settings = _make_settings(Provider.ANTHROPIC)
        with pytest.raises(LLMError):
            _call_anthropic("sys", "usr", settings=settings)

    @patch("pr_split.planner.client.anthropic.Anthropic")
    def test_no_tool_use_block_raises(self, mock_cls: MagicMock) -> None:
        mock_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[],
        )
        mock_cls.return_value.beta.messages.create.return_value = mock_response
        settings = _make_settings(Provider.ANTHROPIC)
        with pytest.raises(LLMError, match="no tool_use block"):
            _call_anthropic("sys", "usr", settings=settings)

    @patch("pr_split.planner.client.anthropic.Anthropic")
    def test_stop_reason_not_tool_use_still_succeeds(self, mock_cls: MagicMock) -> None:
        from anthropic.types.beta import BetaToolUseBlock

        tool_block = BetaToolUseBlock(
            id="tu_1",
            type="tool_use",
            name=SPLIT_TOOL_NAME,
            input={"groups": _SAMPLE_RAW_GROUPS},
        )
        mock_response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[tool_block],
        )
        mock_cls.return_value.beta.messages.create.return_value = mock_response
        settings = _make_settings(Provider.ANTHROPIC)
        result = _call_anthropic("sys", "usr", settings=settings)
        assert result["groups"] == _SAMPLE_RAW_GROUPS


# ---------------------------------------------------------------------------
# _call_openai
# ---------------------------------------------------------------------------
class TestCallOpenai:
    @patch("pr_split.planner.client.openai.OpenAI")
    def test_success(self, mock_cls: MagicMock) -> None:
        fn_item = SimpleNamespace(
            type="function_call",
            name=SPLIT_TOOL_NAME,
            arguments=json.dumps({"groups": _SAMPLE_RAW_GROUPS}),
        )
        mock_response = SimpleNamespace(output=[fn_item])
        mock_cls.return_value.responses.create.return_value = mock_response
        settings = _make_settings(Provider.OPENAI)
        result = _call_openai("sys", "usr", settings=settings)
        assert result["groups"] == _SAMPLE_RAW_GROUPS

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_api_error_raises_llm_error(self, mock_cls: MagicMock) -> None:
        import openai as _openai

        mock_cls.return_value.responses.create.side_effect = _openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
        settings = _make_settings(Provider.OPENAI)
        with pytest.raises(LLMError):
            _call_openai("sys", "usr", settings=settings)

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_no_function_call_raises(self, mock_cls: MagicMock) -> None:
        mock_response = SimpleNamespace(output=[])
        mock_cls.return_value.responses.create.return_value = mock_response
        settings = _make_settings(Provider.OPENAI)
        with pytest.raises(LLMError, match="no function_call"):
            _call_openai("sys", "usr", settings=settings)

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_invalid_json_raises(self, mock_cls: MagicMock) -> None:
        fn_item = SimpleNamespace(
            type="function_call",
            name=SPLIT_TOOL_NAME,
            arguments="not valid json{{{",
        )
        mock_response = SimpleNamespace(output=[fn_item])
        mock_cls.return_value.responses.create.return_value = mock_response
        settings = _make_settings(Provider.OPENAI)
        with pytest.raises(LLMError, match="failed to parse"):
            _call_openai("sys", "usr", settings=settings)


# ---------------------------------------------------------------------------
# _call_llm dispatcher
# ---------------------------------------------------------------------------
class TestCallLlmDispatcher:
    @patch("pr_split.planner.client._call_anthropic")
    def test_dispatches_to_anthropic(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = RawToolOutput(groups=[])
        settings = _make_settings(Provider.ANTHROPIC)
        _call_llm("s", "u", settings=settings)
        mock_fn.assert_called_once()

    @patch("pr_split.planner.client._call_openai")
    def test_dispatches_to_openai(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = RawToolOutput(groups=[])
        settings = _make_settings(Provider.OPENAI)
        _call_llm("s", "u", settings=settings)
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# _call_chunk_with_retry
# ---------------------------------------------------------------------------
class TestCallChunkWithRetry:
    @patch("pr_split.planner.client._call_llm")
    def test_success_first_attempt(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = RawToolOutput(groups=_SAMPLE_RAW_GROUPS)
        settings = _make_settings(Provider.ANTHROPIC)
        groups = _call_chunk_with_retry(
            "sys", "usr", settings=settings, chunk_index=1, total_chunks=2
        )
        assert len(groups) == 1
        assert mock_llm.call_count == 1

    @patch("pr_split.planner.client._call_llm")
    def test_retries_on_llm_error(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = [
            LLMError("fail1"),
            RawToolOutput(groups=_SAMPLE_RAW_GROUPS),
        ]
        settings = _make_settings(Provider.ANTHROPIC)
        groups = _call_chunk_with_retry(
            "sys", "usr", settings=settings, chunk_index=1, total_chunks=1
        )
        assert len(groups) == 1
        assert mock_llm.call_count == 2

    @patch("pr_split.planner.client._call_llm")
    def test_exhausted_retries_raises(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = LLMError("persistent failure")
        settings = _make_settings(Provider.ANTHROPIC)
        with pytest.raises(LLMError, match="persistent failure"):
            _call_chunk_with_retry("sys", "usr", settings=settings, chunk_index=1, total_chunks=1)


# ---------------------------------------------------------------------------
# _plan_split_chunked
# ---------------------------------------------------------------------------
class TestPlanSplitChunked:
    @patch("pr_split.planner.client.recompute_estimated_loc")
    @patch("pr_split.planner.client.assign_uncovered_hunks", return_value=0)
    @patch("pr_split.planner.client._call_chunk_with_retry")
    @patch("pr_split.planner.client.build_chunk_first_prompt", return_value="prompt")
    @patch("pr_split.planner.client.build_chunk_stats_from_hunks", return_value="stats")
    @patch("pr_split.planner.client.build_chunk_diff_from_hunks", return_value="diff")
    @patch("pr_split.planner.client.chunk_hunks")
    @patch("pr_split.planner.client.build_hunk_sequence")
    @patch("pr_split.planner.client._count_tokens", return_value=100)
    def test_single_chunk(
        self,
        mock_count: MagicMock,
        mock_seq: MagicMock,
        mock_chunk: MagicMock,
        mock_diff: MagicMock,
        mock_stats: MagicMock,
        mock_first: MagicMock,
        mock_call: MagicMock,
        mock_assign: MagicMock,
        mock_recompute: MagicMock,
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(Provider.ANTHROPIC)

        hunk = SimpleNamespace(token_estimate=50)
        mock_seq.return_value = [hunk]
        mock_chunk.return_value = [[hunk]]
        group = Group(id="pr-1", title="t", description="d", estimated_loc=3)
        mock_call.return_value = [group]

        result = _plan_split_chunked(parsed, settings, "system", 500)
        assert len(result) == 1
        assert result[0].id == "pr-1"
        mock_call.assert_called_once()

    @patch("pr_split.planner.client.recompute_estimated_loc")
    @patch("pr_split.planner.client.assign_uncovered_hunks", return_value=2)
    @patch("pr_split.planner.client._call_chunk_with_retry")
    @patch("pr_split.planner.client.format_group_catalog", return_value="catalog")
    @patch("pr_split.planner.client.build_chunk_continuation_prompt", return_value="cont")
    @patch("pr_split.planner.client.build_chunk_first_prompt", return_value="prompt")
    @patch("pr_split.planner.client.build_chunk_stats_from_hunks", return_value="stats")
    @patch("pr_split.planner.client.build_chunk_diff_from_hunks", return_value="diff")
    @patch("pr_split.planner.client.chunk_hunks")
    @patch("pr_split.planner.client.build_hunk_sequence")
    @patch("pr_split.planner.client._count_tokens", return_value=100)
    def test_multi_chunk_merges(
        self,
        mock_count: MagicMock,
        mock_seq: MagicMock,
        mock_chunk: MagicMock,
        mock_diff: MagicMock,
        mock_stats: MagicMock,
        mock_first: MagicMock,
        mock_cont: MagicMock,
        mock_catalog: MagicMock,
        mock_call: MagicMock,
        mock_assign: MagicMock,
        mock_recompute: MagicMock,
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(Provider.ANTHROPIC)

        h1 = SimpleNamespace(token_estimate=50)
        h2 = SimpleNamespace(token_estimate=60)
        mock_seq.return_value = [h1, h2]
        mock_chunk.return_value = [[h1], [h2]]

        g1 = Group(id="pr-1", title="t1", description="d1", estimated_loc=3)
        g2 = Group(id="pr-2", title="t2", description="d2", estimated_loc=4)
        mock_call.side_effect = [[g1], [g2]]

        result = _plan_split_chunked(parsed, settings, "system", 500)
        assert len(result) == 2
        assert mock_call.call_count == 2


# ---------------------------------------------------------------------------
# _refine_plan_with_llm — remaining lines (389-403)
# ---------------------------------------------------------------------------
class TestRefinePlanExhausted:
    @patch("pr_split.planner.client._call_llm")
    def test_exhausted_with_remaining_violations(
        self, mock_call_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=5,
            max_loc=10,
            max_refinement_iterations=1,
        )
        # LLM returns same undersized groups so violations persist
        mock_call_llm.return_value = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "feat: add a",
                    "description": "File a",
                    "depends_on": [],
                    "assignments": [
                        {
                            "file_path": "a.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0],
                        }
                    ],
                    "estimated_loc": 3,
                },
                {
                    "id": "pr-2",
                    "title": "feat: add b",
                    "description": "File b",
                    "depends_on": ["pr-1"],
                    "assignments": [
                        {
                            "file_path": "b.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0],
                        }
                    ],
                    "estimated_loc": 4,
                },
            ]
        )
        groups = _undersized_groups()
        result = _refine_plan_with_llm(groups, parsed, settings, system="system")
        assert len(result) == 2
        mock_call_llm.assert_called_once()

    @patch("pr_split.planner.client._call_llm")
    def test_resolved_after_max_iterations(
        self, mock_call_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=5,
            max_loc=10,
            max_refinement_iterations=1,
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
        mock_call_llm.assert_called_once()


# ---------------------------------------------------------------------------
# _plan_split_with_llm
# ---------------------------------------------------------------------------
class TestPlanSplitWithLlm:
    @patch("pr_split.planner.client._refine_plan_with_llm")
    @patch("pr_split.planner.client.recompute_estimated_loc")
    @patch("pr_split.planner.client._call_llm")
    @patch("pr_split.planner.client._count_tokens", return_value=100)
    def test_small_diff_direct_call(
        self,
        mock_count: MagicMock,
        mock_llm: MagicMock,
        mock_recompute: MagicMock,
        mock_refine: MagicMock,
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(Provider.ANTHROPIC, partition_strategy=PartitionStrategy.LLM)
        mock_llm.return_value = RawToolOutput(groups=_SAMPLE_RAW_GROUPS)
        mock_refine.side_effect = lambda groups, *a, **kw: groups

        result = _plan_split_with_llm(parsed, settings)
        assert len(result) == 1
        mock_llm.assert_called_once()

    @patch("pr_split.planner.client._refine_plan_with_llm")
    @patch("pr_split.planner.client._plan_split_chunked")
    @patch("pr_split.planner.client._count_tokens", return_value=999_999_999)
    def test_large_diff_uses_chunking(
        self,
        mock_count: MagicMock,
        mock_chunked: MagicMock,
        mock_refine: MagicMock,
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(Provider.ANTHROPIC, partition_strategy=PartitionStrategy.LLM)
        group = Group(id="pr-1", title="t", description="d", estimated_loc=7)
        mock_chunked.return_value = [group]
        mock_refine.side_effect = lambda groups, *a, **kw: groups

        result = _plan_split_with_llm(parsed, settings)
        assert len(result) == 1
        mock_chunked.assert_called_once()


class TestChunkedPlanningSkipsRefinement:
    @patch("pr_split.planner.client._call_llm")
    @patch("pr_split.planner.client._plan_split_chunked")
    @patch("pr_split.planner.client._count_tokens", return_value=10**9)
    def test_no_refinement_call_after_chunking(
        self, mock_count: MagicMock, mock_chunked: MagicMock, mock_call: MagicMock
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(
            Provider.ANTHROPIC, max_refinement_iterations=2, min_loc=50, max_loc=100
        )
        oversized = Group(id="pr-1", title="t", description="d", estimated_loc=500)
        mock_chunked.return_value = [oversized]

        result = _plan_split_with_llm(parsed, settings)

        assert result == [oversized]
        mock_call.assert_not_called()

    @patch("pr_split.planner.client._refine_plan_with_llm")
    @patch("pr_split.planner.client._call_llm")
    @patch("pr_split.planner.client._count_tokens", return_value=1)
    def test_single_shot_planning_still_refines(
        self, mock_count: MagicMock, mock_call: MagicMock, mock_refine: MagicMock
    ) -> None:
        parsed = parse_diff(_TWO_FILE_DIFF)
        settings = _make_settings(Provider.ANTHROPIC, max_refinement_iterations=2)
        mock_call.return_value = {"groups": []}
        mock_refine.return_value = []

        _plan_split_with_llm(parsed, settings)

        mock_refine.assert_called_once()
