from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pr_split.config import Settings
from pr_split.constants import AssignmentType, Provider
from pr_split.exceptions import LLMError
from pr_split.planner.client import (
    RawToolOutput,
    _call_chunk_with_retry,
    _call_llm,
    _compute_token_ratio,
    _count_tokens,
    _count_tokens_openai,
    _extract_raw_output,
    _merge_chunk_groups,
    _parse_groups,
)
from pr_split.schemas import Group, GroupAssignment

ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PR_SPLIT_PROVIDER",
    "PR_SPLIT_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _raw_group(
    group_id: str = "pr-1",
    title: str = "feat: add feature",
    file_path: str = "src/main.py",
) -> dict:
    return {
        "id": group_id,
        "title": title,
        "description": "some description",
        "depends_on": [],
        "assignments": [
            {
                "file_path": file_path,
                "assignment_type": "whole_file",
                "hunk_indices": [0],
            }
        ],
        "estimated_loc": 50,
    }


def _settings(provider: Provider = Provider.ANTHROPIC) -> Settings:
    match provider:
        case Provider.ANTHROPIC:
            return Settings(ANTHROPIC_API_KEY="sk-ant-test", provider=provider)
        case Provider.OPENAI:
            return Settings(OPENAI_API_KEY="sk-oai-test", provider=provider)


class TestExtractRawOutput:
    def test_valid_groups(self) -> None:
        groups = _extract_raw_output({"groups": [_raw_group()]})
        assert len(groups) == 1
        assert groups[0]["id"] == "pr-1"

    def test_missing_groups_key(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"other": []})

    def test_groups_not_list(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"groups": "not a list"})


class TestParseGroups:
    def test_single_group(self) -> None:
        raw = RawToolOutput(groups=[_raw_group()])
        groups = _parse_groups(raw)
        assert len(groups) == 1
        assert groups[0].id == "pr-1"
        assert groups[0].title == "feat: add feature"
        assert groups[0].assignments[0].assignment_type == AssignmentType.WHOLE_FILE
        assert groups[0].assignments[0].hunk_indices == [0]

    def test_multiple_groups(self) -> None:
        raw = RawToolOutput(groups=[_raw_group("pr-1"), _raw_group("pr-2", title="fix: bug")])
        groups = _parse_groups(raw)
        assert len(groups) == 2
        assert groups[1].id == "pr-2"


class TestMergeChunkGroups:
    def _group(self, group_id: str, file_path: str = "a.py") -> Group:
        return Group(
            id=group_id,
            title=f"group {group_id}",
            description="desc",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path=file_path,
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
            estimated_loc=10,
        )

    def test_new_group_added(self) -> None:
        acc = [self._group("pr-1")]
        new = [self._group("pr-2", file_path="b.py")]
        merged = _merge_chunk_groups(acc, new)
        assert len(merged) == 2

    def test_existing_group_extended(self) -> None:
        acc = [self._group("pr-1", file_path="a.py")]
        new = [self._group("pr-1", file_path="b.py")]
        merged = _merge_chunk_groups(acc, new)
        assert len(merged) == 1
        assert len(merged[0].assignments) == 2

    def test_depends_on_merged(self) -> None:
        g1 = self._group("pr-1")
        g1.depends_on = ["pr-0"]
        g2 = self._group("pr-1")
        g2.depends_on = ["pr-0", "pr-x"]
        merged = _merge_chunk_groups([g1], [g2])
        assert "pr-x" in merged[0].depends_on
        assert merged[0].depends_on.count("pr-0") == 1


class TestComputeTokenRatio:
    def test_normal_ratio(self) -> None:
        diff = MagicMock()
        diff.raw_diff = "a" * 1000
        ratio = _compute_token_ratio(5000, 1000, diff)
        assert ratio == pytest.approx(4.0)

    def test_zero_length_diff(self) -> None:
        diff = MagicMock()
        diff.raw_diff = ""
        ratio = _compute_token_ratio(5000, 1000, diff)
        assert ratio == 0.25


class TestCountTokensOpenai:
    def test_known_model(self) -> None:
        count = _count_tokens_openai(["hello world"], model="gpt-4o")
        assert count > 0

    def test_unknown_model_fallback(self) -> None:
        count = _count_tokens_openai(["hello world"], model="nonexistent-model-xyz")
        assert count > 0

    def test_multiple_texts(self) -> None:
        count_single = _count_tokens_openai(["hello"], model="gpt-4o")
        count_double = _count_tokens_openai(["hello", "world"], model="gpt-4o")
        assert count_double > count_single


class TestCountTokensDispatch:
    @patch("pr_split.planner.client._count_tokens_anthropic", return_value=42)
    def test_dispatches_to_anthropic(self, mock_count: MagicMock) -> None:
        s = _settings(Provider.ANTHROPIC)
        result = _count_tokens("sys", "usr", settings=s)
        assert result == 42
        mock_count.assert_called_once()

    @patch("pr_split.planner.client._count_tokens_openai", return_value=99)
    def test_dispatches_to_openai(self, mock_count: MagicMock) -> None:
        s = _settings(Provider.OPENAI)
        result = _count_tokens("sys", "usr", settings=s)
        assert result == 99
        mock_count.assert_called_once()


class TestCallOpenai:
    def _mock_response(self, arguments: str) -> MagicMock:
        func = MagicMock()
        func.arguments = arguments
        tool_call = MagicMock()
        tool_call.function = func
        message = MagicMock()
        message.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_success(self, mock_openai_cls: MagicMock) -> None:
        import json

        response = self._mock_response(arguments=json.dumps({"groups": [_raw_group()]}))
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        from pr_split.planner.client import _call_openai

        result = _call_openai("sys", "usr", settings=_settings(Provider.OPENAI))
        assert "groups" in result
        assert len(result["groups"]) == 1

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_no_tool_calls(self, mock_openai_cls: MagicMock) -> None:
        message = MagicMock()
        message.tool_calls = []
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        from pr_split.planner.client import _call_openai

        with pytest.raises(LLMError, match="no tool call"):
            _call_openai("sys", "usr", settings=_settings(Provider.OPENAI))

    @patch("pr_split.planner.client.openai.OpenAI")
    def test_invalid_json(self, mock_openai_cls: MagicMock) -> None:
        response = self._mock_response(arguments="not valid json {{{")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        from pr_split.planner.client import _call_openai

        with pytest.raises(LLMError, match="failed to parse tool arguments"):
            _call_openai("sys", "usr", settings=_settings(Provider.OPENAI))


class TestCallLlmDispatch:
    @patch("pr_split.planner.client._call_anthropic")
    def test_dispatches_to_anthropic(self, mock_call: MagicMock) -> None:
        mock_call.return_value = RawToolOutput(groups=[])
        result = _call_llm("sys", "usr", settings=_settings(Provider.ANTHROPIC))
        mock_call.assert_called_once()
        assert result["groups"] == []

    @patch("pr_split.planner.client._call_openai")
    def test_dispatches_to_openai(self, mock_call: MagicMock) -> None:
        mock_call.return_value = RawToolOutput(groups=[])
        result = _call_llm("sys", "usr", settings=_settings(Provider.OPENAI))
        mock_call.assert_called_once()
        assert result["groups"] == []


class TestCallChunkWithRetry:
    @patch("pr_split.planner.client._call_llm")
    def test_success_first_attempt(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = RawToolOutput(groups=[_raw_group()])
        groups = _call_chunk_with_retry(
            "sys", "usr", settings=_settings(), chunk_index=1, total_chunks=1
        )
        assert len(groups) == 1
        assert mock_llm.call_count == 1

    @patch("pr_split.planner.client._call_llm")
    def test_retries_on_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = [
            LLMError("transient"),
            RawToolOutput(groups=[_raw_group()]),
        ]
        groups = _call_chunk_with_retry(
            "sys", "usr", settings=_settings(), chunk_index=1, total_chunks=1
        )
        assert len(groups) == 1
        assert mock_llm.call_count == 2

    @patch("pr_split.planner.client._call_llm")
    def test_exhausts_retries(self, mock_llm: MagicMock) -> None:
        mock_llm.side_effect = LLMError("persistent failure")
        with pytest.raises(LLMError, match="persistent failure"):
            _call_chunk_with_retry(
                "sys", "usr", settings=_settings(), chunk_index=1, total_chunks=1
            )
        assert mock_llm.call_count == 2
