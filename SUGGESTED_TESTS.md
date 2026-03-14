# Suggested Tests

These tests were used as temporary harness tests during a replace-wheels audit.
They can be permanently adopted into the test suite if desired.

## git_ops/branches.py

```python
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.branches import (
    checkout_branch,
    checkout_new_branch,
    is_worktree_clean,
    run_git,
)


class TestRunGitSubprocessWrapper:
    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_passes_args_to_git(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "log", "--oneline"],
            returncode=0,
            stdout="abc123 msg\n",
            stderr="",
        )
        result = run_git("log", "--oneline")
        mock_run.assert_called_once_with(
            ["git", "log", "--oneline"], capture_output=True, text=True
        )
        assert result == "abc123 msg"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_strips_trailing_whitespace(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="  result  \n\n", stderr=""
        )
        assert run_git("status") == "result"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_nonzero_exit_raises_with_stderr(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="  fatal: bad ref  \n"
        )
        with pytest.raises(GitOperationError, match="fatal: bad ref"):
            run_git("rev-parse", "INVALID")

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_empty_stderr_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=1, stdout="", stderr=""
        )
        with pytest.raises(GitOperationError):
            run_git("fail")

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_multiple_args_forwarded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="ok", stderr=""
        )
        run_git("commit", "-m", "message", "--author", "Test <t@t.com>")
        call_args = mock_run.call_args[0][0]
        assert call_args == ["git", "commit", "-m", "message", "--author", "Test <t@t.com>"]


class TestIsWorktreeCleanParsing:
    @patch("pr_split.git_ops.branches.run_git")
    def test_mixed_untracked_and_modified(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "?? file.txt\n M dirty.py"
        assert is_worktree_clean() is False

    @patch("pr_split.git_ops.branches.run_git")
    def test_multiple_untracked_only(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "?? a.txt\n?? b.txt\n?? c.txt"
        assert is_worktree_clean() is True

    @patch("pr_split.git_ops.branches.run_git")
    def test_staged_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "A  new_file.py"
        assert is_worktree_clean() is False

    @patch("pr_split.git_ops.branches.run_git")
    def test_deleted_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = " D deleted.py"
        assert is_worktree_clean() is False

    @patch("pr_split.git_ops.branches.run_git")
    def test_renamed_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "R  old.py -> new.py"
        assert is_worktree_clean() is False


class TestCheckoutWrappers:
    @patch("pr_split.git_ops.branches.run_git")
    def test_checkout_new_branch_args(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        checkout_new_branch("feature/x", "abc123")
        mock_git.assert_called_once_with("checkout", "-b", "feature/x", "abc123")

    @patch("pr_split.git_ops.branches.run_git")
    def test_checkout_branch_args(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        checkout_branch("main")
        mock_git.assert_called_once_with("checkout", "main")
```

## git_ops/prs.py

```python
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.prs import _run_gh, create_pr, fetch_fork_pr


class TestRunGhSubprocessWrapper:
    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_passes_args_to_gh(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh", "api", "repos"], returncode=0, stdout="ok\n", stderr=""
        )
        result = _run_gh("api", "repos")
        mock_run.assert_called_once_with(
            ["gh", "api", "repos"], capture_output=True, text=True
        )
        assert result == "ok"

    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_strips_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh"], returncode=0, stdout="  data  \n", stderr=""
        )
        assert _run_gh("test") == "data"

    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_failure_uses_stderr(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh"], returncode=1, stdout="", stderr="  rate limited  \n"
        )
        with pytest.raises(GitOperationError, match="rate limited"):
            _run_gh("api", "endpoint")


class TestCreatePrUrlParsing:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_extracts_number_from_standard_url(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/123"
        number, url = create_pr("head", "base", "Title", "Body")
        assert number == 123
        assert url == "https://github.com/org/repo/pull/123"

    @patch("pr_split.git_ops.prs._run_gh")
    def test_extracts_number_from_url_with_trailing_slash(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/99/"
        number, url = create_pr("head", "base", "Title", "Body")
        assert number == 99

    @patch("pr_split.git_ops.prs._run_gh")
    def test_multiline_output_uses_last_line(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = (
            "Creating PR...\nhttps://github.com/org/repo/pull/55"
        )
        number, url = create_pr("head", "base", "Title", "Body")
        assert number == 55
        assert url == "https://github.com/org/repo/pull/55"

    @patch("pr_split.git_ops.prs._run_gh")
    def test_wraps_error_with_group_detail(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("API error")
        with pytest.raises(GitOperationError, match="Failed to create PR"):
            create_pr("my-branch", "main", "Title", "Body")


class TestFetchForkPr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_non_fork_raises(self, mock_gh: MagicMock) -> None:
        import json

        pr_data = {
            "head": {
                "ref": "feature",
                "repo": {"fork": False, "clone_url": "https://x", "full_name": "u/r"},
            },
            "base": {"ref": "main"},
        }
        mock_gh.return_value = json.dumps(pr_data)
        with pytest.raises(GitOperationError):
            fetch_fork_pr(42)

    @patch("pr_split.git_ops.prs._run_gh")
    def test_api_failure_raises(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("Not Found")
        with pytest.raises(GitOperationError):
            fetch_fork_pr(999)

    @patch("pr_split.git_ops.prs._run_gh")
    def test_invalid_head_structure_raises(self, mock_gh: MagicMock) -> None:
        import json

        mock_gh.return_value = json.dumps({"head": "not_a_dict", "base": {"ref": "main"}})
        with pytest.raises(GitOperationError):
            fetch_fork_pr(42)
```

## planner/client.py

```python
from __future__ import annotations

import pytest

from pr_split.constants import AssignmentType
from pr_split.exceptions import LLMError
from pr_split.planner.client import (
    RawToolOutput,
    _extract_raw_output,
    _merge_chunk_groups,
    _parse_groups,
)
from pr_split.schemas import Group, GroupAssignment


class TestExtractRawOutputEdgeCases:
    def test_empty_groups_list(self) -> None:
        result = _extract_raw_output({"groups": []})
        assert result == []

    def test_none_value_raises(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"groups": None})

    def test_dict_value_raises(self) -> None:
        with pytest.raises(LLMError, match="missing 'groups'"):
            _extract_raw_output({"groups": {"id": "pr-1"}})

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(LLMError):
            _extract_raw_output({})

    def test_error_message_includes_available_keys(self) -> None:
        with pytest.raises(LLMError, match="alpha.*beta|beta.*alpha"):
            _extract_raw_output({"alpha": 1, "beta": 2})


class TestParseGroupsEdgeCases:
    def test_group_with_multiple_assignments(self) -> None:
        raw = RawToolOutput(
            groups=[
                {
                    "id": "pr-1",
                    "title": "feat: multi",
                    "description": "Multiple files",
                    "depends_on": ["pr-0"],
                    "assignments": [
                        {
                            "file_path": "a.py",
                            "assignment_type": "whole_file",
                            "hunk_indices": [0],
                        },
                        {
                            "file_path": "b.py",
                            "assignment_type": "partial_hunks",
                            "hunk_indices": [0, 2, 3],
                        },
                    ],
                    "estimated_loc": 100,
                }
            ]
        )
        groups = _parse_groups(raw)
        assert len(groups) == 1
        assert len(groups[0].assignments) == 2
        assert groups[0].assignments[0].assignment_type == AssignmentType.WHOLE_FILE
        assert groups[0].assignments[1].assignment_type == AssignmentType.PARTIAL_HUNKS
        assert groups[0].assignments[1].hunk_indices == [0, 2, 3]
        assert groups[0].depends_on == ["pr-0"]

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


class TestMergeChunkGroupsEdgeCases:
    def test_both_empty(self) -> None:
        assert _merge_chunk_groups([], []) == []

    def test_preserves_order(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1")
        g2 = Group(id="pr-2", title="t2", description="d2")
        g3 = Group(id="pr-3", title="t3", description="d3")
        result = _merge_chunk_groups([g1, g2], [g3])
        assert [g.id for g in result] == ["pr-1", "pr-2", "pr-3"]

    def test_duplicate_deps_not_doubled(self) -> None:
        g1 = Group(id="pr-1", title="t", description="d", depends_on=["pr-0", "pr-2"])
        g1_update = Group(id="pr-1", title="t", description="d", depends_on=["pr-0", "pr-3"])
        result = _merge_chunk_groups([g1], [g1_update])
        assert len(result) == 1
        deps = result[0].depends_on
        assert deps.count("pr-0") == 1
        assert "pr-2" in deps
        assert "pr-3" in deps

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
```

## config.py

```python
from __future__ import annotations

import pytest

from pr_split.config import Settings
from pr_split.constants import (
    ANTHROPIC_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    OPENAI_MAX_CONTEXT_TOKENS,
    OPENAI_MODEL,
    Provider,
)


class TestSettingsProviderDispatch:
    def test_anthropic_api_key_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.api_key == "sk-ant-test"

    def test_openai_api_key_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        s = Settings(provider=Provider.OPENAI)
        assert s.api_key == "sk-oai-test"

    def test_anthropic_max_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.max_context_tokens == ANTHROPIC_MAX_CONTEXT_TOKENS

    def test_openai_max_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        s = Settings(provider=Provider.OPENAI)
        assert s.max_context_tokens == OPENAI_MAX_CONTEXT_TOKENS


class TestSettingsDefaultModel:
    def test_anthropic_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.model == DEFAULT_MODEL

    def test_openai_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        s = Settings(provider=Provider.OPENAI)
        assert s.model == OPENAI_MODEL

    def test_explicit_model_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        s = Settings(provider=Provider.ANTHROPIC, model="claude-3-haiku-20240307")
        assert s.model == "claude-3-haiku-20240307"


class TestSettingsValidation:
    def test_anthropic_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            Settings(provider=Provider.ANTHROPIC)

    def test_openai_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings(provider=Provider.OPENAI)

    def test_empty_string_key_raises_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            Settings(provider=Provider.ANTHROPIC)

    def test_empty_string_key_raises_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings(provider=Provider.OPENAI)
```

## cli.py

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pr_split.cli import _render_dag, _render_dag_markdown, _resolve_fork_ref
from pr_split.schemas import Group


class TestRenderDagMarkdownStringBuilding:
    def test_single_root_no_deps(self) -> None:
        groups = [Group(id="pr-1", title="base", description="d")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "pr-1" in result
        assert "base" in result
        assert "<-- this PR" in result

    def test_linear_chain_marks_current(self) -> None:
        g1 = Group(id="pr-1", title="first", description="d")
        g2 = Group(id="pr-2", title="second", description="d", depends_on=["pr-1"])
        result = _render_dag_markdown([g1, g2], "pr-2")
        assert "pr-2" in result
        assert "<-- this PR" in result
        lines = result.splitlines()
        pr1_lines = [l for l in lines if "pr-1" in l]
        for line in pr1_lines:
            assert "<-- this PR" not in line

    def test_contains_dependency_graph_header(self) -> None:
        groups = [Group(id="pr-1", title="t", description="d")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "## Dependency graph" in result
        assert "Merge in this order" in result

    def test_code_block_present(self) -> None:
        groups = [Group(id="pr-1", title="t", description="d")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "```" in result

    def test_diamond_shape(self) -> None:
        g1 = Group(id="pr-1", title="root", description="d")
        g2 = Group(id="pr-2", title="left", description="d", depends_on=["pr-1"])
        g3 = Group(id="pr-3", title="right", description="d", depends_on=["pr-1"])
        g4 = Group(id="pr-4", title="merge", description="d", depends_on=["pr-2", "pr-3"])
        result = _render_dag_markdown([g1, g2, g3, g4], "pr-4")
        assert "pr-4" in result
        assert "pr-1" in result


class TestRenderDagRichTree:
    def test_output_is_string(self) -> None:
        groups = [Group(id="pr-1", title="root", description="d")]
        result = _render_dag(groups)
        assert isinstance(result, str)
        assert "pr-1" in result

    def test_contains_tree_label(self) -> None:
        groups = [Group(id="pr-1", title="root", description="d")]
        result = _render_dag(groups)
        assert "Split Plan" in result

    def test_deps_shown(self) -> None:
        g1 = Group(id="pr-1", title="root", description="d")
        g2 = Group(id="pr-2", title="child", description="d", depends_on=["pr-1"])
        result = _render_dag([g1, g2])
        assert "pr-2" in result
        assert "depends on" in result


class TestResolveForkRef:
    @patch("pr_split.cli.fetch_fork_pr")
    def test_pr_number_format(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": 42, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("#42")
        mock_fetch.assert_called_once_with(42)
        assert result is not None

    @patch("pr_split.cli.fetch_fork_branch")
    def test_colon_format(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": None, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("user:branch")
        mock_fetch.assert_called_once_with("user", "branch")
        assert result is not None

    def test_plain_branch_returns_none(self) -> None:
        result = _resolve_fork_ref("feature/auth")
        assert result is None

    @patch("pr_split.cli.fetch_fork_pr")
    def test_bare_number_treated_as_pr(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": 7, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("7")
        mock_fetch.assert_called_once_with(7)
```

## planner/chunker.py

```python
from __future__ import annotations

import pytest

from pr_split.constants import AssignmentType
from pr_split.planner.chunker import (
    chunk_hunks,
    format_group_catalog,
)
from pr_split.schemas import Group, GroupAssignment
from pr_split.types_defs import HunkRef


class TestChunkHunksAlgorithm:
    def test_exact_fit_single_chunk(self) -> None:
        refs = [
            HunkRef(file_path="a.py", hunk_index=0, token_estimate=50),
            HunkRef(file_path="a.py", hunk_index=1, token_estimate=50),
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_just_over_budget_splits(self) -> None:
        refs = [
            HunkRef(file_path="a.py", hunk_index=0, token_estimate=60),
            HunkRef(file_path="a.py", hunk_index=1, token_estimate=60),
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 2

    def test_single_hunk_per_chunk_when_tight(self) -> None:
        refs = [
            HunkRef(file_path=f"f{i}.py", hunk_index=0, token_estimate=90)
            for i in range(3)
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 3
        assert all(len(c) == 1 for c in result)

    def test_hunk_exceeding_budget_raises(self) -> None:
        refs = [HunkRef(file_path="big.py", hunk_index=0, token_estimate=200)]
        with pytest.raises(ValueError, match="big.py"):
            chunk_hunks(refs, 100)

    def test_empty_sequence(self) -> None:
        assert chunk_hunks([], 100) == []

    def test_many_small_hunks_pack_efficiently(self) -> None:
        refs = [
            HunkRef(file_path=f"f{i}.py", hunk_index=0, token_estimate=10)
            for i in range(10)
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 1
        assert len(result[0]) == 10


class TestFormatGroupCatalogStringBuilding:
    def test_includes_group_id_and_title(self) -> None:
        g = Group(id="pr-1", title="feat: auth", description="Auth module")
        result = format_group_catalog([g])
        assert "pr-1" in result
        assert "feat: auth" in result

    def test_includes_deps(self) -> None:
        g = Group(
            id="pr-2",
            title="feat: api",
            description="API module",
            depends_on=["pr-1"],
        )
        result = format_group_catalog([g])
        assert "depends on" in result
        assert "pr-1" in result

    def test_includes_file_paths(self) -> None:
        g = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="auth.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="models.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
        )
        result = format_group_catalog([g])
        assert "auth.py" in result
        assert "models.py" in result

    def test_empty_groups(self) -> None:
        result = format_group_catalog([])
        assert result == ""

    def test_multiple_groups_separated(self) -> None:
        g1 = Group(id="pr-1", title="t1", description="d1")
        g2 = Group(id="pr-2", title="t2", description="d2")
        result = format_group_catalog([g1, g2])
        assert "pr-1" in result
        assert "pr-2" in result
        lines = result.strip().split("\n")
        assert any(line == "" for line in lines)
```

## diff_ops/reconstructor.py

```python
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.diff_ops.reconstructor import _get_base_file_content
from pr_split.exceptions import GitOperationError


class TestGetBaseFileContentSubprocess:
    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_success_returns_content(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "show"], returncode=0, stdout="file content\n", stderr=""
        )
        result = _get_base_file_content("src/main.py", "abc123")
        assert result == "file content\n"
        mock_run.assert_called_once_with(
            ["git", "show", "abc123:src/main.py"],
            capture_output=True,
            text=True,
        )

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "show"],
            returncode=128,
            stdout="",
            stderr="fatal: path not found",
        )
        with pytest.raises(GitOperationError, match="path not found"):
            _get_base_file_content("nonexistent.py", "abc123")

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_empty_file_returns_empty(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "show"], returncode=0, stdout="", stderr=""
        )
        result = _get_base_file_content("empty.py", "abc123")
        assert result == ""
```

## diff_ops/parser.py

```python
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.diff_ops.parser import ParsedDiff, extract_diff, parse_diff
from pr_split.exceptions import GitOperationError

SAMPLE_DIFF = """\
diff --git a/hello.py b/hello.py
new file mode 100644
--- /dev/null
+++ b/hello.py
@@ -0,0 +1,3 @@
+def hello():
+    return "hello"
+
"""


class TestExtractDiffSubprocess:
    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_success_returns_stdout(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0, stdout=SAMPLE_DIFF, stderr=""
        )
        result = extract_diff("feature", "main")
        assert result == SAMPLE_DIFF
        mock_run.assert_called_once_with(
            ["git", "diff", "main...feature"],
            capture_output=True,
            text=True,
        )

    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"],
            returncode=1,
            stdout="",
            stderr="fatal: bad revision",
        )
        with pytest.raises(GitOperationError, match="bad revision"):
            extract_diff("bad-branch", "main")


class TestParseDiffFunction:
    def test_valid_diff_returns_parsed(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        assert isinstance(parsed, ParsedDiff)
        assert len(parsed.file_paths) == 1
        assert parsed.file_paths[0] == "hello.py"

    def test_empty_diff_returns_empty(self) -> None:
        parsed = parse_diff("")
        assert len(parsed.file_paths) == 0

    def test_raw_diff_preserved(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        assert parsed.raw_diff == SAMPLE_DIFF


class TestParsedDiffLabeledDiff:
    def test_labeled_diff_contains_hunk_index(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        labeled = parsed.labeled_diff
        assert "[hunk_index=0]" in labeled

    def test_labeled_diff_contains_file_header(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        labeled = parsed.labeled_diff
        assert "hello.py" in labeled


class TestParsedDiffStats:
    def test_stats_counts(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        assert stats["total_files"] == 1
        assert stats["total_added"] > 0
        assert stats["total_loc"] == stats["total_added"] + stats["total_removed"]
```

## schemas.py

```python
from __future__ import annotations

import hashlib

import pytest

from pr_split.constants import AssignmentType
from pr_split.schemas import Group, GroupAssignment


class TestGroupPatchHash:
    def test_compute_patch_hash_sha256(self) -> None:
        g = Group(id="pr-1", title="t", description="d", expected_patch="diff content")
        expected = hashlib.sha256(b"diff content").hexdigest()
        assert g.compute_patch_hash() == expected

    def test_auto_sync_on_creation(self) -> None:
        patch = "some patch data"
        g = Group(id="pr-1", title="t", description="d", expected_patch=patch)
        assert g.expected_patch_sha256 == hashlib.sha256(patch.encode()).hexdigest()

    def test_no_patch_no_hash(self) -> None:
        g = Group(id="pr-1", title="t", description="d")
        assert g.expected_patch_sha256 == ""

    def test_explicit_hash_preserved(self) -> None:
        g = Group(
            id="pr-1",
            title="t",
            description="d",
            expected_patch="data",
            expected_patch_sha256="custom_hash",
        )
        assert g.expected_patch_sha256 == "custom_hash"


class TestGroupAssignmentModel:
    def test_default_hunk_indices_empty(self) -> None:
        a = GroupAssignment(
            file_path="f.py",
            assignment_type=AssignmentType.WHOLE_FILE,
        )
        assert a.hunk_indices == []

    def test_assignment_serialization(self) -> None:
        a = GroupAssignment(
            file_path="f.py",
            assignment_type=AssignmentType.PARTIAL_HUNKS,
            hunk_indices=[1, 3, 5],
        )
        d = a.model_dump()
        assert d["file_path"] == "f.py"
        assert d["assignment_type"] == "partial_hunks"
        assert d["hunk_indices"] == [1, 3, 5]
```

## constants.py

```python
from __future__ import annotations

import pytest

from pr_split.constants import (
    BRANCH_PREFIX,
    CHUNK_RETRY_LIMIT,
    CHUNK_TARGET_RATIO,
    DEFAULT_MAX_LOC,
    FORK_REF_PREFIX,
    MAX_OUTPUT_TOKENS,
    PR_REF_PREFIX,
    AssignmentType,
    PRState,
    Priority,
    Provider,
)


class TestStrEnumValues:
    def test_assignment_type_values(self) -> None:
        assert AssignmentType.WHOLE_FILE == "whole_file"
        assert AssignmentType.PARTIAL_HUNKS == "partial_hunks"

    def test_provider_values(self) -> None:
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"

    def test_priority_values(self) -> None:
        assert Priority.ORTHOGONAL == "orthogonal"
        assert Priority.LOGICAL == "logical"

    def test_pr_state_values(self) -> None:
        assert PRState.OPEN == "open"
        assert PRState.CLOSED == "closed"
        assert PRState.MERGED == "merged"

    def test_str_enum_isinstance_str(self) -> None:
        assert isinstance(AssignmentType.WHOLE_FILE, str)
        assert isinstance(Provider.ANTHROPIC, str)
        assert isinstance(Priority.ORTHOGONAL, str)
        assert isinstance(PRState.OPEN, str)


class TestConstantValues:
    def test_branch_prefix(self) -> None:
        assert BRANCH_PREFIX == "pr-split/"

    def test_ref_prefixes(self) -> None:
        assert PR_REF_PREFIX.startswith("refs/")
        assert FORK_REF_PREFIX.startswith("refs/")

    def test_chunk_ratio_between_zero_and_one(self) -> None:
        assert 0 < CHUNK_TARGET_RATIO < 1

    def test_retry_limit_positive(self) -> None:
        assert CHUNK_RETRY_LIMIT > 0

    def test_max_output_tokens_positive(self) -> None:
        assert MAX_OUTPUT_TOKENS > 0

    def test_default_max_loc_positive(self) -> None:
        assert DEFAULT_MAX_LOC > 0
```

## exceptions.py

```python
from __future__ import annotations

import pytest

from pr_split.exceptions import (
    DiffParseError,
    ErrorMsg,
    GitOperationError,
    LLMError,
    PlanValidationError,
    PRSplitError,
)


class TestErrorMsgFormatting:
    def test_format_with_kwargs(self) -> None:
        result = ErrorMsg.COVERAGE_GAP(file="a.py", index=3)
        assert "a.py" in result
        assert "3" in result

    def test_format_without_kwargs(self) -> None:
        result = ErrorMsg.NO_PLAN()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cycle_detected(self) -> None:
        result = ErrorMsg.CYCLE_DETECTED()
        assert "cycle" in result.lower()

    def test_merge_conflict(self) -> None:
        result = ErrorMsg.MERGE_CONFLICT(a="pr-1", b="pr-2", file="shared.py")
        assert "pr-1" in result
        assert "pr-2" in result
        assert "shared.py" in result

    def test_hunk_too_large(self) -> None:
        result = ErrorMsg.HUNK_TOO_LARGE(file="big.py", index=0, tokens=500, budget=100)
        assert "big.py" in result

    def test_pr_create_failed(self) -> None:
        result = ErrorMsg.PR_CREATE_FAILED(group="my-branch", detail="rate limited")
        assert "rate limited" in result


class TestExceptionInheritance:
    def test_all_inherit_from_prsplit(self) -> None:
        assert issubclass(DiffParseError, PRSplitError)
        assert issubclass(PlanValidationError, PRSplitError)
        assert issubclass(GitOperationError, PRSplitError)
        assert issubclass(LLMError, PRSplitError)

    def test_prsplit_is_exception(self) -> None:
        assert issubclass(PRSplitError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PRSplitError):
            raise GitOperationError("test")
```

## types_defs.py

```python
from __future__ import annotations

import pytest

from pr_split.types_defs import DiffStats, FileSummary, ForkPRInfo, HunkInfo, HunkRef


class TestHunkRefNamedTuple:
    def test_fields(self) -> None:
        ref = HunkRef(file_path="a.py", hunk_index=2, token_estimate=100)
        assert ref.file_path == "a.py"
        assert ref.hunk_index == 2
        assert ref.token_estimate == 100

    def test_hashable(self) -> None:
        ref = HunkRef(file_path="a.py", hunk_index=0, token_estimate=50)
        assert hash(ref) is not None
        s = {ref}
        assert ref in s

    def test_equality(self) -> None:
        r1 = HunkRef(file_path="a.py", hunk_index=0, token_estimate=50)
        r2 = HunkRef(file_path="a.py", hunk_index=0, token_estimate=50)
        assert r1 == r2


class TestHunkInfoNamedTuple:
    def test_fields(self) -> None:
        info = HunkInfo(
            index=0,
            source_start=1,
            source_length=10,
            target_start=1,
            target_length=12,
            added_lines=5,
            removed_lines=3,
        )
        assert info.index == 0
        assert info.added_lines == 5
        assert info.removed_lines == 3


class TestFileSummaryTypedDict:
    def test_required_keys(self) -> None:
        fs = FileSummary(
            path="f.py",
            added=10,
            removed=5,
            is_new=False,
            is_deleted=False,
            is_renamed=False,
            hunk_count=2,
        )
        assert fs["path"] == "f.py"
        assert fs["hunk_count"] == 2


class TestDiffStatsTypedDict:
    def test_required_keys(self) -> None:
        ds = DiffStats(
            total_files=3,
            total_added=100,
            total_removed=50,
            total_loc=150,
            file_summaries=[],
        )
        assert ds["total_files"] == 3
        assert ds["total_loc"] == 150


class TestForkPRInfoTypedDict:
    def test_with_pr_number(self) -> None:
        info = ForkPRInfo(
            pr_number=42,
            local_ref="refs/pr-split/pr-42",
            base_branch="main",
            author="User <u@e.com>",
            fork_full_name="user/repo",
        )
        assert info["pr_number"] == 42

    def test_without_pr_number(self) -> None:
        info = ForkPRInfo(
            pr_number=None,
            local_ref="refs/pr-split/fork-user-branch",
            base_branch="main",
            author="User <u@e.com>",
            fork_full_name="user/repo",
        )
        assert info["pr_number"] is None
```

## planner/prompts.py

```python
from __future__ import annotations

import pytest

from pr_split.constants import Priority
from pr_split.planner.prompts import (
    SPLIT_TOOL_NAME,
    SPLIT_TOOL_SCHEMA,
    build_chunk_continuation_prompt,
    build_chunk_first_prompt,
    build_system_prompt,
    build_user_prompt,
)
from pr_split.types_defs import DiffStats, FileSummary


class TestBuildSystemPromptStringBuilding:
    def test_orthogonal_contains_priority(self) -> None:
        result = build_system_prompt(Priority.ORTHOGONAL, 400)
        assert "orthogonal" in result.lower() or "independent" in result.lower()

    def test_logical_contains_priority(self) -> None:
        result = build_system_prompt(Priority.LOGICAL, 400)
        assert "logical" in result.lower()

    def test_contains_max_loc(self) -> None:
        result = build_system_prompt(Priority.ORTHOGONAL, 250)
        assert "250" in result

    def test_returns_nonempty_string(self) -> None:
        result = build_system_prompt(Priority.ORTHOGONAL, 400)
        assert len(result) > 100


class TestBuildUserPromptStringBuilding:
    def test_contains_file_info(self) -> None:
        stats = DiffStats(
            total_files=2,
            total_added=30,
            total_removed=10,
            total_loc=40,
            file_summaries=[
                FileSummary(
                    path="a.py", added=20, removed=5,
                    is_new=False, is_deleted=False, is_renamed=False, hunk_count=1,
                ),
                FileSummary(
                    path="b.py", added=10, removed=5,
                    is_new=True, is_deleted=False, is_renamed=False, hunk_count=1,
                ),
            ],
        )
        result = build_user_prompt(stats, "diff content here")
        assert "a.py" in result
        assert "b.py" in result

    def test_contains_diff_content(self) -> None:
        stats = DiffStats(
            total_files=1, total_added=5, total_removed=0, total_loc=5,
            file_summaries=[
                FileSummary(
                    path="x.py", added=5, removed=0,
                    is_new=True, is_deleted=False, is_renamed=False, hunk_count=1,
                ),
            ],
        )
        result = build_user_prompt(stats, "UNIQUE_DIFF_MARKER")
        assert "UNIQUE_DIFF_MARKER" in result


class TestBuildChunkPrompts:
    def test_first_chunk_contains_chunk_info(self) -> None:
        stats = DiffStats(
            total_files=1, total_added=10, total_removed=0, total_loc=10,
            file_summaries=[
                FileSummary(
                    path="a.py", added=10, removed=0,
                    is_new=True, is_deleted=False, is_renamed=False, hunk_count=1,
                ),
            ],
        )
        result = build_chunk_first_prompt(stats, "diff_data", 3)
        assert "1" in result or "3" in result

    def test_continuation_contains_catalog(self) -> None:
        stats = DiffStats(
            total_files=1, total_added=10, total_removed=0, total_loc=10,
            file_summaries=[
                FileSummary(
                    path="a.py", added=10, removed=0,
                    is_new=True, is_deleted=False, is_renamed=False, hunk_count=1,
                ),
            ],
        )
        catalog = "Group pr-1: feat auth"
        result = build_chunk_continuation_prompt(stats, "diff_data", 2, 3, catalog)
        assert "pr-1" in result or "auth" in result


class TestSplitToolSchema:
    def test_tool_name_is_string(self) -> None:
        assert isinstance(SPLIT_TOOL_NAME, str)
        assert len(SPLIT_TOOL_NAME) > 0

    def test_schema_has_groups_property(self) -> None:
        assert "properties" in SPLIT_TOOL_SCHEMA
        assert "groups" in SPLIT_TOOL_SCHEMA["properties"]
```

## plan_store.py

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pr_split.constants import Priority
from pr_split.plan_store import load_plan, plan_exists, save_plan
from pr_split.schemas import Group, PlanFile, SplitPlan


class TestPlanStoreRoundtrip:
    def test_save_load_preserves_groups(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pr_split.plan_store.PLAN_DIR", str(tmp_path))
        monkeypatch.setattr("pr_split.plan_store.PLAN_FILE", str(tmp_path / "plan.json"))

        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="feature",
                base_branch="main",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                groups=[
                    Group(id="pr-1", title="t1", description="d1"),
                    Group(id="pr-2", title="t2", description="d2", depends_on=["pr-1"]),
                ],
            ),
        )
        save_plan(plan_file)
        loaded = load_plan()
        assert len(loaded.plan.groups) == 2
        assert loaded.plan.groups[1].depends_on == ["pr-1"]

    def test_plan_exists_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pr_split.plan_store.PLAN_FILE", str(tmp_path / "nonexistent.json"))
        assert plan_exists() is False

    def test_plan_exists_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text("{}")
        monkeypatch.setattr("pr_split.plan_store.PLAN_FILE", str(plan_path))
        assert plan_exists() is True


class TestPlanStoreJson:
    def test_saved_file_is_valid_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pr_split.plan_store.PLAN_DIR", str(tmp_path))
        monkeypatch.setattr("pr_split.plan_store.PLAN_FILE", str(tmp_path / "plan.json"))

        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="dev",
                base_branch="main",
                max_loc=400,
                priority=Priority.LOGICAL,
                groups=[],
            ),
        )
        save_plan(plan_file)
        raw = json.loads((tmp_path / "plan.json").read_text())
        assert "plan" in raw
        assert raw["plan"]["priority"] == "logical"
```

## planner/validator.py

```python
from __future__ import annotations

import pytest

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import ParsedDiff, parse_diff
from pr_split.exceptions import PlanValidationError
from pr_split.planner.validator import (
    validate_coverage,
    validate_loc,
    validate_loc_bounds,
)
from pr_split.schemas import Group, GroupAssignment

SAMPLE_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,5 @@
+line1
+line2
+line3
+line4
+line5
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,3 @@
+lineA
+lineB
+lineC
"""


def _make_parsed() -> ParsedDiff:
    return parse_diff(SAMPLE_DIFF)


class TestValidateCoverageAlgorithm:
    def test_full_coverage_passes(self) -> None:
        parsed = _make_parsed()
        groups = [
            Group(
                id="pr-1",
                title="t",
                description="d",
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
            ),
        ]
        validate_coverage(groups, parsed)

    def test_missing_file_raises(self) -> None:
        parsed = _make_parsed()
        groups = [
            Group(
                id="pr-1",
                title="t",
                description="d",
                assignments=[
                    GroupAssignment(
                        file_path="a.py",
                        assignment_type=AssignmentType.WHOLE_FILE,
                        hunk_indices=[0],
                    ),
                ],
            ),
        ]
        with pytest.raises(PlanValidationError, match="b.py"):
            validate_coverage(groups, parsed)

    def test_duplicate_assignment_raises(self) -> None:
        parsed = _make_parsed()
        groups = [
            Group(
                id="pr-1",
                title="t1",
                description="d1",
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
            ),
            Group(
                id="pr-2",
                title="t2",
                description="d2",
                assignments=[
                    GroupAssignment(
                        file_path="a.py",
                        assignment_type=AssignmentType.PARTIAL_HUNKS,
                        hunk_indices=[0],
                    ),
                ],
            ),
        ]
        with pytest.raises(PlanValidationError):
            validate_coverage(groups, parsed)


class TestValidateLocAlgorithm:
    def test_matching_loc_passes(self) -> None:
        parsed = _make_parsed()
        total = parsed.stats["total_loc"]
        groups = [
            Group(id="pr-1", title="t", description="d", estimated_loc=total),
        ]
        validate_loc(groups, parsed)

    def test_mismatch_raises(self) -> None:
        parsed = _make_parsed()
        groups = [
            Group(id="pr-1", title="t", description="d", estimated_loc=999),
        ]
        with pytest.raises(PlanValidationError):
            validate_loc(groups, parsed)


class TestValidateLocBoundsAlgorithm:
    def test_under_limit_no_warnings(self) -> None:
        groups = [Group(id="pr-1", title="t", description="d", estimated_loc=50)]
        warnings = validate_loc_bounds(groups, 100)
        assert warnings == []

    def test_over_limit_returns_warning(self) -> None:
        groups = [
            Group(
                id="pr-1", title="t", description="d",
                estimated_loc=200, estimated_added=150, estimated_removed=50,
            )
        ]
        warnings = validate_loc_bounds(groups, 100)
        assert len(warnings) == 1
        assert "pr-1" in warnings[0]
```
