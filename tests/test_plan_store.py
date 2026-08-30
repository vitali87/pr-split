from __future__ import annotations

import json
from pathlib import Path

import pytest

from pr_split.constants import Priority
from pr_split.exceptions import PRSplitError
from pr_split.plan_store import load_plan, plan_dir, plan_exists, plan_path, save_plan
from pr_split.schemas import (
    BranchRecord,
    GitState,
    Group,
    PlanFile,
    SplitPlan,
)


def _make_plan_file() -> PlanFile:
    return PlanFile(
        plan=SplitPlan(
            dev_branch="feat/big",
            base_branch="main",
            min_loc=50,
            max_loc=400,
            strict_loc_bounds=True,
            priority=Priority.ORTHOGONAL,
            groups=[
                Group(
                    id="pr-1",
                    title="feat: add auth",
                    description="Auth module",
                ),
            ],
        ),
        git_state=GitState(
            branches=[
                BranchRecord(
                    group_id="pr-1",
                    branch_name="pr-split/pr-1",
                    base_branch="main",
                    commit_sha="abc123",
                ),
            ],
        ),
    )


class TestPlanStore:
    def test_save_and_load_roundtrip(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = _make_plan_file()
        save_plan(plan_file)
        loaded = load_plan()
        assert loaded.plan.dev_branch == "feat/big"
        assert len(loaded.plan.groups) == 1
        assert loaded.plan.groups[0].id == "pr-1"
        assert loaded.plan.min_loc == 50
        assert loaded.plan.strict_loc_bounds is True

    def test_plan_exists_false(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert plan_exists() is False

    def test_plan_exists_true(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        save_plan(_make_plan_file())
        assert plan_exists() is True

    def test_load_missing_raises(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(PRSplitError, match="No split plan"):
            load_plan()

    def test_git_state_preserved(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = _make_plan_file()
        save_plan(plan_file)
        loaded = load_plan()
        assert len(loaded.git_state.branches) == 1
        assert loaded.git_state.branches[0].commit_sha == "abc123"


class TestPlanStoreJson:
    def test_saved_file_is_valid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="dev",
                base_branch="main",
                min_loc=25,
                max_loc=400,
                strict_loc_bounds=True,
                priority=Priority.LOGICAL,
                groups=[],
            ),
        )
        save_plan(plan_file)
        raw = json.loads((tmp_path / ".pr-split" / "plan.json").read_text())
        assert "plan" in raw
        assert raw["plan"]["priority"] == "logical"
        assert raw["plan"]["min_loc"] == 25
        assert raw["plan"]["strict_loc_bounds"] is True


class TestPlanStoreCorruptFiles:
    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("", id="empty"),
            pytest.param("{not json", id="not-json"),
            pytest.param('{"plan": {"dev_branch": "x"}}', id="missing-fields"),
            pytest.param('{"plan": {"groups": []}, "git_state": {}}', id="old-schema"),
        ],
    )
    def test_unreadable_plan_raises_pr_split_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan_dir = tmp_path / ".pr-split"
        plan_dir.mkdir()
        (plan_dir / "plan.json").write_text(content)
        with pytest.raises(PRSplitError, match="Cannot load split plan") as excinfo:
            load_plan()
        assert "pr-split split" in str(excinfo.value)

    def test_invalid_utf8_raises_pr_split_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".pr-split").mkdir()
        (tmp_path / ".pr-split" / "plan.json").write_bytes(b'{"plan": "\xff\xfe"}')
        with pytest.raises(PRSplitError, match="Cannot load split plan"):
            load_plan()

    def test_read_error_raises_pr_split_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".pr-split").mkdir()
        (tmp_path / ".pr-split" / "plan.json").mkdir()
        with pytest.raises(PRSplitError, match="Cannot load split plan"):
            load_plan()


class TestPlanPathsResolveAgainstTheRepoRoot:
    def _repo(self, tmp_path: Path) -> Path:
        import subprocess

        repo = tmp_path / "repo"
        (repo / "src").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        return repo

    def test_plan_saved_from_a_subdirectory_is_found_from_the_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pr_split.plan_store import plan_path

        repo = self._repo(tmp_path)
        monkeypatch.chdir(repo / "src")
        save_plan(_make_plan_file())
        assert (repo / ".pr-split" / "plan.json").exists()
        assert not (repo / "src" / ".pr-split").exists()
        assert plan_path() == (repo / ".pr-split" / "plan.json").resolve()

        monkeypatch.chdir(repo)
        assert plan_exists()
        assert load_plan().plan.dev_branch == "feat/big"

    def test_outside_a_repository_the_cwd_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pr_split.plan_store import plan_path

        monkeypatch.chdir(tmp_path)
        assert plan_path() == tmp_path / ".pr-split" / "plan.json"

    def test_template_is_read_from_the_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pr_split.cli import _pr_template_path

        repo = self._repo(tmp_path)
        (repo / ".pr-split").mkdir()
        (repo / ".pr-split" / "template.md").write_text("# {title}")
        monkeypatch.chdir(repo / "src")
        assert _pr_template_path().read_text() == "# {title}"


class TestSavePlanIsAtomic:
    def test_no_temp_file_left_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        save_plan(_make_plan_file())
        leftovers = [p.name for p in plan_dir().iterdir() if p.name != "plan.json"]
        assert leftovers == []

    def test_failed_write_preserves_the_previous_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-write must not destroy the only record of created PRs."""
        monkeypatch.chdir(tmp_path)
        save_plan(_make_plan_file())
        before = plan_path().read_text()

        def exploding_write(self: Path, *args: object, **kwargs: object) -> int:
            # Emulate a crash/disk-full: leave a truncated temp file behind.
            Path(str(self)).parent.mkdir(parents=True, exist_ok=True)
            with open(self, "w") as fh:
                fh.write('{"plan": {"dev_branch"')
            raise OSError(28, "No space left on device")

        # A nested patch context: monkeypatch.undo() would also undo chdir.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "write_text", exploding_write)
            with pytest.raises(OSError):
                save_plan(_make_plan_file())

        assert plan_path().read_text() == before
        loaded = load_plan()
        assert loaded.plan.dev_branch == _make_plan_file().plan.dev_branch
