from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from pr_split.cli import _create_branches_and_commits
from pr_split.constants import AssignmentType
from pr_split.diff_ops import extract_diff, parse_diff
from pr_split.exceptions import PRSplitError
from pr_split.per_group import PerGroupStep, load_per_group_step
from pr_split.schemas import Group, GroupAssignment

# A croft-style gate: raise the version above the parent's, add its notes.
BUMP = textwrap.dedent(
    """\
    set -e
    old=$(git show "$PR_SPLIT_PARENT_REF:Cargo.toml" | sed -n 's/^version = "0.1.\\(.*\\)"/\\1/p')
    new="0.1.$((old + 1))"
    sed -i "s/^version = .*/version = \\"$new\\"/" Cargo.toml
    mkdir -p notes
    printf '%s\\n' "$PR_SPLIT_GROUP_TITLE" > "notes/$new.md"
    """
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@t")
    for key in ("PR_SPLIT_PER_GROUP_RUN", "PR_SPLIT_PER_GROUP_COMMIT_MESSAGE"):
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "Cargo.toml").write_text('[package]\nversion = "0.1.959"\n')
    (repo / "bump.sh").write_text(BUMP)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "checkout", "-q", "-b", "feature")
    for name in ("a", "b", "c"):
        (repo / f"{name}.rs").write_text(f"fn {name}() {{}}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "work")
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)
    return repo


def _groups() -> list[Group]:
    return [
        Group(
            id=f"pr-{i}",
            title=f"Add {name}",
            description="",
            depends_on=[f"pr-{i - 1}"] if i > 1 else [],
            assignments=[
                GroupAssignment(
                    file_path=f"{name}.rs",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        for i, name in enumerate(("a", "b", "c"), start=1)
    ]


def _build(repo: Path) -> dict[str, str]:
    parsed = parse_diff(extract_diff("feature", "main"))
    merge_base = _git(repo, "merge-base", "main", "feature")
    records = _create_branches_and_commits(
        _groups(), parsed, "main", merge_base, "x", stacked=True
    )
    return {r.group_id: r.branch_name for r in records}


def test_every_stacked_layer_gets_its_own_bump_and_notes(repo: Path) -> None:
    (repo / ".pr-split.toml").write_text(
        '[per_group]\nrun = "sh bump.sh"\ncommit_message = "chore: release for {title}"\n'
    )
    branches = _build(repo)

    parent = "main"
    for i, gid in enumerate(("pr-1", "pr-2", "pr-3"), start=960):
        branch = branches[gid]
        assert f'version = "0.1.{i}"' in _git(repo, "show", f"{branch}:Cargo.toml")
        changed = set(_git(repo, "diff", "--name-only", parent, branch).splitlines())
        # Against its own base each layer bumps the version and adds its notes.
        assert {"Cargo.toml", f"notes/0.1.{i}.md"} <= changed
        assert _git(repo, "log", "-1", "--format=%s", branch).startswith("chore: release for")
        parent = branch
    assert _git(repo, "show", f"{branches['pr-3']}:notes/0.1.962.md") == "Add c"


def test_no_step_configured_leaves_one_commit_per_layer(repo: Path) -> None:
    branches = _build(repo)
    assert _git(repo, "rev-list", "--count", f"main..{branches['pr-3']}") == "3"


def test_environment_overrides_the_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / ".pr-split.toml").write_text('[per_group]\nrun = "false"\n')
    monkeypatch.setenv("PR_SPLIT_PER_GROUP_RUN", "touch marker")
    assert load_per_group_step() == PerGroupStep(run="touch marker")
    branches = _build(repo)
    assert "marker" in _git(repo, "ls-tree", "--name-only", branches["pr-1"]).splitlines()


def test_failing_step_names_the_group_and_leaves_no_branch(repo: Path) -> None:
    (repo / ".pr-split.toml").write_text('[per_group]\nrun = "echo nope >&2; exit 3"\n')
    with pytest.raises(PRSplitError, match=r"group 'pr-1' \(exit 3\): nope"):
        _build(repo)
    assert "pr-split/x/pr-1" not in _git(repo, "branch", "--list")


def test_title_is_not_spliced_into_the_command(repo: Path) -> None:
    (repo / ".pr-split.toml").write_text(
        '[per_group]\nrun = "printf %s \\"$PR_SPLIT_GROUP_TITLE\\" > title.txt"\n'
    )
    groups = _groups()
    groups[0].title = "$(touch pwned); `touch pwned2`"
    parsed = parse_diff(extract_diff("feature", "main"))
    records = _create_branches_and_commits(
        groups, parsed, "main", _git(repo, "merge-base", "main", "feature"), "x", stacked=True
    )
    branch = records[0].branch_name
    assert _git(repo, "show", f"{branch}:title.txt") == groups[0].title
    tree = _git(repo, "ls-tree", "--name-only", branch).splitlines()
    assert "pwned" not in tree and "pwned2" not in tree


@pytest.mark.parametrize(
    ("content", "match"),
    [("[per_group\n", "Cannot read"), ("per_group = 3\n", r"\[per_group\]")],
)
def test_bad_config_is_a_clean_error(repo: Path, content: str, match: str) -> None:
    (repo / ".pr-split.toml").write_text(content)
    with pytest.raises(PRSplitError, match=match):
        load_per_group_step()


def test_bad_commit_message_template_is_a_clean_error(repo: Path) -> None:
    (repo / ".pr-split.toml").write_text(
        '[per_group]\nrun = "touch x"\ncommit_message = "release {version}"\n'
    )
    with pytest.raises(PRSplitError, match="commit_message"):
        _build(repo)
