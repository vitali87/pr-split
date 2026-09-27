"""A repository-defined step run on every sub-PR branch before it is pushed.

Some repositories gate each PR on files unique to it, e.g. a version bump
above the PR's base and a release-notes file for that version. With
``--stack`` a layer's base is its parent's branch, so every layer needs its
own. The step runs in the layer's worktree after the layer's own commit; any
changes it leaves are committed to that layer, and children are cut from the
layer after it ran, so each one sees its parent's result.

Configured in ``.pr-split.toml`` at the repository root::

    [per_group]
    run = "scripts/bump-version.sh"
    commit_message = "chore: release notes for {title}"

or with ``PR_SPLIT_PER_GROUP_RUN`` / ``PR_SPLIT_PER_GROUP_COMMIT_MESSAGE``.
The group's details reach the command as environment variables, never
spliced into the command line, so a title cannot inject shell syntax.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from . import logs
from .exceptions import ErrorMsg, PRSplitError
from .git_ops.branches import commit_files_in_dir, run_git, run_git_in_dir

if TYPE_CHECKING:
    from .schemas import Group

CONFIG_FILE = ".pr-split.toml"
DEFAULT_COMMIT_MESSAGE = "chore: per-group step for {title}"
_OUTPUT_TAIL = 800


@dataclass(frozen=True)
class PerGroupStep:
    run: str
    commit_message: str = DEFAULT_COMMIT_MESSAGE


def load_per_group_step() -> PerGroupStep | None:
    """The configured step, environment first, then .pr-split.toml; None if unset."""
    run = os.environ.get("PR_SPLIT_PER_GROUP_RUN", "")
    message = os.environ.get("PR_SPLIT_PER_GROUP_COMMIT_MESSAGE", "")
    config = Path(run_git("rev-parse", "--show-toplevel")) / CONFIG_FILE
    if config.is_file():
        try:
            section = tomllib.loads(config.read_text(encoding="utf-8")).get("per_group", {})
        except tomllib.TOMLDecodeError as exc:
            raise PRSplitError(ErrorMsg.CONFIG_INVALID(path=config, detail=exc)) from exc
        if not isinstance(section, dict):
            raise PRSplitError(ErrorMsg.CONFIG_INVALID(path=config, detail="[per_group]"))
        run = run or str(section.get("run", ""))
        message = message or str(section.get("commit_message", ""))
    if not run.strip():
        return None
    return PerGroupStep(run=run, commit_message=message or DEFAULT_COMMIT_MESSAGE)


def run_per_group_step(
    step: PerGroupStep,
    worktree: str,
    group: Group,
    *,
    index: int,
    pr_base: str,
    parent_ref: str,
    author: str | None = None,
) -> str | None:
    """Run the step in the layer's worktree; return its commit, or None if it changed nothing."""
    logger.info(logs.PER_GROUP_STEP_RUNNING.format(group=group.id, command=step.run))
    env = {
        **os.environ,
        "PR_SPLIT_GROUP_ID": group.id,
        "PR_SPLIT_GROUP_TITLE": group.title,
        "PR_SPLIT_GROUP_INDEX": str(index),
        # The branch the sub-PR is opened against, and the commit the layer
        # was cut from (its parent's head, or the merge base for a root).
        "PR_SPLIT_PR_BASE": pr_base,
        "PR_SPLIT_PARENT_REF": parent_ref,
    }
    result = subprocess.run(
        step.run, shell=True, cwd=worktree, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()[-_OUTPUT_TAIL:]
        raise PRSplitError(
            ErrorMsg.PER_GROUP_STEP_FAILED(group=group.id, code=result.returncode, output=output)
        )
    if not run_git_in_dir(worktree, "status", "--porcelain"):
        return None
    try:
        message = step.commit_message.format(id=group.id, title=group.title, index=index)
    except (KeyError, IndexError, ValueError) as exc:
        raise PRSplitError(
            ErrorMsg.CONFIG_INVALID(path=CONFIG_FILE, detail=f"commit_message: {exc!r}")
        ) from exc
    return commit_files_in_dir(worktree, ["."], message, author=author)
