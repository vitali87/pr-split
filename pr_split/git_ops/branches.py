from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from .. import logs
from ..exceptions import GitOperationError


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout.strip()


def run_git_in_dir(cwd: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout.strip()


def branch_exists(branch: str) -> bool:
    try:
        run_git("rev-parse", "--verify", branch)
    except GitOperationError:
        return False
    return True


def is_worktree_clean() -> bool:
    output = run_git("status", "--porcelain")
    return all(line.startswith("??") for line in output.splitlines())


def push_branch(branch: str) -> None:
    logger.info(logs.PUSHING_BRANCH.format(branch=branch))
    run_git("push", "--force-with-lease", "-u", "origin", branch)


def delete_branch(branch: str, *, remote: bool = False) -> None:
    local_error: GitOperationError | None = None
    try:
        run_git("branch", "-D", branch)
        logger.info(logs.BRANCH_DELETED.format(branch=branch))
    except GitOperationError as exc:
        if not remote:
            raise
        # The local branch may be checked out or already gone; still remove
        # the remote branch so the cleanup is not left half done.
        local_error = exc
    if remote:
        run_git("push", "origin", "--delete", branch)
    if local_error is not None:
        raise local_error


def merge_base(ref_a: str, ref_b: str) -> str:
    return run_git("merge-base", ref_a, ref_b)


def derive_split_namespace(dev_branch_arg: str) -> str:
    raw = dev_branch_arg.split(":", 1)[1] if ":" in dev_branch_arg else dev_branch_arg.lstrip("#")
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "-", raw)
    return sanitized.strip("-")


def add_worktree(path: str, branch_name: str, start_point: str) -> None:
    prev_sha: str | None = None
    if branch_exists(branch_name):
        prev_sha = run_git("rev-parse", branch_name)
        run_git("branch", "-D", branch_name)
    try:
        # A repository post-checkout hook (husky, lint-staged installers)
        # runs inside the throwaway worktree, where it has no toolchain and
        # can only fail; pointing hooksPath at an empty directory for this
        # one command disables it.
        run_git(
            "-c",
            f"core.hooksPath={_no_hooks_dir()}",
            "worktree",
            "add",
            "-b",
            branch_name,
            path,
            start_point,
        )
    except GitOperationError:
        if prev_sha is not None:
            run_git("branch", branch_name, prev_sha)
        raise


def _no_hooks_dir() -> str:
    """An empty directory to use as core.hooksPath (no hooks run)."""
    path = Path(tempfile.gettempdir()) / "pr-split-no-hooks"
    path.mkdir(exist_ok=True)
    return str(path)


def remove_worktree(path: str) -> None:
    run_git("worktree", "remove", "--force", path)


def commit_files_in_dir(
    cwd: str, file_paths: list[str], message: str, *, author: str | None = None
) -> str:
    if not file_paths:
        raise GitOperationError("commit_files_in_dir called with no file paths")
    # -f: the dev branch may track a file that matches .gitignore (added
    # with `git add -f`); the diff materialises it, and without -f `git add`
    # refuses the path and the whole group fails. The path list is explicit,
    # so -f cannot pull in anything unintended; -A still stages deletions.
    run_git_in_dir(cwd, "add", "-A", "-f", "--", *file_paths)
    author_args = ("--author", author) if author else ()
    # The content is a subset of commits already accepted on the dev
    # branch; a pre-commit/commit-msg hook (husky, pre-commit, lint-staged)
    # run inside the throwaway worktree has no node_modules/venv and can
    # only fail, so skip it.
    run_git_in_dir(cwd, "commit", "--no-verify", "-m", message, *author_args)
    return run_git_in_dir(cwd, "rev-parse", "HEAD")
