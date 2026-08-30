from __future__ import annotations

import re
import subprocess

from loguru import logger

from .. import logs
from ..constants import BRANCH_PREFIX
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


def checkout_new_branch(name: str, start_point: str) -> None:
    run_git("checkout", "-b", name, start_point)


def checkout_branch(name: str) -> None:
    run_git("checkout", name)


def commit_files(file_paths: list[str], message: str, *, author: str | None = None) -> str:
    run_git("add", "--", *file_paths)
    author_args = ("--author", author) if author else ()
    try:
        run_git("commit", "-m", message, *author_args)
    except GitOperationError:
        run_git("add", "-u")
        run_git("commit", "-m", message, *author_args)
    return run_git("rev-parse", "HEAD")


def push_branch(branch: str) -> None:
    logger.info(logs.PUSHING_BRANCH.format(branch=branch))
    run_git("push", "--force-with-lease", "-u", "origin", branch)


_REMOTE_REF_MISSING = "remote ref does not exist"


def delete_branch(branch: str, *, remote: bool = False) -> None:
    """Delete a branch locally and, with remote=True, on origin as well.

    The two deletes are independent: a local branch that is already gone
    (``gh pr merge --delete-branch``) or currently checked out must not
    stop the remote branch from being removed, and a remote branch that no
    longer exists counts as deleted.
    """
    try:
        run_git("branch", "-D", branch)
        logger.info(logs.BRANCH_DELETED.format(branch=branch))
    except GitOperationError as exc:
        if not remote:
            raise
        logger.warning(logs.LOCAL_BRANCH_NOT_DELETED.format(branch=branch, error=exc))
    if remote:
        try:
            run_git("push", "origin", "--delete", branch)
        except GitOperationError as exc:
            if _REMOTE_REF_MISSING not in str(exc):
                raise
            logger.info(logs.REMOTE_BRANCH_ALREADY_GONE.format(branch=branch))
        else:
            logger.info(logs.REMOTE_BRANCH_DELETED.format(branch=branch))


def merge_base(ref_a: str, ref_b: str) -> str:
    return run_git("merge-base", ref_a, ref_b)


def derive_split_namespace(dev_branch_arg: str) -> str:
    raw = dev_branch_arg.split(":", 1)[1] if ":" in dev_branch_arg else dev_branch_arg.lstrip("#")
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "-", raw)
    return sanitized.strip("-")


def create_group_branch(group_id: str, base: str, namespace: str) -> str:
    branch_name = f"{BRANCH_PREFIX}{namespace}/{group_id}"
    logger.info(logs.CREATING_BRANCH.format(branch=branch_name, base=base))
    if branch_exists(branch_name):
        checkout_branch(base)
        run_git("branch", "-D", branch_name)
    checkout_new_branch(branch_name, base)
    return branch_name


def add_worktree(path: str, branch_name: str, start_point: str) -> None:
    prev_sha: str | None = None
    if branch_exists(branch_name):
        prev_sha = run_git("rev-parse", branch_name)
        run_git("branch", "-D", branch_name)
    try:
        run_git("worktree", "add", "-b", branch_name, path, start_point)
    except GitOperationError:
        if prev_sha is not None:
            run_git("branch", branch_name, prev_sha)
        raise


def remove_worktree(path: str) -> None:
    run_git("worktree", "remove", "--force", path)


def commit_files_in_dir(
    cwd: str, file_paths: list[str], message: str, *, author: str | None = None
) -> str:
    if not file_paths:
        raise GitOperationError("commit_files_in_dir called with no file paths")
    run_git_in_dir(cwd, "add", "-A", "--", *file_paths)
    author_args = ("--author", author) if author else ()
    run_git_in_dir(cwd, "commit", "-m", message, *author_args)
    return run_git_in_dir(cwd, "rev-parse", "HEAD")
