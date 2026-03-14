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


def delete_branch(branch: str, *, remote: bool = False) -> None:
    run_git("branch", "-D", branch)
    logger.info(logs.BRANCH_DELETED.format(branch=branch))
    if remote:
        run_git("push", "origin", "--delete", branch)


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
