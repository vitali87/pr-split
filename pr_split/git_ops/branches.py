from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

from loguru import logger

from .. import logs
from ..constants import PLAN_DIR
from ..exceptions import ErrorMsg, GitOperationError


def _git_env() -> dict[str, str]:
    # delete_branch inspects git's stderr; pin the UI language so those
    # messages are stable on localised systems.
    return {**os.environ, "LC_ALL": "C", "LANGUAGE": "C"}


def _git(args: tuple[str, ...], cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=_git_env(),
        )
    except FileNotFoundError as exc:
        raise GitOperationError(ErrorMsg.TOOL_NOT_FOUND(tool="git")) from exc
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout.strip()


def run_git(*args: str) -> str:
    return _git(args)


def run_git_in_dir(cwd: str, *args: str) -> str:
    return _git(args, cwd=cwd)


def require_tools(*tools: str) -> str | None:
    """Return the first of ``tools`` that is not on PATH, or None."""
    for tool in tools:
        if shutil.which(tool) is None:
            return tool
    return None


def commit_exists(ref: str) -> bool:
    """True if ``ref`` resolves to a commit object in this repository."""
    try:
        run_git("cat-file", "-e", f"{ref}^{{commit}}")
    except GitOperationError:
        return False
    return True


def branch_exists(branch: str) -> bool:
    try:
        run_git("rev-parse", "--verify", branch)
    except GitOperationError:
        return False
    return True


def adopt_remote_branch(branch: str) -> bool:
    """Create local ``branch`` from its remote-tracking ref when that is unambiguous.

    A fresh clone or worktree has ``origin/<branch>`` but no local branch;
    like ``git checkout <branch>``, adopt the single remote that has it and
    track it, so ``diff_base_ref`` keeps an adopted base fresh from that
    remote. Returns True when a local branch was created.
    """
    if branch_exists(f"refs/heads/{branch}"):
        return False
    try:
        listing = run_git("for-each-ref", "--format=%(refname:short)", f"refs/remotes/*/{branch}")
    except GitOperationError:
        return False
    candidates = [ref for ref in listing.splitlines() if ref.split("/", 1)[1:] == [branch]]
    if len(candidates) != 1:
        return False
    run_git("branch", "--track", branch, candidates[0])
    logger.info(logs.ADOPTED_REMOTE_BRANCH.format(branch=branch, remote_ref=candidates[0]))
    return True


def is_worktree_clean() -> bool:
    """True when nothing tracked is modified, ignoring pr-split's own plan directory.

    `split`/`edit` write `.pr-split/plan.json`; a user who commits the plan
    (to share or review it) then has a modified tracked file, and `execute`
    would refuse to run on the very plan it was asked to execute. Everything
    under the plan directory is therefore excluded from the check.
    """
    output = run_git("status", "--porcelain", "--", f":(top,exclude){PLAN_DIR}")
    return all(line.startswith("??") for line in output.splitlines())


# Server-side failures GitHub reports for a push that can succeed on retry.
_TRANSIENT_PUSH_ERRORS = (
    "fatal error in commit_refs",
    "the remote end hung up unexpectedly",
    "internal server error",
    "http 500",
    "http 502",
    "http 503",
    "connection reset",
    "operation timed out",
)
_PUSH_ATTEMPTS = 3
_PUSH_RETRY_DELAY = 2.0


def _remote_has_local_head(branch: str) -> bool:
    try:
        remote = run_git("ls-remote", "origin", f"refs/heads/{branch}").split()
        return bool(remote) and remote[0] == run_git("rev-parse", branch)
    except GitOperationError:
        return False


def push_branch(branch: str) -> None:
    logger.info(logs.PUSHING_BRANCH.format(branch=branch))
    retried = False
    for attempt in range(1, _PUSH_ATTEMPTS + 1):
        try:
            run_git("push", "--force-with-lease", "-u", "origin", branch)
            return
        except GitOperationError as exc:
            transient = any(marker in str(exc).lower() for marker in _TRANSIENT_PUSH_ERRORS)
            # A push reported as failed (hung-up remote, timeout) may still have
            # landed; the retry is then rejected as a stale lease although the
            # branch is already on the remote.
            if retried and not transient and _remote_has_local_head(branch):
                return
            if not transient or attempt == _PUSH_ATTEMPTS:
                raise
            retried = True
            logger.warning(
                logs.PUSH_RETRY.format(branch=branch, attempt=attempt, error=str(exc).strip())
            )
            time.sleep(_PUSH_RETRY_DELAY * attempt)


_LOCAL_BRANCH_MISSING = "not found"
_REMOTE_REF_MISSING = "remote ref does not exist"


def delete_branch(branch: str, *, remote: bool = False) -> None:
    """Delete ``branch`` locally and, with ``remote``, on origin.

    A branch that is already gone (``pr-split merge`` deletes the local and
    remote branch as part of merging) counts as deleted: cleanup must be
    re-runnable and must not report a completed merge as a failure.
    """
    local_error: GitOperationError | None = None
    try:
        run_git("branch", "-D", branch)
        logger.info(logs.BRANCH_DELETED.format(branch=branch))
    except GitOperationError as exc:
        if _LOCAL_BRANCH_MISSING in str(exc):
            logger.info(logs.BRANCH_ALREADY_GONE.format(branch=branch, where=" locally"))
        elif not remote:
            raise
        else:
            # The local branch may be checked out; still remove the remote
            # branch so the cleanup is not left half done.
            local_error = exc
    if remote:
        try:
            run_git("push", "origin", "--delete", branch)
        except GitOperationError as exc:
            if _REMOTE_REF_MISSING not in str(exc):
                raise
            logger.info(logs.BRANCH_ALREADY_GONE.format(branch=branch, where=" on origin"))
    if local_error is not None:
        raise local_error


def merge_base(ref_a: str, ref_b: str) -> str:
    return run_git("merge-base", ref_a, ref_b)


def _branch_config(branch: str, key: str) -> str:
    try:
        return run_git("config", "--get", f"branch.{branch}.{key}")
    except GitOperationError:
        return ""


def diff_base_ref(base: str) -> str:
    """The ref to diff against for a split whose sub-PRs target ``base``.

    Sub-PRs are opened against the remote's copy of ``base``, but the local
    branch may lag behind it; diffing against the stale local ref would
    present every upstream commit since as branch work. So when ``base``
    tracks a remote branch, fetch it and return the remote-tracking ref
    (e.g. ``origin/main``). A failed fetch (offline) falls back to the
    tracking ref as last fetched; a branch with no upstream is used as is.
    """
    remote = _branch_config(base, "remote")
    merge_ref = _branch_config(base, "merge")
    if not remote or remote == "." or not merge_ref.startswith("refs/heads/"):
        return base
    tracking = f"refs/remotes/{remote}/{merge_ref.removeprefix('refs/heads/')}"
    try:
        run_git("fetch", "--quiet", remote, merge_ref)
    except GitOperationError as exc:
        logger.warning(logs.BASE_FETCH_FAILED.format(remote=remote, base=base, detail=exc))
    if not commit_exists(tracking):
        return base
    upstream = tracking.removeprefix("refs/remotes/")
    counts = run_git("rev-list", "--left-right", "--count", f"{base}...{tracking}").split()
    ahead, behind = int(counts[0]), int(counts[1])
    if behind:
        logger.warning(logs.LOCAL_BASE_BEHIND.format(base=base, upstream=upstream, count=behind))
    if ahead:
        logger.warning(logs.LOCAL_BASE_AHEAD.format(base=base, upstream=upstream, count=ahead))
    return upstream


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
    # The content is a subset of commits already accepted on the dev
    # branch; a pre-commit/commit-msg hook (husky, pre-commit, lint-staged)
    # run inside the throwaway worktree has no node_modules/venv and can
    # only fail, so skip it.
    run_git_in_dir(cwd, "commit", "--no-verify", "-m", message, *author_args)
    return run_git_in_dir(cwd, "rev-parse", "HEAD")
