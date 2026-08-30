from __future__ import annotations

import json
import subprocess

from loguru import logger

from .. import logs
from ..constants import FORK_REF_PREFIX, PR_REF_PREFIX
from ..exceptions import ErrorMsg, GitOperationError
from ..types_defs import ForkPRInfo


def _run_gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout.strip()


def check_gh_auth() -> bool:
    try:
        _run_gh("auth", "status")
    except GitOperationError:
        return False
    return True


GH_STACK_EXTENSION = "github/gh-stack"


def check_gh_stack() -> bool:
    try:
        installed = _run_gh("extension", "list")
    except GitOperationError:
        return False
    return any(GH_STACK_EXTENSION in line.split() for line in installed.splitlines())


def create_pr(
    head: str, base: str, title: str, body: str, *, draft: bool = False
) -> tuple[int, str]:
    args = [
        "pr",
        "create",
        "--base",
        base,
        "--head",
        head,
        "--title",
        title,
        "--body",
        body,
    ]
    if draft:
        args.append("--draft")
    try:
        output = _run_gh(*args)
    except GitOperationError as exc:
        raise GitOperationError(ErrorMsg.PR_CREATE_FAILED(group=head, detail=str(exc))) from exc
    pr_url = output.strip().splitlines()[-1]
    pr_number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])
    logger.info(logs.PR_CREATED.format(number=pr_number, url=pr_url))
    return pr_number, pr_url


def get_pr_state(pr_number: int) -> dict[str, str | bool | None]:
    try:
        raw = _run_gh("pr", "view", str(pr_number), "--json", "state,reviewDecision,isDraft")
        return json.loads(raw)
    except (GitOperationError, json.JSONDecodeError) as exc:
        logger.debug(f"Failed to fetch live state for PR #{pr_number}: {exc}")
        return {}


def merge_pr(pr_number: int, *, auto: bool = False) -> None:
    args = ["pr", "merge", str(pr_number), "--merge", "--delete-branch"]
    if auto:
        args.append("--auto")
    _run_gh(*args)
    logger.info(f"{'Queued' if auto else 'Merged'} PR #{pr_number}")


def close_pr(pr_number: int) -> None:
    _run_gh("pr", "close", str(pr_number))
    logger.info(logs.PR_CLOSED.format(number=pr_number))


def link_stack(pr_numbers: list[int]) -> None:
    try:
        _run_gh("stack", "link", *[str(n) for n in pr_numbers])
    except GitOperationError as exc:
        raise GitOperationError(ErrorMsg.STACK_LINK_FAILED(prs=pr_numbers, detail=exc)) from exc
    logger.info(logs.STACK_LINKED.format(prs=pr_numbers))


def fetch_fork_pr(pr_number: int) -> ForkPRInfo:
    from .branches import run_git

    try:
        raw = _run_gh("api", f"repos/{{owner}}/{{repo}}/pulls/{pr_number}")
    except GitOperationError as exc:
        raise GitOperationError(ErrorMsg.PR_NOT_FOUND(number=pr_number)) from exc

    pr_data: dict[str, object] = json.loads(raw)
    head = pr_data["head"]
    base = pr_data["base"]

    if not isinstance(head, dict) or not isinstance(base, dict):
        raise GitOperationError(ErrorMsg.PR_NOT_FOUND(number=pr_number))

    head_repo = head.get("repo")
    if not isinstance(head_repo, dict) or not head_repo.get("fork"):
        raise GitOperationError(ErrorMsg.PR_NOT_FOUND(number=pr_number))

    clone_url = str(head_repo["clone_url"])
    head_ref = str(head["ref"])
    base_ref = str(base["ref"])
    fork_full_name = str(head_repo["full_name"])

    local_ref = f"{PR_REF_PREFIX}{pr_number}"
    logger.info(logs.FETCHING_FORK_PR.format(number=pr_number, fork=fork_full_name))

    try:
        run_git("fetch", clone_url, f"{head_ref}:{local_ref}")
    except GitOperationError as exc:
        raise GitOperationError(
            ErrorMsg.PR_FETCH_FAILED(number=pr_number, detail=str(exc))
        ) from exc

    author = run_git("log", "-1", "--format=%aN <%aE>", local_ref)
    logger.info(logs.AUTHOR_PRESERVED.format(author=author))

    return ForkPRInfo(
        pr_number=pr_number,
        local_ref=local_ref,
        base_branch=base_ref,
        author=author,
        fork_full_name=fork_full_name,
    )


def fetch_fork_branch(user: str, branch: str) -> ForkPRInfo:
    from .branches import run_git

    repo_name = _run_gh("api", "repos/{owner}/{repo}", "--jq", ".name")

    try:
        raw = _run_gh("api", f"repos/{user}/{repo_name}")
    except GitOperationError as exc:
        raise GitOperationError(
            ErrorMsg.FORK_FETCH_FAILED(user=user, branch=branch, detail=str(exc))
        ) from exc

    repo_data: dict[str, object] = json.loads(raw)
    clone_url = str(repo_data["clone_url"])
    fork_full_name = str(repo_data["full_name"])

    local_ref = f"{FORK_REF_PREFIX}{user}-{branch}"
    logger.info(logs.FETCHING_FORK_BRANCH.format(branch=branch, fork=fork_full_name))

    try:
        run_git("fetch", clone_url, f"{branch}:{local_ref}")
    except GitOperationError as exc:
        raise GitOperationError(
            ErrorMsg.FORK_FETCH_FAILED(user=user, branch=branch, detail=str(exc))
        ) from exc

    author = run_git("log", "-1", "--format=%aN <%aE>", local_ref)
    logger.info(logs.AUTHOR_PRESERVED.format(author=author))

    base_branch = _run_gh("api", "repos/{owner}/{repo}", "--jq", ".default_branch")

    return ForkPRInfo(
        pr_number=None,
        local_ref=local_ref,
        base_branch=base_branch,
        author=author,
        fork_full_name=fork_full_name,
    )
