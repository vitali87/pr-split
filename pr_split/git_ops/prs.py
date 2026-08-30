from __future__ import annotations

import json
import os
import re
import subprocess

from loguru import logger

from .. import logs
from ..constants import FORK_REF_PREFIX, PR_REF_PREFIX
from ..exceptions import ErrorMsg, GitOperationError
from ..types_defs import ForkPRInfo
from .branches import run_git


def _run_gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout.strip()


# Matches the host of an https/ssh/git URL or an scp-like ``user@host:path``.
# Local paths and file:// URLs deliberately do not match: they have no host
# for gh to authenticate against. A dot is required only in the bare
# ``host:path`` form, where it separates real hosts from Windows drives and
# relative paths; scheme and ``user@`` forms accept single-label hosts
# (``git@ghe:org/repo``), which gh itself supports.
_REMOTE_HOST_RE = re.compile(
    r"^(?:"
    r"(?:https?|ssh|git|git\+ssh|ssh\+git)://(?:[^@/]+@)?(?P<scheme_host>[^.:/@][^:/@]*)"
    r"|[^@/:]+@(?P<scp_host>[^.:/@][^:/@]*)"
    r"|(?P<bare_host>[^.:/@][^:/@]*\.[^:/@]+)"
    r")(?::\d+)?[:/]"
)


def gh_host() -> str:
    """The GitHub host pr-split talks to, resolved the way gh does.

    gh targets the host of the repository's remote and only treats GH_HOST
    as an override, so an enterprise-only checkout must be checked against
    its own host, not github.com.
    """
    override = os.environ.get("GH_HOST")
    if override:
        return override.lower()
    try:
        remote = run_git("remote", "get-url", "origin")
    except GitOperationError:
        return "github.com"
    match = _REMOTE_HOST_RE.match(remote.strip())
    if not match:
        return "github.com"
    host = match.group("scheme_host") or match.group("scp_host") or match.group("bare_host")
    # gh lowercases hosts it parses from remotes; `--hostname` is case-sensitive.
    return host.lower()


def check_gh_auth() -> bool:
    # Without --hostname, `gh auth status` exits 1 when *any* configured
    # host is unauthenticated (a stale enterprise token), even though every
    # call pr-split makes targets one host.
    try:
        _run_gh("auth", "status", "--hostname", gh_host())
    except GitOperationError:
        return False
    return True


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
        logger.warning(logs.STACK_LINK_FAILED.format(prs=pr_numbers, detail=exc))
        return
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
