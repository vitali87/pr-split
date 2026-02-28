from __future__ import annotations

import json
import subprocess

from loguru import logger

from .. import logs
from ..constants import BRANCH_PREFIX
from ..exceptions import ErrorMsg, GitOperationError


def run_gh(*args: str) -> str:
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
        run_gh("auth", "status")
    except GitOperationError:
        return False
    return True


def create_pr(head: str, base: str, title: str, body: str) -> tuple[int, str]:
    try:
        output = run_gh(
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
        )
    except GitOperationError as exc:
        raise GitOperationError(ErrorMsg.PR_CREATE_FAILED(group=head, detail=str(exc))) from exc
    pr_url = output.strip().splitlines()[-1]
    pr_number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])
    logger.info(logs.PR_CREATED.format(number=pr_number, url=pr_url))
    return pr_number, pr_url


def close_pr(pr_number: int) -> None:
    run_gh("pr", "close", str(pr_number))
    logger.info(logs.PR_CLOSED.format(number=pr_number))


def list_pr_split_prs() -> list[tuple[int, str]]:
    output = run_gh(
        "pr",
        "list",
        "--json",
        "number,url,headRefName",
        "--limit",
        "200",
    )
    items: list[dict[str, str | int]] = json.loads(output) if output else []
    return [
        (int(item["number"]), str(item["url"]))
        for item in items
        if str(item["headRefName"]).startswith(BRANCH_PREFIX)
    ]
