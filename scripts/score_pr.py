"""Score a PR and generate a markdown comment with the split plan."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


def _set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def _skip(reason: str) -> None:
    print(reason)
    _set_output("total_groups", "1")
    _set_output("objective", "0")
    _set_output("should_split", "false")


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"Error: {name} must be an integer, got '{raw}'.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    max_loc = _parse_int_env("MAX_LOC", 400)
    min_loc_raw = os.environ.get("MIN_LOC", "")
    strategy = os.environ.get("PARTITION_STRATEGY", "graph")
    priority = os.environ.get("PRIORITY", "orthogonal")
    threshold = _parse_int_env("THRESHOLD_GROUPS", 2)
    pr_number = os.environ.get("PR_NUMBER", "")
    base_branch = os.environ["BASE_BRANCH"]
    head_branch = os.environ["HEAD_BRANCH"]

    # Fetch refs — use refs/pull/{n}/head for fork compatibility
    _run(["git", "fetch", "origin", base_branch])
    if pr_number:
        pr_ref = f"refs/pull/{pr_number}/head"
        local_head = f"pr-split/head-{pr_number}"
        _run(["git", "fetch", "origin", f"{pr_ref}:{local_head}"])
    else:
        _run(["git", "fetch", "origin", head_branch])
        local_head = f"origin/{head_branch}"

    # Compute diff stats
    result = _run(["git", "diff", "--numstat", f"origin/{base_branch}...{local_head}"])

    total_added = 0
    total_removed = 0
    file_count = 0
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            added = int(parts[0]) if parts[0] != "-" else 0
            removed = int(parts[1]) if parts[1] != "-" else 0
            total_added += added
            total_removed += removed
            file_count += 1

    total_loc = total_added + total_removed
    _set_output("total_loc", str(total_loc))

    if total_loc <= max_loc:
        _skip(f"PR has {total_loc} LOC — under the {max_loc} threshold, no split needed.")
        return

    # Create local branch refs for pr-split
    _run(["git", "branch", "-f", base_branch, f"origin/{base_branch}"])
    _run(["git", "branch", "-f", head_branch, local_head])

    # Run pr-split in dry-run mode
    cmd = [
        "pr-split", "split", head_branch,
        "--base", base_branch,
        "--partition-strategy", strategy,
        "--priority", priority,
        "--max-loc", str(max_loc),
        "--dry-run",
    ]
    if min_loc_raw:
        cmd.extend(["--min-loc", min_loc_raw])

    result = subprocess.run(cmd, capture_output=True, text=True, input="done\n")
    if result.returncode != 0:
        print(f"pr-split failed:\n{result.stderr}", file=sys.stderr)
        _skip("pr-split failed to generate a plan.")
        return

    plan_path = ".pr-split/plan.json"
    if not os.path.exists(plan_path):
        _skip("No plan file generated.")
        return

    with open(plan_path) as f:
        plan = json.load(f)

    groups = plan.get("groups", [])
    total_groups = len(groups)

    max_group_loc = max((g["estimated_loc"] for g in groups), default=0)
    overflow = sum(max(0, g["estimated_loc"] - max_loc) for g in groups)
    file_groups: dict[str, set[str]] = {}
    for g in groups:
        for a in g.get("assignments", []):
            file_groups.setdefault(a["file_path"], set()).add(g["id"])
    file_scatter = sum(max(0, len(gids) - 1) for gids in file_groups.values())

    objective = overflow * 1000 + file_scatter * 50 + total_groups
    should_split = total_groups >= threshold

    _set_output("total_groups", str(total_groups))
    _set_output("objective", str(objective))
    _set_output("should_split", str(should_split).lower())

    print(f"PR: {total_loc} LOC across {file_count} files")
    print(f"Split plan: {total_groups} groups, objective={objective}")
    print(f"Should split: {should_split}")

    # Generate markdown comment
    lines = [
        "<!-- pr-split-score -->",
        "## pr-split analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total LOC | {total_loc:,} |",
        f"| Files changed | {file_count} |",
        f"| Suggested groups | {total_groups} |",
        f"| Largest group | {max_group_loc:,} LOC |",
        f"| LOC overflow | {overflow:,} |",
        f"| File scatter | {file_scatter} |",
        "",
    ]

    if should_split:
        lines.append(
            f"This PR has **{total_loc:,} LOC** and could be split into "
            f"**{total_groups} smaller PRs**:"
        )
        lines.append("")
        lines.append("| Group | Title | Diff | Depends On | Files |")
        lines.append("|-------|-------|------|------------|-------|")
        for g in groups:
            files = ", ".join(
                f"`{_md_escape(a['file_path'])}`"
                for a in g.get("assignments", [])
            )
            deps = ", ".join(g.get("depends_on", [])) or "—"
            diff_str = (
                f"+{g.get('estimated_added', 0)}/-{g.get('estimated_removed', 0)}"
            )
            title = _md_escape(g["title"])
            gid = _md_escape(g["id"])
            lines.append(f"| {gid} | {title} | {diff_str} | {deps} | {files} |")
        lines.append("")
        lines.append(
            "*Run `pr-split split` locally to create these sub-PRs, "
            "or adjust `--max-loc` to change the target size.*"
        )
    else:
        lines.append("This PR is within acceptable size limits.")

    comment = "\n".join(lines)
    tmp_dir = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
    comment_path = Path(tmp_dir) / "pr-split-comment.md"
    comment_path.write_text(comment)
    _set_output("comment_path", str(comment_path))


if __name__ == "__main__":
    main()
