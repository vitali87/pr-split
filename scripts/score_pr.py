"""Score a PR and generate a markdown comment with the split plan."""

from __future__ import annotations

import os
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def _set_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main() -> None:
    max_loc = os.environ.get("MAX_LOC", "400")
    min_loc = os.environ.get("MIN_LOC", "")
    strategy = os.environ.get("PARTITION_STRATEGY", "graph")
    priority = os.environ.get("PRIORITY", "orthogonal")
    threshold = int(os.environ.get("THRESHOLD_GROUPS", "2"))
    base_branch = os.environ["BASE_BRANCH"]
    head_branch = os.environ["HEAD_BRANCH"]

    # Compute diff stats
    diff_numstat = _run(["git", "diff", "--numstat", f"{base_branch}...{head_branch}"])

    total_added = 0
    total_removed = 0
    file_count = 0
    for line in diff_numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            added = int(parts[0]) if parts[0] != "-" else 0
            removed = int(parts[1]) if parts[1] != "-" else 0
            total_added += added
            total_removed += removed
            file_count += 1

    total_loc = total_added + total_removed

    _set_output("total_loc", str(total_loc))

    # If the diff is tiny, skip scoring
    if total_loc <= int(max_loc):
        _set_output("total_groups", "1")
        _set_output("objective", "0")
        _set_output("should_split", "false")
        print(f"PR has {total_loc} LOC — under the {max_loc} threshold, no split needed.")
        return

    # Run pr-split in dry-run mode with the graph/cp_sat backend
    cmd = [
        "pr-split", "split", head_branch,
        "--base", base_branch,
        "--partition-strategy", strategy,
        "--priority", priority,
        "--max-loc", max_loc,
        "--dry-run",
    ]
    if min_loc:
        cmd.extend(["--min-loc", min_loc])

    # pr-split writes the plan to .pr-split/plan.json
    result = subprocess.run(cmd, capture_output=True, text=True, input="done\n")
    if result.returncode != 0:
        print(f"pr-split failed:\n{result.stderr}", file=sys.stderr)
        _set_output("total_groups", "1")
        _set_output("objective", "0")
        _set_output("should_split", "false")
        return

    # Parse the saved plan
    import json

    plan_path = ".pr-split/plan.json"
    if not os.path.exists(plan_path):
        print("No plan file generated", file=sys.stderr)
        _set_output("total_groups", "1")
        _set_output("objective", "0")
        _set_output("should_split", "false")
        return

    with open(plan_path) as f:
        plan = json.load(f)

    groups = plan.get("groups", [])
    total_groups = len(groups)

    # Compute a simple score
    max_group_loc = max((g["estimated_loc"] for g in groups), default=0)
    overflow = sum(max(0, g["estimated_loc"] - int(max_loc)) for g in groups)
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
                f"`{a['file_path']}`" for a in g.get("assignments", [])
            )
            deps = ", ".join(g.get("depends_on", [])) or "—"
            diff_str = f"+{g.get('estimated_added', 0)}/-{g.get('estimated_removed', 0)}"
            lines.append(f"| {g['id']} | {g['title']} | {diff_str} | {deps} | {files} |")
        lines.append("")
        lines.append(
            "*Run `pr-split split` locally to create these sub-PRs, "
            "or adjust `--max-loc` to change the target size.*"
        )
    else:
        lines.append("This PR is within acceptable size limits.")

    comment = "\n".join(lines)
    with open("/tmp/pr-split-comment.md", "w") as f:
        f.write(comment)


if __name__ == "__main__":
    main()
