"""Symbol-level dependencies between groups, read from the diff's added lines.

A group that uses a name another group newly defines must build on that
group, or its sub-PR cannot compile or import on its own. Shared files are
not enough to see this: a new test file and the module it imports touch no
common file. The analysis is lexical and language-agnostic: it recognises
common definition forms and treats any later whole-word use as a reference.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..diff_ops import ParsedDiff
    from ..schemas import Group

_DEFINITION_PATTERNS = (
    # Python, Ruby: def/class, including async def.
    re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)"),
    # JavaScript/TypeScript, Go, Rust, Swift, Kotlin, PHP functions.
    re.compile(
        r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:function|fn|func|fun)\s+([A-Za-z_]\w*)"
    ),
    # Types across languages.
    re.compile(
        r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:abstract\s+|final\s+|sealed\s+|data\s+)*"
        r"(?:class|struct|enum|interface|trait|type|protocol|record)\s+([A-Za-z_]\w*)"
    ),
    # Top-level constants: UPPER_CASE = ... (Python) or const/let/var NAME = (JS/TS).
    re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=(?!=)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*[:=]"),
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
# Names too generic to tie two groups together.
_MIN_NAME_LENGTH = 4
_IGNORED = frozenset({"main", "self", "this", "init", "test", "setup", "teardown", "None", "True"})


def _added_lines_by_group(groups: list[Group], parsed_diff: ParsedDiff) -> dict[str, list[str]]:
    files = {pf.path: pf for pf in parsed_diff.patch_set}
    added: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        for assignment in group.assignments:
            patch_file = files.get(assignment.file_path)
            if patch_file is None:
                continue
            for idx in assignment.covered_indices(len(patch_file)):
                if idx >= len(patch_file):
                    continue
                added[group.id].extend(
                    line.value.rstrip("\n") for line in patch_file[idx] if line.is_added
                )
    return added


def _definitions(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        for pattern in _DEFINITION_PATTERNS:
            match = pattern.match(line)
            if match:
                names.add(match.group(1))
    return {n for n in names if len(n) >= _MIN_NAME_LENGTH and n not in _IGNORED}


def symbol_dependencies(
    groups: list[Group], parsed_diff: ParsedDiff
) -> dict[str, dict[str, set[str]]]:
    """Map user group -> definer group -> the names it uses from that definer.

    Only names defined by exactly one group count: a name several groups
    define (an overload, a common helper name) says nothing about order.
    """
    added = _added_lines_by_group(groups, parsed_diff)
    defined_by: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        for name in _definitions(added.get(group.id, [])):
            defined_by[name].append(group.id)
    owner = {name: gids[0] for name, gids in defined_by.items() if len(gids) == 1}

    uses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for group in groups:
        own = _definitions(added.get(group.id, []))
        for line in added.get(group.id, []):
            for name in _IDENTIFIER.findall(line):
                definer = owner.get(name)
                if definer is not None and definer != group.id and name not in own:
                    uses[group.id][definer].add(name)
    return {user: dict(definers) for user, definers in uses.items()}
