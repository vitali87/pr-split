"""Make every group a sub-PR that works on its own.

A backend packs hunks into groups by size and affinity, which can leave a
group needing code another group adds: a symbol it uses, the module its
tests import, a ``patch("pkg.mod.name")`` target, a conftest fixture, or a
lockfile's manifest. ``make_groups_standalone`` fixes the plan in steps:

1. Tests are separated from groups of unrelated code (``_peel_tests``).
2. Each group gets a dependency on every group whose code it needs
   (``find_needs``); groups that need each other form a cycle no order can
   satisfy, so they are combined into one (``_condense``).
3. Edits to existing tests move to the group whose code change they follow,
   or that group's own PR would fail its old tests
   (``_attach_test_updates``, ``_carry_imports``); needs are read again.
4. Each remaining test group rejoins the code it tests where that fits
   within the size limit (``_rehome_tests``).

A bigger PR that works beats two smaller ones that each fail on their own.
"""

from __future__ import annotations

import ast
import functools
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from loguru import logger

from .. import logs
from ..constants import AssignmentType
from ..schemas import GroupAssignment
from .chunker import recompute_estimated_loc
from .partitioning import derive_file_order_dependencies, reduce_transitive_dependencies
from .symbols import (
    LOCKFILE_MANIFESTS,
    added_lines_by_file,
    import_bindings,
    is_test_path,
    symbol_dependencies,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from unidiff import Hunk

    from ..diff_ops import ParsedDiff
    from ..schemas import Group

# A dotted string such as "pkg.mod.name": a mock.patch target, or an import path.
_DOTTED_STRING = re.compile(r"""["']([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)["']""")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_MIN_WORD_LENGTH = 4
# Words too common in code to tie a test to one change.
_COMMON_WORDS = frozenset(
    {
        "self",
        "None",
        "True",
        "False",
        "return",
        "assert",
        "import",
        "from",
        "with",
        "patch",
        "mock",
        "MagicMock",
        "call",
        "args",
        "kwargs",
        "result",
        "tests",
        "test",
        "def",
        "class",
        "str",
        "int",
        "list",
        "dict",
        "value",
        "path",
        "name",
        "output",
        "exit_code",
        "pytest",
        "raises",
        "assert_called_once_with",
        "assert_not_called",
        "return_value",
        "side_effect",
    }
)
_FROM_IMPORT = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+\(?([\w\s,*]+)")
_PLAIN_IMPORT = re.compile(r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)")


@dataclass(frozen=True)
class Need:
    """``user`` cannot work without code that ``provider`` adds."""

    user: str
    provider: str
    reason: str


def _hunk_lines(parsed_diff: ParsedDiff, path: str) -> list[str]:
    """Added and context lines of a file's hunks: its imports, as far as the diff shows."""
    for patch_file in parsed_diff.patch_set:
        if patch_file.path == path:
            return [
                line.value.rstrip("\n")
                for hunk in patch_file
                for line in hunk
                if line.is_added or line.is_context
            ]
    return []


def _module_files(module: str) -> tuple[str, str]:
    base = module.replace(".", "/")
    return f"{base}.py", f"{base}/__init__.py"


def _resolve_relative(path: str, module: str) -> str:
    """Turn ``from .x import y`` in ``path`` into an absolute dotted module."""
    level = len(module) - len(module.lstrip("."))
    if not level:
        return module
    package = list(PurePosixPath(path).parent.parts)
    package = package[: len(package) - (level - 1)] if level > 1 else package
    rest = module[level:]
    return ".".join([*package, rest] if rest else package)


def _imported_modules(path: str, source: str) -> list[tuple[str, list[str]]]:
    """(module, imported names) for each import in a Python source."""
    found: list[tuple[str, list[str]]] = []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # A partial view (hunks only) is rarely valid Python; read it line by line.
        for line in source.splitlines():
            if match := _FROM_IMPORT.match(line):
                names = [n.strip().split(" as ")[0] for n in match.group(2).split(",")]
                found.append((_resolve_relative(path, match.group(1)), [n for n in names if n]))
            elif match := _PLAIN_IMPORT.match(line):
                found.extend((m.strip(), []) for m in match.group(1).split(","))
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            found.append((_resolve_relative(path, module), [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
    return found


class _ImportGraph:
    """Python files reachable through imports, read at the dev branch's head."""

    def __init__(
        self, parsed_diff: ParsedDiff, read_file: Callable[[str], str | None] | None
    ) -> None:
        self._parsed_diff = parsed_diff
        self._read_file = read_file
        self._changed = {pf.path for pf in parsed_diff.patch_set}
        self._exists: dict[str, bool] = {}
        self._imports: dict[str, list[str]] = {}

    def _source(self, path: str) -> str | None:
        if self._read_file is not None:
            text = self._read_file(path)
            if text is not None:
                return text
        if path in self._changed:
            return "\n".join(_hunk_lines(self._parsed_diff, path))
        return None

    def text(self, path: str) -> str:
        return self._source(path) or ""

    def _is_file(self, path: str) -> bool:
        if path in self._changed:
            return True
        if self._read_file is None:
            return False
        if path not in self._exists:
            self._exists[path] = self._read_file(path) is not None
        return self._exists[path]

    def direct_imports(self, path: str) -> list[str]:
        if path not in self._imports:
            source = self._source(path) if path.endswith(".py") else None
            targets: list[str] = []
            for module, names in _imported_modules(path, source or ""):
                if not module:
                    continue
                for candidate in _module_files(module):
                    if self._is_file(candidate):
                        targets.append(candidate)
                        break
                for name in names:
                    for candidate in _module_files(f"{module}.{name}"):
                        if self._is_file(candidate):
                            targets.append(candidate)
                            break
            self._imports[path] = list(dict.fromkeys(targets))
        return self._imports[path]

    def direct_targets(self, path: str) -> set[str]:
        """Files a test imports, names in a string or patch target, or is named after."""
        text = self.text(path)
        stem = PurePosixPath(path).stem.removeprefix("test_").removesuffix("_test")
        named = {
            changed
            for changed in self._changed
            if re.search(rf"""["'/]{re.escape(PurePosixPath(changed).name)}["']""", text)
            or PurePosixPath(changed).stem == stem
        }
        patched = {
            target[0]
            for dotted in _DOTTED_STRING.findall(text)
            if (target := _patch_target(dotted, self._changed)) is not None
        }
        return set(self.direct_imports(path)) | named | patched

    def reachable(self, path: str) -> set[str]:
        """Every repository file ``path`` imports, directly or through other modules."""
        seen: set[str] = set()
        pending = [path]
        while pending:
            for target in self.direct_imports(pending.pop()):
                if target not in seen and target != path:
                    seen.add(target)
                    pending.append(target)
        return seen


def _groups_by_file(groups: list[Group]) -> dict[str, list[str]]:
    by_file: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        for assignment in group.assignments:
            if group.id not in by_file[assignment.file_path]:
                by_file[assignment.file_path].append(group.id)
    return by_file


def _symbol_needs(groups: list[Group], parsed_diff: ParsedDiff) -> Iterable[Need]:
    for user, definers in symbol_dependencies(groups, parsed_diff).items():
        for provider, names in definers.items():
            shown = ", ".join(sorted(names)[:3])
            yield Need(user, provider, f"{user} uses {shown} from {provider}")


def _patch_target(dotted: str, changed: Iterable[str]) -> tuple[str, str] | None:
    """(changed module file, attribute) that a string "pkg.mod.name" points at, if any."""
    parts = dotted.split(".")
    changed = set(changed)
    for cut in range(len(parts) - 1, 0, -1):
        for module_file in _module_files(".".join(parts[:cut])):
            if module_file in changed:
                return module_file, parts[cut]
    return None


def _binding_pattern(name: str) -> re.Pattern[str]:
    """Lines that bind ``name`` in a module: import it, define it, or assign it.

    A name the diff merely mentions (``typer.confirm`` where ``typer`` was
    imported long ago) is not one a patch target needs from the diff.
    """
    n = re.escape(name)
    return re.compile(
        rf"^\s*(?:from\s+\S+\s+)?import\b.*\b{n}\b"  # import name / from x import name
        rf"|^\s*(?:async\s+)?(?:def|class)\s+{n}\b"  # def name / class name
        rf"|^\s*{n}\s*(?::[^=]*)?=(?!=)"  # name = ... / name: T = ...
        rf"|^\s*{n}\s*,?\s*(?:#.*)?$"  # a line of a parenthesized import list
    )


def _patch_target_needs(
    groups: list[Group], parsed_diff: ParsedDiff, by_file: dict[str, list[str]]
) -> Iterable[Need]:
    """A string "pkg.mod.name" (a mock.patch target) needs the group that adds ``name``.

    Patching an attribute the module does not have yet fails, and the name
    hides in a string, where symbol analysis does not look for it.
    """
    added = added_lines_by_file(groups, parsed_diff)
    for group in groups:
        for lines in added.get(group.id, {}).values():
            for dotted in {d for line in lines for d in _DOTTED_STRING.findall(line)}:
                target = _patch_target(dotted, by_file)
                if target is None:
                    continue
                module_file, name = target
                binds = _binding_pattern(name)
                for provider in by_file[module_file]:
                    provided = added.get(provider, {}).get(module_file, [])
                    if provider != group.id and any(binds.search(line) for line in provided):
                        reason = f"{group.id} patches {dotted}, which {provider} adds"
                        yield Need(group.id, provider, reason)


def _test_import_needs(
    groups: list[Group],
    by_file: dict[str, list[str]],
    imports: _ImportGraph,
) -> Iterable[Need]:
    """A test-only group needs every changed module its tests import, and their conftests."""
    changed_sources = {path for path in by_file if not is_test_path(path)}
    changed_conftests = [p for p in by_file if PurePosixPath(p).name == "conftest.py"]
    for group in groups:
        # A group of code plus the test updates that follow it (see
        # _attach_test_updates) keeps only the needs its lines name: the
        # whole import reach of a test file, or its conftest, would tie it
        # to every change the tested code sees.
        if not all(is_test_path(a.file_path) for a in group.assignments):
            continue
        for assignment in group.assignments:
            path = assignment.file_path
            providers: dict[str, str] = {}
            if path.endswith(".py"):
                for target in sorted(imports.reachable(path) & changed_sources):
                    for provider in by_file[target]:
                        providers.setdefault(provider, target)
            else:
                # No import analysis for this language: pair test_x/x_test with x.
                stem = PurePosixPath(path).stem.removeprefix("test_").removesuffix("_test")
                for target in sorted(changed_sources):
                    if PurePosixPath(target).stem == stem:
                        for provider in by_file[target]:
                            providers.setdefault(provider, target)
            # A test that loads a file by path (a script, a data file) names it
            # in a string rather than importing it.
            source_text = imports.text(path)
            for target in sorted(changed_sources):
                quoted = rf"""["'/]{re.escape(PurePosixPath(target).name)}["']"""
                if re.search(quoted, source_text):
                    for provider in by_file[target]:
                        providers.setdefault(provider, target)
            test_dir = PurePosixPath(path).parent
            for conftest in changed_conftests:
                conftest_dir = PurePosixPath(conftest).parent
                if conftest != path and (
                    conftest_dir == test_dir or conftest_dir in test_dir.parents
                ):
                    for provider in by_file[conftest]:
                        providers.setdefault(provider, conftest)
            for provider, target in providers.items():
                if provider != group.id:
                    yield Need(group.id, provider, f"{group.id} tests {target}, in {provider}")


_DEF_LINE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s|^\S")


def _same_definition(above: list[str], gap: list[str]) -> bool:
    """Whether the unchanged ``gap`` lines leave the definition ``above`` ends in.

    They do at a new def/class or a top-level line, except the def that a
    decorator stack ``above`` ends in decorates: that is still the same one.
    """
    significant = [line for line in gap if line.strip()]
    last_above = next((line for line in reversed(above) if line.strip()), "")
    in_decorators = last_above.lstrip().startswith("@")
    for line in significant:
        if in_decorators and line.lstrip().startswith("@"):
            continue
        if in_decorators and re.match(r"^\s*(?:async\s+)?def\s", line):
            in_decorators = False  # the decorated def itself
            continue
        if _DEF_LINE.match(line):
            return False
    return True


def _change_span(hunk: Hunk) -> tuple[int, int]:
    """0-based [first, last) target-line range the hunk's added/removed lines cover."""
    position = hunk.target_start - 1
    first: int | None = None
    last = position
    for line in hunk:
        if line.is_added or line.is_removed:
            if first is None:
                first = position
            last = position + (1 if line.is_added else 0)
        if not line.is_removed:
            position += 1
    return (first if first is not None else position), last


def hunk_ties(
    parsed_diff: ParsedDiff, read_file: Callable[[str], str | None] | None
) -> dict[str, list[list[int]]]:
    """Per file, clusters of hunks that edit the same function or class.

    Git splits one edit into two hunks when enough unchanged lines separate
    them: a decorator added above a long signature and the parameter it
    injects, say. Apart, neither half works. Two neighbouring hunks are tied
    when no line between them starts a new definition (or leaves the
    indented block), read from the file at the dev branch's head.
    """
    ties: dict[str, list[list[int]]] = {}
    for patch_file in parsed_diff.patch_set:
        if len(patch_file) < 2 or read_file is None or patch_file.is_removed_file:
            continue
        text = read_file(patch_file.path)
        if text is None:
            continue
        lines = text.splitlines()
        clusters: list[list[int]] = [[0]]
        for idx in range(1, len(patch_file)):
            # Unchanged lines from the end of one edit to the start of the next,
            # including each hunk's own context around the edit.
            end = _change_span(patch_file[idx - 1])[1]
            start = _change_span(patch_file[idx])[0]
            gap = lines[end:start]
            if start > end and _same_definition(lines[:end], gap):
                clusters[-1].append(idx)
            else:
                clusters.append([idx])
        tied = [c for c in clusters if len(c) > 1]
        if tied:
            ties[patch_file.path] = tied
    return ties


def _tie_needs(
    groups: list[Group], ties: dict[str, list[list[int]]], parsed_diff: ParsedDiff
) -> Iterable[Need]:
    """Groups holding hunks of one function need each other: they must be one group."""
    counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    holder: dict[tuple[str, int], str] = {}
    for group in groups:
        for assignment in group.assignments:
            for idx in assignment.covered_indices(counts.get(assignment.file_path, 0)):
                holder[(assignment.file_path, idx)] = group.id
    for path, clusters in ties.items():
        for cluster in clusters:
            owners = sorted({holder[(path, i)] for i in cluster if (path, i) in holder})
            for a, b in itertools.pairwise(owners):
                reason = f"{a} and {b} edit the same definition in {path}"
                yield Need(a, b, reason)
                yield Need(b, a, reason)


def _lockfile_needs(by_file: dict[str, list[str]]) -> Iterable[Need]:
    """A lockfile and its manifest ship together: each needs the other."""
    for lock_path, lock_groups in by_file.items():
        lock = PurePosixPath(lock_path)
        for manifest_name in LOCKFILE_MANIFESTS.get(lock.name, ()):
            manifest_path = str(lock.with_name(manifest_name))
            for lock_group in lock_groups:
                for manifest_group in by_file.get(manifest_path, []):
                    if lock_group == manifest_group:
                        continue
                    reason = f"{lock_path} must ship with {manifest_path}"
                    yield Need(lock_group, manifest_group, reason)
                    yield Need(manifest_group, lock_group, reason)


def find_needs(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    read_file: Callable[[str], str | None] | None = None,
    ties: dict[str, list[list[int]]] | None = None,
) -> list[Need]:
    by_file = _groups_by_file(groups)
    imports = _ImportGraph(parsed_diff, read_file)
    needs: dict[tuple[str, str], Need] = {}
    for need in (
        *_tie_needs(groups, ties or {}, parsed_diff),
        *_symbol_needs(groups, parsed_diff),
        *_patch_target_needs(groups, parsed_diff, by_file),
        *_test_import_needs(groups, by_file, imports),
        *_lockfile_needs(by_file),
    ):
        needs.setdefault((need.user, need.provider), need)
    return list(needs.values())


def _strongly_connected(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative so a long chain cannot exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(edges.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            child = next(children, None)
            if child is not None:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(edges.get(child, ())))))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def _combine(keep: Group, absorbed: list[Group], hunk_counts: dict[str, int]) -> None:
    hunks: dict[str, set[int]] = defaultdict(set)
    for group in (keep, *absorbed):
        for assignment in group.assignments:
            count = hunk_counts.get(assignment.file_path, 0)
            hunks[assignment.file_path].update(assignment.covered_indices(count))
    keep.assignments = [
        GroupAssignment(
            file_path=path,
            assignment_type=(
                AssignmentType.WHOLE_FILE
                if sorted(indices) == list(range(hunk_counts.get(path, 0)))
                else AssignmentType.PARTIAL_HUNKS
            ),
            hunk_indices=sorted(indices),
        )
        for path, indices in sorted(hunks.items())
    ]
    descriptions = [g.description for g in (keep, *absorbed) if g.description]
    keep.description = "\n\n".join(dict.fromkeys(descriptions))


def _condense(
    groups: list[Group],
    edges: dict[str, set[str]],
    needs: list[Need],
    parsed_diff: ParsedDiff,
) -> tuple[list[Group], dict[str, set[str]]]:
    """Combine every set of groups that need each other into one group."""
    order = [g.id for g in groups]
    by_id = {g.id: g for g in groups}
    position = {gid: i for i, gid in enumerate(order)}
    merged_into: dict[str, str] = {}
    hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    for component in _strongly_connected(order, edges):
        if len(component) < 2:
            continue
        members = sorted(component, key=position.__getitem__)
        # Keep the largest group's id and title: it is most of what the PR holds.
        keep_id = max(members, key=lambda gid: (by_id[gid].estimated_loc, -position[gid]))
        others = [gid for gid in members if gid != keep_id]
        member_set = set(members)
        reasons = [n.reason for n in needs if n.user in member_set and n.provider in member_set]
        logger.info(
            logs.GROUPS_COMBINED.format(
                keep=keep_id,
                others=", ".join(others),
                reasons="; ".join(reasons[:3]) or "a test update and the code it adapts to",
            )
        )
        _combine(by_id[keep_id], [by_id[gid] for gid in others], hunk_counts)
        for gid in others:
            merged_into[gid] = keep_id

    kept = [g for g in groups if g.id not in merged_into]
    condensed: dict[str, set[str]] = {}
    for group in kept:
        members = {group.id} | {gid for gid, keep in merged_into.items() if keep == group.id}
        deps: set[str] = set()
        for member in members:
            deps.update(merged_into.get(dep, dep) for dep in edges[member])
        condensed[group.id] = deps - {group.id}
    recompute_estimated_loc(kept, parsed_diff)
    return kept, condensed


def _words(lines: Iterable[str]) -> set[str]:
    return {
        word
        for line in lines
        for word in _IDENTIFIER.findall(line)
        if len(word) >= _MIN_WORD_LENGTH and word not in _COMMON_WORDS
    }


def _attach_test_updates(
    groups: list[Group],
    edges: dict[str, set[str]],
    parsed_diff: ParsedDiff,
    ties: dict[str, list[list[int]]],
    direct_files: Callable[[str], set[str]],
) -> tuple[list[Group], dict[str, set[str]]]:
    """Move each changed existing test into the group whose code change it follows.

    A hunk that rewrites or removes existing test lines is usually adapting
    a test to new behaviour; left in a later test-only PR, the old test
    fails in the PR that changes the behaviour. It moves to the group whose
    changed source lines share the most names with it (the functions it
    calls, the messages it checks). The caller reads needs again afterwards,
    so the moved hunk's own needs (the names it uses) come along.
    """
    files = {pf.path: pf for pf in parsed_diff.patch_set}
    by_id = {g.id: g for g in groups}
    changed_words: dict[str, set[str]] = {}
    for group in groups:
        lines = [
            line.value
            for a in group.assignments
            if not is_test_path(a.file_path) and a.file_path in files
            for idx in a.covered_indices(len(files[a.file_path]))
            for line in files[a.file_path][idx]
        ]
        # Context lines count too: a changed body sits under the unchanged
        # def line that names it, which is what a test of it mentions.
        changed_words[group.id] = _words(lines)

    moved = 0
    for test_group in groups:
        for assignment in list(test_group.assignments):
            patch_file = files.get(assignment.file_path)
            if patch_file is None or not is_test_path(assignment.file_path):
                continue
            held = set(assignment.covered_indices(len(patch_file)))
            cluster_of = {i: c for c in ties.get(assignment.file_path, []) for i in c}
            done: set[int] = set()
            for idx in sorted(held):
                if idx in done:
                    continue
                # Hunks of one test function move together (see hunk_ties).
                cluster = [i for i in cluster_of.get(idx, [idx]) if i in held]
                done.update(cluster)
                hunks = [patch_file[i] for i in cluster]
                # A conftest change or an autouse fixture stubs out or sets up
                # what new code needs, so the code's own existing tests need it
                # even when it only adds lines.
                support = PurePosixPath(assignment.file_path).name == "conftest.py" or any(
                    "autouse" in line.value for hunk in hunks for line in hunk if line.is_added
                )
                if patch_file.is_added_file or not (
                    support or any(map(_changes_existing_tests, hunks))
                ):
                    continue  # new tests only: they can come after the code
                words = _words(line.value for hunk in hunks for line in hunk)
                direct = direct_files(assignment.file_path)
                # What the hunk's new lines patch ("pkg.mod.name") is the change it
                # adapts to; failing that, prefer a group holding a file the
                # test imports or names over one it reaches through others.
                patched = {
                    target[0]
                    for hunk in hunks
                    for line in hunk
                    if line.is_added
                    for dotted in _DOTTED_STRING.findall(line.value)
                    if (target := _patch_target(dotted, files)) is not None
                }
                candidates = [
                    (
                        any(a.file_path in patched for a in by_id[gid].assignments),
                        any(a.file_path in direct for a in by_id[gid].assignments),
                        len(words & changed_words[gid]),
                        gid,
                    )
                    for gid in edges.get(test_group.id, set()) | {test_group.id}
                    if changed_words.get(gid)
                ]
                if not candidates:
                    continue
                *_, score, target = max(candidates)
                if score == 0 or target == test_group.id:
                    continue
                for i in cluster:
                    _move_hunk(test_group, by_id[target], assignment.file_path, i, len(patch_file))
                moved += len(cluster)
    if moved:
        logger.info(logs.TEST_UPDATES_ATTACHED.format(count=moved))
    kept = [g for g in groups if g.assignments]
    kept_ids = {g.id for g in kept}
    for gid in list(edges):
        if gid not in kept_ids:
            # An emptied group's dependents inherit what it depended on.
            for other in kept_ids:
                if gid in edges[other]:
                    edges[other] |= edges[gid] - {other}
            del edges[gid]
    for gid in kept_ids:
        edges[gid] &= kept_ids
    recompute_estimated_loc(kept, parsed_diff)
    return kept, edges


_NEW_BLOCK = re.compile(r"^\s*(?:@|(?:async\s+)?def\s|class\s|$)")


def _changes_existing_tests(hunk: Hunk) -> bool:
    """Whether a test hunk edits existing tests rather than only adding new ones.

    Removed lines edit something. So do added lines that land inside an
    existing block, such as a new expected argument in an assertion: the
    first added line then continues indented code instead of opening a new
    function, class or decorator.
    """
    if hunk.removed:
        return True
    added = [line.value.rstrip("\n") for line in hunk if line.is_added]
    first = next((line for line in added if line.strip()), "")
    return bool(first) and first[:1].isspace() and not _NEW_BLOCK.match(first)


def _ancestors(edges: dict[str, set[str]], gid: str) -> set[str]:
    seen: set[str] = set()
    pending = list(edges.get(gid, ()))
    while pending:
        dep = pending.pop()
        if dep not in seen:
            seen.add(dep)
            pending.extend(edges.get(dep, ()))
    return seen


def _carry_imports(
    groups: list[Group], edges: dict[str, set[str]], parsed_diff: ParsedDiff
) -> None:
    """Move a test file's added imports to the groups that now use them.

    Test updates moved next to their code (``_attach_test_updates``) still
    use names the file's import block binds, and that block stays with the
    remaining tests, which run after all the code: each moved update would
    need a group that needs it back. Each import hunk moves instead to the
    group using it that the other users already build on (or the one with
    the fewest dependencies); the remaining tests then build on that group.
    """
    files = {pf.path: pf for pf in parsed_diff.patch_set}
    holder: dict[tuple[str, int], Group] = {}
    for group in groups:
        for a in group.assignments:
            if a.file_path in files:
                for idx in a.covered_indices(len(files[a.file_path])):
                    holder[(a.file_path, idx)] = group
    for path, patch_file in files.items():
        if not is_test_path(path) or patch_file.is_added_file:
            continue
        for idx, hunk in enumerate(patch_file):
            owner = holder.get((path, idx))
            if owner is None or not all(is_test_path(a.file_path) for a in owner.assignments):
                continue
            bound = import_bindings([line.value.rstrip("\n") for line in hunk if line.is_added])
            if not bound:
                continue
            users = []
            for other_idx, other in enumerate(patch_file):
                user = holder.get((path, other_idx))
                if user is None or user is owner or user in users:
                    continue
                if _words(line.value for line in other if line.is_added) & bound:
                    users.append(user)
            if not users:
                continue
            ancestry = {u.id: _ancestors(edges, u.id) for u in users}
            target = next(
                (u for u in users if all(u.id in ancestry[o.id] or o is u for o in users)),
                min(users, key=lambda u: len(ancestry[u.id])),
            )
            _move_hunk(owner, target, path, idx, len(patch_file))
            holder[(path, idx)] = target
            edges[owner.id].add(target.id)


def _move_hunk(source: Group, target: Group, path: str, idx: int, hunk_count: int) -> None:
    for assignment in list(source.assignments):
        if assignment.file_path != path:
            continue
        remaining = [i for i in assignment.covered_indices(hunk_count) if i != idx]
        if remaining:
            assignment.assignment_type = AssignmentType.PARTIAL_HUNKS
            assignment.hunk_indices = remaining
        else:
            source.assignments.remove(assignment)
    for assignment in target.assignments:
        if assignment.file_path == path:
            indices = sorted(set(assignment.covered_indices(hunk_count)) | {idx})
            assignment.hunk_indices = indices
            assignment.assignment_type = (
                AssignmentType.WHOLE_FILE
                if indices == list(range(hunk_count))
                else AssignmentType.PARTIAL_HUNKS
            )
            return
    target.assignments.append(
        GroupAssignment(
            file_path=path,
            assignment_type=(
                AssignmentType.WHOLE_FILE if hunk_count == 1 else AssignmentType.PARTIAL_HUNKS
            ),
            hunk_indices=[idx],
        )
    )


def _peel_tests(groups: list[Group], keep_declared_deps: bool) -> list[Group]:
    """Split each group that mixes source and test files into a source and a test group.

    A test that sits in an unrelated source group ties that group to the code
    it tests; a few such ties make every group need every other, and fixing
    that by combining would leave one giant PR. Tests are placed again once
    the source groups are in order (see ``_rehome_tests``).
    """
    numbered = all(re.fullmatch(r"pr-\d+", g.id) for g in groups)
    next_number = max((int(g.id[3:]) for g in groups), default=0) + 1 if numbered else 0
    peeled: list[Group] = []
    for group in groups:
        tests = [a for a in group.assignments if is_test_path(a.file_path)]
        if not tests or len(tests) == len(group.assignments):
            peeled.append(group)
            continue
        group.assignments = [a for a in group.assignments if not is_test_path(a.file_path)]
        if numbered:
            test_id = f"pr-{next_number}"
            next_number += 1
        else:
            test_id = f"{group.id}-tests"
        test_group = group.model_copy(
            update={
                "id": test_id,
                "title": f"Tests for {group.title}",
                "assignments": tests,
                "depends_on": list(group.depends_on) if keep_declared_deps else [],
            }
        )
        peeled.extend([group, test_group])
    return peeled


def _is_acyclic(edges: dict[str, set[str]]) -> bool:
    try:
        tuple(TopologicalSorter(edges).static_order())
    except CycleError:
        return False
    return True


def _fold(edges: dict[str, set[str]], absorbed: str, keep: str) -> dict[str, set[str]]:
    """The dependency edges once ``absorbed`` is part of ``keep``."""
    folded: dict[str, set[str]] = {}
    for gid, deps in edges.items():
        if gid == absorbed:
            continue
        mapped = {keep if d == absorbed else d for d in deps}
        if gid == keep:
            mapped |= {keep if d == absorbed else d for d in edges[absorbed]}
        folded[gid] = mapped - {gid}
    return folded


def _rehome_tests(
    groups: list[Group],
    edges: dict[str, set[str]],
    parsed_diff: ParsedDiff,
    max_loc: int | None,
) -> tuple[list[Group], dict[str, set[str]]]:
    """Put each test-only group back with the code it tests, where that fits.

    A test group joins the one group it needs that no other group it needs
    builds on (so the combined PR still comes after all of them), if the
    result stays within ``max_loc``. Otherwise it stays a PR of its own,
    ordered after the code it tests.
    """
    by_id = {g.id: g for g in groups}
    hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    for test_group in list(groups):
        if not all(is_test_path(a.file_path) for a in test_group.assignments):
            continue
        providers = edges.get(test_group.id, set())
        test_stems = {
            PurePosixPath(a.file_path).stem.removeprefix("test_").removesuffix("_test")
            for a in test_group.assignments
        }

        def preference(gid: str, stems: set[str] = test_stems) -> tuple[bool, int]:
            files = {PurePosixPath(a.file_path).stem for a in by_id[gid].assignments}
            return (not files & stems, by_id[gid].estimated_loc)

        for candidate in sorted(providers, key=preference):
            size = by_id[candidate].estimated_loc + test_group.estimated_loc
            if max_loc is not None and size > max_loc:
                continue
            folded = _fold(edges, test_group.id, candidate)
            if not _is_acyclic(folded):
                continue
            _combine(by_id[candidate], [test_group], hunk_counts)
            by_id[candidate].estimated_loc = size
            edges = folded
            groups = [g for g in groups if g.id != test_group.id]
            break
    return groups, edges


def make_groups_standalone(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    read_file: Callable[[str], str | None] | None = None,
    max_loc: int | None = None,
    *,
    keep_declared_deps: bool = True,
) -> list[Group]:
    """Give every group the dependencies it needs, so each sub-PR works on its own.

    Tests are first separated from unrelated code, then each group gets a
    dependency on every group whose code it needs; groups that need each
    other are combined, and each test goes back to the group whose code it
    tests when that fits within ``max_loc``. ``keep_declared_deps`` keeps the
    dependencies a planner chose (an LLM's); without it, the only declared
    edges are file order, which is derived again after tests are separated.
    """
    if len(groups) < 2:
        return groups
    if read_file is not None:
        # Every pass below reads the same files; git show them once.
        read_file = functools.cache(read_file)
    groups = _peel_tests(groups, keep_declared_deps)
    if not keep_declared_deps:
        # A generated plan's edges only encode file order, and peeling moved
        # files between groups; derive them again from the new layout.
        for group in groups:
            group.depends_on = []
        derive_file_order_dependencies(groups)
    ties = hunk_ties(parsed_diff, read_file)
    needs = find_needs(groups, parsed_diff, read_file, ties)
    edges: dict[str, set[str]] = {g.id: set(g.depends_on) for g in groups}
    for need in needs:
        edges[need.user].add(need.provider)
    recompute_estimated_loc(groups, parsed_diff)
    kept, condensed = _condense(groups, edges, needs, parsed_diff)
    imports = _ImportGraph(parsed_diff, read_file)
    kept, condensed = _attach_test_updates(
        kept, condensed, parsed_diff, ties, imports.direct_targets
    )
    _carry_imports(kept, condensed, parsed_diff)
    # A moved hunk brings needs its new group did not have (an import line
    # left behind in its old group, say); read them again from the new layout.
    needs = find_needs(kept, parsed_diff, read_file, ties)
    for need in needs:
        condensed[need.user].add(need.provider)
    kept, condensed = _condense(kept, condensed, needs, parsed_diff)
    kept, condensed = _rehome_tests(kept, condensed, parsed_diff, max_loc)
    for group in kept:
        group.depends_on = sorted(condensed[group.id])
    reduce_transitive_dependencies(kept)
    recompute_estimated_loc(kept, parsed_diff)
    return kept
