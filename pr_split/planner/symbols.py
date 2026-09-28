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
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..diff_ops import ParsedDiff
    from ..schemas import Group

# Class-level fields: dataclass, TypedDict, pydantic and attrs attributes
# (``    name: type`` one indent in). Code reads them as keys and attributes. A
# line ending in a comma is a parameter of a multi-line signature instead.
_FIELD_PATTERN = re.compile(
    r"^(?: {4}|\t)([A-Za-z_]\w*)\s*:\s*[A-Za-z_][\w\[\], .|\"']*(?:=.*[^,\s])?\s*$"
)
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
    # Constants: UPPER_CASE = ... at top level or in a class body (an enum, a
    # message catalogue), or const/let/var NAME = (JS/TS).
    re.compile(r"^(?:(?: {4}|\t))?([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=(?!=)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*[:=]"),
    _FIELD_PATTERN,
)
_FROM_IMPORT = re.compile(r"^\s*from\s+\S+\s+import\s+(.*)$")
_PLAIN_IMPORT = re.compile(r"^\s*import\s+(.*)$")
_ANNOTATED_NAME = re.compile(r"^\s+([A-Za-z_]\w*)\s*:")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
# Names too generic to tie two groups together.
_MIN_NAME_LENGTH = 4
_IGNORED = frozenset({"main", "self", "this", "init", "test", "setup", "teardown", "None", "True"})


def added_lines_by_file(
    groups: list[Group], parsed_diff: ParsedDiff
) -> dict[str, dict[str, list[str]]]:
    """group id -> file path -> the lines that group's hunks add to the file."""
    files = {pf.path: pf for pf in parsed_diff.patch_set}
    added: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for group in groups:
        for assignment in group.assignments:
            patch_file = files.get(assignment.file_path)
            if patch_file is None:
                continue
            for idx in assignment.covered_indices(len(patch_file)):
                if idx >= len(patch_file):
                    continue
                added[group.id][assignment.file_path].extend(
                    line.value.rstrip("\n") for line in patch_file[idx] if line.is_added
                )
    return added


# Lockfiles, each resolved from the manifest(s) named here.
LOCKFILE_MANIFESTS = {
    "uv.lock": ("pyproject.toml",),
    "poetry.lock": ("pyproject.toml",),
    "pdm.lock": ("pyproject.toml",),
    "Pipfile.lock": ("Pipfile",),
    "package-lock.json": ("package.json",),
    "npm-shrinkwrap.json": ("package.json",),
    "yarn.lock": ("package.json",),
    "pnpm-lock.yaml": ("package.json",),
    "bun.lockb": ("package.json",),
    "Cargo.lock": ("Cargo.toml",),
    "go.sum": ("go.mod",),
    "Gemfile.lock": ("Gemfile",),
    "composer.lock": ("composer.json",),
    "mix.lock": ("mix.exs",),
}
LOCKFILES = frozenset(LOCKFILE_MANIFESTS)

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})
_TEST_SUFFIXES = ("_test", "_tests", "_spec", ".test", ".spec", "Test", "Tests", "Spec")


def is_test_path(path: str) -> bool:
    """Whether a path holds tests (or test fixtures such as conftest.py)."""
    pure = PurePosixPath(path)
    stem = pure.name.split(".", 1)[0] if pure.suffix else pure.name
    full_stem = pure.name[: -len(pure.suffix)] if pure.suffix else pure.name
    return (
        any(part in _TEST_DIRS for part in pure.parts[:-1])
        or stem.startswith("test_")
        or stem == "conftest"
        or full_stem.endswith(_TEST_SUFFIXES)
    )


def _definitions(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        for pattern in _DEFINITION_PATTERNS:
            match = pattern.match(line)
            if match:
                name = match.group(1)
                # A field named like a common word (branch, paths) is mentioned
                # all over a codebase; only a specific one ties code together.
                if pattern is _FIELD_PATTERN and (
                    ("_" not in name and len(name) < 8) or line.rstrip().endswith(("[", "("))
                ):
                    continue
                names.add(name)
    return {n for n in names if len(n) >= _MIN_NAME_LENGTH and n not in _IGNORED}


def _bound_name(item: str) -> str | None:
    """The name ``x`` or ``y as x`` binds, or None for noise such as ``(`` or ``*``."""
    parts = item.strip().strip("(),").split()
    name = parts[-1] if parts else ""
    return name.split(".")[0] if _IDENTIFIER.fullmatch(name.split(".")[0] or "-") else None


def import_bindings(lines: list[str]) -> set[str]:
    """Names that the import statements among ``lines`` bind in their file."""
    bound: set[str] = set()
    in_list = False
    for line in lines:
        if in_list:
            items = line.split(")")[0]
            bound.update(filter(None, (_bound_name(i) for i in items.split(","))))
            in_list = ")" not in line
            continue
        match = _FROM_IMPORT.match(line) or _PLAIN_IMPORT.match(line)
        if not match:
            continue
        items = match.group(1).split("#")[0]
        in_list = items.strip().startswith("(") and ")" not in items
        bound.update(filter(None, (_bound_name(i) for i in items.split(","))))
    return {name for name in bound if name != "import"}


def _is_test_module(path: str) -> bool:
    """A test file proper (test_x.py, x_test.go), not a shared fixture or helper module."""
    stem = PurePosixPath(path).stem
    return stem.startswith("test_") or stem.endswith(("_test", ".test", ".spec", "_spec"))


def _can_import(user_path: str, def_path: str, user_is_test: bool) -> bool:
    """Whether code in ``user_path`` can use a name that ``def_path`` defines.

    Non-test code never imports tests. A test file's own helpers stay in that
    file (another test file defining a same-named helper is not using it);
    only conftest files and shared helper modules serve other test files.
    """
    if not is_test_path(def_path):
        return True
    if not user_is_test:
        return False
    return user_path == def_path or not _is_test_module(def_path)


# A keyword argument (``name=value``, no spaces, as PEP 8 writes it) or a
# dict key (``"name": value``): how a TypedDict or dataclass field is filled.
_FIELD_WRITE = re.compile(r"""[(,\s]([A-Za-z_]\w*)=(?!=)|["']([A-Za-z_]\w*)["']\s*:""")


def _new_fields(added: dict[str, dict[str, list[str]]]) -> set[str]:
    return {
        match.group(1)
        for by_file in added.values()
        for lines in by_file.values()
        for line in lines
        if (match := _FIELD_PATTERN.match(line))
    }


def _existing_definitions(parsed_diff: ParsedDiff) -> dict[str, set[str]]:
    """Per file, names its removed lines define: edited, not new, definitions.

    A changed signature or field is rewritten as a removed and an added line;
    the name existed before the diff, so nothing needs the group that edits it.
    """
    existing: dict[str, set[str]] = {}
    for pf in parsed_diff.patch_set:
        removed = [line.value.rstrip("\n") for hunk in pf for line in hunk if line.is_removed]
        # Also any ``name:`` a removed line starts with: a parameter or field
        # in whatever form the old line had it.
        existing[pf.path] = _definitions(removed) | {
            m.group(1) for line in removed if (m := _ANNOTATED_NAME.match(line))
        }
    return existing


def symbol_dependencies(
    groups: list[Group], parsed_diff: ParsedDiff
) -> dict[str, dict[str, set[str]]]:
    """Map user group -> definer group -> the names it uses from that definer.

    Only names defined by exactly one group count: a name several groups
    define (an overload, a common helper name) says nothing about order. A
    name defined in a test file is never a dependency of non-test code, which
    cannot import it; the match is only a same-named attribute or word.
    """
    added = added_lines_by_file(groups, parsed_diff)
    existing = _existing_definitions(parsed_diff)
    defined_by: dict[str, dict[str, str]] = defaultdict(dict)
    for group in groups:
        for path, lines in added.get(group.id, {}).items():
            for name in _definitions(lines) - existing.get(path, set()):
                defined_by[name].setdefault(group.id, path)
    owner = {name: next(iter(by.items())) for name, by in defined_by.items() if len(by) == 1}
    # A new field is only useful once something fills it: readers also need
    # every group that writes it (``name=`` or ``"name":``).
    fields = set(owner) & _new_fields(added)
    writers: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        for path, lines in added.get(group.id, {}).items():
            if is_test_path(path):
                continue  # a test building a fixture fills nothing code reads
            for line in lines:
                for match in _FIELD_WRITE.finditer(line):
                    name = match.group(1) or match.group(2)
                    if name in fields:
                        writers[name].add(group.id)

    # An added import binds a name for its own file only: another group's
    # lines in that file that use the name need the group holding the import.
    importer: dict[tuple[str, str], set[str]] = defaultdict(set)
    for group in groups:
        for path, lines in added.get(group.id, {}).items():
            for name in import_bindings(lines):
                importer[(path, name)].add(group.id)

    uses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for group in groups:
        by_file = added.get(group.id, {})
        for path, lines in by_file.items():
            local = import_bindings(lines)
            for name in {n for line in lines for n in _IDENTIFIER.findall(line)} - local:
                binders = importer.get((path, name), set())
                if len(binders) == 1 and group.id not in binders and len(name) >= 3:
                    uses[group.id][next(iter(binders))].add(name)
        own = _definitions([line for lines in by_file.values() for line in lines])
        for path, lines in by_file.items():
            user_is_test = is_test_path(path)
            for line in lines:
                for name in _IDENTIFIER.findall(line):
                    found = owner.get(name)
                    if found is None or name in own:
                        continue
                    definer, def_path = found
                    if not _can_import(path, def_path, user_is_test):
                        continue
                    for provider in {definer} | writers.get(name, set()):
                        if provider != group.id:
                            uses[group.id][provider].add(name)
    return {user: dict(definers) for user, definers in uses.items()}


def names_in(lines: list[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(names these lines define, identifiers they use), for unit-level affinity."""
    defined = _definitions(lines)
    used = {
        name
        for line in lines
        for name in _IDENTIFIER.findall(line)
        if len(name) >= _MIN_NAME_LENGTH and name not in _IGNORED
    }
    return frozenset(defined), frozenset(used - defined)
