"""Cut an oversized new file into several hunks at top-level boundaries.

Git reports a new file as one hunk, so no plan can bring a new file larger
than ``--max-loc`` under the limit. In a stacked split a later piece of the
file is appended by a child PR on top of its parent, which already holds the
earlier pieces, so the file can be cut anywhere its prefix is still a valid
file: between top-level blocks.

Python files are cut between top-level statements found by ``ast``, with
their decorators and the comments directly above them. Other files are cut
at an unindented line that follows a blank line, outside any bracket,
multi-line string or block comment, which holds for Rust, Go, JavaScript,
TypeScript, Java, C and most other languages.
"""

from __future__ import annotations

import ast
import re
from itertools import pairwise
from typing import TYPE_CHECKING

from unidiff import Hunk

if TYPE_CHECKING:
    from unidiff import PatchedFile
    from unidiff.patch import Line

_NO_NEWLINE_MARKER = "\\"
# A line starting with one of these continues the block above it.
_CONTINUATIONS = ("}", ")", "]", "else", "elif", "except", "finally", "catch")
_DOUBLE_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')
_SINGLE_QUOTED = re.compile(r"'(?:\\.|[^'\\\n])*'")
# Rust uses a lone quote for lifetimes ('a), so only char literals are quoted.
_CHAR_LITERAL = re.compile(r"'(?:\\.[^']*|[^'\\\n])'")
_LINE_COMMENT = re.compile(r"//.*")
_OPENERS = "([{"
_CLOSERS = ")]}"


def _python_block_starts(values: list[str]) -> list[int] | None:
    """Exact top-level statement starts, with their decorators and comments."""
    try:
        tree = ast.parse("".join(values))
    except (SyntaxError, ValueError):
        return None
    starts: list[int] = []
    for node in tree.body[1:]:
        decorators = getattr(node, "decorator_list", [])
        start = min([node.lineno, *(d.lineno for d in decorators)]) - 1
        # Comment lines directly above a statement belong to it.
        while start > 0 and values[start - 1].startswith("#"):
            start -= 1
        starts.append(start)
    return starts


def _strip_literals(code: str, *, rust: bool) -> str:
    code = _DOUBLE_QUOTED.sub("", code)
    code = (_CHAR_LITERAL if rust else _SINGLE_QUOTED).sub("", code)
    return _LINE_COMMENT.sub("", code)


def _generic_block_starts(values: list[str], *, rust: bool = False) -> list[int]:
    """Unindented lines after a blank line, outside brackets, strings and comments."""
    starts: list[int] = []
    depth = 0
    fence: str | None = None  # the delimiter that closes an open string/comment
    for i, line in enumerate(values):
        if (
            i > 0
            and depth == 0
            and fence is None
            and not values[i - 1].strip()
            and line.strip()
            and line[0] not in " \t"
            and not line.startswith(_CONTINUATIONS)
        ):
            starts.append(i)
        rest = line
        while rest:
            if fence is not None:
                end = rest.find(fence)
                if end == -1:
                    break
                rest, fence = rest[end + len(fence) :], None
                continue
            opener = min(
                ((rest.find(tok), tok) for tok in ('"""', "'''", "/*", "`") if tok in rest),
                default=None,
            )
            code = rest if opener is None else rest[: opener[0]]
            code = _strip_literals(code, rust=rust)
            depth = max(0, depth + sum(code.count(c) for c in _OPENERS))
            depth = max(0, depth - sum(code.count(c) for c in _CLOSERS))
            if opener is None:
                break
            index, token = opener
            fence = "*/" if token == "/*" else token
            rest = rest[index + len(token) :]
    return starts


def block_starts(values: list[str], path: str = "") -> list[int]:
    """Line indices (after the first) where a top-level block starts."""
    if path.endswith((".py", ".pyi")):
        starts = _python_block_starts(values)
        if starts is not None:
            return starts
    return _generic_block_starts(values, rust=path.endswith(".rs"))


def cut_points(values: list[str], max_loc: int, path: str = "") -> list[int]:
    """Line indices where a new piece starts, packing whole blocks up to max_loc.

    A single block longer than max_loc stays whole: cutting inside it would
    leave the parent PR with a file that does not parse.
    """
    starts = [s for s in block_starts(values, path) if s > 0]
    cuts: list[int] = []
    piece_start = 0
    block_start = 0
    for block_end in [*starts, len(values)]:
        if block_end - piece_start > max_loc and block_start > piece_start:
            cuts.append(block_start)
            piece_start = block_start
        block_start = block_end
    return cuts


def _piece(lines: list[Line], first_line: int) -> Hunk:
    hunk = Hunk(src_start=0, src_len=0, tgt_start=first_line, tgt_len=len(lines))
    hunk.extend(lines)
    return hunk


def split_new_file(patch_file: PatchedFile, max_loc: int) -> bool:
    """Replace a new file's single hunk with pieces of at most max_loc lines.

    Returns whether the file was split. Only a new file whose one hunk is
    larger than max_loc is touched, and only where a top-level boundary exists.
    """
    if not patch_file.is_added_file or len(patch_file) != 1:
        return False
    hunk = patch_file[0]
    if hunk.added <= max_loc:
        return False
    lines = [line for line in hunk if line.is_added]
    cuts = cut_points([line.value for line in lines], max_loc, patch_file.path)
    if not cuts:
        return False
    bounds = [0, *cuts, len(lines)]
    pieces = [_piece(lines[a:b], a + 1) for a, b in pairwise(bounds)]
    # "\ No newline at end of file" belongs after the file's last line.
    pieces[-1].extend(line for line in hunk if line.line_type == _NO_NEWLINE_MARKER)
    patch_file[:] = pieces
    return True
