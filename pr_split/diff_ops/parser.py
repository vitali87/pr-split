from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import cached_property

from loguru import logger
from unidiff import PatchSet

from .. import logs
from ..exceptions import DiffParseError, GitOperationError
from ..types_defs import DiffStats, FileSummary, HunkInfo

# Flags that pin the diff to the exact unified format the parser and
# reconstructor expect, regardless of the user's git config:
#   --no-color / --no-ext-diff  plain unified output, never an external tool
#   --no-textconv               hunks must be raw blob content, since the
#                               reconstructor applies them to git show output
#   --no-renames                renames become delete + add; the reconstructor
#                               resolves base content by path, so a rename
#                               reported under its new name would fail
#   -U3 and fixed prefixes      override diff.context and diff.mnemonicPrefix,
#                               which would shift hunk positions and paths
DIFF_ARGS: tuple[str, ...] = (
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "-U3",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


def extract_diff(dev_branch: str, base_branch: str) -> str:
    logger.info(logs.EXTRACTING_DIFF.format(base=base_branch, dev=dev_branch))
    # Capture bytes: text mode applies universal-newline translation, which
    # turns CRLF into LF and would make every sub-PR rewrite the file's
    # line endings.
    result = subprocess.run(
        # core.quotePath=false: otherwise git octal-escapes and quotes any
        # non-ASCII path ("b/caf\303\251.txt"), which unidiff cannot parse for
        # new/deleted files and mis-reports as a literal quoted path otherwise.
        ["git", "-c", "core.quotePath=false", "diff", *DIFF_ARGS, f"{base_branch}...{dev_branch}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8")


_C_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    '"': b'"',
}
_C_ESCAPE_RE = re.compile(r"\\([0-7]{1,3}|.)", re.DOTALL)


def unquote_git_path(path: str) -> str:
    """Decode a path git printed in its C-quoted form, e.g. `"a/we\\"ird.py"`.

    Even with core.quotePath=false git quotes any path containing `"`, `\\`
    or a control character. unidiff keeps the quotes and escapes verbatim, so
    without this the a/ b/ prefix is never stripped and the file would be
    written under a literally quoted name.
    """
    if len(path) < 2 or path[0] != '"' or path[-1] != '"':
        return path
    out = bytearray()
    pos = 0
    body = path[1:-1]
    for match in _C_ESCAPE_RE.finditer(body):
        out += body[pos : match.start()].encode("utf-8")
        escape = match.group(1)
        if escape[0] in "01234567":
            # Octal escapes are single bytes; consecutive ones form one
            # multi-byte UTF-8 character, which the final decode reassembles.
            out.append(int(escape, 8))
        else:
            out += _C_ESCAPES.get(escape, escape.encode("utf-8"))
        pos = match.end()
    out += body[pos:].encode("utf-8")
    return bytes(out).decode("utf-8", errors="surrogateescape")


def parse_diff(raw_diff: str) -> ParsedDiff:
    try:
        patch_set = PatchSet(raw_diff)
    except Exception as exc:
        raise DiffParseError(str(exc)) from exc
    for patch_file in patch_set:
        patch_file.source_file = unquote_git_path(patch_file.source_file)
        patch_file.target_file = unquote_git_path(patch_file.target_file)
    return ParsedDiff(patch_set=patch_set, raw_diff=raw_diff)


@dataclass
class ParsedDiff:
    patch_set: PatchSet
    raw_diff: str

    @property
    def file_paths(self) -> list[str]:
        return [pf.path for pf in self.patch_set]

    @cached_property
    def stats(self) -> DiffStats:
        file_summaries: list[FileSummary] = []
        total_added = 0
        total_removed = 0
        for pf in self.patch_set:
            added = pf.added
            removed = pf.removed
            total_added += added
            total_removed += removed
            file_summaries.append(
                FileSummary(
                    path=pf.path,
                    added=added,
                    removed=removed,
                    is_new=pf.is_added_file,
                    is_deleted=pf.is_removed_file,
                    is_renamed=pf.is_rename,
                    hunk_count=len(pf),
                )
            )
        return DiffStats(
            total_files=len(self.patch_set),
            total_added=total_added,
            total_removed=total_removed,
            total_loc=total_added + total_removed,
            file_summaries=file_summaries,
        )

    def hunks_for_file(self, path: str) -> list[HunkInfo]:
        for pf in self.patch_set:
            if pf.path == path:
                return [
                    HunkInfo(
                        index=i,
                        source_start=hunk.source_start,
                        source_length=hunk.source_length,
                        target_start=hunk.target_start,
                        target_length=hunk.target_length,
                        added_lines=hunk.added,
                        removed_lines=hunk.removed,
                    )
                    for i, hunk in enumerate(pf)
                ]
        return []

    @property
    def labeled_diff(self) -> str:
        parts: list[str] = []
        for pf in self.patch_set:
            header = f"--- {pf.source_file}\n+++ {pf.target_file}\n"
            labeled_hunks = [f"[hunk_index={i}]\n{hunk}" for i, hunk in enumerate(pf)]
            parts.append(header + "".join(labeled_hunks))
        return "\n".join(parts)

    def hunk_content(self, path: str, index: int) -> str:
        for pf in self.patch_set:
            if pf.path == path:
                return str(pf[index])
        return ""
