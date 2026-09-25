"""Keep the pieces of a split new file in stack order.

A new file cut into pieces (see diff_ops.new_file_split) is rebuilt by
concatenating the pieces a branch holds, and a stacked child carries its
ancestors' pieces. So the group holding piece k must be the group holding
piece k-1 or build on it; otherwise its sub-PR creates the file without its
beginning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import logs
from ..exceptions import ErrorMsg, PlanValidationError, PRSplitError
from ..graph import PlanDAG

if TYPE_CHECKING:
    from ..diff_ops import ParsedDiff
    from ..schemas import Group


def _piece_owners(groups: list[Group], path: str, pieces: int) -> list[str | None]:
    owners: list[str | None] = [None] * pieces
    for group in groups:
        for assignment in group.assignments:
            if assignment.file_path != path:
                continue
            for idx in assignment.covered_indices(pieces):
                if 0 <= idx < pieces:
                    owners[idx] = group.id
    return owners


def _out_of_order(
    groups: list[Group], parsed_diff: ParsedDiff, dag: PlanDAG
) -> list[tuple[str, int, str, str]]:
    """(file, piece, group, parent) for each piece whose group skips the previous piece's."""
    found: list[tuple[str, int, str, str]] = []
    for path, pieces in sorted(parsed_diff.new_file_pieces.items()):
        owners = _piece_owners(groups, path, pieces)
        for piece in range(1, pieces):
            parent, child = owners[piece - 1], owners[piece]
            if parent is None or child is None or parent == child:
                continue
            if parent not in dag.ancestors(child):
                found.append((path, piece, child, parent))
    return found


def link_new_file_pieces(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    """Make each later piece's group depend on the previous piece's group.

    An edge that would close a cycle is left out; validate_new_file_pieces
    then reports the plan as invalid.
    """
    if not parsed_diff.new_file_pieces:
        return
    by_id = {g.id: g for g in groups}
    for path, _piece, child, parent in _out_of_order(groups, parsed_diff, PlanDAG(groups)):
        group = by_id[child]
        # An earlier edge in this loop may already have ordered the pair.
        if parent in PlanDAG(groups).ancestors(child):
            continue
        group.depends_on.append(parent)
        try:
            PlanDAG(groups).validate_acyclic()
        except PRSplitError:
            group.depends_on.remove(parent)
            continue
        logger.info(logs.NEW_FILE_PIECE_LINKED.format(child=child, file=path, parent=parent))


def validate_new_file_pieces(groups: list[Group], parsed_diff: ParsedDiff, dag: PlanDAG) -> None:
    for path, piece, child, parent in _out_of_order(groups, parsed_diff, dag):
        raise PlanValidationError(
            ErrorMsg.NEW_FILE_PIECE_ORDER(
                group=child, piece=piece + 1, file=path, parent=parent, previous=piece
            )
        )
