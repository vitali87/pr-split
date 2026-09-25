"""Keep the pieces of a split new file in stack order.

A new file cut into pieces (see diff_ops.new_file_split) is rebuilt by
concatenating the pieces a branch holds, and a stacked child carries its
ancestors' pieces. So the group holding piece k must hold piece k-1 too, or
build on the group that does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import logs
from ..exceptions import PRSplitError
from ..graph import PlanDAG
from .validator import new_file_pieces_out_of_order

if TYPE_CHECKING:
    from ..diff_ops import ParsedDiff
    from ..schemas import Group


def link_new_file_pieces(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    """Make each later piece's group depend on the previous piece's group.

    An edge that would close a cycle is left out; validate_new_file_pieces
    then reports the plan as invalid.
    """
    if not parsed_diff.new_file_pieces:
        return
    by_id = {g.id: g for g in groups}
    for path, _piece, child, parent in new_file_pieces_out_of_order(
        groups, parsed_diff, PlanDAG(groups)
    ):
        # An edge added earlier in this loop may already order the pair.
        if parent in PlanDAG(groups).ancestors(child):
            continue
        group = by_id[child]
        group.depends_on.append(parent)
        try:
            PlanDAG(groups).validate_acyclic()
        except PRSplitError:
            group.depends_on.remove(parent)
            continue
        logger.info(logs.NEW_FILE_PIECE_LINKED.format(child=child, file=path, parent=parent))
