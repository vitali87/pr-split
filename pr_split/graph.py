from __future__ import annotations

from collections import deque
from collections.abc import Generator
from graphlib import CycleError, TopologicalSorter

from .exceptions import ErrorMsg, PlanValidationError
from .schemas import Group


class PlanDAG:
    def __init__(self, groups: list[Group]) -> None:
        self._groups: dict[str, Group] = {g.id: g for g in groups}
        self._children: dict[str, list[str]] = {g.id: [] for g in groups}
        self._parents: dict[str, list[str]] = {g.id: list(g.depends_on) for g in groups}
        for g in groups:
            for dep in g.depends_on:
                self._children[dep].append(g.id)

    def _build_sorter(self) -> TopologicalSorter[str]:
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for gid, group in self._groups.items():
            sorter.add(gid, *group.depends_on)
        return sorter

    def validate_acyclic(self) -> None:
        sorter = self._build_sorter()
        try:
            sorter.prepare()
        except CycleError:
            raise PlanValidationError(ErrorMsg.CYCLE_DETECTED()) from None

    def topological_order(self) -> list[str]:
        sorter = self._build_sorter()
        return list(sorter.static_order())

    def iter_ready(self) -> Generator[list[str], None, None]:
        sorter = self._build_sorter()
        sorter.prepare()
        while sorter.is_active():
            batch = list(sorter.get_ready())
            yield batch
            for node in batch:
                sorter.done(node)

    def parents(self, group_id: str) -> list[str]:
        return list(self._parents[group_id])

    def children(self, group_id: str) -> list[str]:
        return list(self._children[group_id])

    def roots(self) -> list[str]:
        return [gid for gid, deps in self._parents.items() if not deps]

    def leaves(self) -> list[str]:
        return [gid for gid, deps in self._children.items() if not deps]

    def is_merge_node(self, group_id: str) -> bool:
        return len(self._parents[group_id]) > 1

    def ancestors(self, group_id: str) -> set[str]:
        result: set[str] = set()
        queue: deque[str] = deque(self._parents[group_id])
        while queue:
            node = queue.popleft()
            if node not in result:
                result.add(node)
                queue.extend(self._parents[node])
        return result

    def descendants(self, group_id: str) -> set[str]:
        result: set[str] = set()
        queue: deque[str] = deque(self._children[group_id])
        while queue:
            node = queue.popleft()
            if node not in result:
                result.add(node)
                queue.extend(self._children[node])
        return result
