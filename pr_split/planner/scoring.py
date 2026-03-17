from __future__ import annotations

from dataclasses import dataclass

from ..graph import PlanDAG
from ..schemas import Group


@dataclass(frozen=True)
class PlanMetrics:
    total_groups: int
    max_group_loc: int
    loc_overflow: int
    dependency_edges: int
    dag_depth: int
    dag_width: int
    file_scatter: int
    tiny_groups: int
    objective: int


def _dag_depth(dag: PlanDAG) -> int:
    depths: dict[str, int] = {}
    for group_id in dag.topological_order():
        parents = dag.parents(group_id)
        depths[group_id] = 1 if not parents else 1 + max(depths[parent] for parent in parents)
    return max(depths.values(), default=0)


def score_plan(groups: list[Group], max_loc: int) -> PlanMetrics:
    dag = PlanDAG(groups)

    max_group_loc = max((group.estimated_loc for group in groups), default=0)
    loc_overflow = sum(max(0, group.estimated_loc - max_loc) for group in groups)
    dependency_edges = sum(len(group.depends_on) for group in groups)
    dag_width = max((len(batch) for batch in dag.iter_ready()), default=0)
    dag_depth = _dag_depth(dag)

    file_groups: dict[str, set[str]] = {}
    for group in groups:
        for assignment in group.assignments:
            file_groups.setdefault(assignment.file_path, set()).add(group.id)
    file_scatter = sum(max(0, len(group_ids) - 1) for group_ids in file_groups.values())

    tiny_threshold = max(1, max_loc // 4)
    tiny_groups = sum(1 for group in groups if 0 < group.estimated_loc < tiny_threshold)

    objective = (
        loc_overflow * 1000
        + dependency_edges * 20
        + file_scatter * 50
        + tiny_groups * 10
        + dag_depth * 5
        + len(groups)
        - dag_width * 2
    )

    return PlanMetrics(
        total_groups=len(groups),
        max_group_loc=max_group_loc,
        loc_overflow=loc_overflow,
        dependency_edges=dependency_edges,
        dag_depth=dag_depth,
        dag_width=dag_width,
        file_scatter=file_scatter,
        tiny_groups=tiny_groups,
        objective=objective,
    )
