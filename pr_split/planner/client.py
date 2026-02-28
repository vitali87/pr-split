from __future__ import annotations

from typing import TypedDict

import anthropic
from loguru import logger

from .. import logs
from ..config import Settings
from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg, LLMError
from ..schemas import Group, GroupAssignment
from .prompts import SPLIT_TOOL_NAME, SPLIT_TOOL_SCHEMA, build_system_prompt, build_user_prompt


class RawAssignment(TypedDict):
    file_path: str
    assignment_type: str
    hunk_indices: list[int]


class RawGroup(TypedDict):
    id: str
    title: str
    description: str
    depends_on: list[str]
    assignments: list[RawAssignment]
    estimated_loc: int


class RawToolOutput(TypedDict):
    groups: list[RawGroup]


def _call_claude(
    system: str,
    user: str,
    tool_name: str,
    tool_schema: dict[str, object],
    *,
    api_key: str,
    model: str,
) -> RawToolOutput:
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Propose a plan to split the diff into groups",
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
    except anthropic.APIError as exc:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail=str(exc))) from exc
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            raw: RawToolOutput = block.input
            return raw
    raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail="no tool_use block in response"))


def _parse_groups(raw: RawToolOutput) -> list[Group]:
    groups: list[Group] = []
    for entry in raw["groups"]:
        assignments = [
            GroupAssignment(
                file_path=a["file_path"],
                assignment_type=a["assignment_type"],
                hunk_indices=a["hunk_indices"],
            )
            for a in entry["assignments"]
        ]
        groups.append(
            Group(
                id=entry["id"],
                title=entry["title"],
                description=entry["description"],
                depends_on=entry["depends_on"],
                assignments=assignments,
                estimated_loc=entry["estimated_loc"],
            )
        )
    return groups


def plan_split(
    parsed_diff: ParsedDiff,
    settings: Settings,
) -> list[Group]:
    diff_stats = parsed_diff.stats
    logger.info(
        logs.DIFF_STATS.format(
            files=diff_stats["total_files"],
            added=diff_stats["total_added"],
            removed=diff_stats["total_removed"],
            loc=diff_stats["total_loc"],
        )
    )

    system = build_system_prompt(settings.priority, settings.max_loc)
    user = build_user_prompt(diff_stats, parsed_diff.raw_diff)

    logger.info(logs.SENDING_TO_LLM.format(model=settings.claude_model))
    raw = _call_claude(
        system=system,
        user=user,
        tool_name=SPLIT_TOOL_NAME,
        tool_schema=SPLIT_TOOL_SCHEMA,
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
    )

    groups = _parse_groups(raw)
    logger.info(logs.LLM_RESPONSE_RECEIVED.format(count=len(groups)))
    return groups
