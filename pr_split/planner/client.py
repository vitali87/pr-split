from __future__ import annotations

from typing import TypedDict

import anthropic
from anthropic.types.beta import BetaToolUseBlock
from loguru import logger

from .. import logs
from ..config import Settings
from ..constants import (
    CHUNK_RETRY_LIMIT,
    CHUNK_TARGET_RATIO,
    CONTEXT_1M_BETA,
    LLM_TIMEOUT_SECONDS,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    AssignmentType,
)
from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg, LLMError
from ..schemas import Group, GroupAssignment
from .chunker import (
    assign_uncovered_hunks,
    build_chunk_diff_from_hunks,
    build_chunk_stats_from_hunks,
    build_hunk_sequence,
    chunk_hunks,
    format_group_catalog,
    recompute_estimated_loc,
)
from .prompts import (
    SPLIT_TOOL_NAME,
    SPLIT_TOOL_SCHEMA,
    build_chunk_continuation_prompt,
    build_chunk_first_prompt,
    build_system_prompt,
    build_user_prompt,
)

_TOOL_DEF = anthropic.types.ToolParam(
    name=SPLIT_TOOL_NAME,
    description="Propose a plan to split the diff into groups",
    input_schema=SPLIT_TOOL_SCHEMA,
)

STOP_REASON_TOOL_USE = "tool_use"


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


def _extract_raw_output(block_input: dict[str, object]) -> list[RawGroup]:
    groups = block_input.get("groups")
    if not isinstance(groups, list):
        raise LLMError(
            ErrorMsg.LLM_PARSE_ERROR(
                detail=f"missing 'groups' in tool output (keys: {list(block_input.keys())})"
            )
        )
    return groups  # type: ignore[return-value]


def _log_truncation(response: object) -> None:
    stop_reason = getattr(response, "stop_reason", "unknown")
    keys: list[str] = []
    for block in getattr(response, "content", []):
        if isinstance(block, BetaToolUseBlock) and isinstance(block.input, dict):
            keys = list(block.input.keys())
            break
    logger.warning(
        logs.LLM_OUTPUT_TRUNCATED.format(stop_reason=stop_reason, keys=keys)
    )


def _count_tokens(
    system: str,
    user: str,
    *,
    api_key: str,
    model: str,
) -> int:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[_TOOL_DEF],
    )
    return response.input_tokens


def _call_claude(
    system: str,
    user: str,
    *,
    api_key: str,
    model: str,
) -> RawToolOutput:
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_TOOL_DEF],
            tool_choice={"type": "tool", "name": SPLIT_TOOL_NAME},
            betas=[CONTEXT_1M_BETA],
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except anthropic.APIError as exc:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail=str(exc))) from exc
    if response.stop_reason != STOP_REASON_TOOL_USE:
        _log_truncation(response)
    for block in response.content:
        if isinstance(block, BetaToolUseBlock) and block.name == SPLIT_TOOL_NAME:
            return RawToolOutput(groups=_extract_raw_output(block.input))
    raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail="no tool_use block in response"))


def _call_chunk_with_retry(
    system: str,
    user: str,
    *,
    api_key: str,
    model: str,
    chunk_index: int,
    total_chunks: int,
) -> list[Group]:
    last_error: LLMError | None = None
    for attempt in range(1, CHUNK_RETRY_LIMIT + 1):
        try:
            raw = _call_claude(system=system, user=user, api_key=api_key, model=model)
            return _parse_groups(raw)
        except LLMError as exc:
            last_error = exc
            if attempt < CHUNK_RETRY_LIMIT:
                logger.warning(
                    logs.CHUNK_RETRY.format(
                        index=chunk_index,
                        total=total_chunks,
                        attempt=attempt,
                        error=str(exc),
                    )
                )
    raise last_error  # type: ignore[misc]


def _parse_groups(raw: RawToolOutput) -> list[Group]:
    groups: list[Group] = []
    for entry in raw["groups"]:
        assignments = [
            GroupAssignment(
                file_path=a["file_path"],
                assignment_type=AssignmentType(a["assignment_type"]),
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


def _merge_chunk_groups(accumulated: list[Group], chunk_groups: list[Group]) -> list[Group]:
    acc_map = {g.id: g for g in accumulated}
    for cg in chunk_groups:
        if cg.id in acc_map:
            existing = acc_map[cg.id]
            existing.assignments.extend(cg.assignments)
            for dep in cg.depends_on:
                if dep not in existing.depends_on:
                    existing.depends_on.append(dep)
        else:
            acc_map[cg.id] = cg
    return list(acc_map.values())


def _compute_token_ratio(
    full_token_count: int,
    overhead_tokens: int,
    parsed_diff: ParsedDiff,
) -> float:
    diff_tokens = full_token_count - overhead_tokens
    diff_chars = len(parsed_diff.raw_diff)
    if diff_chars <= 0:
        return 0.25
    return diff_tokens / diff_chars


def _plan_split_chunked(
    parsed_diff: ParsedDiff,
    settings: Settings,
    system: str,
    full_token_count: int,
) -> list[Group]:
    overhead = _count_tokens(
        system, ".", api_key=settings.anthropic_api_key, model=settings.claude_model
    )
    chunk_limit = int(MAX_CONTEXT_TOKENS * CHUNK_TARGET_RATIO)
    diff_budget = chunk_limit - overhead - MAX_OUTPUT_TOKENS
    token_ratio = _compute_token_ratio(full_token_count, overhead, parsed_diff)

    logger.info(
        logs.CALIBRATING_CHUNKS.format(overhead=overhead, budget=diff_budget, ratio=token_ratio)
    )

    hunk_sequence = build_hunk_sequence(parsed_diff, token_ratio)
    chunks = chunk_hunks(hunk_sequence, diff_budget)
    total_chunks = len(chunks)

    logger.info(logs.CHUNKED_MODE.format(chunks=total_chunks, hunks=len(hunk_sequence)))

    accumulated: list[Group] = []

    for i, chunk_refs in enumerate(chunks):
        chunk_index = i + 1
        chunk_diff = build_chunk_diff_from_hunks(parsed_diff, chunk_refs)
        chunk_stats = build_chunk_stats_from_hunks(parsed_diff, chunk_refs)
        chunk_tokens = sum(h.token_estimate for h in chunk_refs)

        logger.info(
            logs.CHUNK_SENDING.format(
                index=chunk_index,
                total=total_chunks,
                hunks=len(chunk_refs),
                tokens=chunk_tokens,
            )
        )

        if chunk_index == 1:
            user = build_chunk_first_prompt(chunk_stats, chunk_diff, total_chunks)
        else:
            catalog = format_group_catalog(accumulated)
            user = build_chunk_continuation_prompt(
                chunk_stats, chunk_diff, chunk_index, total_chunks, catalog
            )

        chunk_groups = _call_chunk_with_retry(
            system=system,
            user=user,
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

        prev_count = len(accumulated)
        if chunk_index == 1:
            accumulated = chunk_groups
        else:
            accumulated = _merge_chunk_groups(accumulated, chunk_groups)

        logger.info(
            logs.CHUNK_RECEIVED.format(
                index=chunk_index,
                total=total_chunks,
                new_groups=len(accumulated) - prev_count,
                total_groups=len(accumulated),
            )
        )

    auto_assigned = assign_uncovered_hunks(accumulated, parsed_diff)
    if auto_assigned:
        logger.warning(logs.UNCOVERED_HUNKS_FIXED.format(count=auto_assigned))
    recompute_estimated_loc(accumulated, parsed_diff)
    return accumulated


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
    user = build_user_prompt(diff_stats, parsed_diff.labeled_diff)

    logger.info(logs.COUNTING_TOKENS.format(model=settings.claude_model))
    token_count = _count_tokens(
        system, user, api_key=settings.anthropic_api_key, model=settings.claude_model
    )
    effective_limit = MAX_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS
    logger.info(logs.TOKEN_COUNT.format(tokens=token_count, limit=effective_limit))

    if token_count > effective_limit:
        logger.warning(logs.DIFF_TOO_LARGE.format(tokens=token_count, limit=effective_limit))
        groups = _plan_split_chunked(parsed_diff, settings, system, token_count)
        logger.info(logs.LLM_RESPONSE_RECEIVED.format(count=len(groups)))
        return groups

    logger.info(logs.SENDING_TO_LLM.format(model=settings.claude_model))
    raw = _call_claude(
        system=system,
        user=user,
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
    )
    groups = _parse_groups(raw)
    recompute_estimated_loc(groups, parsed_diff)
    logger.info(logs.LLM_RESPONSE_RECEIVED.format(count=len(groups)))
    return groups
