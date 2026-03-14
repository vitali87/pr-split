from __future__ import annotations

import json
from typing import TypedDict

import anthropic
import openai
import tiktoken
from anthropic.types.beta import BetaToolUseBlock
from loguru import logger

from .. import logs
from ..config import Settings
from ..constants import (
    CHUNK_RETRY_LIMIT,
    CHUNK_TARGET_RATIO,
    MAX_OUTPUT_TOKENS,
    AssignmentType,
    Provider,
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

_ANTHROPIC_TOOL_DEF = anthropic.types.ToolParam(
    name=SPLIT_TOOL_NAME,
    description="Propose a plan to split the diff into groups",
    input_schema=SPLIT_TOOL_SCHEMA,
)

_OPENAI_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": SPLIT_TOOL_NAME,
        "description": "Propose a plan to split the diff into groups",
        "parameters": SPLIT_TOOL_SCHEMA,
    },
}


class _RawAssignment(TypedDict):
    file_path: str
    assignment_type: str
    hunk_indices: list[int]


class _RawGroup(TypedDict):
    id: str
    title: str
    description: str
    depends_on: list[str]
    assignments: list[_RawAssignment]
    estimated_loc: int


class RawToolOutput(TypedDict):
    groups: list[_RawGroup]


def _extract_raw_output(block_input: dict[str, object]) -> list[_RawGroup]:
    groups = block_input.get("groups")
    if not isinstance(groups, list):
        raise LLMError(
            ErrorMsg.LLM_PARSE_ERROR(
                detail=f"missing 'groups' in tool output (keys: {list(block_input.keys())})"
            )
        )
    return groups  # type: ignore[return-value]


def _count_tokens_anthropic(system: str, user: str, *, settings: Settings) -> int:
    client = anthropic.Anthropic(api_key=settings.api_key)
    response = client.messages.count_tokens(
        model=settings.model,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[_ANTHROPIC_TOOL_DEF],
    )
    return response.input_tokens


def _count_tokens_openai(texts: list[str], *, model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return sum(len(enc.encode(t)) for t in texts)


def _count_tokens(system: str, user: str, *, settings: Settings) -> int:
    match settings.provider:
        case Provider.ANTHROPIC:
            return _count_tokens_anthropic(system, user, settings=settings)
        case Provider.OPENAI:
            return _count_tokens_openai([system, user], model=settings.model)


def _call_anthropic(system: str, user: str, *, settings: Settings) -> RawToolOutput:
    client = anthropic.Anthropic(api_key=settings.api_key)
    try:
        response = client.beta.messages.create(
            model=settings.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_ANTHROPIC_TOOL_DEF],
            tool_choice={"type": "tool", "name": SPLIT_TOOL_NAME},
            betas=["context-1m-2025-08-07"],
            timeout=600,
        )
    except anthropic.APIError as exc:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail=str(exc))) from exc
    if response.stop_reason != "tool_use":
        stop_reason = getattr(response, "stop_reason", "unknown")
        keys: list[str] = []
        for block in getattr(response, "content", []):
            if isinstance(block, BetaToolUseBlock) and isinstance(block.input, dict):
                keys = list(block.input.keys())
                break
        logger.warning(logs.LLM_OUTPUT_TRUNCATED.format(stop_reason=stop_reason, keys=keys))
    for block in response.content:
        if isinstance(block, BetaToolUseBlock) and block.name == SPLIT_TOOL_NAME:
            return RawToolOutput(groups=_extract_raw_output(block.input))
    raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail="no tool_use block in response"))


def _call_openai(system: str, user: str, *, settings: Settings) -> RawToolOutput:
    client = openai.OpenAI(api_key=settings.api_key)
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[_OPENAI_TOOL_DEF],
            tool_choice={
                "type": "function",
                "function": {"name": SPLIT_TOOL_NAME},
            },
        )
    except openai.APIError as exc:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail=str(exc))) from exc
    if not response.choices:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail="no choices in response"))
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise LLMError(ErrorMsg.LLM_PARSE_ERROR(detail="no tool call in response"))
    raw_args = tool_calls[0].function.arguments
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise LLMError(
            ErrorMsg.LLM_PARSE_ERROR(detail=f"failed to parse tool arguments: {exc}")
        ) from exc
    return RawToolOutput(groups=_extract_raw_output(parsed))


def _call_llm(system: str, user: str, *, settings: Settings) -> RawToolOutput:
    match settings.provider:
        case Provider.ANTHROPIC:
            return _call_anthropic(system, user, settings=settings)
        case Provider.OPENAI:
            return _call_openai(system, user, settings=settings)


def _call_chunk_with_retry(
    system: str,
    user: str,
    *,
    settings: Settings,
    chunk_index: int,
    total_chunks: int,
) -> list[Group]:
    last_error: LLMError | None = None
    for attempt in range(1, CHUNK_RETRY_LIMIT + 1):
        try:
            raw = _call_llm(system=system, user=user, settings=settings)
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


def _plan_split_chunked(
    parsed_diff: ParsedDiff,
    settings: Settings,
    system: str,
    full_token_count: int,
) -> list[Group]:
    overhead = _count_tokens(system, ".", settings=settings)
    chunk_limit = int(settings.max_context_tokens * CHUNK_TARGET_RATIO)
    diff_budget = chunk_limit - overhead - MAX_OUTPUT_TOKENS
    diff_chars = len(parsed_diff.raw_diff)
    token_ratio = (full_token_count - overhead) / diff_chars if diff_chars > 0 else 0.25

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
            settings=settings,
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

    system = build_system_prompt(settings.priority, settings.max_loc)
    user = build_user_prompt(diff_stats, parsed_diff.labeled_diff)

    logger.info(logs.COUNTING_TOKENS.format(model=settings.model))
    token_count = _count_tokens(system, user, settings=settings)
    effective_limit = settings.max_context_tokens - MAX_OUTPUT_TOKENS
    logger.info(logs.TOKEN_COUNT.format(tokens=token_count, limit=effective_limit))

    if token_count > effective_limit:
        logger.warning(logs.DIFF_TOO_LARGE.format(tokens=token_count, limit=effective_limit))
        groups = _plan_split_chunked(parsed_diff, settings, system, token_count)
        logger.info(logs.LLM_RESPONSE_RECEIVED.format(count=len(groups)))
        return groups

    logger.info(logs.SENDING_TO_LLM.format(model=settings.model))
    raw = _call_llm(system=system, user=user, settings=settings)
    groups = _parse_groups(raw)
    recompute_estimated_loc(groups, parsed_diff)
    logger.info(logs.LLM_RESPONSE_RECEIVED.format(count=len(groups)))
    return groups
