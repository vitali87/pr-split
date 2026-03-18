from __future__ import annotations

from itertools import product

from pr_split.planner.chunker import (
    _chunk_boundary_penalty,
    _chunk_slack_penalty,
    chunk_hunks_dynamic_programming,
)
from pr_split.types_defs import HunkRef


def _chunk_plan_cost(
    hunk_sequence: list[HunkRef],
    chunks: list[list[HunkRef]],
    token_budget: int,
) -> float:
    cost = 0.0
    end_index = 0
    for chunk in chunks:
        used_tokens = sum(ref.token_estimate for ref in chunk)
        file_mix_penalty = max(0, len({ref.file_path for ref in chunk}) - 1) * 50.0
        end_index += len(chunk)
        cost += (
            100.0
            + _chunk_slack_penalty(used_tokens, token_budget)
            + file_mix_penalty
            + _chunk_boundary_penalty(hunk_sequence, end_index - 1)
        )
    return cost


def _all_valid_chunkings(
    hunk_sequence: list[HunkRef],
    token_budget: int,
) -> list[list[list[HunkRef]]]:
    def _walk(start: int) -> list[list[list[HunkRef]]]:
        if start == len(hunk_sequence):
            return [[]]

        results: list[list[list[HunkRef]]] = []
        used_tokens = 0
        for end in range(start + 1, len(hunk_sequence) + 1):
            used_tokens += hunk_sequence[end - 1].token_estimate
            if used_tokens > token_budget:
                break
            chunk = hunk_sequence[start:end]
            for suffix in _walk(end):
                results.append([chunk, *suffix])
        return results

    return _walk(0)


def _make_hunk_sequence(
    file_names: tuple[str, ...],
    token_estimates: tuple[int, ...],
) -> list[HunkRef]:
    return [
        HunkRef(file_path=file_path, hunk_index=index, token_estimate=token_estimate)
        for index, (file_path, token_estimate) in enumerate(
            zip(file_names, token_estimates, strict=True)
        )
    ]


class TestChunkHunksDynamicProgrammingExhaustive:
    def test_matches_bruteforce_optimum_for_small_instances(self) -> None:
        file_variants = ("a.py", "b.py")
        token_variants = (1, 2, 3)

        checked_cases = 0
        for length in range(1, 5):
            for file_names in product(file_variants, repeat=length):
                for token_estimates in product(token_variants, repeat=length):
                    hunk_sequence = _make_hunk_sequence(file_names, token_estimates)
                    for token_budget in range(2, 6):
                        if any(token > token_budget for token in token_estimates):
                            continue

                        actual_chunks = chunk_hunks_dynamic_programming(
                            hunk_sequence, token_budget
                        )
                        actual_cost = _chunk_plan_cost(
                            hunk_sequence,
                            actual_chunks,
                            token_budget,
                        )
                        brute_force_cost = min(
                            _chunk_plan_cost(hunk_sequence, candidate, token_budget)
                            for candidate in _all_valid_chunkings(hunk_sequence, token_budget)
                        )

                        assert actual_cost == brute_force_cost
                        checked_cases += 1

        assert checked_cases > 5_000
