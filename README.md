<p align="center">
  <img src="logo.png" alt="pr-split logo" width="180">
</p>

<h1 align="center">pr-split</h1>

<p align="center">
  Decompose large PRs into a DAG of small, reviewable PRs
</p>

<p align="center">
  <a href="https://github.com/vitali87/pr-split/blob/main/LICENSE"><img src="https://img.shields.io/github/license/vitali87/pr-split" alt="License"></a>
</p>

## Why pr-split?

Vibe coding with AI assistants can produce massive PRs that no one wants to review. A 2,000 line PR with changes across dozens of files is a review bottleneck: teammates skim it, rubber stamp it, or just ignore it. `pr-split` turns that monolith into a set of focused, bite-sized PRs your team can actually review with confidence. Each sub-PR has a clear purpose, minimal scope, and explicit dependencies, so reviewers know exactly what changed and why.

## How it works

`pr-split` takes a large pull request (local branch, fork PR number, or `user:branch`), sends the diff to an LLM for analysis, and produces a split plan: a set of smaller, focused PRs arranged in a dependency DAG. Each sub-PR gets its own branch, commit, and GitHub PR targeting the correct base.

<img src="pr-split.png" alt="pr-split system design" width="100%">

## Installation

```bash
# With uv (recommended)
uv tool install pr-split

# With pip
pip install pr-split
```

## Prerequisites

- Python 3.12+
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated via `gh auth login`
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable set when using the `llm` partition backend

## Usage

### Split a local branch

```bash
pr-split split feature-branch --base main
```

### Split a fork PR by number

```bash
pr-split split '#42' --base main
```

### Split a fork PR by user:branch

```bash
pr-split split someuser:feature-branch --base main
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base` | `main` | Base branch for the diff |
| `--max-loc` | `400` | Soft limit on diff lines per sub-PR |
| `--priority` | `orthogonal` | Grouping priority (`orthogonal` or `logical`) |
| `--chunk-strategy` | `dynamic_programming` | Large-diff chunking strategy (`dynamic_programming` or `greedy`) |
| `--partition-strategy` | `llm` | Hunk-to-PR partition backend (`llm`, `graph`, or `cp_sat`) |

### Other commands

```bash
# Show status of an existing split
pr-split status

# Clean up all pr-split branches and close PRs
pr-split clean
```

## Configuration

Settings can be set via environment variables with the `PR_SPLIT_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `PR_SPLIT_PROVIDER` | `anthropic` | LLM provider (`anthropic` or `openai`) |
| `ANTHROPIC_API_KEY` | (required for Anthropic) | Anthropic API key |
| `OPENAI_API_KEY` | (required for OpenAI) | OpenAI API key |
| `PR_SPLIT_MODEL` | auto per provider | Model name (defaults to best available model for the chosen provider) |
| `PR_SPLIT_MAX_LOC` | `400` | Default soft limit on diff lines |
| `PR_SPLIT_PRIORITY` | `orthogonal` | Default grouping priority |
| `PR_SPLIT_CHUNK_STRATEGY` | `dynamic_programming` | Large-diff chunking strategy |
| `PR_SPLIT_PARTITION_STRATEGY` | `llm` | Hunk-to-PR partition backend |

## Planning backends

`pr-split` now separates two optimization layers:

- **Chunking**: for diffs that exceed the model context window, `dynamic_programming` chooses chunk boundaries to avoid splitting the same file when possible. `greedy` keeps the previous first-fit behavior.
- **Partitioning**: `llm` preserves the original semantic planner, `graph` uses deterministic affinity-based grouping, and `cp_sat` uses an optimization model to balance group count, LOC, and cohesion.

The `cp_sat` backend requires the optional [`ortools`](https://developers.google.com/optimization) package to be installed in the runtime environment.

For a deeper explanation of the planning model, optimization methods, scoring, and research directions, see [METHODOLOGY.md](METHODOLOGY.md).

## What it does

1. Extracts the merge-base diff between your branch and the base (same view as GitHub's PR page)
2. Sends the diff to the configured LLM, which groups hunks into logical sub-PRs with dependency ordering
3. Validates the plan: full coverage (every hunk assigned exactly once), no cycles, no merge conflicts between independent groups
4. Shows you the plan (table + dependency tree) and asks for confirmation
5. Creates branches, commits, pushes, and opens GitHub PRs in topological order
6. For diffs exceeding the model's context window, uses the configured chunking strategy and processes chunks sequentially while carrying forward the group catalog across chunks

## License

[MIT](LICENSE)
