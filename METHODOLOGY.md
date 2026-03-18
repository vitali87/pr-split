## pr-split Methodology

This document describes the methodology behind `pr-split`: what problem it solves, how the current system models that problem, which optimization methods are used, and where contributors can push the design further.

It is written for advanced users and contributors who want more than a usage guide. The intent is to make the planner legible as an engineering system and as an optimization problem.

## 1. Problem Statement

`pr-split` takes a large diff and decomposes it into a set of small, reviewable pull requests with explicit merge-order dependencies.

The goal is not simply to make more PRs. The goal is to produce a split that is:

- reviewable: each PR is small enough to inspect carefully
- coherent: related changes stay together
- low-friction: unnecessary dependencies and file scattering are minimized
- mergeable: the resulting plan respects ordering constraints when changes touch the same parts of the codebase

In `pr-split`, the dependency graph is a **merge-order DAG**, not a stack of branch ancestry. An edge `A -> B` means `A` should land before `B`.

## 2. Core Representation

The planner operates on progressively richer representations:

1. Raw diff
2. Parsed files and hunks
3. `HunkRef` sequence for chunking
4. `PartitionUnit` list for deterministic partitioning
5. `Group` objects for final review PRs
6. `PlanDAG` for dependency analysis, validation, and scoring

This matters because different algorithms operate on different abstractions:

- chunking works on ordered hunks
- deterministic partitioning works on bounded units
- validation and scoring work on final groups and dependency edges

## 3. Formalized Planning Problem

At a high level, the split problem is a constrained partitioning problem.

### Inputs

- a diff with files and hunks
- a soft group size budget, `max_loc`
- a priority mode:
  - `orthogonal`: prefer separation and lower coupling
  - `logical`: prefer semantic cohesion
- optional context-window constraints for LLM planning

### Outputs

- a set of groups `G = {g1, ..., gk}`
- hunk assignments into those groups
- dependency edges between groups, forming an acyclic merge-order DAG

### Hard constraints

- every hunk is assigned exactly once
- the final dependency graph is acyclic
- independent groups must not overlap on the same file hunk
- total estimated LOC across groups must match the diff

### Soft objectives

- minimize LOC overflow above `max_loc`
- minimize the number of groups when possible
- minimize file scatter across groups
- minimize unnecessary dependency depth
- avoid very tiny groups
- preserve semantic or structural cohesion
- maximize parallel review width when it does not create fragmentation

These objectives compete with one another. There is no single perfect split. The planner therefore uses a mix of heuristics, explicit optimization, and scoring.

## 4. End-to-End Pipeline

The current pipeline is:

1. Extract the merge-base diff against the chosen base branch.
2. Parse the diff into files and hunks.
3. If using the `llm` backend and the diff is too large for one prompt:
   - chunk the ordered hunk sequence with the configured chunking strategy
   - plan chunk by chunk while carrying a catalog of already-created groups
4. Produce groups using the selected partition backend:
   - `llm`
   - `graph`
   - `cp_sat`
5. Recompute estimated LOC from the parsed diff.
6. Derive merge-order dependencies.
7. Validate the plan.
8. Score the plan.
9. Present the plan and, if confirmed, create branches, commits, and PRs.

## 5. Method 1: Dynamic Programming for Chunking

Chunking is only needed when an LLM cannot see the entire diff in a single prompt.

The chunking problem is:

> Given an ordered sequence of hunks with token estimates, split the sequence into contiguous chunks that fit within a token budget while minimizing chunking cost.

This is a classic dynamic-programming problem because:

- hunk order is fixed
- chunk boundaries are contiguous
- the total objective can be decomposed into per-chunk costs

### State

Let `dp[i]` be the minimum cost of chunking the first `i` hunks.

### Transition

For each `i`, consider every earlier cut `j < i` such that the chunk `j+1 .. i` fits in the token budget:

`dp[i] = min(dp[j] + cost(j+1, i))`

Backpointers recover the optimal boundaries.

### Current chunk cost components

The implemented DP cost includes:

- a base cost per chunk
- a slack penalty for wasting too much token budget
- a file-mix penalty when many files are mixed into one chunk
- a strong boundary penalty for splitting adjacent hunks from the same file

### Why this helps

Greedy chunking simply fills a bucket until it overflows. DP instead places boundaries where they hurt the downstream planner least.

This is not the final PR split. It is a pre-processing optimization to improve the quality of chunked LLM planning.

## 6. Method 2: LLM Planning

The `llm` backend asks a language model to propose groups, assignments, and dependencies directly from the diff.

The LLM is useful because it can infer high-level structure:

- feature boundaries
- test-to-source coupling
- likely implementation layers
- documentation or refactor relationships

The LLM is not treated as a formal optimizer. Instead, it acts as a semantic planner.

### Strengths

- strongest semantic understanding
- can group related code across files when the relation is not obvious from paths alone
- flexible for unusual diffs

### Weaknesses

- hard to guarantee globally optimal tradeoffs
- large diffs may require chunking
- behavior depends on prompt quality and model behavior

## 7. Method 3: Graph-Based Partitioning

The `graph` backend is a deterministic heuristic partitioner.

### Step 1: Build partition units

Files are broken into `PartitionUnit`s. A unit contains one or more contiguous hunks from a file, capped so the unit itself does not exceed `max_loc`.

This transforms the raw hunk problem into a smaller bounded grouping problem.

### Step 2: Compute affinity implicitly

The graph heuristic scores how much two units should belong together. Current signals include:

- same file
- shared directory depth
- same file suffix
- source/test pairing heuristic
- priority-specific bias:
  - `logical` rewards grouped related units more aggressively
  - `orthogonal` penalizes mixing files more aggressively

### Step 3: Greedy grouping

The algorithm:

1. seeds a group from the largest remaining unit
2. repeatedly adds the best-scoring candidate that still fits the LOC budget
3. stops when no positive addition remains

This is a deterministic clustering-style heuristic, not an exact solver.

### Why keep it

- no LLM required
- cheap and fast
- stable and reproducible
- good baseline for experimentation

## 8. Method 4: CP-SAT Partitioning

The `cp_sat` backend uses constraint programming with OR-Tools CP-SAT.

This is the most formal optimization backend in the current system.

### Decision variables

For each unit `u` and group slot `g`:

- `x[u, g] in {0,1}` indicates whether unit `u` is assigned to slot `g`

For each slot `g`:

- `y[g] in {0,1}` indicates whether the slot is used
- `overflow[g] >= 0` measures soft violation above `max_loc`

For unit pairs:

- auxiliary variables represent whether two units are colocated in the same group

### Constraints

- every unit is assigned to exactly one group slot
- load in each used slot is bounded by a large upper bound
- overflow captures the amount above `max_loc`
- unused slots must remain empty
- slots are left-packed to reduce symmetry

### Objective

The current objective is a weighted combination of:

- number of groups used
- total LOC overflow
- cross-file grouping penalties in `orthogonal` mode
- affinity rewards for colocating related units

This is a standard weighted-objective formulation: the model converts multiple soft goals into a single scalar objective.

### Why CP-SAT

- supports integer and Boolean structure directly
- handles hard combinatorial constraints better than an LLM
- gives a deterministic optimization baseline

### Current practical limit

The CP-SAT model still scales superlinearly in the number of units. It is suitable for medium-sized instances, but it is not yet a fully tuned large-scale formulation.

## 9. Dependency Derivation

After grouping, `pr-split` derives merge-order dependencies.

The current rule is intentionally conservative for same-file sequencing:

- for a file appearing in multiple groups, groups are ordered by their earliest hunk index in that file
- adjacent groups in that order get dependency edges
- transitive reduction removes redundant edges

This produces a merge-order DAG that is easier to review than a fully dense dependency graph.

## 10. Validation Methodology

Validation is deterministic and currently enforces the hard correctness properties of a split.

The validator checks:

- acyclicity
- hunk coverage
- no duplicate assignment of the same hunk
- LOC conservation between groups and the original diff
- no overlap conflicts between independent groups

The validator also emits soft warnings when groups exceed `max_loc`.

This separation between hard validity and soft quality is important:

- validity says whether the plan is acceptable
- scoring says whether the plan is good

## 11. Scoring Methodology

`pr-split` includes a deterministic scoring layer to quantify plan quality.

Current metrics include:

- total number of groups
- max group LOC
- total overflow above `max_loc`
- dependency edge count
- DAG depth
- DAG width
- file scatter
- tiny-group count

These are combined into a scalar objective used for comparison between candidate plans.

The score is not a proof of optimality. It is a planner-quality heuristic that makes tradeoffs explicit and measurable.

## 12. Why Multiple Methods Exist

There is no single universally best solver for this problem.

Different parts of the workflow benefit from different methods:

- dynamic programming is a natural fit for ordered chunk segmentation
- graph heuristics are fast and deterministic for structural grouping
- CP-SAT is appropriate for constrained combinatorial assignment
- LLMs are best used for semantic interpretation and explanation

This is a hybrid system by design.

## 13. Scientific Framing

From a research perspective, `pr-split` sits at the intersection of:

- constrained clustering
- graph partitioning
- integer optimization
- approximate semantic segmentation of code changes
- human-centered review tooling

The system should be understood as a multi-objective optimization problem with practical engineering constraints:

- incomplete semantic information
- noisy estimates of reviewability
- real-world runtime budgets
- human reviewer preferences that are only partially observable

## 14. Evaluation Methodology for Contributors

If you want to contribute scientifically, do not evaluate changes only on a few anecdotal diffs. Build and compare methods systematically.

Recommended evaluation loop:

1. Collect a benchmark corpus of real diffs.
2. For each diff, run all supported planning backends.
3. Record:
   - validity pass/fail
   - runtime
   - score metrics
   - number of PRs
   - reviewer-facing artifacts
4. Compare plans both quantitatively and qualitatively.
5. Run ablations on new heuristics or objective terms.

### Suggested quantitative metrics

- validity rate
- average score
- average overflow
- average file scatter
- average DAG depth and width
- runtime by backend
- stability across repeated runs

### Suggested qualitative metrics

- human judgment of semantic coherence
- perceived review effort
- whether titles and descriptions match the grouped changes
- whether merge order feels intuitive

## 15. Limitations of the Current Methodology

The current system is useful, but it is not the final word.

Important limitations:

- the graph affinity function is still heuristic and path-based
- CP-SAT weights are hand-tuned, not learned
- LLM chunked planning is still sequential and path-dependent
- scoring is deterministic but not yet calibrated against human review outcomes
- merge-order dependencies are conservative and file-centric

## 16. High-Value Research Directions

The most promising next directions are:

### A. Candidate generation and selection

Generate multiple candidate plans, validate and score them, and choose the best valid plan instead of committing to a single backend invocation.

### B. Repair loops

Turn validation failures into structured feedback and allow one or more repair passes, especially for the LLM backend.

### C. Better semantic affinity

Improve grouping signals using:

- import graphs
- symbol references
- test-to-source mapping
- AST or static-analysis relationships
- LLM-derived affinity hints

### D. Stronger optimization models

Improve the CP-SAT formulation or add alternative partition solvers where useful, but only when the formulation is correctness-safe and benchmarked.

### E. Empirical calibration

Use a benchmark corpus and human review feedback to tune weights in the scoring and CP-SAT objectives.

## 17. Practical Contribution Rules

If you change the methodology, contributors should generally provide:

- the code change
- tests
- rationale for the new heuristic or constraint
- before/after examples
- benchmark evidence when changing objectives or optimization structure

For optimization changes, a good default standard is:

- preserve validity
- do not silently weaken constraints
- show measurable improvement on at least one meaningful metric without major regressions elsewhere

## 18. Code Map

Current code locations:

- chunking: `pr_split/planner/chunker.py`
- planner dispatch and LLM flow: `pr_split/planner/client.py`
- deterministic partitioning and CP-SAT: `pr_split/planner/partitioning.py`
- scoring: `pr_split/planner/scoring.py`
- validation: `pr_split/planner/validator.py`
- CLI and configuration: `pr_split/cli.py`, `pr_split/config.py`

## 19. Summary

`pr-split` is best understood as a hybrid planning system for review decomposition:

- dynamic programming improves large-diff chunking
- graph heuristics provide a deterministic structural splitter
- CP-SAT provides a formal combinatorial optimizer
- LLMs provide semantic planning
- validation and scoring give the system correctness checks and measurable quality criteria

That hybrid design is the methodology.
