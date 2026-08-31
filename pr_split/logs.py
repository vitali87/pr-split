EXTRACTING_DIFF = "Extracting diff between {base} and {dev}"
DIFF_STATS = "Diff: {files} files, +{added}/-{removed} lines ({loc} LOC)"
SENDING_TO_LLM = "Sending diff to LLM for analysis ({model})"
LLM_RESPONSE_RECEIVED = "Received split plan with {count} groups"
VALIDATING_PLAN = "Validating split plan"
VALIDATION_PASSED = "Plan validation passed"
LOC_MIN_WARN = "Group '{group}' has {loc} diff lines (+{added}/-{removed}) below minimum: {limit}"
LOC_MAX_WARN = "Group '{group}' has {loc} diff lines (+{added}/-{removed}) above maximum: {limit}"
PRESENTING_PLAN = "Split plan ready for review"
CREATING_BRANCH = "Creating branch {branch} from {base}"
CREATING_MERGE_BASE = "Creating merge base {branch} from parents: {parents}"
MATERIALIZING_FILES = "Materializing {count} file(s) for group '{group}'"
COMMITTING_GROUP = "Committing group '{group}': {title}"
PUSHING_BRANCH = "Pushing {branch} to origin"
CREATING_PR = "Creating PR for group '{group}'"
PR_CREATED = "PR #{number} created: {url}"
SAVING_PLAN = "Saving plan to {path}"
PLAN_LOADED = "Loaded plan with {count} groups from {path}"
CLEANING_BRANCHES = "Cleaning up pr-split branches"
BRANCH_DELETED = "Deleted branch {branch}"
PR_CLOSED = "Closed PR #{number}"
CLEAN_COMPLETE = "Cleanup complete: {branches} branches, {prs} PRs"
CLEAN_INCOMPLETE = (
    "Some PRs or branches could not be cleaned up; the plan file was kept"
    " so 'pr-split clean' can be re-run"
)
FETCHING_FORK_PR = "Fetching PR #{number} from fork {fork}"
FETCHING_FORK_BRANCH = "Fetching branch {branch} from fork {fork}"
AUTHOR_PRESERVED = "Preserving author: {author}"
COUNTING_TOKENS = "Counting input tokens ({model})"
TOKEN_COUNT = "Token count: {tokens} (limit: {limit})"
PLANNING_WITH_BACKEND = "Planning split with backend '{backend}'"
CHUNK_STRATEGY_SELECTED = "Using chunking strategy '{strategy}'"
DIFF_TOO_LARGE = (
    "Diff exceeds context window ({tokens} tokens > {limit} limit), switching to chunked mode"
)
CALIBRATING_CHUNKS = (
    "Overhead: {overhead} tokens, diff budget per chunk: {budget} tokens,"
    " ratio: {ratio:.4f} tokens/char"
)
CHUNKED_MODE = "Using chunked processing ({chunks} chunks, {hunks} total hunks)"
CHUNK_SENDING = "Sending chunk {index}/{total} ({hunks} hunks, ~{tokens} tokens)"
CHUNK_RECEIVED = "Chunk {index}/{total}: {new_groups} new groups, {total_groups} total"
LLM_OUTPUT_TRUNCATED = (
    "LLM output truncated (stop_reason: {stop_reason}), keys in partial output: {keys}"
)
CHUNK_RETRY = "Chunk {index}/{total} failed (attempt {attempt}), retrying: {error}"
INVALID_HUNK_INDEX = (
    "Group '{group}': invalid hunk index {index} for {file} (max: {max}), skipping"
)
HUNK_AUTO_ASSIGNED = "Auto-assigned uncovered hunk {file}[{index}] to group '{group}'"
UNCOVERED_HUNKS_FIXED = "Auto-assigned {count} uncovered hunk(s) to existing groups"
PLAN_METRICS = (
    "Plan metrics: groups={groups}, max_group_loc={max_loc}, underflow={underflow}, "
    "overflow={overflow}, width={width}, depth={depth}, scatter={scatter}, objective={objective}"
)
REFINEMENT_START = (
    "LOC bound violations detected ({count}), starting refinement iteration {iteration}"
)
REFINEMENT_RESOLVED = (
    "All LOC bound violations resolved after {iterations} refinement iteration(s)"
)
REFINEMENT_EXHAUSTED = (
    "Refinement iteration limit reached ({iterations}), {remaining} violation(s) remain"
)
STACK_LINKED = "Linked stack for PRs {prs}"
MERGE_NODE_NOT_STACKED = (
    "Group '{group}' depends on multiple groups; native stacks are linear, so its"
    " branch and PR target the base branch directly, carrying every ancestor's"
    " changes until those PRs merge"
)
PR_SKIPPED_BASE_NOT_PUSHED = (
    "Skipping PR for group '{group}': its base branch '{base}' was not pushed"
)
CP_SAT_NOT_OPTIMAL = (
    "CP-SAT stopped at the {timeout:.1f}s limit with a feasible but unproven-optimal plan "
    "({units} units, {groups} groups); raise --cp-sat-timeout for a better partition"
)
