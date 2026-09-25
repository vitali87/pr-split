EXTRACTING_DIFF = "Extracting diff between {base} and {dev}"
DIFF_STATS = "Diff: {files} files, +{added}/-{removed} lines ({loc} LOC)"
NEW_FILE_SPLIT = (
    "New file '{file}' ({loc} lines) is over --max-loc; split it into {pieces} pieces"
    " at top-level boundaries, each stacked on the one before"
)
NEW_FILE_PIECE_LINKED = (
    "Group '{child}' holds a later piece of new file '{file}'; it now depends on '{parent}'"
)
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
MOVE_REVERTED = "Removed {file} hunk {index} from {branch}"
MOVE_APPLIED = "Applied {file} hunk {index} to {branch}"
LLM_UNAVAILABLE_USING_GRAPH = (
    "LLM planner unavailable ({reason}); planning with the graph backend, which needs no model."
    " Set the key, or PR_SPLIT_PROVIDER=local for a local model, to use the LLM planner"
)
REPAIR_UNKNOWN_FILE = (
    "Plan repair: group '{group}' named '{file}', which is not in the diff; dropped"
)
REPAIR_UNKNOWN_HUNKS = (
    "Plan repair: group '{group}' named hunks {indices} of '{file}', which do not exist; dropped"
)
REPAIR_DUPLICATE_HUNKS = (
    "Plan repair: hunks {indices} of '{file}' were also in group '{owner}';"
    " removed from group '{group}'"
)
REPAIR_UNKNOWN_DEPENDENCY = "Plan repair: group '{group}' depended on unknown ids {deps}; dropped"
REPAIR_EMPTY_GROUP = "Plan repair: group '{group}' holds no hunks; dropped"
PER_GROUP_STEP_RUNNING = "Running per-group step for {group}: {command}"
RESTACKED_LAYER = "Rebased {branch} onto the current head of {parent}"
RESTACK_FETCH_FAILED = "Could not fetch {branch} ({detail}); using the local branch"
LAYER_BEHIND_PARENT = (
    "{group} ({branch}) does not contain the current head of {parent};"
    " run 'pr-split restack' to carry the parent's changes up"
)
BASE_FETCH_FAILED = (
    "Could not fetch '{base}' from '{remote}' ({detail}); splitting against its last fetched state"
)
LOCAL_BASE_BEHIND = (
    "Local '{base}' is {count} commit(s) behind '{upstream}'; splitting against '{upstream}',"
    " which the sub-PRs target, so upstream commits are not split as branch work"
)
LOCAL_BASE_AHEAD = (
    "Local '{base}' has {count} commit(s) not on '{upstream}'; the sub-PRs target '{upstream}',"
    " so any of them on the dev branch are split as branch work. Push '{base}' first if that is"
    " not intended"
)
CREATING_PR = "Creating PR for group '{group}'"
PR_CREATED = "PR #{number} created: {url}"
SAVING_PLAN = "Saving plan to {path}"
PLAN_LOADED = "Loaded plan with {count} groups from {path}"
CLEANING_BRANCHES = "Cleaning up pr-split branches"
BRANCH_DELETED = "Deleted branch {branch}"
PR_CLOSED = "Closed PR #{number}"
PR_ALREADY_DONE = "PR #{number} is already {state}, nothing to close"
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
LOCAL_SERVER_OFF_MACHINE = (
    "PR_SPLIT_LOCAL_BASE_URL points at '{host}', which is not this machine;"
    " the diff will be sent to that host"
)
LLM_OUTPUT_INCOMPLETE = "LLM output incomplete (status: {status}, reason: {reason})"
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
REFINEMENT_SKIPPED_CHUNKED = (
    "Skipping LOC refinement: the diff exceeded the context window and was planned in"
    " chunks, and the refinement prompt embeds the full diff; {remaining} violation(s)"
    " remain. Adjust --min-loc/--max-loc or use the interactive editor"
)
REFINEMENT_REJECTED = (
    "Refinement iteration {iteration} produced an invalid plan ({reason}); "
    "keeping the current plan, {remaining} violation(s) remain"
)
REFINEMENT_NO_IMPROVEMENT = (
    "Refinement iteration {iteration} did not reduce violations ({before} -> {after}); "
    "keeping the current plan"
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
    "CP-SAT stopped at the {timeout:g}s limit with a feasible but unproven-optimal plan "
    "({units} units, {groups} groups); raise --cp-sat-timeout for a better partition"
)
