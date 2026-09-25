from __future__ import annotations

from pr_split.cli import _move_assignment
from pr_split.config import Settings
from pr_split.constants import AssignmentType, PartitionStrategy
from pr_split.diff_ops.parser import parse_diff
from pr_split.planner.partitioning import partition_diff, refresh_generated_description
from pr_split.schemas import Group, GroupAssignment

# The #220 repro: a new module and the main.rs hunk that declares it.
_DIFF = """\
diff --git a/src/search_editor.rs b/src/search_editor.rs
new file mode 100644
--- /dev/null
+++ b/src/search_editor.rs
@@ -0,0 +1,6 @@
+pub struct SearchEditor {
+    query: String,
+}
+
+impl SearchEditor {
+}
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,4 @@
 mod app;
+mod search_editor;
 fn main() {
 }
"""


def _group(gid: str, description: str, *paths: str) -> Group:
    return Group(
        id=gid,
        title=gid,
        description=description,
        assignments=[
            GroupAssignment(
                file_path=path, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[0]
            )
            for path in paths
        ],
    )


def test_move_in_the_editor_relists_generated_descriptions() -> None:
    parsed = parse_diff(_DIFF)
    settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=6, min_loc=1)
    groups = partition_diff(parsed, settings)
    owner = {a.file_path: g for g in groups for a in g.assignments}
    src, dst = owner["src/main.rs"], owner["src/search_editor.rs"]
    assert src.id != dst.id

    assert _move_assignment(groups, parsed, "src/main.rs", 0, src.id, dst.id)

    assert dst.description == "graph partition over 2 file(s): src/main.rs, src/search_editor.rs"
    assert src.description == "graph partition over 0 file(s): "


def test_cp_sat_description_is_relisted() -> None:
    group = _group("pr-1", "cp_sat partition over 1 file(s): old.py", "a.py", "b.py")
    refresh_generated_description(group)
    assert group.description == "cp_sat partition over 2 file(s): a.py, b.py"


def test_llm_and_hand_written_descriptions_are_kept() -> None:
    for text in [
        "Add the search editor and wire it into main",
        "graph partition over many file(s): a.py",
        "graph partition over 3 file(s): a.py",
        "llm partition over 1 file(s): a.py",
        "",
    ]:
        group = _group("pr-1", text, "a.py", "b.py")
        refresh_generated_description(group)
        assert group.description == text
