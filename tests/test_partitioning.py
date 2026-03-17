from __future__ import annotations

import pytest

from pr_split.config import Settings
from pr_split.constants import PartitionStrategy, Priority
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import PRSplitError
from pr_split.planner.partitioning import build_partition_units, partition_diff

SAMPLE_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,4 @@
+lineA
+lineB
+lineC
+lineD
"""

TWO_HUNK_DIFF = """\
diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1,3 +1,4 @@
 old1
+new1
 old2
 old3
@@ -10,3 +11,4 @@
 old10
+new10
 old11
 old12
"""


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_loc: int,
    partition_strategy: PartitionStrategy,
    priority: Priority = Priority.ORTHOGONAL,
) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return Settings(
        max_loc=max_loc,
        partition_strategy=partition_strategy,
        priority=priority,
    )


class TestBuildPartitionUnits:
    def test_splits_large_file_by_hunks(self) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        units = build_partition_units(parsed, 1)
        assert len(units) == 2
        assert units[0].file_path == "c.py"
        assert units[0].hunk_indices == (0,)
        assert units[1].hunk_indices == (1,)


class TestPartitionDiffGraph:
    def test_orthogonal_mode_keeps_unrelated_files_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = partition_diff(parsed, settings)
        assert len(groups) == 2
        assert all(group.depends_on == [] for group in groups)

    def test_split_file_groups_get_merge_order_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=1,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = partition_diff(parsed, settings)
        assert len(groups) == 2
        assert groups[1].depends_on == ["pr-1"]


class TestPartitionDiffCpSat:
    def test_missing_ortools_raises_clean_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
        )
        try:
            import ortools  # noqa: F401
        except ImportError:
            with pytest.raises(PRSplitError, match="ortools"):
                partition_diff(parsed, settings)
        else:
            groups = partition_diff(parsed, settings)
            assert groups
