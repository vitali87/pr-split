from __future__ import annotations

from unidiff import PatchSet

from pr_split.diff_ops.reconstructor import apply_hunks

PATCH_TEXT = """\
--- a/example.py
+++ b/example.py
@@ -1,5 +1,6 @@
 line1
+inserted_after_1
 line2
 line3
 line4
 line5
@@ -10,4 +11,5 @@
 line10
 line11
+inserted_after_11
 line12
 line13
"""


def _base_content() -> str:
    return "\n".join(f"line{i}" for i in range(1, 16)) + "\n"


class TestApplyHunks:
    def test_apply_all_hunks(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0, 1])
        lines = result.splitlines()
        assert "inserted_after_1" in lines
        assert "inserted_after_11" in lines

    def test_apply_first_hunk_only(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0])
        lines = result.splitlines()
        assert "inserted_after_1" in lines
        assert "inserted_after_11" not in lines

    def test_apply_second_hunk_only(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [1])
        lines = result.splitlines()
        assert "inserted_after_1" not in lines
        assert "inserted_after_11" in lines

    def test_apply_no_hunks(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [])
        assert result == _base_content()

    def test_line_count_after_one_hunk(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        base = _base_content()
        base_lines = base.splitlines()
        result = apply_hunks(base, pf, [0])
        result_lines = result.splitlines()
        assert len(result_lines) == len(base_lines) + 1

    def test_preserves_untouched_lines(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0])
        lines = result.splitlines()
        assert "line7" in lines
        assert "line8" in lines
        assert "line15" in lines
