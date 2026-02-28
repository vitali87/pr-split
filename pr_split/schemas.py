from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

from .constants import AssignmentType, Priority, PRState


class GroupAssignment(BaseModel):
    file_path: str
    assignment_type: AssignmentType
    hunk_indices: list[int] = Field(default_factory=list)


class Group(BaseModel):
    id: str
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    assignments: list[GroupAssignment] = Field(default_factory=list)
    estimated_loc: int = 0
    estimated_added: int = 0
    estimated_removed: int = 0
    expected_patch: str = ""
    expected_patch_sha256: str = ""

    def compute_patch_hash(self) -> str:
        return hashlib.sha256(self.expected_patch.encode()).hexdigest()

    @model_validator(mode="after")
    def sync_patch_hash(self) -> Group:
        if self.expected_patch and not self.expected_patch_sha256:
            self.expected_patch_sha256 = self.compute_patch_hash()
        return self


class SplitPlan(BaseModel):
    dev_branch: str
    base_branch: str
    max_loc: int
    priority: Priority
    groups: list[Group] = Field(default_factory=list)
    author: str | None = None


class BranchRecord(BaseModel):
    group_id: str
    branch_name: str
    base_branch: str
    commit_sha: str = ""
    merge_base_branch: str | None = None
    merge_base_parents: list[str] | None = None


class PRRecord(BaseModel):
    group_id: str
    pr_number: int
    pr_url: str
    state: PRState = PRState.OPEN


class GitState(BaseModel):
    branches: list[BranchRecord] = Field(default_factory=list)
    prs: list[PRRecord] = Field(default_factory=list)


class PlanFile(BaseModel):
    plan: SplitPlan
    git_state: GitState = Field(default_factory=GitState)
