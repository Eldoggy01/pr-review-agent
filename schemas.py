from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PRDetails(BaseModel):
    author: str
    title: str
    body: str = ""
    diff_url: str
    state: str
    head_sha: str
    commit_shas: List[str] = Field(default_factory=list)


class ChangedFile(BaseModel):
    filename: str
    status: Literal["added", "modified", "removed", "renamed", "copied", "changed", "unchanged"] | str
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None


class CommitDetails(BaseModel):
    sha: str
    files: List[ChangedFile] = Field(default_factory=list)
