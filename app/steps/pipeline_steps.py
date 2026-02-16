from typing import Any, Dict, List, Optional
from app.client.github_client import GitHubClient
from llama_index.core.workflow import Context
from app.schemas.schemas import PRDetails


#Просто маппер
def pr_to_details_dict(pr: Any, commit_shas: List[str]) -> Dict[str, Any]:
    """
    Mapper: PyGithub PR -> dict (validated via PRDetails).
    """
    details = PRDetails(
        author=pr.user.login if pr.user else "unknown",
        title=pr.title or "",
        body=pr.body or "",
        diff_url=pr.diff_url,
        state=pr.state,
        head_sha=pr.head.sha if pr.head else "",
        commit_shas=commit_shas,
    )
    return details.model_dump()


def get_pr_details_step(client: GitHubClient, pr_number: int) -> Dict[str, Any]:
    """
    Tool step: fetch PR details + commit SHAs.
    """
    pr = client.get_pull_request(pr_number)

    commit_shas: List[str] = []
    for c in pr.get_commits():
        commit_shas.append(c.sha)

    return pr_to_details_dict(pr, commit_shas)


def get_file_contents_step(client: GitHubClient, file_path: str, ref: Optional[str] = None) -> str:
    """
    Tool step: fetch file contents from repo.
    """

    print("FILE PATH: "+file_path)
    return client.get_file_contents(file_path=file_path, ref=ref)


def get_commit_details_step(client: GitHubClient, commit_sha: str) -> list[dict[str, Any]]:
    """
    Tool step: fetch commit details including changed files and patches.
    """
    commit = client.repo.get_commit(commit_sha)

    changed_files: list[dict[str, any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch or "",
        })

    return changed_files


async def add_contexts_to_state(ctx: Context, gathered_contexts: str) -> str:
    await ctx.store.set("gathered_contexts", gathered_contexts)
    return "State updated with gathered_contexts."


async def add_review_comment_to_state(ctx: Context, review_comment: str) -> str:
    """Useful for saving the draft PR review comment into shared workflow state."""
    await ctx.store.set("review_comment", review_comment)
    return "State updated with review_comment."

async def add_final_review_to_state(ctx: Context, final_review_comment: str) -> str:
    """Useful for saving the final reviewed PR comment into shared workflow state."""
    await ctx.store.set("final_review_comment", final_review_comment)
    return "State updated with final_review_comment."

def add_review_comment_to_pr_review_state(state, review_comment):
    state["review_comment"] = review_comment

def add_final_review_to_pr_review_state(state, final_review_comment):
    state["final_review_comment"] = final_review_comment
