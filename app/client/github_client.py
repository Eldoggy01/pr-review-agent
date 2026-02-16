import os
from typing import Optional, Any
import dotenv
from github import Github


dotenv.load_dotenv("../../config.env")


class GitHubClient:
    """
    Wrapper around PyGithub. GitHub API calls are here.
    """

    def __init__(self, repo_url: str, token: Optional[str] = None):
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN is required (env var or passed to GitHubClient).")

        self.git = Github(self.token)

        repo_name = repo_url.split("/")[-1].replace(".git", "")
        username = repo_url.split("/")[-2]
        self.full_repo_name = f"{username}/{repo_name}"
        self.repo = self.git.get_repo(self.full_repo_name)

    def close(self) -> None:
        try:
            self.git.close()
        except Exception:
            pass #TODO


    # --- Pull Requests ---

    def get_pull_request(self, pr_number: int) -> Any:
        return self.repo.get_pull(pr_number)

    def get_pull_request_commits(self, pr_number: int) -> list[Any]:
        pr = self.get_pull_request(pr_number)
        return list(pr.get_commits())

    # --- Files ---

    def get_file_contents(self, file_path: str, ref: Optional[str] = None) -> str:
        """
        Returns decoded file content as utf-8 text.
        """
        contents = self.repo.get_contents(file_path, ref=ref or self.repo.default_branch)
        return contents.decoded_content.decode("utf-8")

    # --- Commits ---

    def get_commit(self, sha: str) -> Any:
        return self.repo.get_commit(sha)

    # --- Posting ---
    def post_pr_review(self, pr_number: int, body: str) -> str:
        """
        Posts a PR review with body text. Uses PyGithub PullRequest.create_review.
        Returns a short status string for tool output.
        """
        pr = self.get_pull_request(pr_number)
        pr.create_review(body=body)
        return f"Posted PR review on PR #{pr_number}."
