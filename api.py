import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from service import run_pr_review_workflow

app = FastAPI(title="pr-review-agent API")

DEFAULT_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/Eldoggy01/recipes-api.git")

class ReviewRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    repo_url: Optional[str] = None
    github_token: Optional[str] = None  # можно не передавать, если есть env var

class ReviewResponse(BaseModel):
    final_response: str
    state: dict
    tool_calls: list
    events: list

@app.post("/review", response_model=ReviewResponse)
async def review(req: ReviewRequest):
    repo_url = req.repo_url or DEFAULT_REPO_URL
    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url is required (or set GITHUB_REPO_URL).")

    # токен можно брать из env, если не передан
    github_token = req.github_token or os.getenv("GITHUB_TOKEN")

    result = await run_pr_review_workflow(
        repo_url=repo_url,
        user_prompt=req.prompt,
        github_token=github_token,
    )
    return result