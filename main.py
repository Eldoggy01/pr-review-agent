
import asyncio
import os
from typing import Any

import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate

from github_client import GitHubClient
from prompts import CONTEXT_AGENT_SYSTEM_PROMPT, COMMENTOR_AGENT_SYSTEM_PROMPT, REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT
from pipeline_steps import (
    get_pr_details_step,
    get_file_contents_step,
    get_commit_details_step, add_contexts_to_state, add_review_comment_to_state,
    add_final_review_to_state,
)

dotenv.load_dotenv("config.env")


def build_llm() -> OpenAI:
    return OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

def build_github_tools(client: GitHubClient):
    def get_pr_details(pr_number: int) -> dict:
        """Fetch pull request details (author, title, body, diff_url, state, head_sha) and commit SHAs."""
        return get_pr_details_step(client, pr_number)

    def get_file_contents(file_path: str, ref: str | None = None) -> str:
        """Fetch file contents by path from the repository. Optionally provide ref (branch/sha)."""
        return get_file_contents_step(client, file_path=file_path, ref=ref)

    def get_pr_commit_details(commit_sha: str) -> list[dict[str, Any]]:
        """Fetch commit details by SHA, including changed files and patch diffs."""
        return get_commit_details_step(client, commit_sha)

    def post_pr_review(pr_number: int, final_comment: str) -> str:
        """Post a PR final review comment to GitHub for a given PR number."""
        return client.post_pr_review(pr_number=pr_number, body=final_comment)

    return {
        "get_pr_details": FunctionTool.from_defaults(get_pr_details),
        "get_file_contents": FunctionTool.from_defaults(get_file_contents),
        "get_pr_commit_details": FunctionTool.from_defaults(get_pr_commit_details),
        "post_pr_review": FunctionTool.from_defaults(post_pr_review),
    }


def build_state_tools():
    return {
        "add_contexts_to_state": FunctionTool.from_defaults(add_contexts_to_state),
        "add_review_comment_to_state": FunctionTool.from_defaults(add_review_comment_to_state),
        "add_final_review_to_state": FunctionTool.from_defaults(add_final_review_to_state),
    }


def build_agents(llm: OpenAI, client: GitHubClient):
    gh = build_github_tools(client)
    st = build_state_tools()

    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers all needed PR/repo context (PR details, commit diffs, file contents) and saves summary into shared state.",
        tools=[
            gh["get_pr_details"],
            gh["get_file_contents"],
            gh["get_pr_commit_details"],
            st["add_contexts_to_state"],
        ],
        system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
        can_handoff_to=["CommentorAgent"],
    )

    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Uses the context gathered by the context agent to draft a pull review comment.",
        tools=[
            st["add_review_comment_to_state"],
        ],
        system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    )

    review_and_posting_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Reviews the drafted PR comment, requests comment rewrites if needed, saves final review to state, and posts it to GitHub.",
        tools=[
            st["add_final_review_to_state"],
            gh["post_pr_review"],
        ],
        system_prompt=REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT,
        can_handoff_to=["CommentorAgent"],
    )

    return context_agent, commentor_agent, review_and_posting_agent


def build_workflow(context_agent: FunctionAgent, commentor_agent: FunctionAgent, review_and_posting_agent: FunctionAgent) -> AgentWorkflow:
    return AgentWorkflow(
        agents=[context_agent, commentor_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review_comment": "",
        },
    )


async def main():
    repo_url = os.getenv("GITHUB_REPO_URL")
    if not repo_url:
        raise ValueError("Set GITHUB_REPO_URL env var (e.g. https://github.com/user/repo.git).")

    client = GitHubClient(repo_url=repo_url)
    try:
        llm = build_llm()
        context_agent, commentor_agent, review_and_posting_agent = build_agents(llm, client)
        workflow_agent = build_workflow(context_agent, commentor_agent, review_and_posting_agent)

        query = input().strip()
        prompt = RichPromptTemplate(query)

        handler = workflow_agent.run(prompt.format())

        current_agent = None
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"Current agent: {current_agent}")

            elif isinstance(event, AgentOutput):
                if event.response and event.response.content:
                    print("\n\nFinal response:", event.response.content)
                if event.tool_calls:
                    print("Selected tools: ", [call.tool_name for call in event.tool_calls])

            elif isinstance(event, ToolCall):
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")

    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
