import os
from typing import Any, Dict, Optional

from llama_index.core.agent import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from github_client import GitHubClient
from pipeline_steps import add_contexts_to_state, add_review_comment_to_state, add_final_review_to_state, \
    get_commit_details_step, get_file_contents_step, get_pr_details_step
from prompts import CONTEXT_AGENT_SYSTEM_PROMPT, COMMENTOR_AGENT_SYSTEM_PROMPT, REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT


def build_llm() -> OpenAI:
    return OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

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

async def run_pr_review_workflow(
    repo_url: str,
    user_prompt: str,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Запускаем AgentWorkflow и возвращаем:
      - final_response
      - финальные значения state (gathered_contexts/review_comment/final_review_comment)
      - (опционально) список tool calls и лог событий
    """
    client = GitHubClient(repo_url=repo_url, token=github_token)
    try:
        llm: OpenAI = build_llm()
        context_agent, commentor_agent, review_and_posting_agent = build_agents(llm, client)
        workflow_agent = build_workflow(context_agent, commentor_agent, review_and_posting_agent)

        prompt = RichPromptTemplate(user_prompt)
        handler = workflow_agent.run(prompt.format())

        final_text = ""
        tool_calls = []
        events_log = []

        current_agent = None
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                events_log.append({"type": "agent", "name": event.current_agent_name})

            if isinstance(event, ToolCall):
                tool_calls.append({"tool": event.tool_name, "args": event.tool_kwargs})
                events_log.append({"type": "tool_call", "tool": event.tool_name})

            if isinstance(event, ToolCallResult):
                events_log.append({"type": "tool_result", "tool": event.tool_name})

            if isinstance(event, AgentOutput):
                if event.response and event.response.content:
                    final_text = event.response.content  # обновляем на последнюю финальную
                    events_log.append({"type": "output", "content": final_text})

        gathered_contexts = await handler.ctx.store.get("gathered_contexts")
        review_comment = await handler.ctx.store.get("review_comment")
        final_review_comment = await handler.ctx.store.get("final_review_comment")

        return {
            "final_response": final_text,
            "state": {
                "gathered_contexts": gathered_contexts or "",
                "review_comment": review_comment or "",
                "final_review_comment": final_review_comment or "",
            },
            "tool_calls": tool_calls,
            "events": events_log,
        }

    finally:
        client.close()