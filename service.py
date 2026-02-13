from typing import Any, Dict, Optional

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult

from github_client import GitHubClient
from main import build_llm, build_agents, build_workflow  # или перенесите эти функции в отдельный module

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