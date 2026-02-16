# service_langgraph.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, List, TypedDict, Literal
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
import json


from app.client.github_client import GitHubClient
from app.steps.pipeline_steps import (
    get_pr_details_step,
    get_commit_details_step,
    get_file_contents_step,
    add_review_comment_to_pr_review_state,
    add_final_review_to_pr_review_state,
)
from app.prompts.prompts import (
    COMMENTOR_AGENT_SYSTEM_PROMPT,
    REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT,
)


# ---------
# 1) STATE
# ---------

class PRReviewState(TypedDict, total=False):
    repo_url: str
    github_token: Optional[str]
    user_prompt: str
    pr_number: int

    # shared artifacts
    gathered_contexts: str
    review_comment: str
    final_review_comment: str

    # reviewer control
    review_ok: bool
    revision_notes: str  # что исправить, если review_ok=False
    needs_more_context: bool  # если комментор запросил контекст
    requested_files: List[str]  # какие файлы ещё нужны
    reviewer_attempts: int
    force_commentor_final: bool
    # observability
    tool_calls: List[Dict[str, Any]]
    events: List[Dict[str, Any]]

def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )

def _init_obs(state: PRReviewState) -> None:
    state.setdefault("tool_calls", [])
    state.setdefault("events", [])
    state.setdefault("reviewer_attempts", 0)


def _log_agent(state: PRReviewState, name: str) -> None:
    state["events"].append({"type": "agent", "name": name})


def _log_tool(state: PRReviewState, tool: str, args: Dict[str, Any]) -> None:
    state["tool_calls"].append({"tool": tool, "args": args})
    state["events"].append({"type": "tool_call", "tool": tool})


def _log_output(state: PRReviewState, content: str) -> None:
    state["events"].append({"type": "output", "content": content})


def _extract_pr_number(user_prompt: str) -> int:
    m = re.search(r"(?:#|PR\s*)(\d+)", user_prompt, re.IGNORECASE)
    if not m:
        raise ValueError("PR number not found in user_prompt")
    return int(m.group(1))


# -------------------
# 2) GRAPH NODES
# -------------------

async def context_node(state: PRReviewState) -> PRReviewState:
    """
    ЖЁСТКИЙ шаг: собрать то, что разрешено собирать (PR details, changed files, requested files).
    Решение 'какие файлы запросили' приходит из state.requested_files (т.е. не хаос).
    Собираем всё вручную.
    """
    _init_obs(state)
    _log_agent(state, "ContextAgent")

    client = GitHubClient(repo_url=state["repo_url"], token=state.get("github_token"))
    try:
        pr_number = state.get("pr_number") or _extract_pr_number(state["user_prompt"])
        state["pr_number"] = pr_number

        _log_tool(state, "get_pr_details_step", {"pr_number": pr_number})
        pr_details = get_pr_details_step(client, pr_number)

        # changed files via commits (как у тебя)
        changed_files_all: List[Dict[str, Any]] = []
        for sha in pr_details.get("commit_shas", []):
            _log_tool(state, "get_commit_details_step", {"commit_sha": sha})
            changed_files_all.extend(get_commit_details_step(client, sha))

        # optional extra file fetches (только если их запросили)
        extra_files_text: List[str] = []
        for path in state.get("requested_files", []) or []:
            _log_tool(state, "get_file_contents_step", {"file_path": path, "ref": pr_details.get("head_sha")})
            content = get_file_contents_step(client, file_path=path, ref=pr_details.get("head_sha"))
            extra_files_text.append(f"\n\n### File: {path}\n```text\n{content}\n```")

        gathered = (
            f"## PR Details\n{pr_details}\n\n"
            f"## Changed Files (from commits)\n{changed_files_all}\n"
            f"{''.join(extra_files_text)}"
        )

        # сохраняем в graph state (как и раньше)
        state["gathered_contexts"] = gathered

        # сброс флагов
        state["needs_more_context"] = False
        state["requested_files"] = []

        _log_output(state, "Context gathered.")
        return state
    finally:
        client.close()


async def commentor_node(state: PRReviewState) -> PRReviewState:
    """
    Агентный шаг, но bounded:
    - он может: попросить доп.файлы (вернуть needs_more_context + requested_files)
    - или: выдать draft review_comment
    """
    _init_obs(state)
    _log_agent(state, "CommentorAgent")

    llm = build_llm()

    revision_notes = (state.get("revision_notes") or "").strip()
    gathered = (state.get("gathered_contexts") or "").strip()

    prompt = f"""
{COMMENTOR_AGENT_SYSTEM_PROMPT}

USER REQUEST:
{state["user_prompt"]}

AVAILABLE CONTEXT:
{gathered}

REVISION NOTES FROM REVIEWER (if any):
{revision_notes}

TASK:
1) If you need more repository files to write a correct review, output JSON:
   {{"action":"need_context","requested_files":["path/a.py","path/b.md"]}}
2) Otherwise output JSON:
   {{"action":"draft","review_comment":"...markdown 200-300 words..."}}
"""

    # лучше попробовать llm.with_structured_output(...)
    raw = await llm.ainvoke(prompt)

    # примитивный парсинг, в проде заменить на строгую схему
    try:
        data = json.loads(raw.content if hasattr(raw, "content") else str(raw))
    except Exception:
        # fallback: считаем что это draft
        data = {"action": "draft", "review_comment": raw.content if hasattr(raw, "content") else str(raw)}

    if data.get("action") == "need_context":
        state["needs_more_context"] = True
        state["requested_files"] = data.get("requested_files", [])
        state["revision_notes"] = ""  # не мешаем
        _log_output(state, f"Needs more context: {state['requested_files']}")
        return state

    review_comment = data.get("review_comment", "")
    state["review_comment"] = review_comment
    add_review_comment_to_pr_review_state(state, review_comment)

    if state.get("force_commentor_final"):
        state["final_review_comment"] = review_comment
        state["review_ok"] = True
        _log_output(state, "Force final: skipping reviewer")
        return state

    state["review_ok"] = False
    _log_output(state, review_comment)
    return state


async def reviewer_node(state: PRReviewState) -> PRReviewState:

    """
    Контроль качества.
    Он НЕ переписывает текст сам бесконтрольно — он либо принимает, либо выдаёт список правок.
    """

    _init_obs(state)
    _log_agent(state, "ReviewAndPostingAgent")

    state["reviewer_attempts"] += 1


    llm = build_llm()
    draft = (state.get("review_comment") or "").strip()

    prompt = f"""
{REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT}

DRAFT REVIEW:
{draft}

OUTPUT STRICT JSON:
- If OK:
  {{"ok": true, "final_review_comment": "...(maybe lightly edited)..."}}
- If NOT OK:
  {{"ok": false, "revision_notes": "what to fix конкретно по пунктам"}}
"""

    raw = await llm.ainvoke(prompt)

    import json
    try:
        data = json.loads(raw.content if hasattr(raw, "content") else str(raw))
    except Exception:
        # если LLM не дал JSON — считаем, что не ок и просим переписать
        data = {"ok": False, "revision_notes": "Output must be strict JSON and meet the review criteria (200-300 words, markdown, quoted lines, tests/docs/migrations notes)."}

    if data.get("ok") is True:
        state["review_ok"] = True
        final_text = data.get("final_review_comment", draft)
        state["final_review_comment"] = final_text
        add_final_review_to_pr_review_state(state, final_text)
        _log_output(state, final_text)
        return state

    state["review_ok"] = False
    state["revision_notes"] = data.get("revision_notes", "Please rewrite to meet the criteria.")
    _log_output(state, state["revision_notes"])

    if state["reviewer_attempts"] >=  1:
        state["force_commentor_final"] = True
    return state


async def post_node(state: PRReviewState) -> PRReviewState:
    """
    Отдельный жёсткий узел: постинг возможен ТОЛЬКО если review_ok=True.
    """
    _init_obs(state)
    _log_agent(state, "PostToGitHub")

    if not state.get("review_ok"):
        # защита от неправильной связки
        _log_output(state, "Skipped posting because review_ok=False")
        return state

    client = GitHubClient(repo_url=state["repo_url"], token=state.get("github_token"))
    try:
        pr_number = state["pr_number"]
        body = state.get("final_review_comment") or state.get("review_comment") or ""

        # заменить на твой реальный метод постинга
        _log_tool(state, "post_review_comment", {"pr_number": pr_number, "body_len": len(body)})
        client.post_pr_review(pr_number=pr_number, body=body)

        _log_output(state, "Posted to GitHub.")
        return state
    finally:
        client.close()


# -------------------
# 3) ROUTING (edges)
# -------------------

def route_after_commentor(state: PRReviewState) -> Literal["context", "reviewer", "post"]:
    if state.get("needs_more_context"):
        return "context"
    if state.get("force_commentor_final"):
        return "post"
    return "reviewer"


def route_after_reviewer(state: PRReviewState) -> Literal["commentor", "post"]:
    return "post" if state.get("review_ok") else "commentor"


# -------------------
# 4) BUILD GRAPH
# -------------------

def build_pr_review_graph():
    g = StateGraph(PRReviewState)

    g.add_node("context", context_node)
    g.add_node("commentor", commentor_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("post", post_node)

    g.set_entry_point("context")
    g.add_edge("context", "commentor")

    g.add_conditional_edges("commentor", route_after_commentor, {
        "context": "context",
        "reviewer": "reviewer",
        "post": "post",
    })

    g.add_conditional_edges("reviewer", route_after_reviewer, {
        "commentor": "commentor",
        "post": "post",
    })

    g.add_edge("post", END)

    return g.compile()

    # поработать с чекпоинт (можно memory, можно redis/sqlite). Для сохранения состояния на случай падений.
    # return g.compile(checkpointer=MemorySaver())


# --------------------------------
# 5) PUBLIC SERVICE
# --------------------------------

async def run_pr_review_workflow_langgraph(
    repo_url: str,
    user_prompt: str,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    graph = build_pr_review_graph()

    init_state: PRReviewState = {
        "repo_url": repo_url,
        "github_token": github_token,
        "user_prompt": user_prompt,
        "tool_calls": [],
        "events": [],
    }

    final_state: PRReviewState = {}
    async for event in graph.astream_events(init_state, version="v2"):
        if event.get("event") != "on_chat_model_stream":
         init_state["events"].append({"type": "lg_event", "event": event.get("event"), "name": event.get("name")})

    final_state = await graph.ainvoke(init_state)

    final_text = (
        final_state.get("final_review_comment")
        or final_state.get("review_comment")
        or ""
    )

    return {
        "final_response": final_text,
        "state": {
            "gathered_contexts": final_state.get("gathered_contexts") or "",
            "review_comment": final_state.get("review_comment") or "",
            "final_review_comment": final_state.get("final_review_comment") or "",
        },
        "tool_calls": final_state.get("tool_calls") or [],
        "events": final_state.get("events") or [],
    }