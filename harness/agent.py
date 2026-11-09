import json
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .llm import build_llm
from .tools import run_tool

SYSTEM_PROMPT = (
    "You are a research assistant. Gather facts with the tools available, then answer "
    "concisely and cite the arXiv id of every paper you rely on. "
    "Do not repeat a search you have already run. If the results are thin or empty, say so "
    "plainly rather than searching again. Answer as soon as you have enough."
)

RECURSION_LIMIT = 12
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DB = str(PROJECT_ROOT / "harness.sqlite")

# Tools that change something outside the process. These pause for human approval.
SIDE_EFFECTING = {"save_report"}

# Hard per-run ceiling on each tool. The model can reword a query to dodge the
# duplicate check, so the budget counts calls regardless of arguments.
TOOL_BUDGETS = {"search_arxiv": 2, "fetch_url": 3, "save_report": 1}
APPROVALS = {"yes", "y", "approve", "approved", "ok"}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def make_checkpointer(path: str = CHECKPOINT_DB) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def agent_node(state: AgentState) -> dict:
    reply = build_llm().invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [reply]}


def _tool_use_counts(messages: list) -> dict:
    """How many times each tool has run so far in this thread."""
    counts: dict = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            counts[call["name"]] = counts.get(call["name"], 0) + 1
    return counts


def _calls_already_made(messages: list) -> set:
    """Every (tool, arguments) pair executed earlier in this run."""
    seen = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            seen.add((call["name"], json.dumps(call["args"], sort_keys=True)))
    return seen


def tools_node(state: AgentState) -> dict:
    messages = state["messages"]
    already = _calls_already_made(messages[:-1])
    used = _tool_use_counts(messages[:-1])
    results = []
    for call in messages[-1].tool_calls:
        name = call["name"]
        budget = TOOL_BUDGETS.get(name)
        if budget is not None and used.get(name, 0) >= budget:
            results.append(ToolMessage(
                content=f"Budget exhausted: {name} may run at most {budget} times per request "
                        f"and has already run {used[name]} times. Answer from what you have, or "
                        "state plainly that you could not find what was asked for.",
                tool_call_id=call["id"],
            ))
            continue
        used[name] = used.get(name, 0) + 1
        key = (name, json.dumps(call["args"], sort_keys=True))
        if key in already:
            results.append(ToolMessage(
                content=f"Already called {call['name']} with these exact arguments earlier in "
                        "this run; the result is unchanged. Do not call it again. Answer from "
                        "what you have, or state plainly that the search found nothing relevant.",
                tool_call_id=call["id"],
            ))
            continue
        if call["name"] in SIDE_EFFECTING:
            decision = interrupt({
                "tool": call["name"],
                "args": call["args"],
                "prompt": f"Approve {call['name']}?",
            })
            if str(decision).strip().lower() not in APPROVALS:
                results.append(ToolMessage(
                    content="Refused by the human reviewer. Do not retry this action; "
                            "report to the user that it was declined.",
                    tool_call_id=call["id"],
                ))
                continue
        results.append(ToolMessage(
            content=run_tool(call["name"], call["args"]),
            tool_call_id=call["id"],
        ))
    return {"messages": results}


def route(state: AgentState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


def build_agent(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def run(question: str, config: dict | None = None) -> dict:
    merged = {"recursion_limit": RECURSION_LIMIT}
    if config:
        merged.update(config)
    try:
        return build_agent().invoke({"messages": [HumanMessage(question)]}, config=merged)
    except GraphRecursionError:
        return {"messages": [AIMessage(
            f"Halted after {merged['recursion_limit']} steps without an answer. The question "
            "may be too broad, or a tool may be returning unusable results."
        )]}
