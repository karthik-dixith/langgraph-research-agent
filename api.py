"""HTTP interface to the research harness.

    uvicorn api:app --reload

Two endpoints, because the agent can pause. POST /research starts a run; if the
agent wants a side-effecting tool it returns status=awaiting_approval with a
thread_id, and POST /approve resumes that thread with the decision.
"""

import uuid

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from harness.agent import RECURSION_LIMIT, build_agent, make_checkpointer
from harness.trace import citations_are_grounded, extract_trajectory

app = FastAPI(title="LangGraph Research Harness", version="0.1.0")
agent = build_agent(make_checkpointer())


class Ask(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    thread_id: str | None = Field(default=None, description="Continue an existing thread")


class Decision(BaseModel):
    thread_id: str
    decision: str = Field(description="yes to approve, anything else to refuse")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _shape(result: dict, thread_id: str) -> dict:
    if "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "tool": request["tool"],
            "arguments": request["args"],
        }
    trajectory = extract_trajectory(result.get("messages", []))
    grounded, ungrounded = citations_are_grounded(
        trajectory["answer"], trajectory["observations"]
    )
    return {
        "status": "complete",
        "thread_id": thread_id,
        "answer": trajectory["answer"],
        "tools_used": [call["name"] for call in trajectory["tool_calls"]],
        "citations_grounded": grounded,
        "ungrounded_citations": ungrounded,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research")
def research(ask: Ask) -> dict:
    thread_id = ask.thread_id or uuid.uuid4().hex[:12]
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": ask.question}]},
            config=_config(thread_id),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    return _shape(result, thread_id)


@app.post("/approve")
def approve(decision: Decision) -> dict:
    try:
        result = agent.invoke(
            Command(resume=decision.decision), config=_config(decision.thread_id)
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    return _shape(result, decision.thread_id)
