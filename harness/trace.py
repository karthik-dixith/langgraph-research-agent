import json
import re
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / "traces"
ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


def extract_trajectory(messages: list) -> dict:
    """Reconstruct what the agent did from the final message list."""
    calls, observations = [], []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append({"name": call["name"], "args": call["args"]})
        if type(message).__name__ == "ToolMessage":
            observations.append(str(message.content))
    answer = ""
    for message in reversed(messages):
        if type(message).__name__ == "AIMessage" and message.content:
            answer = str(message.content)
            break
    return {"tool_calls": calls, "observations": observations, "answer": answer}


def citations_are_grounded(answer: str, observations: list) -> tuple[bool, list]:
    """Every arXiv id cited in the answer must appear in something a tool returned."""
    haystack = "\n".join(observations)
    cited = sorted(set(ARXIV_ID.findall(answer)))
    ungrounded = [c for c in cited if c.split("v")[0] not in haystack]
    return (not ungrounded), ungrounded


RATE_LIMIT_MARKERS = ("429", "413", "rate limit", "ratelimit", "quota", "too many requests",
                      "request too large", "tokens per minute", "tpm", "overloaded",
                      "503", "502", "timeout", "connection")


def is_infra_failure(error: str) -> bool:
    """True when a run died from quota or transport, not from agent behaviour."""
    low = error.lower()
    return bool(error) and any(marker in low for marker in RATE_LIMIT_MARKERS)


def traced_run(agent, question: str, config: dict) -> dict:
    """Invoke an agent and record a structured trace of the run."""
    run_id = uuid.uuid4().hex[:12]
    started = time.time()
    interrupted = False
    error = ""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
        interrupted = "__interrupt__" in result
        messages = result.get("messages", [])
        halted = False
    except Exception as exc:
        halted = True
        error = f"{type(exc).__name__}: {exc}"
        result = {"error": error}
        # The run died, but the checkpointer still holds what it did.
        try:
            messages = agent.get_state(config).values.get("messages", [])
        except Exception:
            messages = []

    elapsed = time.time() - started
    trajectory = extract_trajectory(messages)
    grounded, ungrounded = citations_are_grounded(trajectory["answer"], trajectory["observations"])

    trace = {
        "run_id": run_id,
        "question": question,
        "tools": [c["name"] for c in trajectory["tool_calls"]],
        "tool_args": [c["args"] for c in trajectory["tool_calls"]],
        "n_tool_calls": len(trajectory["tool_calls"]),
        "answer": trajectory["answer"],
        "grounded": grounded,
        "ungrounded_citations": ungrounded,
        "interrupted": interrupted,
        "halted": halted,
        "error": error,
        "infra_failure": is_infra_failure(error),
        "seconds": round(elapsed, 2),
        "raw_result": result,
    }
    return trace


def save_trace(trace: dict) -> Path:
    TRACE_DIR.mkdir(exist_ok=True)
    path = TRACE_DIR / f"{trace['run_id']}.json"
    serialisable = {k: v for k, v in trace.items() if k != "raw_result"}
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    return path
