"""Live demo of the research harness.

Shows the thing an agent demo usually hides: the approval gate. When the agent
decides to write a file, the run stops and waits for a human.
"""

import uuid

import streamlit as st
from langgraph.types import Command

from harness.agent import RECURSION_LIMIT, build_agent, make_checkpointer
from harness.trace import citations_are_grounded, extract_trajectory

st.set_page_config(page_title="Research Harness", page_icon="*", layout="centered")


@st.cache_resource
def get_agent():
    return build_agent(make_checkpointer(":memory:"))


agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:12]
if "pending" not in st.session_state:
    st.session_state.pending = None

config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "recursion_limit": RECURSION_LIMIT,
}

st.title("LangGraph Research Harness")
st.caption(
    "A tool-using agent over arXiv. Searches, reads pages, and writes briefs — "
    "but never writes anything without asking first."
)

with st.sidebar:
    st.subheader("Guards")
    st.markdown(
        "- Typed tool contracts (Pydantic)\n"
        "- Retry with backoff on transient failure\n"
        "- Per-tool call budgets\n"
        "- Human approval before any write\n"
        "- Durable state (checkpointer)"
    )
    if st.button("New thread"):
        st.session_state.thread_id = uuid.uuid4().hex[:12]
        st.session_state.pending = None
        st.rerun()
    st.caption(f"thread: `{st.session_state.thread_id}`")


def render(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
        st.rerun()

    trajectory = extract_trajectory(result.get("messages", []))
    tools_used = [call["name"] for call in trajectory["tool_calls"]]
    grounded, ungrounded = citations_are_grounded(
        trajectory["answer"], trajectory["observations"]
    )

    st.subheader("Answer")
    st.markdown(trajectory["answer"] or "_No answer produced._")

    left, right = st.columns(2)
    left.metric("Tool calls", len(tools_used))
    right.metric("Citations grounded", "yes" if grounded else "no")
    if ungrounded:
        st.warning(f"Cited but not found in any tool output: {', '.join(ungrounded)}")

    with st.expander("Trajectory"):
        for index, call in enumerate(trajectory["tool_calls"], 1):
            st.markdown(f"**{index}. `{call['name']}`**")
            st.json(call["args"])


if st.session_state.pending:
    request = st.session_state.pending
    st.warning(f"The agent wants to run **{request['tool']}**. Approve?")
    for key, value in request["args"].items():
        with st.expander(key, expanded=(key != "markdown")):
            st.text(str(value)[:2000])

    approve, refuse = st.columns(2)
    if approve.button("Approve", type="primary", use_container_width=True):
        st.session_state.pending = None
        with st.spinner("Running..."):
            render(agent.invoke(Command(resume="yes"), config=config))
    if refuse.button("Refuse", use_container_width=True):
        st.session_state.pending = None
        with st.spinner("Running..."):
            render(agent.invoke(Command(resume="no"), config=config))

else:
    question = st.text_input(
        "Question",
        placeholder="Find two recent papers on evaluating LLM agent trajectories.",
    )
    if st.button("Run", type="primary") and question:
        with st.spinner("Researching..."):
            try:
                render(agent.invoke(
                    {"messages": [{"role": "user", "content": question}]}, config=config
                ))
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
