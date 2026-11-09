"""Live demo of the research harness.

Shows the thing an agent demo usually hides: the approval gate. When the agent
decides to write a file, the run stops and waits for a human.
"""

import uuid

import streamlit as st
from langgraph.types import Command

from harness.agent import RECURSION_LIMIT, TOOL_BUDGETS, build_agent, make_checkpointer
from harness.trace import citations_are_grounded, extract_trajectory

st.set_page_config(
    page_title="Research Harness",
    page_icon="*",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    ("Search and cite",
     "Find two recent papers on evaluating LLM agent trajectories and cite their arXiv ids."),
    ("Read a web page",
     "Read https://example.com and tell me in one sentence what the page says."),
    ("Try to write a file",
     "Find one recent paper on LLM agent evaluation and save a short brief as "
     "agent-eval.md using the save_report tool."),
]


@st.cache_resource
def get_agent():
    return build_agent(make_checkpointer(":memory:"))


agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:12]
if "pending" not in st.session_state:
    st.session_state.pending = None
if "qbox" not in st.session_state:
    st.session_state.qbox = ""

config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "recursion_limit": RECURSION_LIMIT,
}


def use_example(text: str) -> None:
    st.session_state.qbox = text


with st.sidebar:
    st.subheader("What this is")
    st.markdown(
        "A research agent built on LangGraph. It decides which tool to call, calls it, "
        "reads the result, and repeats until it can answer."
    )

    st.subheader("Its tools")
    st.markdown(
        f"- **search_arxiv** — searches papers (max {TOOL_BUDGETS['search_arxiv']} calls per question)\n"
        f"- **fetch_url** — reads a web page (max {TOOL_BUDGETS['fetch_url']})\n"
        f"- **save_report** — writes a file. **Always asks you first.**"
    )

    st.subheader("Guards")
    st.markdown(
        "- Typed tool contracts (Pydantic), validated before anything runs\n"
        "- Retry with backoff on transient failure\n"
        "- Per-tool call budgets, so it cannot loop\n"
        "- Human approval before any write\n"
        "- Durable state, so a crashed run is still debuggable"
    )

    st.divider()
    st.markdown("[Source on GitHub](https://github.com/karthik-dixith/langgraph-research-agent)")
    if st.button("Start a new thread"):
        st.session_state.thread_id = uuid.uuid4().hex[:12]
        st.session_state.pending = None
        st.rerun()
    st.caption(f"thread: `{st.session_state.thread_id}`")


st.title("LangGraph Research Harness")
st.markdown(
    "Ask a research question. The agent searches **arXiv**, reads pages, and can write "
    "a brief to a file — but it stops and asks before writing anything."
)

with st.expander("How it works", expanded=False):
    st.markdown(
        "1. You ask a question.\n"
        "2. The model decides which tool to call and with what arguments. Those arguments "
        "are validated against a typed schema before anything executes.\n"
        "3. The tool runs, the result goes back to the model, and it decides again.\n"
        "4. If it wants to **write a file**, the run pauses and waits for your approval.\n"
        "5. When it answers, every arXiv id it cites is checked against what the tools "
        "actually returned — so you can see whether it made anything up.\n\n"
        "Every answer below shows the tool calls it made and whether its citations held up."
    )

if not st.session_state.pending:
    st.caption("Not sure what to ask? Try one of these:")
    columns = st.columns(len(EXAMPLES))
    for column, (label, text) in zip(columns, EXAMPLES):
        column.button(label, on_click=use_example, args=(text,), use_container_width=True)


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

    with st.expander("What the agent did", expanded=True):
        if not trajectory["tool_calls"]:
            st.caption("It answered without calling any tool.")
        for index, call in enumerate(trajectory["tool_calls"], 1):
            st.markdown(f"**{index}. `{call['name']}`**")
            st.json(call["args"])


if st.session_state.pending:
    request = st.session_state.pending
    st.warning(
        f"**The agent wants to run `{request['tool']}`.** It cannot write anything "
        "unless you approve it. This is the permission gate."
    )
    for key, value in request["args"].items():
        with st.expander(key, expanded=(key != "markdown")):
            st.text(str(value)[:2000])

    approve, refuse = st.columns(2)
    if approve.button("Approve", type="primary", use_container_width=True):
        st.session_state.pending = None
        with st.spinner("Writing..."):
            render(agent.invoke(Command(resume="yes"), config=config))
    if refuse.button("Refuse", use_container_width=True):
        st.session_state.pending = None
        with st.spinner("Resuming..."):
            render(agent.invoke(Command(resume="no"), config=config))

else:
    question = st.text_input(
        "Your question",
        key="qbox",
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
                st.caption(
                    "This demo runs on a free tier with a daily token limit. If that is a "
                    "rate-limit error, it will clear on its own."
                )
