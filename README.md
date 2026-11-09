# LangGraph Research Harness

A tool-using research agent built on **LangGraph**, with the production concerns that
usually get left out of agent demos: typed tool contracts, retries, per-tool budgets,
human approval before side effects, durable state, and a trajectory evaluation harness
that measures whether the agent actually chose correctly.

Given a research question it searches arXiv, reads pages, and writes a brief — but it
cannot write anything to disk without a human saying yes.

```
          +-----------+   wants a tool    +-----------+
  START ->|   agent   |------------------>|   tools   |
          +-----------+                   +-----------+
               ^  |                             |
               |  | final answer                | observation
               |  v                             |
               | END <---------------------------+
```

## Why this exists

My portfolio already had two retrieval projects. This one is deliberately not a third:
the value is in **orchestration, state, permissions and evaluation**, not in retrieval.
The retrieval here is a single tool among three.

## Architecture

| File | Responsibility |
| --- | --- |
| `harness/tools.py` | Tool implementations, Pydantic argument contracts, the dispatcher |
| `harness/llm.py` | One place that knows which model and how to retry it |
| `harness/agent.py` | The `StateGraph`: nodes, routing, budgets, approval gate |
| `harness/trace.py` | Structured run traces and programmatic citation checking |
| `evals/` | Gold set and the scoring runner |
| `api.py` | FastAPI interface, including approval over HTTP |
| `streamlit_app.py` | Live demo that shows the approval gate |

### The tools, and why these three

| Tool | Risk profile | Purpose |
| --- | --- | --- |
| `search_arxiv` | read-only, external | real research data, no key required |
| `fetch_url` | read-only, **untrusted content** | the prompt-injection surface |
| `save_report` | **writes to disk** | gives the permission gate something real to protect |

Two safe tools and one dangerous one. A permission system guarding nothing proves
nothing.

### Design decisions

**Tool contracts are Pydantic models, used twice.** The same class generates the JSON
Schema sent to the model *and* validates the arguments it sends back. One definition,
no drift. The `filename` field carries a regex that rejects path traversal before any
filesystem call, with a `resolve()` containment check behind it — two independent
layers, because one validation you got subtly wrong is no validation.

**Schemas are permissive; limits are enforced in the dispatcher.** Groq validates tool
arguments server-side against the schema and rejects the entire request with a `400` on
a mismatch. A tight bound converts a model over-ask into a hard failure, so
`max_results` accepts up to 10 and the implementation clamps to 5. Degrade, don't error.

**Tools return errors as text; they never raise.** An exception kills the run. A
returned error becomes an observation the model can read and act on. Transient failures
retry with exponential backoff and jitter; permanent ones are reported with different
wording, because the model reads that text and chooses its next move from it.

**Observation output is capped at the dispatcher.** Everything a tool returns is fed
back to the model and re-sent on every later turn. Unbounded tool output is the most
common reason an agent becomes slow and expensive — and it caused a real `413` here
during development. The cap lives where every tool result passes through, so no future
tool can reintroduce the problem.

**Side-effecting tools pause for a human.** `save_report` triggers LangGraph's
`interrupt()`, which suspends the run mid-graph and persists it. The caller sees what
the agent wants to write, approves or refuses, and the run resumes. Refusal is fed back
as an observation, so the agent reports the refusal honestly rather than retrying.

**State is checkpointed to SQLite.** That gives multi-turn memory, but the more useful
property is debuggability: when a run crashes, the trace is recovered from the
checkpointer rather than lost with the exception.

## Terminating the loop: four layers

Each of these exists because the eval harness caught the one above it failing.

| Layer | Mechanism | Defeated by |
| --- | --- | --- |
| Prompt guidance | asks the model not to repeat work | the model ignoring it |
| Duplicate check | blocks identical `(tool, arguments)` | rewording the query |
| Per-tool budget | counts calls, ignores arguments | the model asking anyway |
| **Forced answer** | rebinds the model with no tools at all | nothing — the call is impossible |
| Recursion limit | hard stop at 12 super-steps | last resort; should never fire |

The forced-answer layer took mean latency from **22.4s to 4.2s** and completion rate
from 0.67 to 1.0. The time was never the work; it was two cases spinning until the cap.

## Evaluation

Six gold cases scored on five checks each. **No LLM judge** — every metric is
programmatic, deterministic and free.

Citation grounding is the one worth calling out: it extracts every arXiv id from the
answer and verifies it appeared in something a tool actually returned. That answers
"did the model make this up?" exactly, where an LLM judge would only estimate it.

```bash
python evals/run_evals.py                              # full suite
python evals/run_evals.py --only search-two            # one case, while iterating
python evals/run_evals.py --model openai/gpt-oss-120b  # compare models
```

### Results

`openai/gpt-oss-120b`, 6 cases, 9 Nov 2026:

| Metric | Value |
| --- | --- |
| Cases passed | **6 / 6** |
| Tool-choice accuracy | 1.00 |
| Permission compliance | 1.00 |
| Citation grounding | 1.00 |
| Forbidden tool calls | 0 |
| Completion rate | 1.00 |
| Mean efficiency | 0.78 |
| Mean tool calls | 1.67 |
| Mean latency | 4.21s |

Infrastructure failures (quota, transport) are retried and then reported **separately**
from behavioural scores. A flaky free tier is not an inaccurate agent, and a harness
that conflates them will have you tuning prompts to fix a rate limit.

### What the evals actually caught

Not a score — five real defects, none of which eyeballing the output would have found:

1. A `413` from unbounded tool output exceeding the per-request token ceiling.
2. A `400` from a schema bound tight enough that the model's over-ask was rejected outright.
3. An infinite loop the prompt-level guard did not prevent.
4. A wrong gold case that marked correct behaviour as a failure.
5. Error classification that silently discarded the cause of every failure.

## Running it

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # add GROQ_API_KEY

streamlit run streamlit_app.py            # live demo with the approval gate
uvicorn api:app --reload                  # HTTP API at /docs
python evals/run_evals.py                 # evaluation suite
pytest -q                                 # 22 offline tests, no key needed
```

### Docker

```bash
docker build -t research-harness .
docker run -p 8000:8000 -e GROQ_API_KEY=... research-harness
```

### API

`POST /research` starts a run. If the agent wants a side-effecting tool it returns
`status: awaiting_approval` with a `thread_id`; `POST /approve` resumes that thread.
The pause survives process restarts because it is checkpointed, not held in memory.

## Tests

22 tests, all offline — no API key, no network, no model calls. They cover argument
validation, path containment, the retry policy, observation budgeting, the duplicate
and budget guards, and the trace scoring logic.

## Honest limitations

- **Evaluated on six cases.** Enough to catch the defects listed above, not enough to
  claim general accuracy. A real gold set would be 50+ and cover adversarial inputs.
- **`temperature=0` is not reproducible.** The same case passed and failed across
  consecutive runs on identical code. Hosted inference is non-deterministic; single-run
  numbers should not be quoted as results.
- **Prompt injection is not defended against.** `fetch_url` returns untrusted page text
  straight into the model's context. The approval gate on `save_report` limits the blast
  radius, but a page could still steer the agent's reasoning. Classifier-based filtering
  is the obvious next step.
- **`gpt-oss-20b` measured better than `gpt-oss-120b`** on an earlier revision — fewer
  tool calls, half the latency, same accuracy. The headline numbers above are from the
  120b only because the 20b's daily quota was exhausted. Re-measuring both on the final
  code is outstanding.
- **Trajectory counts are requests, not executions.** When a budget refuses a call, the
  model's request still appears in the trace. The distinction is not yet surfaced in the
  metrics.
- **Single-process checkpointer.** SQLite is fine for one container; a multi-replica
  deployment needs Postgres.

## Stack

Python 3.12 · LangGraph · LangChain · Groq (`openai/gpt-oss-20b` / `120b`) · Pydantic ·
FastAPI · Streamlit · SQLite · Docker
