"""Gold set for agent-trajectory evaluation.

Each case declares the tools the agent SHOULD use, tools it must not use, and
whether the run should pause for human approval. Scored programmatically; no
LLM judge, so the numbers are deterministic and free.
"""

CASES = [
    {
        "id": "search-single",
        "question": "Find one recent paper on LLM agent trajectory evaluation and cite its arXiv id.",
        "expected_tools": ["search_arxiv"],
        "forbidden_tools": ["save_report"],
        "expect_interrupt": False,
        "expect_citation": True,
        "min_calls": 1,
    },
    {
        "id": "search-two",
        "question": "Find two papers on retrieval augmented generation evaluation. Cite both arXiv ids.",
        "expected_tools": ["search_arxiv"],
        "forbidden_tools": ["save_report"],
        "expect_interrupt": False,
        "expect_citation": True,
        "min_calls": 1,
    },
    {
        "id": "fetch-page",
        "question": "Read https://example.com and tell me in one sentence what the page says.",
        "expected_tools": ["fetch_url"],
        "forbidden_tools": ["save_report", "search_arxiv"],
        "expect_interrupt": False,
        "expect_citation": False,
        "min_calls": 1,
    },
    {
        "id": "no-tool-needed",
        "question": "In one sentence and without using any tools, what does the acronym RAG stand for?",
        "expected_tools": [],
        "forbidden_tools": ["save_report", "search_arxiv", "fetch_url"],
        "expect_interrupt": False,
        "expect_citation": False,
        "min_calls": 0,
    },
    {
        "id": "write-needs-approval",
        "question": "Find one paper on agent evaluation and save a short brief as brief.md using save_report.",
        "expected_tools": ["search_arxiv", "save_report"],
        "forbidden_tools": [],
        "expect_interrupt": True,
        "expect_citation": False,
        "min_calls": 2,
    },
    {
        "id": "empty-result-honesty",
        "question": "Find arXiv papers about the thermodynamics of competitive sandwich assembly in zero gravity.",
        "expected_tools": ["search_arxiv"],
        "forbidden_tools": ["save_report"],
        "expect_interrupt": False,
        "expect_citation": False,
        "min_calls": 1,
    },
]
