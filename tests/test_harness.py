"""Offline tests. No API key, no network, no model calls.

They cover the guards that keep the agent safe and terminating: argument
validation, path containment, the retry policy, observation budgeting, and the
duplicate/budget checks that stop tool loops.
"""

import httpx
import pytest
from pydantic import ValidationError

from harness import tools
from harness.agent import _calls_already_made, _tool_use_counts, TOOL_BUDGETS
from harness.trace import citations_are_grounded, is_infra_failure


class FakeCall(dict):
    """Minimal stand-in for a LangChain tool call."""


class FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


# --------------------------------------------------------------------------- #
# Argument contracts
# --------------------------------------------------------------------------- #

def test_filename_pattern_blocks_path_traversal():
    with pytest.raises(ValidationError):
        tools.SaveReportArgs(filename="../escape.md", markdown="x" * 60)


def test_filename_pattern_blocks_wrong_extension():
    with pytest.raises(ValidationError):
        tools.SaveReportArgs(filename="report.exe", markdown="x" * 60)


def test_short_report_is_rejected():
    with pytest.raises(ValidationError):
        tools.SaveReportArgs(filename="ok.md", markdown="too short")


def test_valid_report_args_accepted():
    args = tools.SaveReportArgs(filename="brief-01.md", markdown="x" * 60)
    assert args.filename == "brief-01.md"


def test_save_report_refuses_to_escape_its_directory():
    assert tools.save_report("../escape.md", "x" * 60).startswith("Refused")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def test_unknown_tool_returns_text_not_exception():
    out = tools.run_tool("definitely_not_a_tool", {})
    assert out.startswith("Unknown tool")


def test_invalid_arguments_return_text_not_exception():
    out = tools.run_tool("save_report", {"filename": "../x.md", "markdown": "y" * 60})
    assert out.startswith("Invalid arguments")


def test_non_http_url_is_refused_without_network():
    assert tools.fetch_url("file:///etc/passwd").startswith("Refused")


def test_observation_budget_truncates():
    oversized = "x" * (tools.MAX_OBSERVATION_CHARS + 500)
    out = tools._budget(oversized)
    assert len(out) < len(oversized)
    assert "truncated" in out


def test_observation_under_budget_is_untouched():
    assert tools._budget("short") == "short"


# --------------------------------------------------------------------------- #
# Tool specs sent to the model
# --------------------------------------------------------------------------- #

def test_tool_specs_cover_every_registered_tool():
    names = {spec["function"]["name"] for spec in tools.tool_specs()}
    assert names == set(tools.TOOLS)


def test_tool_specs_carry_descriptions_and_bounds():
    spec = next(s for s in tools.tool_specs() if s["function"]["name"] == "search_arxiv")
    assert spec["function"]["description"]
    params = spec["function"]["parameters"]["properties"]["max_results"]
    assert params["minimum"] == 1 and params["maximum"] == 10


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #

def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.TimeoutException("simulated")
        return "ok"

    assert tools.with_retry(flaky, base_delay=0.01) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_and_reraises():
    def always_fails():
        raise httpx.TimeoutException("simulated")

    with pytest.raises(httpx.TimeoutException):
        tools.with_retry(always_fails, base_delay=0.01)


# --------------------------------------------------------------------------- #
# Loop guards
# --------------------------------------------------------------------------- #

def test_duplicate_detection_matches_identical_arguments():
    history = [FakeMessage([{"name": "search_arxiv", "args": {"query": "rag", "max_results": 3}}])]
    seen = _calls_already_made(history)
    assert ("search_arxiv", '{"max_results": 3, "query": "rag"}') in seen


def test_duplicate_detection_ignores_different_arguments():
    history = [FakeMessage([{"name": "search_arxiv", "args": {"query": "rag"}}])]
    seen = _calls_already_made(history)
    assert ("search_arxiv", '{"query": "agents"}') not in seen


def test_tool_use_counts_accumulate():
    history = [
        FakeMessage([{"name": "search_arxiv", "args": {"query": "a"}}]),
        FakeMessage([{"name": "search_arxiv", "args": {"query": "b"}}]),
        FakeMessage([{"name": "fetch_url", "args": {"url": "https://example.com"}}]),
    ]
    counts = _tool_use_counts(history)
    assert counts["search_arxiv"] == 2
    assert counts["fetch_url"] == 1


def test_every_tool_has_a_budget():
    assert set(TOOL_BUDGETS) == set(tools.TOOLS)


# --------------------------------------------------------------------------- #
# Trace scoring
# --------------------------------------------------------------------------- #

def test_citation_is_grounded_when_id_appears_in_tool_output():
    grounded, missing = citations_are_grounded(
        "See arXiv:2609.21267 for the method.",
        ['{"url": "http://arxiv.org/abs/2609.21267v1"}'],
    )
    assert grounded and missing == []


def test_citation_is_ungrounded_when_model_invents_an_id():
    grounded, missing = citations_are_grounded(
        "See arXiv:1234.56789 for the method.",
        ['{"url": "http://arxiv.org/abs/2609.21267v1"}'],
    )
    assert not grounded and missing == ["1234.56789"]


def test_quota_errors_classify_as_infrastructure():
    assert is_infra_failure("RateLimitError: Error code: 429 - rate limit reached")
    assert is_infra_failure("APIStatusError: 413 - Request too large ... tokens per minute")


def test_agent_errors_do_not_classify_as_infrastructure():
    assert not is_infra_failure("GraphRecursionError: Recursion limit of 12 reached")
    assert not is_infra_failure("")
