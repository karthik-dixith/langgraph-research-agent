from typing import List

import random
import time

import arxiv
from pydantic import BaseModel, Field


from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from pydantic import ValidationError


class SearchArxivArgs(BaseModel):
    """Search arXiv for papers. Returns title, authors, date, full abstract and URL for each
    result. The abstract is normally sufficient to answer a question — do not fetch the arXiv
    page afterwards unless you specifically need the full body text."""

    query: str = Field(description="Search terms, e.g. 'retrieval augmented generation evaluation'")
    max_results: int = Field(default=3, ge=1, le=10, description="How many papers to return.  Ask for only as many as the question needs.")


class Paper(BaseModel):
    title: str
    authors: List[str]
    published: str
    summary: str
    url: str


MAX_PAPERS = 5


def search_arxiv(query: str, max_results: int = 3) -> List[Paper]:
    max_results = min(max_results, MAX_PAPERS)
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in client.results(search):
        papers.append(Paper(
            title=result.title.strip().replace("\n", " "),
            authors=[a.name for a in result.authors][:4],
            published=result.published.date().isoformat(),
            summary=result.summary.strip().replace("\n", " ")[:600],
            url=result.entry_id,
        ))
    return papers

# Anchored to the package, not the working directory, so paths survive a
# move and are correct whatever directory the process is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MAX_PAGE_CHARS = 4000
MAX_OBSERVATION_CHARS = 2500

class FetchUrlArgs(BaseModel):
    """Fetch a web page and return its readable text. Use for articles and documentation.
    Do NOT use on arXiv abstract pages — search_arxiv already returns those abstracts."""

    url: str = Field(description="Full http(s) URL of the page to read")


def fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return f"Refused: not an http(s) URL: {url!r}"
    try:
        response = httpx.get(url, timeout=15.0, follow_redirects=True,
                             headers={"User-Agent": "langgraph-research-harness/0.1"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Fetch failed: {exc}"
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())[:MAX_PAGE_CHARS]


class SaveReportArgs(BaseModel):
    """Save a finished research brief to disk as markdown. Only call when the brief is complete."""

    filename: str = Field(
        description="File name only, no folders. Letters, digits, hyphens, underscores, ending .md",
        pattern=r"^[A-Za-z0-9_-]{1,60}\.md$",
    )
    markdown: str = Field(description="The full report body in markdown", min_length=50)


def save_report(filename: str, markdown: str) -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    target = (REPORTS_DIR / filename).resolve()
    if REPORTS_DIR.resolve() not in target.parents:
        return "Refused: path escapes the reports directory."
    target.write_text(markdown, encoding="utf-8")
    return f"Saved {target.name} ({len(markdown)} chars) to {REPORTS_DIR}/"

TOOLS = {
    "search_arxiv": (SearchArxivArgs, search_arxiv),
    "fetch_url": (FetchUrlArgs, fetch_url),
    "save_report": (SaveReportArgs, save_report),
}

def tool_specs() -> List[dict]:
    """The exact tool payload sent to the model on every call."""
    specs = []
    for name, (args_model, _) in TOOLS.items():
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (args_model.__doc__ or "").strip(),
                "parameters": args_model.model_json_schema(),
            },
        })
    return specs

TRANSIENT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
)


def with_retry(fn, attempts: int = 3, base_delay: float = 1.0, **kwargs):
    """Call fn, retrying transient failures with exponential backoff and jitter."""
    last_error = None
    for attempt in range(attempts):
        try:
            return fn(**kwargs)
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    raise last_error

def _budget(observation: str) -> str:
    """Cap what a tool returns. Observations are re-sent on every later turn."""
    if len(observation) <= MAX_OBSERVATION_CHARS:
        return observation
    return observation[:MAX_OBSERVATION_CHARS] + f"\n[truncated at {MAX_OBSERVATION_CHARS} chars]"


def run_tool(name: str, raw_args: dict) -> str:
    """Validate the model's arguments, then execute with retries. Never raises."""
    if name not in TOOLS:
        return f"Unknown tool: {name!r}. Available: {', '.join(TOOLS)}"

    args_model, fn = TOOLS[name]
    try:
        args = args_model(**raw_args)
    except ValidationError as exc:
        return f"Invalid arguments for {name}: {exc}"

    try:
        result = with_retry(fn, **args.model_dump())
    except TRANSIENT_ERRORS as exc:
        return f"Tool {name} unavailable after 3 attempts: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return f"Tool {name} failed: {type(exc).__name__}: {exc}"

    if isinstance(result, str):
        return _budget(result)
    return _budget("\n".join(p.model_dump_json() for p in result))