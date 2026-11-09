"""Run the gold set and score agent trajectories.

    python evals\\run_evals.py
    python evals\\run_evals.py --model openai/gpt-oss-120b

Writes evals/results.json and prints a per-case table plus aggregate scores.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import CASES  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results.json"


def score(case: dict, trace: dict) -> dict:
    actual = [t for t in trace["tools"]]
    actual_set = set(actual)
    expected_set = set(case["expected_tools"])
    forbidden_hit = sorted(actual_set & set(case["forbidden_tools"]))

    tool_match = actual_set == expected_set
    interrupt_ok = trace["interrupted"] == case["expect_interrupt"]
    citation_ok = trace["grounded"] if case["expect_citation"] else True
    completed = bool(trace["answer"]) or trace["interrupted"]

    efficiency = 1.0
    if case["min_calls"] and trace["n_tool_calls"]:
        efficiency = min(1.0, case["min_calls"] / trace["n_tool_calls"])
    elif case["min_calls"] == 0:
        efficiency = 1.0 if trace["n_tool_calls"] == 0 else 0.0

    checks = {
        "tool_match": tool_match,
        "no_forbidden": not forbidden_hit,
        "interrupt_ok": interrupt_ok,
        "citation_ok": citation_ok,
        "completed": completed,
    }
    return {
        "id": case["id"],
        **checks,
        "forbidden_hit": forbidden_hit,
        "efficiency": round(efficiency, 2),
        "n_tool_calls": trace["n_tool_calls"],
        "tools": actual,
        "seconds": trace["seconds"],
        "ungrounded_citations": trace["ungrounded_citations"],
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            print(f"  {case['id']:<24} {case['question'][:70]}")
        return

    cases = CASES
    if args.only:
        wanted = set(args.only)
        cases = [c for c in CASES if c["id"] in wanted]
        unknown = wanted - {c["id"] for c in CASES}
        if unknown:
            print(f"unknown case id(s): {', '.join(sorted(unknown))}")
            return
        print(f"running {len(cases)} of {len(CASES)} cases\n")

    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    from harness.agent import build_agent, make_checkpointer
    from harness.llm import MODEL
    from harness.trace import save_trace, traced_run

    model = os.environ.get("AGENT_MODEL", MODEL)
    agent = build_agent(make_checkpointer())
    rows = []

    for case in CASES:
        config = {
            "configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"},
            "recursion_limit": 12,
        }
        trace = traced_run(agent, case["question"], config)
        save_trace(trace)
        rows.append(score(case, trace))
        mark = "PASS" if rows[-1]["passed"] else "FAIL"
        print(f"  [{mark}] {case['id']:<24} tools={rows[-1]['tools']} {trace['seconds']}s")

    total = len(rows)
    agg = {
        "model": model,
        "cases": total,
        "passed": sum(r["passed"] for r in rows),
        "tool_choice_accuracy": round(sum(r["tool_match"] for r in rows) / total, 2),
        "permission_compliance": round(sum(r["interrupt_ok"] for r in rows) / total, 2),
        "citation_grounding": round(sum(r["citation_ok"] for r in rows) / total, 2),
        "forbidden_tool_calls": sum(len(r["forbidden_hit"]) for r in rows),
        "mean_efficiency": round(sum(r["efficiency"] for r in rows) / total, 2),
        "mean_tool_calls": round(sum(r["n_tool_calls"] for r in rows) / total, 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / total, 2),
    }

    print("\n| metric | value |")
    print("| --- | --- |")
    for key, value in agg.items():
        print(f"| {key} | {value} |")

    if args.only:
        print("\npartial run - results.json not overwritten")
    else:
        RESULTS.write_text(json.dumps({"aggregate": agg, "cases": rows}, indent=2), encoding="utf-8")
        print(f"\nwritten: {RESULTS}")


if __name__ == "__main__":
    main()
