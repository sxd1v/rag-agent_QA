"""对 vector RAG、hybrid RAG 与 enhanced ReAct Agent 运行真实批量对比评估。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

from app.agent.react_loop import run_react_loop
from app.core.cache import clear as clear_cache
from app.core.llm_client import get_llm_call_count, reset_llm_call_count
from app.services.qa_service import answer_question
from app.services.ragas_eval import evaluate, evaluate_agent_behavior
from app.services.retriever import search_docs

PIPELINES = ("vector_rag", "hybrid_rag", "react_agent")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "eval_report.json")


def check_keywords(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    return hits / len(keywords)


def run_pipeline(question: str, pipeline: str) -> dict:
    if pipeline == "react_agent":
        return run_react_loop(question, retrieval_strategy="enhanced")

    strategy = "vector" if pipeline == "vector_rag" else "hybrid"
    reset_llm_call_count()
    response = answer_question(question, retrieval_strategy=strategy)
    docs = search_docs(question, strategy=strategy)
    return {
        "answer": response.answer,
        "abstained": response.abstained,
        "citations": response.citations,
        "context": docs,
        "retrieval_attempts": 1,
        "history": [],
        "llm_calls": get_llm_call_count(),
    }


def case_passed(case: dict, result: dict, keyword_score: float, scores: dict) -> bool:
    if not case.get("answerable", True):
        return result.get("abstained", False)
    return (
        not result.get("abstained", False)
        and keyword_score >= 0.4
        and scores["faithfulness"] >= 0.5
        and scores["answer_relevancy"] >= 0.5
    )


def evaluate_case(case: dict, pipeline: str) -> dict:
    try:
        started = time.perf_counter()
        result = run_pipeline(case["question"], pipeline)
        latency_ms = round((time.perf_counter() - started) * 1000)
        scores = evaluate(case["question"], result["answer"], result["context"])
        keyword_score = check_keywords(result["answer"], case["expected_keywords"])
        passed = case_passed(case, result, keyword_score, scores)
        behavior = (
            evaluate_agent_behavior(result, case.get("answerable", True))
            if pipeline == "react_agent" else {}
        )
        return {
            "pipeline": pipeline,
            "question": case["question"],
            "category": case["category"],
            "answerable": case.get("answerable", True),
            "abstained": result.get("abstained", False),
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "keyword_score": keyword_score,
            "latency_ms": latency_ms,
            "llm_calls": result.get("llm_calls", 0),
            "retrieval_attempts": result["retrieval_attempts"],
            "agent_metrics": behavior,
            "pass": passed,
        }
    except Exception as exc:
        return {
            "pipeline": pipeline,
            "question": case["question"],
            "category": case["category"],
            "error": str(exc),
            "pass": False,
        }


def save_report(rows: list[dict]):
    temp_path = f"{REPORT_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, REPORT_PATH)


def evaluate_pipeline(
    cases: list[dict],
    pipeline: str,
    rows: list[dict],
    completed: set[tuple[str, str]],
    workers: int,
):
    pending = [
        case for case in cases
        if (pipeline, case["question"]) not in completed
    ]
    print(f"\n[{pipeline}] pending={len(pending)} completed={len(cases) - len(pending)}", flush=True)
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_case, case, pipeline): case
            for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows.append(row)
            completed.add((pipeline, case["question"]))
            save_report(rows)
            outcome = "PASS" if row.get("pass") else ("ERROR" if "error" in row else "FAIL")
            print(f"  {outcome} {case['question']}", flush=True)


def print_summary(rows: list[dict], pipeline: str):
    pipeline_rows = [row for row in rows if row["pipeline"] == pipeline]
    valid = [row for row in pipeline_rows if "error" not in row]
    if not valid:
        print(f"{pipeline}: no valid results")
        return
    unanswerable = [row for row in valid if not row["answerable"]]
    refusal_accuracy = (
        sum(row["abstained"] for row in unanswerable) / len(unanswerable)
        if unanswerable else 0.0
    )
    average = lambda field: sum(row[field] for row in valid) / len(valid)
    passed = sum(row["pass"] for row in valid)
    print(
        f"{pipeline:12} pass={passed}/{len(valid)} "
        f"faith={average('faithfulness'):.2f} "
        f"relevancy={average('answer_relevancy'):.2f} "
        f"precision={average('context_precision'):.2f} "
        f"refusal={refusal_accuracy:.2f} "
        f"latency_ms={average('latency_ms'):.0f} "
        f"llm_calls={average('llm_calls'):.2f} "
        f"failure_rate={(len(pipeline_rows) - len(valid)) / len(pipeline_rows):.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="clear service caches before each pipeline for a controlled cold-cache comparison",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")

    cases_path = os.path.join(os.path.dirname(__file__), "data", "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as file:
        cases = json.load(file)

    rows = []
    if args.resume and os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        rows = [
            row for row in loaded
            if row.get("pipeline") in args.pipelines
            and "question" in row
            and "error" not in row
        ]
    else:
        save_report(rows)
    completed = {(row["pipeline"], row["question"]) for row in rows}

    for pipeline in args.pipelines:
        if args.cold_cache and not args.resume:
            clear_cache()
        evaluate_pipeline(cases, pipeline, rows, completed, args.workers)

    print("\nSummary")
    for pipeline in args.pipelines:
        print_summary(rows, pipeline)
    print(f"\nDetailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
