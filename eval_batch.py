"""对 vector RAG、hybrid RAG 与 enhanced ReAct Agent 运行真实批量对比评估。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

from app.agent.react_loop import run_react_loop
from app.core.cache import clear as clear_cache, get_stats as get_cache_stats, reset_stats as reset_cache_stats
from app.core.llm_client import get_chat_llm, get_llm_call_count, reset_llm_call_count
from app.agent.tools import GenerateAnswerTool
from app.services.ragas_eval import evaluate, evaluate_agent_behavior
from app.services.retriever import search_docs

PIPELINES = ("vector_rag", "hybrid_rag", "react_agent")
DEFAULT_REPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "eval_report.json")
FATAL_ERROR_MARKERS = (
    "insufficient balance",
    "account balance is insufficient",
    "authentication",
    "invalid api key",
    "unauthorized",
    "forbidden",
)


class FatalEvalError(RuntimeError):
    """外部服务不可用时中止批量评估，避免生成大批无效 ERROR 行。"""


def is_fatal_external_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in FATAL_ERROR_MARKERS)


def check_keywords(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    return hits / len(keywords)


def run_pipeline(question: str, pipeline: str, enable_rerank: bool | None) -> dict:
    if pipeline == "react_agent":
        return run_react_loop(
            question,
            retrieval_strategy="enhanced",
            enable_rerank=enable_rerank,
        )

    strategy = "vector" if pipeline == "vector_rag" else "hybrid"
    reset_llm_call_count()
    docs = search_docs(question, top_k=5, strategy=strategy, enable_rerank=enable_rerank)
    response = GenerateAnswerTool().execute(question, docs)
    return {
        "answer": response["answer"],
        "abstained": response["abstained"],
        "citations": response["citations"],
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


def context_recall(retrieved_docs: list, expected_chunk_ids: list[str]) -> float | str:
    if not expected_chunk_ids:
        return "N/A"
    retrieved_ids = {doc.metadata.get("chunk_id") for doc in retrieved_docs}
    return round(len(retrieved_ids & set(expected_chunk_ids)) / len(expected_chunk_ids), 2)


def evaluate_case(case: dict, pipeline: str, enable_rerank: bool | None) -> dict:
    try:
        reset_cache_stats()
        started = time.perf_counter()
        result = run_pipeline(case["question"], pipeline, enable_rerank)
        latency_ms = round((time.perf_counter() - started) * 1000)
        cache_stats = get_cache_stats()
        scores = evaluate(case["question"], result["answer"], result["context"])
        scores["context_recall"] = context_recall(
            result["context"],
            case.get("ground_truth_chunk_ids", []),
        )
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
            "question_type": case.get("question_type", case["category"]),
            "ground_truth_chunk_ids": case.get("ground_truth_chunk_ids", []),
            "answerable": case.get("answerable", True),
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "abstained": result.get("abstained", False),
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
            "keyword_score": keyword_score,
            "latency_ms": latency_ms,
            "llm_calls": result.get("llm_calls", 0),
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_gets": cache_stats["gets"],
            "cache_hits": cache_stats["hits"],
            "routed_to": result.get("routed_to"),
            "retrieval_attempts": result["retrieval_attempts"],
            "agent_metrics": behavior,
            "pass": passed,
        }
    except Exception as exc:
        error = str(exc)
        if is_fatal_external_error(error):
            raise FatalEvalError(error) from exc
        return {
            "pipeline": pipeline,
            "question": case["question"],
            "category": case["category"],
            "error": error,
            "pass": False,
        }


def save_report(rows: list[dict], report_path: str):
    temp_path = f"{report_path}.tmp"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, report_path)


def evaluate_pipeline(
    cases: list[dict],
    pipeline: str,
    rows: list[dict],
    completed: set[tuple[str, str]],
    workers: int,
    enable_rerank: bool | None,
    report_path: str,
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
            executor.submit(evaluate_case, case, pipeline, enable_rerank): case
            for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except FatalEvalError as exc:
                for pending_future in futures:
                    pending_future.cancel()
                save_report(rows, report_path)
                raise FatalEvalError(
                    f"{pipeline} aborted on '{case['question']}': {exc}"
                ) from exc
            rows.append(row)
            completed.add((pipeline, case["question"]))
            save_report(rows, report_path)
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
    average = lambda field: (
        sum(row.get(field, 0) for row in valid) / len(valid)
    )
    numeric_recalls = [
        row["context_recall"] for row in valid
        if isinstance(row.get("context_recall"), (int, float))
    ]
    avg_recall = sum(numeric_recalls) / len(numeric_recalls) if numeric_recalls else 0.0
    passed = sum(row["pass"] for row in valid)
    print(
        f"{pipeline:12} pass={passed}/{len(valid)} "
        f"faith={average('faithfulness'):.2f} "
        f"relevancy={average('answer_relevancy'):.2f} "
        f"precision={average('context_precision'):.2f} "
        f"recall={avg_recall:.2f} "
        f"refusal={refusal_accuracy:.2f} "
        f"latency_ms={average('latency_ms'):.0f} "
        f"llm_calls={average('llm_calls'):.2f} "
        f"cache_hit_rate={average('cache_hit_rate'):.2f} "
        f"failure_rate={(len(pipeline_rows) - len(valid)) / len(pipeline_rows):.2f}"
    )


def preflight_external_services(enable_rerank: bool | None):
    """在正式批跑前验证 embedding 与 chat API 可用，失败则不覆盖报告。"""
    reset_cache_stats()
    reset_llm_call_count()
    try:
        search_docs(
            "什么是 RAG？",
            top_k=1,
            strategy="hybrid",
            enable_rerank=enable_rerank,
        )
        get_chat_llm().invoke("只回复 ok")
    except Exception as exc:
        message = str(exc)
        if is_fatal_external_error(message):
            raise FatalEvalError(f"Preflight failed: {message}") from exc
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--questions-file", help="optional UTF-8 text file with one question per line")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip external API preflight check before starting a fresh report",
    )
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
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as file:
            selected_questions = {line.strip() for line in file if line.strip()}
        cases = [case for case in cases if case["question"] in selected_questions]
        if not cases:
            parser.error("--questions-file did not match any test cases")

    rows = []
    report_path = os.path.abspath(args.report_path)
    enable_rerank = False if args.disable_rerank else None

    if args.resume and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        rows = [
            row for row in loaded
            if row.get("pipeline") in args.pipelines
            and "question" in row
            and "error" not in row
        ]
    else:
        if not args.skip_preflight:
            preflight_external_services(enable_rerank)
        save_report(rows, report_path)
    completed = {(row["pipeline"], row["question"]) for row in rows}

    for pipeline in args.pipelines:
        if args.cold_cache and not args.resume:
            clear_cache()
        evaluate_pipeline(
            cases,
            pipeline,
            rows,
            completed,
            args.workers,
            enable_rerank=enable_rerank,
            report_path=report_path,
        )

    print("\nSummary")
    for pipeline in args.pipelines:
        print_summary(rows, pipeline)
    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
