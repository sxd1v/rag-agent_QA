"""
批量评估：对测试集中的每条 case 跑 Agent + RAGAs 四指标 + 关键词检查。

用法：python eval_batch.py
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.react_loop import run_react_loop
from app.services.ragas_eval import evaluate
from app.services.retriever import search_docs


def check_keywords(answer: str, keywords: list[str]) -> float:
    """检查答案中是否包含预期关键词，返回命中率"""
    if not answer or not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hits / len(keywords)


def is_pass(keyword_score: float, faithfulness: float, relevancy: float) -> bool:
    """判定这条 case 是否通过"""
    return keyword_score >= 0.4 and faithfulness >= 0.5 and relevancy >= 0.5


def main():
    # 加载测试集
    cases_path = os.path.join(os.path.dirname(__file__), "data", "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    passed = 0
    all_scores = []

    print(f"\\n{'='*60}")
    print(f"  批量评估：{total} 条测试用例")
    print(f"{'='*60}\\n")

    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"[{i}/{total}] {q}...", end=" ", flush=True)

        try:
            # 跑 Agent
            result = run_react_loop(q)
            answer = result["answer"]

            # 跑 RAGAs 评估
            docs = search_docs(q, top_k=5)
            scores = evaluate(q, answer, docs)

            # 关键词检查
            kw_score = check_keywords(answer, case["expected_keywords"])

            # 判定
            ok = is_pass(kw_score, scores["faithfulness"], scores["answer_relevancy"])
            if ok:
                passed += 1
                print("✅ PASS")
            else:
                print(f"❌ FAIL (kw={kw_score:.2f} faith={scores['faithfulness']} rel={scores['answer_relevancy']})")

            all_scores.append({
                "question": q,
                "category": case["category"],
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_precision": scores["context_precision"],
                "keyword_score": kw_score,
                "pass": ok,
            })

        except Exception as e:
            print(f"💥 ERROR: {e}")
            all_scores.append({
                "question": q,
                "category": case["category"],
                "error": str(e),
                "pass": False,
            })

    # 统计报告
    print(f"\\n{'='*60}")
    print(f"  评估结果汇总")
    print(f"{'='*60}")

    valid_scores = [s for s in all_scores if "error" not in s]
    if valid_scores:
        avg_faith = sum(s["faithfulness"] for s in valid_scores) / len(valid_scores)
        avg_rel = sum(s["answer_relevancy"] for s in valid_scores) / len(valid_scores)
        avg_prec = sum(s["context_precision"] for s in valid_scores) / len(valid_scores)
        avg_kw = sum(s["keyword_score"] for s in valid_scores) / len(valid_scores)

        print(f"  总测试: {total} 条")
        print(f"  通过: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  Faithfulness 均值: {avg_faith:.2f}")
        print(f"  Answer Relevancy 均值: {avg_rel:.2f}")
        print(f"  Context Precision 均值: {avg_prec:.2f}")
        print(f"  关键词命中率均值: {avg_kw:.2f}")

        # 按分类统计
        by_category = {}
        for s in valid_scores:
            cat = s["category"]
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0}
            by_category[cat]["total"] += 1
            if s["pass"]:
                by_category[cat]["passed"] += 1

        print(f"\\n  按分类统计:")
        for cat, stats in by_category.items():
            print(f"    {cat}: {stats['passed']}/{stats['total']} 通过")

    # 保存详细结果
    report_path = os.path.join(os.path.dirname(__file__), "data", "eval_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"\\n  详细报告已保存: {report_path}")

    # 输出 Bad Cases
    failed = [s for s in all_scores if not s.get("pass", False)]
    if failed:
        print(f"\\n  Bad Cases ({len(failed)} 条):")
        for s in failed:
            err = s.get("error", f"kw={s.get('keyword_score','?')} faith={s.get('faithfulness','?')}")
            print(f"    ❌ {s['question']} — {err}")


if __name__ == "__main__":
    main()
