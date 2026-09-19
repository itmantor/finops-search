#!/usr/bin/env python3
"""
اجراکننده‌ی محک (benchmark) برای ایندکس واژگانی
------------------------------------
benchmark.json را می‌خواند، هر پرس‌وجو را با جستجوی واژگانی اجرا می‌کند
و recall@40 ،recall@5، MRR و جدول جزئیات هر مورد را چاپ می‌کند.
"""
import json
from pathlib import Path

from search.lexical import LexicalIndex

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = BASE_DIR / "benchmark.json"

TOP_K = 40
TOP_K_STRICT = 5
SHOW_ON_FAIL = 5


def find_rank(results, expected_descriptions):
    """رتبه‌ی (۱-پایه) اولین شرح مطابق در نتایج، به همراه خود شرح؛ اگر یافت نشد None."""
    expected_set = set(expected_descriptions)
    for rank, (item, _score) in enumerate(results, start=1):
        if item.description in expected_set:
            return rank, item.description
    return None, None


def main():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    cases = benchmark["cases"]

    print("در حال ساخت ایندکس واژگانی...")
    index = LexicalIndex()
    print(f"ایندکس آماده شد ({len(index.items)} شرح یکتا).\n")

    hits_40 = 0
    hits_5 = 0
    reciprocal_ranks = []
    rows = []

    for case in cases:
        query = case["query"]
        expected = case["expected_descriptions"]
        results = index.search(query, k=TOP_K)
        rank, matched_desc = find_rank(results, expected)

        if rank is not None:
            hits_40 += 1
            if rank <= TOP_K_STRICT:
                hits_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append((query, expected, rank, matched_desc, results))

    n = len(cases)
    recall_40 = hits_40 / n
    recall_5 = hits_5 / n
    mrr = sum(reciprocal_ranks) / n

    print("=" * 100)
    print(f"{'#':>3}  {'رتبه':>6}  {'پرس‌وجو':<45} {'شرح یافت‌شده / منتظره'}")
    print("=" * 100)
    for i, (query, expected, rank, matched_desc, results) in enumerate(rows, start=1):
        rank_str = str(rank) if rank is not None else "✗"
        match_str = matched_desc if matched_desc else " / ".join(expected)
        print(f"{i:>3}  {rank_str:>6}  {query:<45} {match_str}")

        if rank is None:
            print(f"      انتظار: {' / '.join(expected)}")
            print(f"      {SHOW_ON_FAIL} نتیجه‌ی برتر برگشتی به‌جای آن:")
            for j, (item, score) in enumerate(results[:SHOW_ON_FAIL], start=1):
                print(f"        {j}. ({score:.2f}) {item.description}")
            print()

    print("=" * 100)
    print(f"تعداد موارد:     {n}")
    print(f"recall@{TOP_K}:      {recall_40:.3f}  ({hits_40}/{n})")
    print(f"recall@{TOP_K_STRICT}:       {recall_5:.3f}  ({hits_5}/{n})")
    print(f"MRR:             {mrr:.3f}")


if __name__ == "__main__":
    main()
