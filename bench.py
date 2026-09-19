#!/usr/bin/env python3
"""
اجراکننده‌ی محک (benchmark) برای کانال‌های جستجو
------------------------------------
benchmark.json را می‌خواند، هر پرس‌وجو را با یک یا چند کانال جستجو
(lexical / semantic / hybrid) اجرا می‌کند و recall@40، recall@5، MRR و
جدول جزئیات هر مورد را چاپ می‌کند.
"""
import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = BASE_DIR / "benchmark.json"

TOP_K = 40
TOP_K_STRICT = 5
SHOW_ON_FAIL = 5

MODES = ["lexical", "semantic", "hybrid"]


def find_rank(results, expected_descriptions):
    """رتبه‌ی (۱-پایه) اولین شرح مطابق در نتایج، به همراه خود شرح؛ اگر یافت نشد None."""
    expected_set = set(expected_descriptions)
    for rank, (item, _score) in enumerate(results, start=1):
        if item.description in expected_set:
            return rank, item.description
    return None, None


def build_indices(modes):
    """ساخت فقط ایندکس‌های لازم؛ semantic بین حالت semantic و hybrid مشترک است."""
    indices = {}
    lexical_index = None
    semantic_index = None

    if "lexical" in modes or "hybrid" in modes:
        from search.lexical import LexicalIndex
        lexical_index = LexicalIndex()
        indices["lexical"] = lexical_index

    if "semantic" in modes or "hybrid" in modes:
        from search.semantic import SemanticIndex
        semantic_index = SemanticIndex()
        indices["semantic"] = semantic_index

    if "hybrid" in modes:
        from search.hybrid import HybridIndex
        indices["hybrid"] = HybridIndex(lexical_index, semantic_index)

    return indices


def run_mode(mode, index, cases):
    """اجرای محک برای یک کانال؛ چاپ جدول جزئیات و بازگرداندن آمار."""
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
    stats = {
        "recall@40": hits_40 / n,
        "recall@5": hits_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
    }

    print("=" * 100)
    print(f"حالت: {mode}")
    print("=" * 100)
    print(f"{'#':>3}  {'رتبه':>6}  {'پرس‌وجو':<45} {'شرح یافت‌شده / منتظره'}")
    print("-" * 100)
    for i, (query, expected, rank, matched_desc, results) in enumerate(rows, start=1):
        rank_str = str(rank) if rank is not None else "✗"
        match_str = matched_desc if matched_desc else " / ".join(expected)
        print(f"{i:>3}  {rank_str:>6}  {query:<45} {match_str}")

        if rank is None:
            print(f"      انتظار: {' / '.join(expected)}")
            print(f"      {SHOW_ON_FAIL} نتیجه‌ی برتر برگشتی به‌جای آن:")
            for j, (item, score) in enumerate(results[:SHOW_ON_FAIL], start=1):
                print(f"        {j}. ({score:.3f}) {item.description}")
            print()

    print("-" * 100)
    print(f"تعداد موارد:     {n}")
    print(f"recall@{TOP_K}:      {stats['recall@40']:.3f}  ({hits_40}/{n})")
    print(f"recall@{TOP_K_STRICT}:       {stats['recall@5']:.3f}  ({hits_5}/{n})")
    print(f"MRR:             {stats['mrr']:.3f}")
    print()

    return stats


def print_comparison(all_stats):
    print("=" * 60)
    print("جدول مقایسه‌ی کانال‌ها")
    print("=" * 60)
    print(f"{'حالت':<12} {'recall@40':>10} {'recall@5':>10} {'MRR':>8}")
    for mode in MODES:
        if mode not in all_stats:
            continue
        s = all_stats[mode]
        print(f"{mode:<12} {s['recall@40']:>10.3f} {s['recall@5']:>10.3f} {s['mrr']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="اجرای محک روی کانال‌های جستجو")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--all", action="store_true", help="اجرای هر سه حالت و چاپ جدول مقایسه")
    args = parser.parse_args()

    modes = MODES if args.all else [args.mode]

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    cases = benchmark["cases"]

    print("در حال ساخت ایندکس(ها)...")
    indices = build_indices(modes)
    print("ایندکس(ها) آماده شد.\n")

    all_stats = {}
    for mode in modes:
        all_stats[mode] = run_mode(mode, indices[mode], cases)

    if args.all:
        print_comparison(all_stats)


if __name__ == "__main__":
    main()
