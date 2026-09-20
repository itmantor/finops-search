#!/usr/bin/env python3
"""
اجراکننده‌ی محک (benchmark) برای کانال‌های جستجو
------------------------------------
benchmark.json را می‌خواند، هر پرس‌وجو را با یک یا چند کانال جستجو
(lexical / semantic / hybrid) اجرا می‌کند و recall@40، recall@5، MRR و
جدول جزئیات هر مورد را چاپ می‌کند.

با پرچم --understand، هر پرس‌وجو پیش از بازیابی از search/understand.py
عبور می‌کند: کانال واژگانی عبارت پاک‌شده + کلیدواژه‌ها را می‌گیرد (شبکه‌ی
توکن گسترده برای BM25) و کانال معنایی فقط عبارت پاک‌شده را (یک بردار
یکپارچه، بدون رقیق‌شدن).

با پرچم --enriched {no-examples,with-examples,all}، متن ایندکس از
pipeline/enrich_catalog.py (canonical/synonyms/examples/uses) استفاده
می‌کند (نگاه کنید به search/enrichment.py). --enriched all هر سه حالت
(none / no-examples / with-examples) را کنار هم مقایسه می‌کند تا مشخص
شود آیا examples واقعاً کمک می‌کند یا فقط نویز/ریسک اضافه می‌کند.
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
ENRICH_STATES = ["none", "no-examples", "with-examples"]
ENRICH_LABELS = {
    "none": "بدون غنی‌سازی",
    "no-examples": "غنی‌سازی بدون examples",
    "with-examples": "غنی‌سازی با examples",
}


def find_rank(results, expected_descriptions):
    """رتبه‌ی (۱-پایه) اولین شرح مطابق در نتایج، به همراه خود شرح؛ اگر یافت نشد None."""
    expected_set = set(expected_descriptions)
    for rank, (item, _score) in enumerate(results, start=1):
        if item.description in expected_set:
            return rank, item.description
    return None, None


def build_indices(mode_state_pairs, examples_source="reviewed"):
    """ساخت فقط ایندکس‌های لازم برای مجموعه‌ای از جفت‌های (mode, enrich_state).

    dict برگشتی با کلید (mode, state) پر می‌شود. برای state != "none"، متن
    واژگانی از search/enrichment.py ساخته می‌شود و کانال معنایی از بردارهای
    غنی‌شده‌ی جداگانه (checkpoints/catalog_embeddings_enriched.jsonl) استفاده
    می‌کند — این فایل هرگز بردارهای اصلی را جای‌گزین نمی‌کند.
    هر دو حالت no-examples و with-examples از همان بردار معنایی مشترک
    استفاده می‌کنند (چون متن معنایی فقط description+canonical است و examples
    اصلاً در کانال معنایی دخیل نیست)؛ پس فقط یک بار Embedding غنی‌شده لود می‌شود.
    """
    needed_states = {state for _, state in mode_state_pairs}
    enrichment = None
    if needed_states - {"none"}:
        from search.enrichment import load_enrichment
        enrichment = load_enrichment(examples_source=examples_source)

    lexical_cache = {}
    semantic_cache = {}
    hybrid_cache = {}

    def semantic_variant(state):
        return "none" if state == "none" else "enriched"

    def get_lexical(state):
        if state not in lexical_cache:
            from search.lexical import LexicalIndex
            if state == "none":
                lexical_cache[state] = LexicalIndex()
            else:
                from search.enrichment import make_lexical_text_fn
                include_examples = state == "with-examples"
                lexical_cache[state] = LexicalIndex(
                    text_fn=make_lexical_text_fn(enrichment, include_examples)
                )
        return lexical_cache[state]

    def get_semantic(state):
        variant = semantic_variant(state)
        if variant not in semantic_cache:
            from search.semantic import SemanticIndex
            if variant == "none":
                semantic_cache[variant] = SemanticIndex()
            else:
                from common.config import CHECKPOINT_DIR
                path = CHECKPOINT_DIR / "catalog_embeddings_enriched.jsonl"
                semantic_cache[variant] = SemanticIndex(embeddings_path=path)
        return semantic_cache[variant]

    indices = {}
    for mode, state in mode_state_pairs:
        if mode == "lexical":
            indices[(mode, state)] = get_lexical(state)
        elif mode == "semantic":
            indices[(mode, state)] = get_semantic(state)
        elif mode == "hybrid":
            if state not in hybrid_cache:
                from search.hybrid import HybridIndex
                hybrid_cache[state] = HybridIndex(get_lexical(state), get_semantic(state))
            indices[(mode, state)] = hybrid_cache[state]

    return indices


def make_search_fn(mode, index, understander):
    """ساخت تابع search(query) -> results برای یک کانال، با یا بدون فهم پرس‌وجو."""
    if mode == "hybrid":
        def fn(query):
            if understander is None:
                return index.search(query, k=TOP_K)
            lex_q = understander.lexical_query(query)
            sem_q = understander.semantic_query(query)
            return index.search_split(lex_q, sem_q, k=TOP_K)
        return fn

    def fn(query):
        if understander is None:
            q = query
        elif mode == "lexical":
            q = understander.lexical_query(query)
        else:  # semantic
            q = understander.semantic_query(query)
        return index.search(q, k=TOP_K)
    return fn


def run_mode(label, search_fn, cases):
    """اجرای محک برای یک کانال (با label دلخواه)؛ چاپ جدول جزئیات و بازگرداندن آمار."""
    hits_40 = 0
    hits_5 = 0
    reciprocal_ranks = []
    rows = []

    for case in cases:
        query = case["query"]
        expected = case["expected_descriptions"]
        results = search_fn(query)
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
    print(f"حالت: {label}")
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
    print(f"{'حالت':<20} {'recall@40':>10} {'recall@5':>10} {'MRR':>8}")
    for label, s in all_stats.items():
        print(f"{label:<20} {s['recall@40']:>10.3f} {s['recall@5']:>10.3f} {s['mrr']:>8.3f}")


def print_understanding(cases, understander):
    print("=" * 100)
    print("خروجی فهم پرس‌وجو (query understanding)")
    print("=" * 100)
    for i, case in enumerate(cases, start=1):
        query = case["query"]
        u = understander.understand(query)
        print(f"{i:>3}. خام:      {query}")
        print(f"     برند:      {u.get('brand', '')}")
        print(f"     پاک‌شده:   {u.get('clean', '')}")
        print(f"     کلیدواژه‌ها: {'، '.join(u.get('keywords', []))}")
        print()


def main():
    parser = argparse.ArgumentParser(description="اجرای محک روی کانال‌های جستجو")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--all", action="store_true", help="اجرای هر سه حالت و چاپ جدول مقایسه")
    parser.add_argument("--understand", action="store_true",
                         help="عبور پرس‌وجو از فهم پرس‌وجو (search/understand.py) پیش از بازیابی")
    parser.add_argument("--enriched", choices=ENRICH_STATES + ["all"], default="none",
                         help="متن ایندکس از غنی‌سازی کاتالوگ استفاده کند؛ all هر سه حالت را مقایسه می‌کند")
    parser.add_argument("--examples-source", choices=["reviewed", "raw"], default="reviewed",
                         help="reviewed: examples پاس ۲ (بازبینی‌شده)؛ raw: examples_raw پاس ۱ (بدون بازبینی، برای همه‌ی شرح‌ها)")
    args = parser.parse_args()

    modes = MODES if args.all else [args.mode]
    enrich_states = ENRICH_STATES if args.enriched == "all" else [args.enriched]

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    cases = benchmark["cases"]

    understander = None
    if args.understand:
        from search.understand import QueryUnderstander
        understander = QueryUnderstander()
        print_understanding(cases, understander)

    # با --all --understand، هر حالت هم بدون و هم با فهم پرس‌وجو اجرا می‌شود.
    if args.all and args.understand:
        understand_variants = [(None, "بدون فهم"), (understander, "با فهم")]
    else:
        understand_variants = [(understander, "با فهم" if args.understand else None)]

    variants = []  # (mode, state, understander, label)
    show_state_label = len(enrich_states) > 1 or args.enriched != "none"
    for mode in modes:
        for state in enrich_states:
            for u, u_label in understand_variants:
                label_bits = [mode]
                if show_state_label:
                    label_bits.append(f"[{ENRICH_LABELS[state]}]")
                if u_label:
                    label_bits.append(f"({u_label})")
                variants.append((mode, state, u, " ".join(label_bits)))

    print("در حال ساخت ایندکس(ها)...")
    mode_state_pairs = {(mode, state) for mode, state, _, _ in variants}
    indices = build_indices(mode_state_pairs, examples_source=args.examples_source)
    print("ایندکس(ها) آماده شد.\n")

    all_stats = {}
    for mode, state, u, label in variants:
        search_fn = make_search_fn(mode, indices[(mode, state)], u)
        all_stats[label] = run_mode(label, search_fn, cases)

    if args.all or len(variants) > 1:
        print_comparison(all_stats)


if __name__ == "__main__":
    main()
