#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرور جستجوی شناسه کالا — نسخه هیبرید (لایه بازیابی جدید)
------------------------------------
دو حالت جستجو:
  - "fast"  (جستجوی سریع)  : فقط کانال واژگانی BM25، بدون فراخوانی OpenAI، فوری و رایگان.
  - "smart" (جستجوی هوشمند): فهم پرس‌وجو (search/understand.py) + بازیابی هیبرید
                              (search/hybrid.py؛ BM25 + معنایی، ترکیب با RRF).

مرحله‌ی انتخاب نهایی (search/select.py) عمداً استفاده نمی‌شود — طبق محک،
دقت آن از گرفتن رتبه‌ی اول بازیابی بهتر نبود.

کاتالوگ، ایندکس واژگانی و ایندکس معنایی همگی در سطح ماژول (یک‌بار، هنگام
import) بارگذاری می‌شوند تا با gunicorn --preload بین همه‌ی workerها به
اشتراک گذاشته شوند.

اجرا (توسعه):
    python server/app.py
اجرا (تولید): از طریق systemd + gunicorn، نگاه کنید به README.md
"""
import datetime
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR, LOGS_DIR, OPENAI_API_KEY, PORT
from search.enrichment import load_enrichment, make_lexical_text_fn
from search.hybrid import HybridIndex
from search.lexical import LexicalIndex
from search.semantic import SemanticIndex
from search.understand import QueryUnderstander

MODE_FAST = "fast"
MODE_SMART = "smart"
RESULT_LIMIT = 8

SEMANTIC_INDEX_PATH = CHECKPOINT_DIR / "catalog_semantic.index"
SEMANTIC_POSITIONS_PATH = CHECKPOINT_DIR / "catalog_semantic_positions.npy"
SEARCH_LOG_PATH = LOGS_DIR / "search_log.jsonl"

app = Flask(__name__)
_log_lock = threading.Lock()

print("⏳ در حال بارگذاری کاتالوگ و ایندکس‌ها ...")

ITEMS = load_catalog()
ENRICHMENT = load_enrichment()  # پیکربندی انتخاب‌شده طبق بنچمارک: غنی‌سازی با examples
LEXICAL = LexicalIndex(items=ITEMS, text_fn=make_lexical_text_fn(ENRICHMENT, include_examples=True))

SEMANTIC = None
HYBRID = None
if SEMANTIC_INDEX_PATH.exists() and SEMANTIC_POSITIONS_PATH.exists():
    SEMANTIC = SemanticIndex.from_prebuilt(SEMANTIC_INDEX_PATH, SEMANTIC_POSITIONS_PATH, items=ITEMS)
    HYBRID = HybridIndex(LEXICAL, SEMANTIC)
else:
    print("⚠️  ایندکس معنایی از پیش‌ساخته پیدا نشد. اجرا کنید: python pipeline/build_search_index.py")

UNDERSTANDER = QueryUnderstander()

print(f"✅ آماده — {len(ITEMS):,} شرح یکتا بارگذاری شد.")


def _to_scored_percent(results):
    """تبدیل [(item, خام‌امتیاز)] به [(item, درصد نسبی به بهترین نتیجه‌ی همین فهرست)]."""
    if not results:
        return []
    top = results[0][1]
    if top <= 0:
        return [(item, 0) for item, _ in results]
    return [(item, round(max(0.0, min(1.0, score / top)) * 100)) for item, score in results]


def make_record(item, score):
    return {
        "description": item.description,
        "domestic_ids": list(item.domestic_ids),
        "imported_ids": list(item.imported_ids),
        "score": score,
    }


def _log_search(query, mode, results):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "results": [
            {
                "description": r["description"],
                "domestic_ids": r["domestic_ids"],
                "imported_ids": r["imported_ids"],
            }
            for r in results
        ],
    }
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _log_lock:
            with open(SEARCH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        print(f"⚠️  خطا در ثبت لاگ جستجو: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "record_count": len(ITEMS),
        "has_smart": HYBRID is not None and bool(OPENAI_API_KEY),
        "has_api_key": bool(OPENAI_API_KEY),
    })


@app.route("/api/search", methods=["POST"])
def search():
    payload = request.get_json(force=True, silent=True) or {}
    query = (payload.get("query") or "").strip()
    mode = payload.get("mode") or MODE_FAST

    if not query:
        return jsonify({"results": []})

    if mode == MODE_SMART:
        if not OPENAI_API_KEY:
            return jsonify({"error": "برای «جستجوی هوشمند»، کلید OpenAI در .env لازم است."})
        if HYBRID is None:
            return jsonify({"error": "ایندکس معنایی آماده نیست. ابتدا python pipeline/build_search_index.py را اجرا کنید."})
        try:
            lexical_query = UNDERSTANDER.lexical_query(query)
            semantic_query = UNDERSTANDER.semantic_query(query)
            raw_results = HYBRID.search_split(lexical_query, semantic_query, k=RESULT_LIMIT)
        except Exception as e:
            return jsonify({"error": f"خطا در ارتباط با OpenAI: {e}"})
    else:
        mode = MODE_FAST
        raw_results = LEXICAL.search(query, k=RESULT_LIMIT)

    results = [make_record(item, pct) for item, pct in _to_scored_percent(raw_results)]
    _log_search(query, mode, results)

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
