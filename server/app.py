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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR, LOGS_DIR, OPENAI_API_KEY, PORT
from common.textnorm import tokenize
from search.enrichment import load_enrichment, make_lexical_text_fn
from search.hybrid import HybridIndex
from search.lexical import LexicalIndex
from search.semantic import SemanticIndex
from search.taxonomy_labels import label_for
from search.understand import QueryUnderstander

MODE_FAST = "fast"
MODE_SMART = "smart"

# عمق بازیابی داخلی (هر دو حالت) در برابر عمق نمایش اولیه در کلاینت:
# قبلاً هر دو ۸ بودند، پس یک تطابق در رتبه‌ی ۲۸ (مثل «شیر لبنیات» زیر ۷۶
# شیرآلات صنعتی) هرگز دیده نمی‌شد. حالا ۲۰۰ مورد بازیابی و به کلاینت
# برگردانده می‌شود؛ کلاینت فقط ۲۰ تای اول را نشان می‌دهد و بقیه را با دکمه‌ی
# «نمایش بیشتر» بدون فراخوانی دوباره‌ی API آشکار می‌کند.
RETRIEVE_LIMIT = 200
DISPLAY_LIMIT = 20  # فقط برای برش لاگ؛ کلاینت مقدار خودش را دارد

# تشخیص ابهام دسته‌بندی: بین ۵۰ نتیجه‌ی برتر، اگر حداقل دو گروه level1 هرکدام
# حداقل ۳ عضو داشته باشند و بزرگ‌ترین گروه کمتر از ۷۰٪ کل باشد، یعنی نتایج
# بین چند دسته‌ی واقعاً متفاوت پخش شده‌اند (نه یک نویز جزئی داخل یک دسته).
#
# «۵۰ نتیجه‌ی برتر» فقط از میان مواردی انتخاب می‌شود که همه‌ی توکن‌های
# پرس‌وجو را دارند (AND، نه OR): BM25/RRF برای بازیابی خوب عمداً OR است
# (هر سند حاوی حتی یک توکن امتیاز می‌گیرد)، اما همین باعث می‌شود برای
# پرس‌وجوی دوکلمه‌ای مثل «تیرچه بلوک»، اسنادی که فقط کلمه‌ی عمومی «بلوک»
# را دارند (مثلاً بلوک سیلندر، بلوک تغذیه) وارد ۵۰تای برتر شوند و ابهام
# قلابی بسازند — با آزمایش مستقیم روی کاتالوگ تأیید شد: هیچ شرحی هر دو
# توکن «تیرچه»/«بلوک» را با هم ندارد، پس فیلتر AND این مورد را کاملاً
# حذف می‌کند، در حالی که برای «شیر» (یک توکنی) AND همان OR است و ابهام
# واقعی (شیر لبنیات/شیرآلات صنعتی) دست‌نخورده باقی می‌ماند.
AMBIGUITY_TOP_N = 50
AMBIGUITY_MIN_GROUP = 3
AMBIGUITY_MAX_SHARE = 0.7

SEMANTIC_INDEX_PATH = CHECKPOINT_DIR / "catalog_semantic.index"
SEMANTIC_POSITIONS_PATH = CHECKPOINT_DIR / "catalog_semantic_positions.npy"
SEARCH_LOG_PATH = LOGS_DIR / "search_log.jsonl"

app = Flask(__name__)
_log_lock = threading.Lock()

print("⏳ در حال بارگذاری کاتالوگ و ایندکس‌ها ...")

ITEMS = load_catalog()
ENRICHMENT = load_enrichment(examples_source="raw")  # پیکربندی انتخاب‌شده طبق بنچمارک: غنی‌سازی با examples (پاس ۱، ۱۰۰٪ کامل — پاس ۲ رهاشده و فقط ۶.۷٪ پوشش دارد)
LEXICAL_TEXT_FN = make_lexical_text_fn(ENRICHMENT, include_examples=True)
LEXICAL = LexicalIndex(items=ITEMS, text_fn=LEXICAL_TEXT_FN)

# توکن‌های هر شرح، از پیش محاسبه‌شده؛ فقط برای فیلتر AND در تشخیص ابهام
# (نگاه کنید به compute_ambiguity) استفاده می‌شود، نه برای رتبه‌بندی BM25.
ITEM_TOKENS = {item: frozenset(tokenize(LEXICAL_TEXT_FN(item))) for item in ITEMS}

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
        # فقط برای گروه‌بندی کلاینت هنگام ابهام؛ در نتایج عادی نمایش داده نمی‌شود.
        "level1": item.level1,
    }


def compute_ambiguity(ranked_items, query_tokens):
    """تشخیص قطعی و بدون فراخوانی مدل: آیا ۵۰ نتیجه‌ی برتر بین چند دسته‌ی
    level1 واقعاً متفاوت پخش شده‌اند؟ فقط گروه‌های حداقل ۳عضوی به‌عنوان چیپ
    برگردانده می‌شوند (یک مورد پرت تک‌عضوی گزینه‌ی معناداری برای انتخاب نیست).

    فقط مواردی در نظر گرفته می‌شوند که همه‌ی query_tokens را دارند (AND) —
    نه هر موردی که در بازیابی OR-محور رتبه گرفته؛ دلیل را در تعریف
    AMBIGUITY_TOP_N بالا ببینید."""
    if query_tokens:
        candidates = [item for item in ranked_items if query_tokens <= ITEM_TOKENS.get(item, frozenset())]
    else:
        candidates = ranked_items

    top = candidates[:AMBIGUITY_TOP_N]
    if not top:
        return False, []

    counts = Counter(item.level1 for item in top)
    ordered = counts.most_common()
    largest = ordered[0][1]
    qualifying = [(level1, c) for level1, c in ordered if c >= AMBIGUITY_MIN_GROUP]

    ambiguous = len(qualifying) >= 2 and largest < AMBIGUITY_MAX_SHARE * len(top)
    groups = [
        {"label": label_for(level1), "level1": level1, "count": c}
        for level1, c in qualifying
    ]
    return ambiguous, groups


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
            for r in results[:DISPLAY_LIMIT]
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
            raw_results = HYBRID.search_split(lexical_query, semantic_query, k=RETRIEVE_LIMIT)
            # عبارت پاک‌شده (بدون برند/کلیدواژه‌های گسترش‌یافته) برای فیلتر AND
            # ابهام؛ از کش UNDERSTANDER می‌آید، فراخوانی اضافه‌ای ندارد.
            query_tokens = set(tokenize(semantic_query))
        except Exception as e:
            return jsonify({"error": f"خطا در ارتباط با OpenAI: {e}"})
    else:
        mode = MODE_FAST
        raw_results = LEXICAL.search(query, k=RETRIEVE_LIMIT)
        query_tokens = set(tokenize(query))

    ambiguous, groups = compute_ambiguity([item for item, _ in raw_results], query_tokens)
    results = [make_record(item, pct) for item, pct in _to_scored_percent(raw_results)]
    _log_search(query, mode, results)

    return jsonify({"results": results, "ambiguous": ambiguous, "groups": groups})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
