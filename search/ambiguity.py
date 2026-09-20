# -*- coding: utf-8 -*-
"""
تشخیص ابهام دسته‌بندی — مشترک بین سرور وب و پردازش گروهی اکسل
------------------------------------
بین N نتیجه‌ی برتر، اگر حداقل دو گروه level1 هرکدام حداقل ۳ عضو داشته
باشند و بزرگ‌ترین گروه کمتر از ۷۰٪ کل باشد، یعنی نتایج بین چند دسته‌ی
واقعاً متفاوت پخش شده‌اند (نه یک نویز جزئی داخل یک دسته).

«N نتیجه‌ی برتر» فقط از میان مواردی انتخاب می‌شود که همه‌ی توکن‌های
پرس‌وجو را دارند (AND، نه OR): BM25/RRF برای بازیابی خوب عمداً OR است
(هر سند حاوی حتی یک توکن امتیاز می‌گیرد)، اما همین باعث می‌شود برای
پرس‌وجوی دوکلمه‌ای مثل «تیرچه بلوک»، اسنادی که فقط کلمه‌ی عمومی «بلوک»
را دارند (مثلاً بلوک سیلندر، بلوک تغذیه) وارد Nتای برتر شوند و ابهام
قلابی بسازند — با آزمایش مستقیم روی کاتالوگ تأیید شد: هیچ شرحی هر دو
توکن «تیرچه»/«بلوک» را با هم ندارد، پس فیلتر AND این مورد را کاملاً
حذف می‌کند، در حالی که برای «شیر» (یک توکنی) AND همان OR است و ابهام
واقعی (شیر لبنیات/شیرآلات صنعتی) دست‌نخورده باقی می‌ماند.

هم سرور وب (server/app.py) و هم پردازش گروهی (pipeline/bulk_lookup.py) از
همین تابع استفاده می‌کنند تا هر دو مسیر دقیقاً یک قانون را اجرا کنند.
"""
from collections import Counter

from common.textnorm import tokenize
from search.taxonomy_labels import label_for

TOP_N = 50
MIN_GROUP = 3
MAX_SHARE = 0.7


def compute_ambiguity(ranked_items, query_tokens, item_tokens):
    """تشخیص قطعی و بدون فراخوانی مدل.

    ranked_items: موارد بازیابی‌شده، به ترتیب رتبه (بهترین اول).
    query_tokens: مجموعه‌ی توکن‌های پرس‌وجو (برای فیلتر AND).
    item_tokens: نگاشت item -> frozenset توکن‌های متن ایندکس‌شده‌ی آن.

    خروجی: (ambiguous: bool, groups: list[{"label","level1","count"}],
             top_candidates: list[item]) — گروه‌های حداقل ۳عضوی به‌عنوان
    چیپ برگردانده می‌شوند (یک مورد پرت تک‌عضوی گزینه‌ی معناداری برای
    انتخاب نیست)؛ top_candidates برای انتخاب یک نماینده به ازای هر گروه
    در پردازش گروهی اکسل به کار می‌رود.
    """
    if query_tokens:
        candidates = [item for item in ranked_items if query_tokens <= item_tokens.get(item, frozenset())]
    else:
        candidates = ranked_items

    top = candidates[:TOP_N]
    if not top:
        return False, [], []

    counts = Counter(item.level1 for item in top)
    ordered = counts.most_common()
    largest = ordered[0][1]
    qualifying = [(level1, c) for level1, c in ordered if c >= MIN_GROUP]

    ambiguous = len(qualifying) >= 2 and largest < MAX_SHARE * len(top)
    groups = [
        {"label": label_for(level1), "level1": level1, "count": c}
        for level1, c in qualifying
    ]
    return ambiguous, groups, top


def representative_per_group(top_candidates, groups):
    """اولین (بالاترین‌رتبه) نماینده‌ی هر گروه واجد شرایط، برای «گزینه‌های دیگر»."""
    wanted = {g["level1"] for g in groups}
    reps = {}
    for item in top_candidates:
        if item.level1 in wanted and item.level1 not in reps:
            reps[item.level1] = item
        if len(reps) == len(wanted):
            break
    return reps


def resolve_ambiguous_group_by_context(top_candidates, groups, context_tokens):
    """وقتی compute_ambiguity ابهام واقعی بین چند گروه را تشخیص داده (مثلاً
    «شیر» بین خوراکی/شیرآلات صنعتی)، اگر زمینه دقیقاً با نام یکی از آن گروه‌ها
    هم‌پوشانی توکن داشته باشد و با هیچ گروه دیگری هم‌پوشانی نداشته باشد، همان
    گروه را قطعی می‌کند و بالاترین‌رتبه عضوش را برمی‌گرداند.

    این جدا از apply_context_boost است: آن تابع برای «جواب قاطع اما بی‌ربط»
    است (وقتی اصلاً بین چند گروه واقعی پخش نشده، فقط رتبه‌ی اول جواب اشتباه
    یک دسته‌ی دیگر است)؛ این تابع برای وقتی است که واقعاً چند گروه بزرگ و
    رقیب داریم (مثلاً هم شیرآلات صنعتی هم خوراکی هرکدام ده‌ها عضو) — boost
    نرم به‌تنهایی تعداد اعضای هر گروه را عوض نمی‌کند، پس ابهام هنوز تشخیص
    داده می‌شود؛ این تابع دقیقاً همان ابهام تشخیص‌داده‌شده را با زمینه رفع
    می‌کند.

    عمداً روی خودِ نام گروه (level1) مقایسه می‌شود، نه متن غنی‌شده‌ی تک‌تک
    اعضا: متن غنی‌شده گاهی کلمه‌ای عمومی و نامرتبط را به یک کالای دیگر
    می‌چسباند (مثلاً «پزشکی» در uses یک ادویه‌ی سنتی) که می‌توانست چند گروه
    را به‌غلط «پشتیبانی‌شده» نشان دهد.

    اگر بیش از یک گروه یا هیچ گروهی پشتیبانی نشود، None (ابهام باقی می‌ماند
    — رفع قطعی فقط وقتی مجاز است که سیگنال یکتا باشد)."""
    if not context_tokens:
        return None

    supported_levels = [
        g["level1"] for g in groups
        if set(tokenize(g["level1"])) & context_tokens
    ]
    if len(supported_levels) != 1:
        return None

    chosen = supported_levels[0]
    for item in top_candidates:
        if item.level1 == chosen:
            return item
    return None


CONTEXT_BOOST = 1.03  # ضریب افزایش امتیاز کاندیدهایی که level1شان با زمینه هم‌پوشانی دارد


def apply_context_boost(raw_results, context_tokens, boost=CONTEXT_BOOST):
    """امتیاز هر کاندید که level1اش با یک زمینه‌ی جانبی (context_tokens —
    «دسته کالا»/«توضیحات بیشتر» در جستجوی گروهی اکسل) هم‌پوشانی توکن دارد را
    boost می‌کند و دوباره مرتب می‌کند؛ raw_results همان [(item, score), ...]
    بازگشتی از LexicalIndex.search/HybridIndex.search_split است.

    این یک BOOST نرم است، نه فیلتر سخت — تفاوت مهمی که نسخه‌ی قبلی
    (resolve_with_context، حالا حذف شده) نداشت: آن نسخه هر وقت زمینه با یک
    دسته هم‌پوشانی داشت، بی‌قیدوشرط به آن دسته می‌پرید، حتی وقتی کاندید
    واقعی رتبه‌ی اول به آن دسته هیچ ربطی نداشت (مثلاً «روتختی» یک‌بار به
    غلط «روتختی بیمارستانی» شد فقط چون توضیحات فایل «...پزشکی» هم داشت).
    اینجا فقط امتیاز کمی بالا می‌رود؛ اگر بهترین کاندید هم‌پوشان خیلی
    ضعیف‌تر از بهترین کاندید هم‌پوشان‌نشده باشد (فاصله‌ی امتیاز از قابلیت
    boost فراتر باشد)، رتبه‌ی اول عوض نمی‌شود — یعنی زمینه فقط تساوی‌های
    نزدیک را می‌شکند، جواب‌های قاطع را زیر و رو نمی‌کند.

    خروجی همان ساختار raw_results است (لیست [(item, score)]، مرتب‌شده بر
    اساس امتیاز boost‌شده)، پس بلافاصله جای‌گزین raw_results در ادامه‌ی
    مسیر (ranked_items، compute_ambiguity، انتخاب رتبه‌ی اول) می‌شود."""
    if not context_tokens:
        return raw_results

    boosted = [
        (item, score * boost if set(tokenize(item.level1)) & context_tokens else score)
        for item, score in raw_results
    ]
    boosted.sort(key=lambda pair: pair[1], reverse=True)
    return boosted
