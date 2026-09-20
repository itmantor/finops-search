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


CONTEXT_TOP_K = 50
CONTEXT_MIN_SUPPORT = 3


def resolve_with_context(ranked_items, context_tokens, top_k=CONTEXT_TOP_K, min_support=CONTEXT_MIN_SUPPORT):
    """با یک زمینه‌ی جانبی (context_tokens — مثلاً «دسته کالا»/«توضیحات بیشتر»
    در جستجوی گروهی اکسل)، رتبه‌ی اول بازیابی را تأیید یا اصلاح می‌کند —
    مستقل از این‌که compute_ambiguity ابهام را تشخیص داده یا نه.

    چرا مستقل از تشخیص ابهام: تشخیص ابهام روی کاندیدهای فیلترشده‌ی AND کار
    می‌کند (فقط مواردی که همه‌ی توکن‌های پرس‌وجو را دارند)، اما برای عبارات
    ترکیبی که توکنایز آن‌ها را تکه‌تکه می‌کند (مثلاً «ژل ضد جوش»)، این فیلتر
    گاهی آن‌قدر سخت‌گیر می‌شود که فقط ۰ یا ۱ کاندید واقعی از دسته‌ی درست باقی
    می‌ماند — خیلی کم‌تر از آستانه‌ی گروه‌بندی (۳عضوی) — در حالی که رتبه‌ی
    اولِ خامِ نامرتبط (مثلاً «ضد جوش وسایل نقلیه») هیچ‌وقت ابهام را فعال
    نمی‌کند چون تنها بازیگر میدان است. این تابع مستقیماً روی رتبه‌بندی خام
    (بدون فیلتر AND) کار می‌کند تا این حالت را هم بگیرد.

    منطق:
    - اگر level1 رتبه‌ی اول با context_tokens هم‌پوشانی داشته باشد، زمینه
      همان را تأیید می‌کند — همان رتبه‌ی اول برگردانده می‌شود (حتی اگر
      compute_ambiguity ابهام را هم تشخیص داده باشد، دیگر نیازی به بررسی
      کاربر نیست: زمینه از قبل تأییدش کرده).
    - وگرنه، بین top_k نتیجه‌ی برتر خام، اگر حداقل min_support مورد level1ی
      داشته باشند که با زمینه هم‌پوشانی دارد، بالاترین‌رتبه‌ی آن‌ها به‌جای
      رتبه‌ی اول انتخاب می‌شود — یعنی زمینه فقط وقتی رتبه‌ی اول را عوض
      می‌کند که پشتوانه‌ی واقعی و چندتایی داشته باشد، نه یک مورد پرت تک‌تایی
      (که می‌تواند تصادفی/نامرتبط باشد؛ مثلاً یک وسیله‌ی پزشکی بی‌ربط که فقط
      به‌خاطر کلمه‌ای مشترک در بین ۵۰تای برتر یک پرس‌وجوی دارویی افتاده).
    - اگر هیچ‌کدام صدق نکند، None (زمینه نمی‌تواند با اطمینان کافی چیزی را
      تعیین کند — فراخوان باید به رفتار پیش‌فرض برگردد: اگر ابهام تشخیص داده
      شده «نیاز به بررسی»، وگرنه همان رتبه‌ی اول خام)."""
    if not ranked_items or not context_tokens:
        return None

    top1 = ranked_items[0]
    if set(tokenize(top1.level1)) & context_tokens:
        return top1

    matching = [item for item in ranked_items[:top_k] if set(tokenize(item.level1)) & context_tokens]
    if len(matching) >= min_support:
        return matching[0]
    return None
