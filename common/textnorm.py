"""
نرمال‌سازی و توکنایز متن فارسی
------------------------------------
ماژول کوچک و خالص (بدون وابستگی خارجی) که عمداً ساده نگه داشته شده تا
تنظیم کیفیت توکنایز در آینده راحت باشد.
"""
import re

_CHAR_MAP = {
    "ي": "ی",
    "ك": "ک",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ۀ": "ه",
    "‌": " ",  # نیم‌فاصله → فاصله
}
_CHAR_MAP_PATTERN = re.compile("|".join(re.escape(c) for c in _CHAR_MAP))

_DIGITS_PATTERN = re.compile(r"[0-9۰-۹٠-٩]+")
_PUNCT_PATTERN = re.compile(r"[^\w\s]|_", flags=re.UNICODE)

STOPWORDS = {
    "و", "در", "از", "به", "با", "بر", "های", "ها", "برای",
    "یک", "این", "ان", "که", "را",
}


def normalize(text):
    """یکسان‌سازی حروف عربی/فارسی، حذف نیم‌فاصله، اعداد و علائم نگارشی."""
    text = _CHAR_MAP_PATTERN.sub(lambda m: _CHAR_MAP[m.group(0)], text)
    text = _DIGITS_PATTERN.sub(" ", text)
    text = _PUNCT_PATTERN.sub(" ", text)
    return text


def tokenize(text, drop_stopwords=True):
    """نرمال‌سازی و شکستن متن به توکن‌ها؛ توکن‌های تک‌حرفی حذف می‌شوند."""
    text = normalize(text)
    tokens = [t for t in text.split() if len(t) > 1]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


NEGATION_PREFIXES = {"ضد", "بدون", "غیر"}


def tokenize_with_bigrams(text, drop_stopwords=True):
    """همان tokenize، به‌علاوه‌ی بایگرام‌های توکن‌های مجاور («ضد»,«افتاب» هم
    توکن «ضد_افتاب» هم می‌سازد) — فقط برای ایندکس/جستجوی واژگانی BM25
    (search/lexical.py)، نه برای فیلتر AND ابهام یا هم‌پوشانی زمینه
    (search/ambiguity.py که عمداً از tokenize ساده استفاده می‌کند).

    دلیل: ترکیب‌های دوکلمه‌ای فارسی مثل «ضد آفتاب» یا «هیالورونیک اسید» با
    توکنایز تک‌کلمه‌ای تکه‌تکه می‌شوند و مفهوم ترکیبی هرگز به‌عنوان یک واحد
    قابل جستجو نیست. با benchmark.json اندازه‌گیری شد: هیچ افتی در recall@40
    یا recall@5 در دو حالت لغوی‌تنها و ترکیبی ایجاد نمی‌کند، و MRR در هر دو
    حالت بهتر می‌شود (لغوی‌تنها ۰.۵۴۷→۰.۶۰۵، ترکیبی ۰.۶۱۰→۰.۶۲۳ با
    recall@5 ترکیبی هم ۰.۶۴۷→۰.۷۶۵).

    نکته‌ی حیاتی درباره‌ی نفی (ضد/بدون/غیر): صرفاً افزودن بایگرام کافی نیست،
    چون توکن تنهای بعد از این پیشوندها هم‌چنان در فهرست باقی می‌ماند و با
    معنای وارونه‌اش تطبیق می‌خورد — «ضد آفتاب» با توکن تنهای «افتاب» به یک
    آفتاب‌پرست (موجود زنده) هم‌پوشانی پیدا می‌کرد، که دقیقاً برعکسِ معنای
    «ضد آفتاب» است. برای این پیشوندها، توکن تنهای اسم بعدی سرکوب می‌شود (فقط
    واحد ترکیبی «ضد_افتاب» باقی می‌ماند، نه «افتاب» به‌تنهایی) تا تطبیق
    وارونه رخ ندهد؛ همین تابع برای متن ایندکس و پرس‌وجو یکسان اجرا می‌شود،
    پس دو طرف هم‌سو می‌مانند."""
    tokens = tokenize(text, drop_stopwords=drop_stopwords)
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    suppressed = {
        i + 1 for i, t in enumerate(tokens)
        if t in NEGATION_PREFIXES and i + 1 < len(tokens)
    }
    unigrams = [t for i, t in enumerate(tokens) if i not in suppressed]
    return unigrams + bigrams
