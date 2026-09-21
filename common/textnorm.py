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


def _bigrams(tokens):
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def tokenize_with_bigrams(text, drop_stopwords=True):
    """برای متن ایندکس (سمت سند): unigram + bigram معمولی، بدون سرکوب —
    فقط search/lexical.py هنگام ساخت ایندکس از این استفاده می‌کند، نه
    search/ambiguity.py (که عمداً از tokenize ساده استفاده می‌کند).

    دلیل وجود بایگرام: ترکیب‌های دوکلمه‌ای فارسی مثل «ضد آفتاب» یا
    «هیالورونیک اسید» با توکنایز تک‌کلمه‌ای تکه‌تکه می‌شوند و مفهوم ترکیبی
    هرگز به‌عنوان یک واحد قابل جستجو نیست. با benchmark.json اندازه‌گیری شد:
    هیچ افتی در recall@40 یا recall@5 ایجاد نمی‌کند، MRR بهتر می‌شود.

    چرا اینجا سرکوب نفی نمی‌شود (برخلاف tokenize_query_with_bigrams): یک
    سند ممکن است خودش عبارت نفی را داشته باشد (مثلاً «تونر ضد جوش») و در
    عین حال با توکن تنهای غیرنفی‌شده‌ی همان کلمه هم باید قابل‌یافتن بماند
    (پرس‌وجوی «... جوش دار» که «جوش» در آن نفی نشده، یعنی پوست جوش‌دار، نه
    ضدجوش). اگر سرکوب این‌جا هم اجرا شود، سند توکن تنهای «جوش» را از دست
    می‌دهد و دیگر با چنین پرس‌وجویی هم‌پوشانی ندارد — دقیقاً همین رگرسیون
    با اندازه‌گیری روی benchmark.json پیدا شد (recall@40 حالت ترکیبی
    ۰.۹۴۱→۰.۸۸۲). راه‌حل: سند هر دو شکل را نگه می‌دارد (جوش و ضد_جوش)؛
    فقط سمت پرس‌وجو سرکوب می‌کند (پایین را ببینید)."""
    tokens = tokenize(text, drop_stopwords=drop_stopwords)
    return tokens + _bigrams(tokens)


def tokenize_query_with_bigrams(text, drop_stopwords=True):
    """برای پرس‌وجو (سمت کاربر): unigram + bigram، به‌علاوه‌ی سرکوب توکن
    تنهای بعد از پیشوند نفی (ضد/بدون/غیر) — فقط search/lexical.py هنگام
    جستجو از این استفاده می‌کند.

    وقتی کاربر «ضد آفتاب» می‌نویسد، دقیقاً یعنی نقیض «آفتاب»، نه یک
    زیرمجموعه‌اش؛ اگر توکن تنهای «افتاب» هم در پرس‌وجو بماند، با هر سندی که
    فقط کلمه‌ی «افتاب» را دارد (مثلاً «آفتاب‌پرست موجود زنده»، یک حیوان) هم
    هم‌پوشانی پیدا می‌کند — دقیقاً برعکسِ معنای مورد نظر کاربر. پس اینجا،
    فقط سمت پرس‌وجو، توکن تنهای اسم بعد از نفی سرکوب می‌شود و فقط واحد
    ترکیبی («ضد_افتاب») باقی می‌ماند.

    نامتقارن با tokenize_with_bigrams (سمت سند) عمدی است: سند «جوش» و
    «ضد_جوش» هر دو را نگه می‌دارد، پرس‌وجوی «ضد آفتاب» فقط «ضد_افتاب» را
    می‌خواهد. این‌طور هم «ضد آفتاب» دیگر به آفتاب‌پرست نمی‌رسد، هم «... جوش
    دار» هنوز به سند «تونر ضد جوش» (که توکن تنهای جوش را نگه داشته) می‌رسد."""
    tokens = tokenize(text, drop_stopwords=drop_stopwords)
    bigrams = _bigrams(tokens)

    suppressed = {
        i + 1 for i, t in enumerate(tokens)
        if t in NEGATION_PREFIXES and i + 1 < len(tokens)
    }
    unigrams = [t for i, t in enumerate(tokens) if i not in suppressed]
    return unigrams + bigrams
