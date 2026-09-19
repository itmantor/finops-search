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
