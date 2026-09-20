"""
ایندکس واژگانی BM25 روی شرح‌های یکتای کاتالوگ
------------------------------------
ایندکس در حافظه و در زمان استارت ساخته می‌شود (چند ثانیه طول می‌کشد)،
بنابراین نیازی به ذخیره‌ی آن روی دیسک نیست.
"""
from rank_bm25 import BM25Okapi

from common.catalog import load_catalog
from common.textnorm import tokenize_with_bigrams

K1 = 1.5
B = 0.75


class LexicalIndex:
    def __init__(self, items=None, k1=K1, b=B, text_fn=None):
        """text_fn(item) -> str متن ایندکس هر مورد را می‌سازد؛ پیش‌فرض همان item.description
        قبلی است (بدون تغییر رفتار). برای متن غنی‌شده، یک text_fn سفارشی بده
        (نگاه کنید به search/enrichment.py).

        توکنایز شامل بایگرام‌های توکن‌های مجاور هم هست (common/textnorm.
        tokenize_with_bigrams) تا ترکیب‌های دوکلمه‌ای مثل «ضد آفتاب» به‌عنوان
        یک واحد هم قابل تطبیق باشند، نه فقط تکه‌های جدا («ضد»، «افتاب»)."""
        self.items = items if items is not None else load_catalog()
        text_fn = text_fn or (lambda item: item.description)
        tokenized = [tokenize_with_bigrams(text_fn(item)) for item in self.items]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query, k=40):
        """جستجو و بازگرداندن k مورد برتر به‌صورت جفت (item, score)."""
        tokens = tokenize_with_bigrams(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:k]
        return [(self.items[i], float(scores[i])) for i in top_idx]
