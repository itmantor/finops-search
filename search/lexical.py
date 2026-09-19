"""
ایندکس واژگانی BM25 روی شرح‌های یکتای کاتالوگ
------------------------------------
ایندکس در حافظه و در زمان استارت ساخته می‌شود (چند ثانیه طول می‌کشد)،
بنابراین نیازی به ذخیره‌ی آن روی دیسک نیست.
"""
from rank_bm25 import BM25Okapi

from common.catalog import load_catalog
from common.textnorm import tokenize

K1 = 1.5
B = 0.75


class LexicalIndex:
    def __init__(self, items=None, k1=K1, b=B):
        self.items = items if items is not None else load_catalog()
        tokenized = [tokenize(item.description) for item in self.items]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query, k=40):
        """جستجو و بازگرداندن k مورد برتر به‌صورت جفت (item, score)."""
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:k]
        return [(self.items[i], float(scores[i])) for i in top_idx]
