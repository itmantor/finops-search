"""
ترکیب کانال واژگانی و معنایی با Reciprocal Rank Fusion (RRF)
------------------------------------
امتیازهای BM25 و شباهت کسینوسی روی مقیاس‌های متفاوتی هستند و قابل مقایسه
مستقیم نیستند، به همین دلیل به‌جای ترکیب وزن‌دار امتیازها (مثل فرمول قدیمی
0.7*semantic + 0.3*keyword) از RRF استفاده می‌شود که فقط به رتبه‌ی هر
کانال وابسته است:

    score(item) = sum over channels of  1 / (RRF_K + rank_in_that_channel)
"""
from collections import defaultdict

from search.lexical import LexicalIndex
from search.semantic import SemanticIndex

RRF_K = 60
TOP_PER_CHANNEL = 100


class HybridIndex:
    def __init__(self, lexical=None, semantic=None):
        self.lexical = lexical if lexical is not None else LexicalIndex()
        self.semantic = semantic if semantic is not None else SemanticIndex()

    def search(self, query, k=40):
        """جستجو در هر دو کانال (با همان پرس‌وجو) و ترکیب نتایج با RRF."""
        return self.search_split(query, query, k=k)

    def search_split(self, lexical_query, semantic_query, k=40):
        """جستجو با پرس‌وجوی متفاوت برای هر کانال (مثلاً بعد از فهم پرس‌وجو) و ترکیب با RRF."""
        channel_results = [
            self.lexical.search(lexical_query, k=TOP_PER_CHANNEL),
            self.semantic.search(semantic_query, k=TOP_PER_CHANNEL),
        ]

        scores = defaultdict(float)
        items_by_desc = {}
        for results in channel_results:
            for rank, (item, _score) in enumerate(results, start=1):
                scores[item.description] += 1.0 / (RRF_K + rank)
                items_by_desc[item.description] = item

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(items_by_desc[desc], score) for desc, score in ranked[:k]]
