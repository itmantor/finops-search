"""
ایندکس معنایی (Embedding) روی شرح‌های یکتای کاتالوگ
------------------------------------
بردارهای از پیش محاسبه‌شده در checkpoints/catalog_embeddings.jsonl خوانده
می‌شوند (به‌وسیله‌ی pipeline/embed_catalog.py ساخته شده‌اند)، L2-نرمال
می‌شوند و در یک IndexFlatIP فیس ذخیره می‌شوند. Embedding پرس‌وجو در زمان
جستجو و با فراخوانی API انجام می‌شود؛ نتیجه در حافظه کش می‌شود تا اجراهای
تکراری محک هزینه‌ی دوباره نداشته باشند.

برای سرور تولید، به‌جای پارس‌کردن jsonl خام (که برای نسخه‌ی غنی‌شده حدود
۷۴۲ مگابایت است) از SemanticIndex.from_prebuilt استفاده کنید که یک ایندکس
FAISS و آرایه‌ی numpy از پیش‌ساخته را می‌خواند (نگاه کنید به
pipeline/build_search_index.py).
"""
import hashlib
import json

import faiss
import numpy as np

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR, EMBED_MODEL, OPENAI_API_KEY
from common.utils import with_retry

EMBEDDINGS_PATH = CHECKPOINT_DIR / "catalog_embeddings.jsonl"


def _desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def _load_embeddings(path):
    by_hash = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_hash[rec["desc_hash"]] = rec["embedding"]
    return by_hash


class SemanticIndex:
    def __init__(self, items=None, embeddings_path=None):
        self.items = items if items is not None else load_catalog()
        embeddings_path = embeddings_path or EMBEDDINGS_PATH
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"فایل بردارها پیدا نشد: {embeddings_path}\n"
                "ابتدا `python pipeline/embed_catalog.py` را اجرا کنید."
            )

        by_hash = _load_embeddings(embeddings_path)
        vectors = []
        matched_items = []
        for item in self.items:
            vec = by_hash.get(_desc_hash(item.description))
            if vec is not None:
                vectors.append(vec)
                matched_items.append(item)

        self.items = matched_items
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self._matrix = matrix
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

        self._client = None
        self._query_cache = {}

    @classmethod
    def from_prebuilt(cls, index_path, positions_path, items=None):
        """بارگذاری سریع از ایندکس FAISS و آرایه‌ی موقعیت‌های از پیش‌ساخته
        (pipeline/build_search_index.py)، بدون پارس‌کردن jsonl خام بردارها."""
        obj = cls.__new__(cls)
        all_items = items if items is not None else load_catalog()
        positions = np.load(positions_path)
        obj.items = [all_items[i] for i in positions]
        obj._matrix = None
        obj._index = faiss.read_index(str(index_path))
        obj._client = None
        obj._query_cache = {}
        return obj

    def _embed_query(self, query):
        if query in self._query_cache:
            return self._query_cache[query]

        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)

        def call():
            return self._client.embeddings.create(model=EMBED_MODEL, input=[query])

        resp = with_retry(call)
        vec = np.asarray([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vec)
        self._query_cache[query] = vec
        return vec

    def search(self, query, k=40):
        """جستجو و بازگرداندن k مورد برتر به‌صورت جفت (item, score)."""
        vec = self._embed_query(query)
        scores, idx = self._index.search(vec, k)
        results = []
        for i, score in zip(idx[0], scores[0]):
            if i == -1:
                continue
            results.append((self.items[i], float(score)))
        return results
