# -*- coding: utf-8 -*-
"""
راه‌اندازی مشترک کل پشته‌ی جستجو (کاتالوگ + ایندکس‌ها + فهم پرس‌وجو)
------------------------------------
هم سرور وب (server/app.py) و هم پردازش گروهی اکسل (pipeline/bulk_lookup.py،
که در یک پردازه‌ی جدا اجرا می‌شود و پشته‌ی preload-شده‌ی gunicorn را در
اختیار ندارد) از این تابع استفاده می‌کنند تا هر دو مسیر دقیقاً همان
ایندکس‌ها و همان غنی‌سازی را به کار ببرند.
"""
from collections import namedtuple

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR
from common.textnorm import tokenize
from search.enrichment import load_enrichment, make_lexical_text_fn
from search.hybrid import HybridIndex
from search.lexical import LexicalIndex
from search.semantic import SemanticIndex
from search.understand import QueryUnderstander

SEMANTIC_INDEX_PATH = CHECKPOINT_DIR / "catalog_semantic.index"
SEMANTIC_POSITIONS_PATH = CHECKPOINT_DIR / "catalog_semantic_positions.npy"

Engine = namedtuple("Engine", [
    "items", "enrichment", "lexical_text_fn", "lexical",
    "semantic", "hybrid", "understander", "item_tokens",
])


def load_engine():
    items = load_catalog()
    # پیکربندی انتخاب‌شده طبق بنچمارک: غنی‌سازی با examples (پاس ۱، ۱۰۰٪
    # کامل — پاس ۲ رهاشده و فقط بخشی پوشش دارد، پس raw استفاده می‌شود).
    enrichment = load_enrichment(examples_source="raw")
    lexical_text_fn = make_lexical_text_fn(enrichment, include_examples=True)
    lexical = LexicalIndex(items=items, text_fn=lexical_text_fn)

    # فقط برای فیلتر AND در تشخیص ابهام (search/ambiguity.py)، نه رتبه‌بندی BM25.
    item_tokens = {item: frozenset(tokenize(lexical_text_fn(item))) for item in items}

    semantic = None
    hybrid = None
    if SEMANTIC_INDEX_PATH.exists() and SEMANTIC_POSITIONS_PATH.exists():
        semantic = SemanticIndex.from_prebuilt(SEMANTIC_INDEX_PATH, SEMANTIC_POSITIONS_PATH, items=items)
        hybrid = HybridIndex(lexical, semantic)

    understander = QueryUnderstander()

    return Engine(items, enrichment, lexical_text_fn, lexical, semantic, hybrid, understander, item_tokens)
