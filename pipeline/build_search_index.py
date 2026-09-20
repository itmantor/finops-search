#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ساخت ایندکس معنایی از پیش‌ساخته برای سرور
------------------------------------
checkpoints/catalog_embeddings_enriched.jsonl حدود ۷۴۲ مگابایت است. پارس‌کردن
آن در هر بار استارت سرور کند است و اگر gunicorn چند worker فورک کند، این کار
چند برابر هم تکرار می‌شود. این اسکریپت یک‌بار آن فایل را می‌خواند و دو خروجی
سبک و سریع‌بارگذاری روی دیسک می‌سازد:

  checkpoints/catalog_semantic.index      — ایندکس FAISS (IndexFlatIP، بردارها L2-نرمال‌شده)
  checkpoints/catalog_semantic_positions.npy — آرایه‌ی numpy از موقعیت هر بردار در خروجی
                                                common.catalog.load_catalog() (چون ممکن
                                                است بعضی شرح‌ها بردار نداشته باشند)

سرور (server/app.py) این دو فایل را می‌خواند، نه jsonl خام را.

هر وقت کاتالوگ یا Embedding غنی‌شده عوض شد، این اسکریپت را دوباره اجرا کنید:
    python pipeline/build_search_index.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR

EMBEDDINGS_PATH = CHECKPOINT_DIR / "catalog_embeddings_enriched.jsonl"
INDEX_PATH = CHECKPOINT_DIR / "catalog_semantic.index"
POSITIONS_PATH = CHECKPOINT_DIR / "catalog_semantic_positions.npy"


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def main():
    if not EMBEDDINGS_PATH.exists():
        print(f"❌ فایل بردارهای غنی‌شده پیدا نشد: {EMBEDDINGS_PATH}")
        print("   ابتدا pipeline/embed_enriched.py را اجرا کنید.")
        return

    items = load_catalog()
    print(f"📦 {len(items):,} شرح یکتا از کاتالوگ خوانده شد.")

    hash_to_pos = {desc_hash(item.description): pos for pos, item in enumerate(items)}

    positions = []
    vectors = []
    seen = set()
    print(f"📖 در حال خواندن {EMBEDDINGS_PATH.name} ...")
    with open(EMBEDDINGS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pos = hash_to_pos.get(rec["desc_hash"])
            if pos is None or pos in seen:
                continue
            seen.add(pos)
            positions.append(pos)
            vectors.append(rec["embedding"])

    if not vectors:
        print("❌ هیچ بردار منطبقی با کاتالوگ پیدا نشد.")
        return

    matrix = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(POSITIONS_PATH, np.asarray(positions, dtype="int64"))

    missing = len(items) - len(positions)
    print(f"✅ {len(positions):,}/{len(items):,} شرح دارای بردار — ایندکس ساخته شد.")
    if missing:
        print(f"⚠️  {missing:,} شرح بدون بردار غنی‌شده‌اند و در جستجوی معنایی نمی‌آیند.")
    print(f"   → {INDEX_PATH}")
    print(f"   → {POSITIONS_PATH}")


if __name__ == "__main__":
    main()
