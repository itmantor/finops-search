#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۷ — ساخت ایندکس FAISS برای جستجوی معنایی سریع
------------------------------------
با مقیاس فعلی داده (~۴۷ هزار رکورد)، IndexFlatIP (جستجوی دقیق، بدون تقریب)
هم به‌اندازه کافی سریع است و از پیچیدگی ایندکس‌های تقریبی (IVF و ...) بی‌نیاز می‌کند.

ورودی:  checkpoints/embeddings_final.jsonl ، outputs/categorized_goods.xlsx
خروجی:  outputs/faiss.index ، outputs/metadata.pkl
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import openpyxl

from common.config import CHECKPOINT_DIR, OUTPUT_DIR
from common.utils import JsonlCheckpoint

IN_PATH = CHECKPOINT_DIR / "embeddings_final.jsonl"
CAT_PATH = OUTPUT_DIR / "categorized_goods.xlsx"
INDEX_OUT = OUTPUT_DIR / "faiss.index"
META_OUT = OUTPUT_DIR / "metadata.pkl"


def load_categories():
    cat = {}
    if CAT_PATH.exists():
        wb = openpyxl.load_workbook(CAT_PATH, read_only=True, data_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            sid, desc, typ, l1, l2, l3, kw = row
            cat[str(sid)] = {"type": typ or "", "level1": l1 or "", "level2": l2 or "", "level3": l3 or ""}
    return cat


def main():
    ckpt = JsonlCheckpoint(IN_PATH, key_field="id")
    records = ckpt.load_all()
    if not records:
        print("❌ ابتدا مرحله ۶ (Embedding نهایی) را اجرا کنید.")
        return

    categories = load_categories()

    ids = [r["id"] for r in records]
    descs = [r["desc"] for r in records]
    types = [categories.get(i, {}).get("type", "") for i in ids]
    level1 = [categories.get(i, {}).get("level1", "") for i in ids]

    X = np.array([r["embedding"] for r in records], dtype="float32")
    faiss.normalize_L2(X)  # برای استفاده از Inner Product به‌جای شباهت کسینوسی

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_OUT))

    with open(META_OUT, "wb") as f:
        pickle.dump({"ids": ids, "descs": descs, "types": types, "level1": level1}, f)

    print(f"✅ ایندکس FAISS با {len(ids):,} رکورد ساخته شد → {INDEX_OUT}")


if __name__ == "__main__":
    main()
