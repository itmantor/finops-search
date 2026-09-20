#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۶ — Embedding نهایی بر اساس متن غنی‌شده (Embedding_Text)
------------------------------------
به‌جای شرح خام، متن غنی‌شده (شامل دسته‌بندی و کلمات کلیدی) embed می‌شود
تا جستجوی معنایی دقیق‌تر شود. قابلیت Resume دارد.

ورودی:  outputs/embedding_data.xlsx
خروجی:  checkpoints/embeddings_final.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl
from openai import OpenAI

from common.config import OPENAI_API_KEY, EMBED_MODEL, CHECKPOINT_DIR, OUTPUT_DIR
from common.utils import JsonlCheckpoint, batched, with_retry

BATCH_SIZE = 100
IN_PATH = OUTPUT_DIR / "embedding_data.xlsx"
OUT_PATH = CHECKPOINT_DIR / "embeddings_final.jsonl"


def main():
    if not IN_PATH.exists():
        print("❌ ابتدا مرحله ۵ (ساخت Embedding_Text) را اجرا کنید.")
        return
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    wb = openpyxl.load_workbook(IN_PATH, read_only=True, data_only=True)
    ws = wb.active
    records = [
        {"id": str(r[0]), "desc": r[1], "text": r[2]}
        for r in ws.iter_rows(min_row=2, values_only=True) if r[0]
    ]
    print(f"📦 {len(records):,} رکورد خوانده شد.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    ckpt = JsonlCheckpoint(OUT_PATH, key_field="id")
    remaining = [r for r in records if not ckpt.is_done(r["id"])]
    already = len(records) - len(remaining)
    if already:
        print(f"⏭  {already:,} رکورد قبلاً پردازش شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} رکورد باقی‌مانده.")

    if not remaining:
        print("✅ همه رکوردها از قبل پردازش شده‌اند.")
        return

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, chunk in enumerate(batched(remaining, BATCH_SIZE), start=1):
        texts = [r["text"] for r in chunk]

        def call():
            return client.embeddings.create(model=EMBED_MODEL, input=texts)

        resp = with_retry(call)
        for r, item in zip(chunk, resp.data):
            ckpt.append({"id": r["id"], "desc": r["desc"], "embedding": item.embedding})

        done = min(bi * BATCH_SIZE, len(remaining))
        print(f"  ✓ دسته {bi}/{total_batches} — {done:,}/{len(remaining):,}")

    print(f"✅ Embedding نهایی کامل شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
