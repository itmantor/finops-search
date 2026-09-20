#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۱ — Embedding اولیه
------------------------------------
شرح خام هر کالا (ستون Description) را به بردار معنایی تبدیل می‌کند.
قابلیت Resume دارد: در صورت قطع شدن، با اجرای دوباره‌ی همین اسکریپت
فقط رکوردهای باقی‌مانده پردازش می‌شوند.

خروجی: checkpoints/embeddings_initial.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from common.config import OPENAI_API_KEY, EMBED_MODEL, INPUT_FILE, CHECKPOINT_DIR
from common.utils import load_records, JsonlCheckpoint, batched, with_retry

BATCH_SIZE = 100
OUT_PATH = CHECKPOINT_DIR / "embeddings_initial.jsonl"


def main():
    if not INPUT_FILE:
        print("❌ متغیر INPUT_FILE در .env تنظیم نشده. مسیر فایل کالاها را در .env مشخص کنید.")
        return
    if not Path(INPUT_FILE).exists():
        print(f"❌ فایل پیدا نشد: {INPUT_FILE}")
        return
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    records = load_records(INPUT_FILE)
    print(f"📦 {len(records):,} رکورد از {INPUT_FILE} خوانده شد.")

    ckpt = JsonlCheckpoint(OUT_PATH, key_field="id")
    remaining = [r for r in records if not ckpt.is_done(r["id"])]
    already = len(records) - len(remaining)
    if already:
        print(f"⏭  {already:,} رکورد قبلاً پردازش شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} رکورد باقی‌مانده برای Embedding.")

    if not remaining:
        print("✅ همه رکوردها از قبل پردازش شده‌اند.")
        return

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, chunk in enumerate(batched(remaining, BATCH_SIZE), start=1):
        texts = [r["desc"] for r in chunk]

        def call():
            return client.embeddings.create(model=EMBED_MODEL, input=texts)

        resp = with_retry(call)
        for r, item in zip(chunk, resp.data):
            ckpt.append({
                "id": r["id"], "desc": r["desc"], "type": r.get("type", ""),
                "embedding": item.embedding,
            })

        done = min(bi * BATCH_SIZE, len(remaining))
        print(f"  ✓ دسته {bi}/{total_batches} — {done:,}/{len(remaining):,}")

    print(f"✅ Embedding اولیه کامل شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
