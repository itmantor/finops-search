#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۲ — Embedding کاتالوگ شرح‌های یکتا
------------------------------------
هر شرح یکتای کاتالوگ (common.catalog) را به بردار معنایی تبدیل می‌کند.
قابلیت Resume دارد: هر رکورد با هش شرح کلید می‌خورد و در صورت قطع شدن،
با اجرای دوباره‌ی همین اسکریپت فقط شرح‌های باقی‌مانده پردازش می‌شوند.

خروجی: checkpoints/catalog_embeddings.jsonl
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR, EMBED_MODEL, OPENAI_API_KEY
from common.utils import JsonlCheckpoint, batched, with_retry

BATCH_SIZE = 100
OUT_PATH = CHECKPOINT_DIR / "catalog_embeddings.jsonl"


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def main():
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    items = load_catalog()
    print(f"📦 {len(items):,} شرح یکتا از کاتالوگ خوانده شد.")

    ckpt = JsonlCheckpoint(OUT_PATH, key_field="desc_hash")
    remaining = [item for item in items if not ckpt.is_done(desc_hash(item.description))]
    already = len(items) - len(remaining)
    if already:
        print(f"⏭  {already:,} شرح قبلاً Embed شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} شرح باقی‌مانده برای Embedding.")

    if not remaining:
        print("✅ همه‌ی شرح‌ها از قبل Embed شده‌اند.")
        return

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, chunk in enumerate(batched(remaining, BATCH_SIZE), start=1):
        texts = [item.description for item in chunk]

        def call():
            return client.embeddings.create(model=EMBED_MODEL, input=texts)

        resp = with_retry(call)
        for item, data in zip(chunk, resp.data):
            ckpt.append({
                "desc_hash": desc_hash(item.description),
                "description": item.description,
                "embedding": data.embedding,
            })

        done = min(bi * BATCH_SIZE, len(remaining))
        print(f"  ✓ دسته {bi}/{total_batches} — {done:,}/{len(remaining):,}")

    print(f"✅ Embedding کاتالوگ کامل شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
