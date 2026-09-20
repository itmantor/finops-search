#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۴ (ادامه) — Embedding معنایی غنی‌شده
------------------------------------
همان کاتالوگ را دوباره Embed می‌کند، اما این‌بار متن ورودی «شرح + canonical»
است (نه فقط شرح خام) — طبق پیشنهاد مرحله ۴: متن معنایی باید یک عبارت
یکپارچه بماند، نه رقیق شود با synonyms/examples/uses.

هرگز checkpoints/catalog_embeddings.jsonl (بردارهای اصلی، فقط شرح) را
دست نمی‌زند؛ خروجی این اسکریپت در فایل جداگانه‌ای ذخیره می‌شود تا بشود
هر دو را در بنچمارک مقایسه کرد.

قابلیت Resume دارد: هر رکورد با هش شرح اصلی کلید می‌خورد (همان کلیدی که
SemanticIndex برای یافتن بردار هر item.description جست‌وجو می‌کند).

خروجی: checkpoints/catalog_embeddings_enriched.jsonl
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR, EMBED_MODEL, OPENAI_API_KEY
from common.utils import JsonlCheckpoint, batched, with_retry
from search.enrichment import load_enrichment, semantic_text, desc_hash as enrichment_desc_hash

BATCH_SIZE = 100
OUT_PATH = CHECKPOINT_DIR / "catalog_embeddings_enriched.jsonl"


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def main():
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    items = load_catalog()
    enrichment = load_enrichment()
    print(f"📦 {len(items):,} شرح یکتا از کاتالوگ خوانده شد.")
    print(f"🧬 {len(enrichment):,} شرح دارای غنی‌سازی است.")

    ckpt = JsonlCheckpoint(OUT_PATH, key_field="desc_hash")
    remaining = [item for item in items if not ckpt.is_done(desc_hash(item.description))]
    already = len(items) - len(remaining)
    if already:
        print(f"⏭  {already:,} شرح قبلاً Embed شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} شرح باقی‌مانده برای Embedding غنی‌شده.")

    if not remaining:
        print("✅ همه‌ی شرح‌ها از قبل Embed شده‌اند.")
        return

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, chunk in enumerate(batched(remaining, BATCH_SIZE), start=1):
        texts = []
        for item in chunk:
            rec = enrichment.get(enrichment_desc_hash(item.description), {})
            texts.append(semantic_text(item, rec))

        def call():
            return client.embeddings.create(model=EMBED_MODEL, input=texts)

        resp = with_retry(call)
        for item, text, data in zip(chunk, texts, resp.data):
            ckpt.append({
                "desc_hash": desc_hash(item.description),
                "description": item.description,
                "embedding_text": text,
                "embedding": data.embedding,
            })

        done = min(bi * BATCH_SIZE, len(remaining))
        print(f"  ✓ دسته {bi}/{total_batches} — {done:,}/{len(remaining):,}")

    print(f"✅ Embedding غنی‌شده کامل شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
