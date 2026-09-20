#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۳ — تولید Taxonomy برای هر خوشه با کمک GPT
------------------------------------
برای هر خوشه، نمونه‌ای از شرح کالاها به GPT فرستاده می‌شود و یک دسته‌بندی
سه‌سطحی + کلمات کلیدی برمی‌گردد. قابلیت Resume دارد (بر اساس ClusterID).

ورودی:  checkpoints/clusters.pkl
خروجی:  outputs/taxonomy.xlsx ، outputs/taxonomy.json
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl
from openai import OpenAI

from common.config import OPENAI_API_KEY, CHAT_MODEL, CHECKPOINT_DIR, OUTPUT_DIR
from common.utils import JsonlCheckpoint, with_retry

IN_PATH = CHECKPOINT_DIR / "clusters.pkl"
CKPT_PATH = CHECKPOINT_DIR / "taxonomy.jsonl"
XLSX_OUT = OUTPUT_DIR / "taxonomy.xlsx"
JSON_OUT = OUTPUT_DIR / "taxonomy.json"

SAMPLE_SIZE = 15

PROMPT_TEMPLATE = """نمونه‌ای از شرح کالاهای یک دسته را می‌بینی. بر اساس مفهوم و کاربرد این کالاها،
یک دسته‌بندی سه‌سطحی (از کلی به جزئی) و ۴ تا ۶ کلمه‌ی کلیدی مرتبط که در جستجو مفید هستند ارائه بده.

فقط یک شیء JSON با دقیقاً این ساختار برگردان، بدون هیچ توضیح یا متن اضافه:
{{"level1": "...", "level2": "...", "level3": "...", "keywords": ["...", "..."]}}

نمونه کالاها:
{samples}
"""


def main():
    import pickle
    if not IN_PATH.exists():
        print("❌ ابتدا مرحله ۲ (خوشه‌بندی) را اجرا کنید.")
        return
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    with open(IN_PATH, "rb") as f:
        data = pickle.load(f)

    clusters = defaultdict(list)
    for desc, label in zip(data["descs"], data["labels"]):
        clusters[label].append(desc)

    client = OpenAI(api_key=OPENAI_API_KEY)
    ckpt = JsonlCheckpoint(CKPT_PATH, key_field="cluster_id")

    cluster_ids = sorted(clusters.keys())
    remaining = [c for c in cluster_ids if not ckpt.is_done(str(c))]
    already = len(cluster_ids) - len(remaining)
    if already:
        print(f"⏭  {already:,} خوشه قبلاً پردازش شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} خوشه از {len(cluster_ids):,} باقی‌مانده.")

    for idx, cid in enumerate(remaining, start=1):
        if cid == -1:
            taxonomy = {"level1": "دسته‌بندی‌نشده", "level2": "", "level3": "", "keywords": []}
        else:
            sample = random.sample(clusters[cid], min(SAMPLE_SIZE, len(clusters[cid])))
            prompt = PROMPT_TEMPLATE.format(samples="\n".join(sample))

            def call():
                return client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )

            resp = with_retry(call)
            try:
                taxonomy = json.loads(resp.choices[0].message.content)
            except Exception:
                taxonomy = {"level1": "نامشخص", "level2": "", "level3": "", "keywords": []}

        ckpt.append({"cluster_id": str(cid), "count": len(clusters[cid]), **taxonomy})
        print(f"  ✓ خوشه {cid} ({idx}/{len(remaining)}) → {taxonomy.get('level1', '')}")

    # ساخت خروجی نهایی از روی همه‌ی رکوردهای checkpoint (شامل اجراهای قبلی)
    all_records = ckpt.load_all()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Taxonomy"
    ws.append(["Level1", "Level2", "Level3", "Keywords", "Count", "ClusterID"])
    taxonomy_map = {}
    for rec in all_records:
        ws.append([rec.get("level1", ""), rec.get("level2", ""), rec.get("level3", ""),
                   ", ".join(rec.get("keywords", [])), rec.get("count", 0), rec.get("cluster_id")])
        taxonomy_map[rec["cluster_id"]] = rec
    wb.save(XLSX_OUT)

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(taxonomy_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Taxonomy ساخته شد → {XLSX_OUT}")


if __name__ == "__main__":
    main()
