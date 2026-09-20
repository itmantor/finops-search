#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۴ — تخصیص دسته‌بندی به هر کالا بر اساس خوشه‌اش
------------------------------------
ورودی:  checkpoints/clusters.pkl ، outputs/taxonomy.json
خروجی:  outputs/categorized_goods.xlsx
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl

from common.config import CHECKPOINT_DIR, OUTPUT_DIR

CLUSTERS_PATH = CHECKPOINT_DIR / "clusters.pkl"
TAXONOMY_JSON = OUTPUT_DIR / "taxonomy.json"
XLSX_OUT = OUTPUT_DIR / "categorized_goods.xlsx"


def main():
    if not CLUSTERS_PATH.exists():
        print("❌ ابتدا مرحله ۲ (خوشه‌بندی) را اجرا کنید.")
        return
    if not TAXONOMY_JSON.exists():
        print("❌ ابتدا مرحله ۳ (Taxonomy) را اجرا کنید.")
        return

    with open(CLUSTERS_PATH, "rb") as f:
        data = pickle.load(f)
    with open(TAXONOMY_JSON, encoding="utf-8") as f:
        taxonomy_map = json.load(f)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CategorizedGoods"
    ws.append(["StuffCod", "Description", "Type", "Level1", "Level2", "Level3", "Keywords"])

    for sid, desc, typ, label in zip(data["ids"], data["descs"], data["types"], data["labels"]):
        tax = taxonomy_map.get(str(label), {})
        ws.append([sid, desc, typ, tax.get("level1", ""), tax.get("level2", ""), tax.get("level3", ""),
                   ", ".join(tax.get("keywords", []))])

    wb.save(XLSX_OUT)
    print(f"✅ دسته‌بندی {len(data['ids']):,} کالا انجام شد → {XLSX_OUT}")


if __name__ == "__main__":
    main()
