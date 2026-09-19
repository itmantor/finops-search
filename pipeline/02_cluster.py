#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۲ — خوشه‌بندی خودکار
------------------------------------
از UMAP برای کاهش بُعد و HDBSCAN برای خوشه‌بندی خودکار استفاده می‌کند
(به‌جای KMeans با تعداد ثابت خوشه، چون تعداد دسته‌های واقعی از قبل مشخص نیست).

نکته: این مرحله محاسباتی و محلی است (بدون فراخوانی API)، پس نسبتاً سریع
است و در صورت قطع شدن، کافی‌ست دوباره از ابتدا اجرا شود؛ نیازی به
Checkpoint افزایشی ندارد.

ورودی:  checkpoints/embeddings_initial.jsonl
خروجی:  checkpoints/clusters.pkl
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.config import CHECKPOINT_DIR
from common.utils import JsonlCheckpoint

IN_PATH = CHECKPOINT_DIR / "embeddings_initial.jsonl"
OUT_PATH = CHECKPOINT_DIR / "clusters.pkl"


def main():
    ckpt = JsonlCheckpoint(IN_PATH, key_field="id")
    records = ckpt.load_all()
    if not records:
        print("❌ ابتدا مرحله ۱ (Embedding اولیه) را اجرا کنید.")
        return

    print(f"📦 {len(records):,} بردار بارگذاری شد.")

    ids = [r["id"] for r in records]
    descs = [r["desc"] for r in records]
    types = [r.get("type", "") for r in records]
    X = np.array([r["embedding"] for r in records], dtype="float32")

    print("🔻 کاهش بُعد با UMAP (ممکن است چند دقیقه طول بکشد)...")
    import umap
    n_neighbors = min(15, max(2, len(X) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=30, metric="cosine", random_state=42)
    X_reduced = reducer.fit_transform(X)

    print("🧩 خوشه‌بندی با HDBSCAN...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"✅ {n_clusters:,} خوشه ساخته شد. {n_noise:,} رکورد بدون خوشه (نویز) باقی ماند "
          f"و بعداً به‌عنوان «دسته‌بندی‌نشده» علامت‌گذاری می‌شوند.")

    result = {"ids": ids, "descs": descs, "types": types, "labels": labels.tolist()}
    with open(OUT_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"💾 ذخیره شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
