#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اجرای کامل پایپ‌لاین (مراحل ۱ تا ۷) به‌ترتیب.

هر مرحله‌ی API-محور (۱، ۳، ۶) به‌صورت مستقل Resume می‌شود، پس اگر برنامه
قطع شد (قطع اینترنت، بسته‌شدن پنجره، ری‌استارت سیستم)، کافی‌ست همین
اسکریپت را دوباره اجرا کنید — از همان‌جا که متوقف شده بود ادامه می‌دهد.
"""
import subprocess
import sys
from pathlib import Path

STAGES = [
    "01_embed_initial.py",
    "02_cluster.py",
    "03_taxonomy.py",
    "04_categorize.py",
    "05_build_embedding_text.py",
    "06_embed_final.py",
    "07_build_faiss.py",
]


def main():
    base = Path(__file__).resolve().parent
    for stage in STAGES:
        print(f"\n{'=' * 50}\n▶ اجرای {stage}\n{'=' * 50}")
        result = subprocess.run([sys.executable, str(base / stage)])
        if result.returncode != 0:
            print(f"\n❌ مرحله {stage} با خطا متوقف شد.")
            print("   دوباره همین اسکریپت (run_all.py) را اجرا کنید تا از همین‌جا ادامه یابد.")
            sys.exit(1)
    print("\n✅ کل پایپ‌لاین با موفقیت اجرا شد! حالا می‌توانید سرور جستجو را اجرا کنید:")
    print("   python server/app.py")


if __name__ == "__main__":
    main()
