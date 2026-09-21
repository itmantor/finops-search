#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
غنی‌سازی هدفمند واژگان بازاری — فقط شاخه‌ی بهداشتی/آرایشی
------------------------------------
مشکلی که این اسکریپت حل می‌کند: پاس ۱ (pipeline/enrich_catalog.py) تقریباً
همه‌ی smoke testها روی شاخه‌ی دارویی انجام شد؛ شاخه‌ی آرایشی/بهداشتی از نظر
حجم synonyms کم نیست (میانگین ۳.۳۰ در برابر ۳.۲۴ کل کاتالوگ)، اما واژگان
تولیدشده اصطلاحات رسمی/بروکراتیک است، نه چیزی که مشتری/دکتر داروخانه واقعاً
می‌گوید — «مام»، «ژیلت»، «ضد آفتاب»، «هیالورونیک» هیچ‌کدام در ۲۳٬۷۴۲ شرح
کاتالوگ حتی یک‌بار هم نیامده‌اند (با tokenize مستقیم تأیید شد).

فقط شاخه‌ی TARGET_LEVEL1 (۳۸۸ شرح) را دوباره غنی‌سازی می‌کند، فقط فیلد
synonyms را (canonical/examples_raw/uses پاس ۱ دست‌نخورده می‌ماند)، و خروجی
در فایل جداگانه‌ای ذخیره می‌شود (checkpoints/catalog_enrichment_cosmetics.
jsonl) که search/enrichment.py آن را به‌عنوان افزونه (نه جای‌گزین) روی پاس ۱
بارگذاری می‌کند — یعنی بقیه‌ی کاتالوگ کاملاً دست‌نخورده می‌ماند.

Resume دارد (JsonlCheckpoint، کلید desc_hash)، با ThreadPoolExecutor هم‌زمان
(~۸ کارگر) اجرا می‌شود، و اگر سهمیه‌ی روزانه‌ی API تمام شود متوقف می‌شود
(نه این‌که هر ردیف باقی‌مانده را با retry/backoff کامل هدر بدهد) — همان
درسی که از اجرای پاس ۱/جستجوی گروهی گرفته شد.

اجرا: python pipeline/reenrich_cosmetics.py
"""
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.catalog import load_catalog
from common.config import CHAT_MODEL, CHECKPOINT_DIR, OPENAI_API_KEY
from common.utils import JsonlCheckpoint, with_retry

TARGET_LEVEL1 = "اقلام بهداشتی و ارایشی"
MAX_WORKERS = 8
OUT_PATH = CHECKPOINT_DIR / "catalog_enrichment_cosmetics.jsonl"
PASS1_PATH = CHECKPOINT_DIR / "catalog_enrichment.jsonl"

SYSTEM_PROMPT = """تو یک متخصص واژگان بازار محصولات آرایشی/بهداشتی در ایران هستی —
دقیقاً همان چیزی که یک مشتری در داروخانه یا فروشگاه لوازم آرایشی می‌گوید، یا
یک تکنسین/فروشنده‌ی داروخانه برای پیدا کردن کد یک کالا تایپ می‌کند.

برای هر شرح کاتالوگ (رسمی/بروکراتیک، طبق نامگذاری گمرکی) که با synonyms
فعلی‌اش (تولید یک پاس قبلی‌تر، عمومی‌تر) داده می‌شود، فهرستی از واژگان
بازاری واقعی برگردان که در synonyms فعلی نیستند.

تمرکز اصلی: نام‌های ژنریک/برند-عمومی‌شده که مردم واقعاً به‌جای عبارت رسمی
به کار می‌برند — نه توصیف رسمی دوباره. نمونه‌های دقیقاً از همین نوع که باید
پیدا کنی (نه لزوماً همین کالاها، اصل قیاس مهم است):
- «رول ضد عرق لوازم ارایشی» باید «مام» و «دئودورانت» را هم داشته باشد
  (نام برند قدیمی Mum که در فارسی به کل این دسته تبدیل شده).
- «دسته تیغ اصلاح ( خود تراش )» / کالاهای تیغ اصلاح باید «ژیلت» را هم داشته
  باشد (نام برند Gillette که مترادف عمومی «تیغ اصلاح» شده).
- کرم‌های عمومی صورت/بدن که می‌توانند برای محافظت در برابر نور خورشید هم
  استفاده شوند باید «ضد آفتاب»، «سان اسکرین»، «کرم آفتاب» را هم داشته باشند
  (حتی اگر شرح رسمی این را نگوید — کاتالوگ کد اختصاصی «ضدآفتاب» ندارد، پس
  نزدیک‌ترین کرم/ژل عمومی باید با این کلمات هم پیدا شدنی باشد).
- سرم‌های آرایشی/بهداشتی عمومی باید نام ماده‌ی موثره‌ی رایج مثل «هیالورونیک»،
  «ویتامین سی»، «نیاسینامید»، «رتینول» را هم داشته باشند (وقتی واقعاً با نوع
  سرم هم‌خوان است).

قوانین:
- فقط فارسی (به‌جز نام‌های لاتین رایج مثل «ژیلت» که خودشان اسم خاص هستند).
- ۲ تا ۸ واژه‌ی تازه به ازای هر شرح؛ اگر واقعاً چیزی معنادار نمی‌شناسی،
  آرایه‌ی خالی برگردان — چیزی اختراع نکن.
- هرگز چیزی را که از قبل در synonyms فعلی هست دوباره تکرار نکن.
- فقط واژه‌ای اضافه کن که یک مشتری واقعی برای *همین* کالای خاص جستجو می‌کند،
  نه یک اصطلاح عمومیِ کل دسته که ربط مستقیم ندارد.
- خروجی باید دقیقاً یک شیء JSON با کلید "items" باشد: آرایه‌ای به همان تعداد
  و ترتیب ورودی، هرکدام فقط {"synonyms_market": ["...", "..."]}.
"""


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def _looks_like_rate_limit(exc):
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text


def build_user_content(items, pass1_map):
    lines = ["برای هر شرح زیر، synonyms_market تازه پیدا کن:\n"]
    for i, item in enumerate(items, start=1):
        rec = pass1_map.get(desc_hash(item.description), {})
        lines.append(f"{i}. شرح: {item.description}")
        lines.append(f"   synonyms فعلی: {json.dumps(rec.get('synonyms', []), ensure_ascii=False)}")
    return "\n".join(lines)


def call_batch(client, items, pass1_map):
    def call():
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(items, pass1_map)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    resp = with_retry(call)
    parsed = json.loads(resp.choices[0].message.content)
    results = parsed["items"]
    if len(results) != len(items):
        raise ValueError(f"پاسخ نامعتبر ({len(results)} در برابر {len(items)})")
    return results


def main():
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    items = load_catalog()
    target_items = [i for i in items if i.level1 == TARGET_LEVEL1]
    print(f"🎯 {len(target_items):,} شرح در شاخه‌ی «{TARGET_LEVEL1}».")

    pass1_map = {}
    with open(PASS1_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pass1_map[rec["desc_hash"]] = rec

    ckpt = JsonlCheckpoint(OUT_PATH, key_field="desc_hash")
    remaining = [i for i in target_items if not ckpt.is_done(desc_hash(i.description))]
    already = len(target_items) - len(remaining)
    if already:
        print(f"⏭  {already:,} شرح قبلاً انجام شده بود (Resume) — رد شد.")
    print(f"🚀 {len(remaining):,} شرح باقی‌مانده.")

    if not remaining:
        print("✅ همه‌ی شرح‌های این شاخه از قبل غنی‌سازی شده‌اند.")
        return

    quota_exhausted = threading.Event()
    lock = threading.Lock()

    def worker(item):
        if quota_exhausted.is_set():
            return item, None
        try:
            result = call_batch(client, [item], pass1_map)[0]
            return item, result
        except Exception as e:
            print(f"  ⚠️  «{item.description}» خطا داد: {e}")
            if _looks_like_rate_limit(e):
                quota_exhausted.set()
            return item, None

    done_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, item) for item in remaining]
        for future in as_completed(futures):
            item, result = future.result()
            if result is not None:
                ckpt.append({
                    "desc_hash": desc_hash(item.description),
                    "description": item.description,
                    "synonyms_market": result.get("synonyms_market", []),
                })
                with lock:
                    done_count += 1
                print(f"  ✓ {done_count}/{len(remaining)} — {item.description[:40]}")
            if quota_exhausted.is_set():
                for f in futures:
                    f.cancel()

    if quota_exhausted.is_set():
        remaining_count = len(remaining) - done_count
        print(f"⏸  سهمیه‌ی روزانه‌ی API تمام شد — {done_count}/{len(remaining)} پردازش شد، "
              f"{remaining_count} باقی مانده. بعداً همین اسکریپت را دوباره اجرا کنید تا ادامه یابد.")
    else:
        print(f"✅ کامل شد → {OUT_PATH}")


if __name__ == "__main__":
    main()
