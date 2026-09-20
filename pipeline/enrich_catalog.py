#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مرحله ۴ — غنی‌سازی کاتالوگ با واژگان بازار (دو پاس)
------------------------------------
مشکلی که این مرحله حل می‌کند: هیچ شرح کاتالوگی حاوی نام‌های بازاری دارو
(مثل «استامینوفن») نیست، پس هیچ کانال جستجویی نمی‌تواند از این نام‌ها به
شرح صحیح («قرص داروی شیمیایی موثر بر سیستم اعصاب و عضلات») برسد.

دو پاس، عمداً جدا از هم، چون هر کدام یک نوع خرابی متفاوت را جلو می‌گیرند:

  پاس ۱ (عمق) — هر شرح تک یا در دسته‌های کوچک (۳ تا ۵ تایی) غنی‌سازی می‌شود
  تا canonical/synonyms/examples/uses عمیق و دقیق بمانند. اگر شرح‌های زیاد و
  مشابه در یک فراخوانی با هم دیده شوند، مدل کوتاه و کلی‌گو می‌شود (دقیقاً
  همان الگوی رقیق‌شدنی که در فهم پرس‌وجوی نویزدار و در سند بلند دیدیم).

  پاس ۲ (سازگاری) — فقط فیلد examples را، برای هر برگ تاکسونومی (ردیف‌های
  خواهر که فقط شکل دارویی/فیزیکی‌شان فرق دارد)، در یک فراخوانی بازبینی
  می‌کند: نام‌های واقعی اما نامربوط را حذف، شکل دارویی/علامت را از examples
  حذف، و در صورت وجود شواهد بازار، پوشش را بین خواهر و برادرها هم‌سو می‌کند.
  canonical/synonyms/uses را دست نمی‌زند.

examples به‌صورت جداگانه ذخیره می‌شود (checkpoints/catalog_examples.jsonl)
تا بعداً بشود آن‌ها را به‌صورت مستقل در ایندکس واژگانی روشن/خاموش کرد و اثر
واقعی‌شان روی recall را جدا از canonical/synonyms اندازه گرفت.

هر دو پاس Resume دارند (کلید: هش شرح) و با ThreadPoolExecutor به‌صورت
هم‌زمان (~۸ کارگر) اجرا می‌شوند؛ نوشتن روی فایل checkpoint با قفل محافظت
می‌شود.

خروجی:
  checkpoints/catalog_enrichment.jsonl  — canonical, synonyms, uses, examples_raw (پاس ۱)
  checkpoints/catalog_examples.jsonl    — examples نهایی، پس از بازبینی (پاس ۲)
"""
import argparse
import hashlib
import json
import re
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.catalog import load_catalog
from common.config import CHAT_MODEL, CHECKPOINT_DIR, OPENAI_API_KEY
from common.utils import JsonlCheckpoint, batched, with_retry

PASS1_BATCH_SIZE = 5    # دسته‌های کوچک و دلبخواه (بدون گروه‌بندی) برای حفظ عمق
PASS2_GROUP_CAP = 40    # سقف اندازه‌ی هر فراخوانی بازبینی برای یک برگ تاکسونومی
MAX_WORKERS = 8

PASS1_PATH = CHECKPOINT_DIR / "catalog_enrichment.jsonl"
PASS2_PATH = CHECKPOINT_DIR / "catalog_examples.jsonl"

PHARMA_LEVEL1 = "دارو، فراورده های بیولوژیک و مکمل های دارویی و غذایی"
SMOKE_PREFIX_COUNT = 500

SYSTEM_PROMPT_PASS1 = """تو یک دستیار غنی‌سازی کاتالوگ کالا برای یک موتور جستجوی فارسی هستی.
برای هر شرح کاتالوگ که همراه با مسیر دسته‌بندی‌اش (level1 > level2 > level3 > level4) داده می‌شود،
یک شیء JSON با دقیقاً این ساختار تولید کن:

{"canonical": "...", "synonyms": ["...", "..."], "examples": ["...", "..."], "uses": ["...", "..."]}

- "canonical": یک عبارت کوتاه فارسی (حداکثر ~۸ کلمه) که دقیقاً بگوید این کد چیست. تمام بخش
  تمایزدهنده‌ی شرح اصلی را حفظ کن — آن را به یک عبارت عمومی کوتاه‌تر خلاصه نکن. مثلاً اگر شرح
  «موثر بر سیستم اعصاب و عضلات» دارد، canonical هم باید همین تمایز را نگه دارد، نه صرفاً
  «قرص داروی شیمیایی».
- "synonyms": ۳ تا ۸ عبارت بازاری فارسی که مردم واقعاً برای این چیز به کار می‌برند.
- "examples": ۰ تا ۱۰ نام مشخص محصول/برند/نام ژنریک واقعی که زیر این کد قرار می‌گیرند و هم با
  دسته‌ی درمانی/کاربردی و هم با شکل دارویی/فیزیکی این شرح مطابقت دارند.
  هرگز شکل دارویی/فیزیکی را به‌عنوان نمونه تکرار نکن (مثلاً «کپسول نرم» زیر ردیف «سافت ژل» غلط است).
  هرگز علامت/نشانه یا کاربرد را به‌عنوان نمونه نیاور — آن‌ها باید در uses بیایند، نه اینجا.
  این فیلد برای داروها حیاتی است: برای «قرص داروی شیمیایی موثر بر سیستم اعصاب و عضلات» باید
  شامل نام‌هایی مثل استامینوفن، پاراستامول، ایبوپروفن، ژلوفن، ناپروکسن و مشابه آن‌ها باشد.
- "uses": ۰ تا ۵ کاربرد یا علامت/نشانه‌ی کوتاه، هرجا معنادار باشد.

قوانین:
- فقط فارسی.
- متراکم و دقیق باش. هیچ حشو یا کلمه‌ی عمومی بی‌معنا مثل «محصول»، «کالا»، «انواع»، «مختلف»، «مرغوب» ننویس.
  آرایه‌ی خالی همیشه بهتر از آرایه‌ی مبهم یا اختراعی است.
- اگر این کد را نمی‌شناسی یا مطمئن نیستی، آرایه‌های خالی برگردان؛ چیزی اختراع نکن.
- هرگز از جای‌نگه‌دار (placeholder) مثل «برند X»، «مدل ABC»، «برند A» استفاده نکن. یا یک نام واقعی
  بنویس، یا آن ورودی را کلاً حذف کن — چیزی بینابین این دو نباشد.
- دقت به دامنه‌ی دسته حیاتی است: بسیاری از ردیف‌های دارویی یک شکل دارویی را برای یک گروه درمانی
  کامل و گسترده توصیف می‌کنند، نه یک زیرشاخه‌ی خاص؛ نمونه‌ها باید طیف رایج آن گروه درمانی را نشان
  دهند (نه فقط یک زیرگروه تصادفی)، تا جایی که واقعاً می‌شناسی‌شان.
- به تفاوت «شیمیایی» در برابر «گیاهی» و «طبیعی و سنتی» در شرح دقت کن: در ردیف‌هایی که «داروی
  شیمیایی» نوشته شده فقط داروهای شیمیایی/ترکیبات دارویی مصنوعی بیاور، هرگز گیاهی؛ در ردیف‌های
  «داروی گیاهی» یا «طبیعی و سنتی» فقط نمونه‌های گیاهی/سنتی بیاور، هرگز شیمیایی.
- خروجی باید دقیقاً یک شیء JSON با کلید "items" باشد که آرایه‌ای از این اشیاء
  به همان تعداد و به همان ترتیب ورودی است — هیچ موردی را رد نکن و ترتیب را عوض نکن:
  {"items": [{"canonical": "...", "synonyms": [...], "examples": [...], "uses": [...]}, ...]}
"""

SYSTEM_PROMPT_PASS2 = """تو در حال بازبینی (reconciliation) فیلد "examples" برای مجموعه‌ای از
ردیف‌های خواهر یک کاتالوگ دارویی/کالایی فارسی هستی. همه‌ی این ردیف‌ها مسیر تاکسونومی یکسانی
دارند و معمولاً فقط در شکل دارویی/فیزیکی (قرص، کپسول، شربت، قطره، پماد، ...) با هم فرق دارند.

برای هر ردیف، فهرست فعلی examples (خروجی خام یک پاس قبلی) داده شده. فقط examples را اصلاح کن؛
هیچ فیلد دیگری وجود ندارد و نباید تولید شود.

اولویت اصلاح، از مهم‌ترین به کم‌اهمیت‌ترین:
۱. نام واقعی داروی/محصولی که به دسته‌ی درمانی یا شکل دارویی این ردیف تعلق ندارد را حذف کن. این
   خطرناک‌ترین نوع خطاست، چون جستجوی واقعی کاربر را به کد اشتباه می‌فرستد (مثلاً «استرپتوکیناز»،
   یک داروی تزریقی ترومبولیتیک، هرگز نباید زیر یک ردیف «قطره» بماند).
۲. موردی که فقط تکرار شکل دارویی خودش است را حذف کن (مثلاً «کپسول نرم» به‌عنوان نمونه‌ی زیر ردیف
   «سافت ژل» — این توصیف شکل داروست، نه نام دارو).
۳. موردی که در واقع یک علامت/نشانه یا کاربرد است، نه نام محصول، را حذف کن (مثلاً «خارش» زیر یک
   کرم — این باید در uses باشد، نه examples).
۴. نام اختراعی/ناشناخته‌ای که هیچ داروی واقعی شناخته‌شده‌ای نیست را هم حذف کن، اما این کم‌اهمیت‌ترین
   مورد است — یک نام اختراعی و نامربوط تقریباً بی‌ضرر است چون کسی آن را جستجو نمی‌کند؛ نام واقعیِ
   اشتباه‌جا‌افتاده به‌مراتب خطرناک‌تر است و باید اول اصلاح شود.
۵. اگر یک ماده‌ی موثره یا نام بازاری واقعاً زیر یک ردیف خواهر دیگر آمده و به‌صورت واقعی در بازار به
   شکل دارویی این ردیف هم موجود است اما اینجا نیامده، می‌توانی اضافه‌اش کنی تا پوشش بین خواهر و
   برادرها هم‌سو شود.

قوانین:
- فقط نام محصول/برند/ماده‌ی موثره‌ی واقعی که هم با دسته‌ی درمانی/کاربردی این مسیر و هم با شکل
  دارویی/فیزیکی همان ردیف مطابقت دارد نگه‌دار یا اضافه کن.
- هرگز شکل دارویی/فیزیکی را به‌عنوان نمونه نگه ندار یا اضافه نکن.
- هرگز علامت/نشانه/کاربرد را به‌عنوان نمونه نگه ندار یا اضافه نکن.
- وقتی مطمئن نیستی، آرایه را خالی بگذار — از اختراع‌کردن یا حدس‌زدن خودداری کن.
- خروجی باید دقیقاً یک شیء JSON با کلید "items" باشد: آرایه‌ای از اشیاء به همان تعداد و ترتیب
  ورودی، هرکدام فقط {"examples": ["...", "..."]}.
"""


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def taxonomy_path(item):
    parts = [p for p in (item.level1, item.level2, item.level3, item.level4) if p]
    return " > ".join(parts)


# پاکسازی قطعی (بدون فراخوانی مدل) روی هر فیلد لیستی قبل از نوشتن در checkpoint:
# دو کلاس خرابی که مدل گاه‌به‌گاه تولید می‌کند و پاس ۲ تضمینی آن‌ها را نمی‌گیرد
# (چون پاس ۲ فقط روی سازگاری معنایی examples تمرکز دارد، نه روی زباله‌ی نویسه‌ای).
_NON_PERSIAN_LATIN_SCRIPT_PAT = re.compile(
    r"[Ѐ-ӿ぀-ヿ一-鿿가-힣ᄀ-ᇿ]"
)  # سیریلیک، هیراگانا/کاتاکانا، هان (چینی/کانجی)، هانگول
_PLACEHOLDER_WORD_PAT = re.compile(r"(برند|مدل)\s+[A-Za-z]{1,4}\b")  # «برند X»، «مدل ABC»
_BARE_TRAILING_LETTER_PAT = re.compile(r"(?:^|\s)[A-Za-z]$")  # به یک حرف لاتین تنها ختم شود


def sanitize_list(entries, stats):
    """حذف قطعی ورودی‌های خالی/اسکریپت غیرفارسی-غیرلاتین/جای‌نگه‌دار از یک فیلد لیستی.
    stats با شمار کل و حذف‌شده به‌روزرسانی می‌شود تا نرخ پاکسازی قابل گزارش باشد."""
    cleaned = []
    for raw in entries:
        stats["total"] += 1
        entry = (raw or "").strip()
        if not entry:
            stats["removed"] += 1
            continue
        if _NON_PERSIAN_LATIN_SCRIPT_PAT.search(entry):
            stats["removed"] += 1
            continue
        if _PLACEHOLDER_WORD_PAT.search(entry) or _BARE_TRAILING_LETTER_PAT.search(entry):
            stats["removed"] += 1
            continue
        cleaned.append(entry)
    return cleaned


def group_by_taxonomy_leaf(items):
    """گروه‌بندی شرح‌ها بر اساس (level1, level2, level3)، با حفظ ترتیب کاتالوگ."""
    groups = OrderedDict()
    for item in items:
        key = (item.level1, item.level2, item.level3)
        groups.setdefault(key, []).append(item)
    return groups


def make_leaf_batches(items, cap):
    """هر دسته فقط از یک برگ تاکسونومی می‌آید؛ برگ‌های بزرگ‌تر از cap تکه‌تکه می‌شوند."""
    batches = []
    for group_items in group_by_taxonomy_leaf(items).values():
        for chunk in batched(group_items, cap):
            batches.append(chunk)
    return batches


def call_json(client, system_prompt, user_content):
    def call():
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

    resp = with_retry(call)
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------- پاس ۱: عمق

def build_user_content_pass1(items):
    lines = ["شرح‌های زیر را غنی‌سازی کن:\n"]
    for i, item in enumerate(items, start=1):
        path = taxonomy_path(item)
        lines.append(f"{i}. شرح: {item.description}\n   مسیر دسته‌بندی: {path}")
    return "\n".join(lines)


def enrich_batch_pass1(client, items):
    """غنی‌سازی عمیق یک دسته‌ی کوچک؛ در صورت عدم تطابق تعداد، تک‌به‌تک دوباره تلاش می‌کند."""
    try:
        parsed = call_json(client, SYSTEM_PROMPT_PASS1, build_user_content_pass1(items))
        results = parsed["items"]
        if len(results) != len(items) or not all(isinstance(r, dict) for r in results):
            raise ValueError(f"پاسخ نامعتبر یا با تعداد نامطابق ({len(results)} در برابر {len(items)})")
        return results
    except Exception as e:
        if len(items) == 1:
            print(f"    ⚠️  پاس ۱ شکست خورد برای «{items[0].description}»: {e} — آرایه‌های خالی ثبت می‌شود.")
            return [{"canonical": "", "synonyms": [], "examples": [], "uses": []}]
        print(f"    ⚠️  دسته‌ی پاس ۱ با خطا مواجه شد ({e}) — تک‌به‌تک تلاش می‌شود.")
        results = []
        for item in items:
            results.extend(enrich_batch_pass1(client, [item]))
        return results


# ------------------------------------------------------------ پاس ۲: سازگاری

def build_user_content_pass2(items, pass1_map):
    path = taxonomy_path(items[0])
    lines = [
        f"مسیر دسته‌بندی مشترک همه‌ی ردیف‌های زیر: {path}",
        "",
        "ردیف‌های زیر خواهر و برادر هم در همین دسته‌اند. برای هرکدام examples فعلی داده شده؛ "
        "طبق قوانین بازبینی کن:",
        "",
    ]
    for i, item in enumerate(items, start=1):
        rec = pass1_map.get(desc_hash(item.description), {})
        ex = rec.get("examples_raw", [])
        lines.append(f"{i}. شرح: {item.description}")
        lines.append(f"   examples فعلی: {json.dumps(ex, ensure_ascii=False)}")
    return "\n".join(lines)


def enrich_batch_pass2(client, items, pass1_map):
    """بازبینی examples یک گروه خواهر؛ در صورت عدم تطابق تعداد، تک‌به‌تک دوباره تلاش می‌کند."""
    try:
        parsed = call_json(client, SYSTEM_PROMPT_PASS2, build_user_content_pass2(items, pass1_map))
        results = parsed["items"]
        if len(results) != len(items) or not all(isinstance(r, dict) for r in results):
            raise ValueError(f"پاسخ نامعتبر یا با تعداد نامطابق ({len(results)} در برابر {len(items)})")
        return results
    except Exception as e:
        if len(items) == 1:
            # روی خطای فنی، به‌جای خالی‌کردن، فهرست خام پاس ۱ دست‌نخورده می‌ماند (امن‌تر از حذف کامل).
            fallback = pass1_map.get(desc_hash(items[0].description), {}).get("examples_raw", [])
            print(f"    ⚠️  پاس ۲ شکست خورد برای «{items[0].description}»: {e} — examples پاس ۱ دست‌نخورده ثبت می‌شود.")
            return [{"examples": fallback}]
        print(f"    ⚠️  دسته‌ی پاس ۲ با خطا مواجه شد ({e}) — تک‌به‌تک تلاش می‌شود.")
        results = []
        for item in items:
            results.extend(enrich_batch_pass2(client, [item], pass1_map))
        return results


# ------------------------------------------------------------------ اجرا

def select_smoke_items(items):
    """۵۰۰ شرح اول + هر شرحی که level1 آن دارویی است."""
    prefix = items[:SMOKE_PREFIX_COUNT]
    prefix_descs = {item.description for item in prefix}
    pharma = [item for item in items if item.level1 == PHARMA_LEVEL1 and item.description not in prefix_descs]
    return prefix + pharma


def run_concurrent(batches, worker_fn, on_result, label):
    total = len(batches)
    lock = threading.Lock()
    done_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker_fn, chunk) for chunk in batches]
        for bi, future in enumerate(as_completed(futures), start=1):
            chunk, results = future.result()
            with lock:
                on_result(chunk, results)
                done_count += len(chunk)
                print(f"  ✓ [{label}] دسته {bi}/{total} — {done_count:,} شرح")


def run_pass1(client, target_items):
    ckpt1 = JsonlCheckpoint(PASS1_PATH, key_field="desc_hash")
    remaining = [item for item in target_items if not ckpt1.is_done(desc_hash(item.description))]
    already = len(target_items) - len(remaining)
    if already:
        print(f"⏭  [پاس ۱] {already:,} شرح قبلاً غنی‌سازی شده بود (Resume) — رد شد.")
    print(f"🚀 [پاس ۱] {len(remaining):,} شرح باقی‌مانده.")

    if remaining:
        batches = list(batched(remaining, PASS1_BATCH_SIZE))
        sanitize_stats = {"total": 0, "removed": 0}

        def worker(chunk):
            return chunk, enrich_batch_pass1(client, chunk)

        def on_result(chunk, results):
            for item, result in zip(chunk, results):
                ckpt1.append({
                    "desc_hash": desc_hash(item.description),
                    "description": item.description,
                    "canonical": result.get("canonical", ""),
                    "synonyms": sanitize_list(result.get("synonyms", []), sanitize_stats),
                    "examples_raw": sanitize_list(result.get("examples", []), sanitize_stats),
                    "uses": sanitize_list(result.get("uses", []), sanitize_stats),
                })

        run_concurrent(batches, worker, on_result, "پاس ۱")

        if sanitize_stats["total"]:
            rate = 100 * sanitize_stats["removed"] / sanitize_stats["total"]
            print(f"🧹 [پاس ۱] پاکسازی قطعی: {sanitize_stats['removed']:,} از "
                  f"{sanitize_stats['total']:,} ورودی حذف شد ({rate:.2f}٪).")

    print(f"✅ [پاس ۱] کامل شد → {PASS1_PATH}")
    return ckpt1


def run_pass2(client, target_items, ckpt1):
    pass1_map = {r["desc_hash"]: r for r in ckpt1.load_all()}

    ckpt2 = JsonlCheckpoint(PASS2_PATH, key_field="desc_hash")
    remaining = [item for item in target_items if not ckpt2.is_done(desc_hash(item.description))]
    already = len(target_items) - len(remaining)
    if already:
        print(f"⏭  [پاس ۲] {already:,} شرح قبلاً بازبینی شده بود (Resume) — رد شد.")
    print(f"🚀 [پاس ۲] {len(remaining):,} شرح باقی‌مانده.")

    if remaining:
        batches = make_leaf_batches(remaining, PASS2_GROUP_CAP)
        sanitize_stats = {"total": 0, "removed": 0}

        def worker(chunk):
            return chunk, enrich_batch_pass2(client, chunk, pass1_map)

        def on_result(chunk, results):
            for item, result in zip(chunk, results):
                ckpt2.append({
                    "desc_hash": desc_hash(item.description),
                    "description": item.description,
                    "examples": sanitize_list(result.get("examples", []), sanitize_stats),
                })

        run_concurrent(batches, worker, on_result, "پاس ۲")

        if sanitize_stats["total"]:
            rate = 100 * sanitize_stats["removed"] / sanitize_stats["total"]
            print(f"🧹 [پاس ۲] پاکسازی قطعی: {sanitize_stats['removed']:,} از "
                  f"{sanitize_stats['total']:,} ورودی حذف شد ({rate:.2f}٪).")

    print(f"✅ [پاس ۲] کامل شد → {PASS2_PATH}")
    return ckpt2


def print_samples(pass1_map, pass2_map, items, n=15):
    print("\n" + "=" * 100)
    print(f"نمونه‌های غنی‌سازی‌شده ({n} مورد) — examples نهایی پس از پاس ۲")
    print("=" * 100)

    pharma_items = [i for i in items if i.level1 == PHARMA_LEVEL1]
    generic_items = [i for i in items if i.level1 != PHARMA_LEVEL1]
    n_pharma = min(8, len(pharma_items))
    n_generic = n - n_pharma
    sample_items = generic_items[:n_generic] + pharma_items[:n_pharma]

    for item in sample_items:
        h = desc_hash(item.description)
        r1 = pass1_map.get(h)
        r2 = pass2_map.get(h)
        if not r1:
            continue
        print(f"\nشرح:    {item.description}")
        print(f"مسیر:   {taxonomy_path(item)}")
        print(f"canonical: {r1.get('canonical', '')}")
        print(f"synonyms:  {'، '.join(r1.get('synonyms', []))}")
        print(f"examples:  {'، '.join((r2 or {}).get('examples', []))}")
        print(f"uses:      {'، '.join(r1.get('uses', []))}")


def print_target_group(pass1_map, pass2_map, items, level3):
    """چاپ کنار هم قرص/کپسول/شربت (یا هر برگی) برای بازبینی دستی، همراه با تغییرات پاس ۲."""
    group = [i for i in items if i.level1 == PHARMA_LEVEL1 and i.level3 == level3]
    print("\n" + "=" * 100)
    print(f"ردیف‌های خواهر برای «{level3}» — قبل/بعد از پاس ۲")
    print("=" * 100)

    any_diff_printed = False
    for item in group:
        h = desc_hash(item.description)
        r1 = pass1_map.get(h, {})
        r2 = pass2_map.get(h, {})
        raw = r1.get("examples_raw", [])
        final = r2.get("examples", [])
        removed = [x for x in raw if x not in set(final)]
        added = [x for x in final if x not in set(raw)]

        print(f"\n{item.description}")
        print(f"  canonical: {r1.get('canonical', '')}")
        print(f"  synonyms:  {'، '.join(r1.get('synonyms', []))}")
        print(f"  uses:      {'، '.join(r1.get('uses', []))}")
        print(f"  examples (خام پاس ۱):   {'، '.join(raw)}")
        print(f"  examples (نهایی پاس ۲): {'، '.join(final)}")
        if removed or added:
            any_diff_printed = True
            if removed:
                print(f"  ✂️  حذف‌شده: {'، '.join(removed)}")
            if added:
                print(f"  ➕ افزوده‌شده: {'، '.join(added)}")

    if not any_diff_printed:
        print("\n(پاس ۲ در این گروه چیزی تغییر نداد)")


def print_pass2_diff_summary(pass1_map, pass2_map, items):
    print("\n" + "=" * 100)
    print("خلاصه‌ی تمام تغییرات پاس ۲ (فقط ردیف‌هایی که چیزی حذف یا اضافه شد)")
    print("=" * 100)

    n_changed = 0
    for item in items:
        h = desc_hash(item.description)
        raw = pass1_map.get(h, {}).get("examples_raw", [])
        final = pass2_map.get(h, {}).get("examples", [])
        removed = [x for x in raw if x not in set(final)]
        added = [x for x in final if x not in set(raw)]
        if not removed and not added:
            continue
        n_changed += 1
        print(f"\n{item.description}")
        if removed:
            print(f"  ✂️  حذف‌شده: {'، '.join(removed)}")
        if added:
            print(f"  ➕ افزوده‌شده: {'، '.join(added)}")

    print(f"\nجمع: {n_changed:,} ردیف تغییر کرد (از {len(items):,} ردیف بررسی‌شده).")


def main():
    parser = argparse.ArgumentParser(description="غنی‌سازی کاتالوگ با واژگان بازار (دو پاس)")
    parser.add_argument("--smoke", action="store_true",
                         help="اجرا فقط روی ۵۰۰ شرح اول + همه‌ی شرح‌های دارویی")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY در .env تنظیم نشده است.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    items = load_catalog()
    print(f"📦 {len(items):,} شرح یکتا از کاتالوگ خوانده شد.")

    target_items = select_smoke_items(items) if args.smoke else items
    if args.smoke:
        print(f"🧪 حالت آزمایشی (smoke): {len(target_items):,} شرح "
              f"({SMOKE_PREFIX_COUNT} شرح اول + شرح‌های دارویی).")

    ckpt1 = run_pass1(client, target_items)
    ckpt2 = run_pass2(client, target_items, ckpt1)

    if args.smoke:
        pass1_map = {r["desc_hash"]: r for r in ckpt1.load_all()}
        pass2_map = {r["desc_hash"]: r for r in ckpt2.load_all()}

        print_samples(pass1_map, pass2_map, target_items, n=15)
        print_target_group(pass1_map, pass2_map, target_items, "سیستم عصبی (گروه N)")
        print_pass2_diff_summary(pass1_map, pass2_map, target_items)


if __name__ == "__main__":
    main()
