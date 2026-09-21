#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
پردازش پس‌زمینه‌ی جستجوی گروهی از فایل اکسل/CSV
------------------------------------
به‌صورت یک پردازه‌ی جدا اجرا می‌شود (نه ترد داخل gunicorn) تا از ری‌استارت
وب‌سرور یا timeout درخواست HTTP آسیب نبیند: server/app.py این اسکریپت را
با subprocess.Popen جدا (detached) اجرا می‌کند و بلافاصله برمی‌گردد؛ این
اسکریپت خودش وضعیت را در status.json به‌روزرسانی می‌کند تا صفحه‌ی وب با
polling پیشرفت را نشان دهد.

اجرا: python pipeline/bulk_lookup.py <job_dir>
"""
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import OPENAI_API_KEY
from common.textnorm import tokenize
from common.utils import JsonlCheckpoint
from search.ambiguity import (
    apply_context_boost, compute_ambiguity,
    representative_per_group, resolve_ambiguous_group_by_context,
)
from search.engine import load_engine
from search.taxonomy_labels import label_for

from pipeline.bulk_shared import (
    COL_BRAND, COL_CATEGORY_HINT, COL_DESC, COL_NOTES, COL_OUT_CATEGORY,
    COL_OUT_ID, COL_OUT_OFFICIAL_DESC, COL_OUT_OTHER_OPTIONS,
    COL_OUT_STATUS, COL_OUT_USER_CHOICE, COL_TYPE, STATUS_FOUND,
    STATUS_NOT_FOUND, STATUS_REVIEW, TYPE_DOMESTIC, TYPE_IMPORTED, VALID_TYPES,
    output_header_order, read_rows, write_result_xlsx,
)

MAX_WORKERS = 8
RETRIEVE_LIMIT = 200
OPTION_SEP = " | "
ROW_INDEX_KEY = "__row_index__"  # فقط داخلی؛ در نوشتن result.xlsx نادیده گرفته می‌شود
CHECKPOINT_FILENAME = "progress.jsonl"


def _looks_like_rate_limit(exc):
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text


def load_status(job_dir):
    with open(job_dir / "status.json", encoding="utf-8") as f:
        return json.load(f)


def save_status(job_dir, status):
    tmp = job_dir / "status.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    tmp.replace(job_dir / "status.json")


def build_core_text(row):
    """شرح کالا با برند-زدایی قطعی (نه حدسی، چون برندها هرگز در کاتالوگ نیستند)."""
    desc = (row.get(COL_DESC) or "").strip()
    brand = (row.get(COL_BRAND) or "").strip()
    if brand:
        desc = desc.replace(brand, " ")
        desc = re.sub(r"\s+", " ", desc).strip()
    return desc


def build_context_text(row):
    """دسته کالا + توضیحات بیشتر — زمینه‌ی جانبی برای رفع ابهام (نه بخشی از خود شرح)."""
    return " ".join(
        (row.get(col) or "").strip()
        for col in (COL_CATEGORY_HINT, COL_NOTES)
        if (row.get(col) or "").strip()
    )


def build_query_text(row):
    """متنی که به فهم پرس‌وجو فرستاده می‌شود: شرح(برند-زدوده) + زمینه، هم‌جا."""
    core = build_core_text(row)
    context = build_context_text(row)
    return f"{core} {context}".strip()


def row_type(row):
    t = (row.get(COL_TYPE) or "").strip()
    return t if t in VALID_TYPES else ""


def pick_ids(item, rtype):
    if rtype == TYPE_IMPORTED:
        return list(item.imported_ids)
    if rtype == TYPE_DOMESTIC:
        return list(item.domestic_ids)
    return list(item.domestic_ids) + list(item.imported_ids)


def _found_record(item, rtype):
    return {
        COL_OUT_ID: "، ".join(pick_ids(item, rtype)),
        COL_OUT_OFFICIAL_DESC: item.description,
        COL_OUT_CATEGORY: label_for(item.level1),
        COL_OUT_OTHER_OPTIONS: "",
        COL_OUT_STATUS: STATUS_FOUND,
    }


def _not_found_record():
    return {
        COL_OUT_ID: "", COL_OUT_OFFICIAL_DESC: "", COL_OUT_CATEGORY: "",
        COL_OUT_OTHER_OPTIONS: "", COL_OUT_STATUS: STATUS_NOT_FOUND,
    }


def process_row(engine, mode, row):
    """جستجوی واقعی یک ردیف (۱ یا ۲ فراخوانی API در حالت هوشمند). فقط برای
    ردیف‌هایی صدا زده می‌شود که قبلاً نتیجه‌ی قطعی نداشته‌اند.

    شرح(core) و زمینه(دسته کالا/توضیحات بیشتر) دو نقش کاملاً جدا دارند:
    - core (شرح با برند-زدایی قطعی) تنها چیزی است که به فهم پرس‌وجو/بازیابی
      داده می‌شود و فیلتر AND ابهام را می‌سازد.
    - زمینه هرگز وارد بازیابی (فهم پرس‌وجو/BM25/embedding) نمی‌شود — با
      آزمایش مستقیم تأیید شد که اگر زمینه هم داخل متنی برود که به مدل داده
      می‌شود، مدل کلماتِ زمینه را وارد keywords خودش می‌کند و همین BM25/RRF
      را به‌سمت زمینه منحرف می‌کند، حتی برای سطرهایی که واقعاً به آن دسته
      تعلق ندارند. زمینه یک متن ثابت است که معمولاً یکسان روی همه‌ی سطرهای
      فایل اعمال می‌شود، پس نباید بتواند نتیجه‌ی سطرهایی را که واقعاً به آن
      ربطی ندارند خراب کند. به‌جایش، بعد از بازیابی خالص core، فقط امتیاز
      کاندیدهای هم‌دسته با زمینه کمی boost می‌شود (search.ambiguity.
      apply_context_boost) — یک فیلتر سخت نیست: اگر بهترین کاندید boost‌شده
      خیلی ضعیف‌تر از بهترین کاندید boost‌نشده باشد، رتبه‌ی اول عوض نمی‌شود
      (مثلاً «روتختی» با boost هم به‌غلط «روتختی بیمارستانی» نمی‌شود، چون
      نسخه‌ی عمومی‌اش به‌وضوح قوی‌تر است؛ اما «چشم‌بند» که تقریباً مساوی
      رتبه‌بندی شده، به نسخه‌ی پزشکی/زمینه‌دار می‌رود).
    - حتی بعد از boost، اگر رتبه‌ی اول نهایی هیچ توکن معناداری از پرس‌وجو را
      در متن خودش نداشته باشد (has_token_support)، «یافت شد» قاطع اما بی‌ربط
      اعلام نمی‌شود — به‌جایش «نیاز به بررسی» با چند گزینه‌ی نماینده.
    """
    core = build_core_text(row)
    context = build_context_text(row)
    rtype = row_type(row)
    core_tokens = set(tokenize(core))
    context_tokens = set(tokenize(context))

    if not core_tokens:
        return _not_found_record()

    if mode == "smart" and engine.hybrid is not None:
        lexical_q = engine.understander.lexical_query(core)
        semantic_q = engine.understander.semantic_query(core)
        raw_results = engine.hybrid.search_split(lexical_q, semantic_q, k=RETRIEVE_LIMIT)
        query_tokens = set(tokenize(semantic_q)) or core_tokens
    else:
        raw_results = engine.lexical.search(core, k=RETRIEVE_LIMIT)
        query_tokens = core_tokens

    if not raw_results:
        return _not_found_record()

    raw_results = apply_context_boost(raw_results, context_tokens)
    ranked_items = [item for item, _ in raw_results]

    ambiguous, groups, top = compute_ambiguity(ranked_items, query_tokens, engine.item_tokens)
    if ambiguous:
        resolved = resolve_ambiguous_group_by_context(top, groups, context_tokens)
        if resolved is not None:
            return _found_record(resolved, rtype)

        reps = representative_per_group(top, groups)
        lines = []
        for g in groups:
            item = reps.get(g["level1"])
            if item is None:
                continue
            ids = pick_ids(item, rtype)
            lines.append(f"{g['label']}{OPTION_SEP}{item.description}{OPTION_SEP}{'، '.join(ids)}")
        return {
            COL_OUT_ID: "", COL_OUT_OFFICIAL_DESC: "", COL_OUT_CATEGORY: "",
            COL_OUT_OTHER_OPTIONS: "\n".join(lines), COL_OUT_STATUS: STATUS_REVIEW,
        }

    return _found_record(ranked_items[0], rtype)


def resolve_from_user_choice(row):
    """آیا کاربر در «انتخاب شما» یکی از گزینه‌های ذخیره‌شده‌ی «گزینه‌های دیگر» را
    نوشته؟ در صورت تطابق دقیقاً یکی از گزینه‌ها، بدون هیچ فراخوانی API قطعی
    می‌شود؛ در غیر این صورت (بدون تطابق یا تطابق مبهم با چند گزینه) ردیف همچنان
    «نیاز به بررسی» می‌ماند."""
    choice = (row.get(COL_OUT_USER_CHOICE) or "").strip()
    options_text = row.get(COL_OUT_OTHER_OPTIONS) or ""
    if not choice or not options_text:
        return None

    needle = choice.casefold()
    matches = []
    for line in options_text.splitlines():
        parts = line.split(OPTION_SEP)
        if len(parts) != 3:
            continue
        label, desc, ids = (p.strip() for p in parts)
        haystack = f"{label} {desc} {ids}".casefold()
        if needle in haystack:
            matches.append((label, desc, ids))

    if len(matches) != 1:
        return None

    label, desc, ids = matches[0]
    return {
        COL_OUT_ID: ids,
        COL_OUT_OFFICIAL_DESC: desc,
        COL_OUT_CATEGORY: label,
        COL_OUT_OTHER_OPTIONS: "",
        COL_OUT_STATUS: STATUS_FOUND,
    }


def classify_rows(rows):
    """هر ردیف را به یکی از سه دسته می‌برد:
    - already_done: وضعیت قبلی «یافت شد»/«یافت نشد»، یا «نیاز به بررسی» بدون
      انتخاب تازه‌ی قابل‌تطبیق — دست‌نخورده می‌ماند. جستجو قطعی است، پس تکرار آن
      روی همان متن ورودی دقیقاً همان نتیجه‌ی مبهم قبلی را می‌دهد و فقط هزینه‌ی
      API را هدر می‌دهد؛ برای اجبار به جستجوی دوباره، کاربر فقط کافی است ستون
      «وضعیت» همان ردیف را قبل از آپلود مجدد خالی کند.
    - resolved_locally: «نیاز به بررسی» بوده و «انتخاب شما» به‌طور قطعی با یکی از
      گزینه‌ها تطابق دارد — بدون فراخوانی API قطعی می‌شود.
    - needs_search: بقیه (فقط ردیف‌های کاملاً تازه، بدون وضعیت قبلی)."""
    already_done, resolved_locally, needs_search = [], [], []
    for row in rows:
        status = (row.get(COL_OUT_STATUS) or "").strip()
        if status in (STATUS_FOUND, STATUS_NOT_FOUND):
            already_done.append(row)
        elif status == STATUS_REVIEW:
            resolution = resolve_from_user_choice(row)
            if resolution:
                resolved_locally.append((row, resolution))
            else:
                already_done.append(row)
        else:
            needs_search.append(row)
    return already_done, resolved_locally, needs_search


def main():
    job_dir = Path(sys.argv[1])
    status = load_status(job_dir)
    mode = status["mode"]

    try:
        if mode == "smart" and not OPENAI_API_KEY:
            raise RuntimeError("برای «هوشمند»، کلید OpenAI در .env لازم است.")

        input_path = job_dir / status["input_filename"]
        headers, rows = read_rows(input_path)
        header_order = output_header_order(headers)
        for i, row in enumerate(rows):
            row[ROW_INDEX_KEY] = i  # کلید پایدار برای checkpoint، فارغ از خروجی ستون‌ها

        already_done, resolved_locally, needs_search = classify_rows(rows)

        # checkpoint سطح-ردیف (نه ستون‌های خروجی خودِ فایل): اگر این اسکریپت روی
        # همین job_dir دوباره اجرا شود (بعد از crash یا ری‌استارت سرور، نگاه کنید
        # به server/app.py:_resume_stale_bulk_jobs)، ردیف‌هایی که قبلاً همین‌جا
        # تمام شده‌اند دوباره پردازش/فراخوانی API نمی‌شوند.
        ckpt = JsonlCheckpoint(job_dir / CHECKPOINT_FILENAME, key_field=ROW_INDEX_KEY)
        ckpt_by_index = {r[ROW_INDEX_KEY]: r for r in ckpt.load_all()}
        still_needs_search = []
        for row in needs_search:
            cached = ckpt_by_index.get(row[ROW_INDEX_KEY])
            if cached is not None:
                row.update({k: v for k, v in cached.items() if k != ROW_INDEX_KEY})
            else:
                still_needs_search.append(row)

        already_done_count = len(already_done) + len(resolved_locally) + (len(needs_search) - len(still_needs_search))
        status.update(status="processing", processed=already_done_count,
                       total=len(rows), total_to_search=len(still_needs_search), stopped_reason=None)
        save_status(job_dir, status)

        print(f"📦 {len(rows)} ردیف — {already_done_count} قبلاً نهایی (شامل checkpoint اجرای قبلی)، "
              f"{len(still_needs_search)} نیاز به جستجو.")

        for row, resolution in resolved_locally:
            row.update(resolution)

        quota_exhausted = threading.Event()

        if still_needs_search:
            engine = load_engine()
            lock = threading.Lock()
            done_count = already_done_count

            def worker(row):
                if quota_exhausted.is_set():
                    return row, None
                try:
                    return row, process_row(engine, mode, row)
                except Exception as e:
                    print(f"  ⚠️  ردیف {row[ROW_INDEX_KEY]} خطا داد: {e}")
                    if _looks_like_rate_limit(e):
                        quota_exhausted.set()
                    return row, None

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(worker, row) for row in still_needs_search]
                for future in as_completed(futures):
                    row, result = future.result()
                    if result is not None:
                        row.update(result)
                        ckpt.append({ROW_INDEX_KEY: row[ROW_INDEX_KEY], **result})
                        with lock:
                            done_count += 1
                            status["processed"] = done_count
                            save_status(job_dir, status)
                        print(f"  ✓ {done_count}/{len(rows)} — {result[COL_OUT_STATUS]}")
                    if quota_exhausted.is_set():
                        for f in futures:
                            f.cancel()  # فقط آن‌هایی که هنوز شروع نشده‌اند لغو می‌شوند

        result_path = job_dir / "result.xlsx"
        write_result_xlsx(result_path, header_order, rows)

        if quota_exhausted.is_set():
            remaining = len(rows) - status["processed"]
            status.update(status="done", result_filename="result.xlsx",
                           stopped_reason="quota_exhausted")
            save_status(job_dir, status)
            print(f"⏸  سهمیه‌ی روزانه‌ی API تمام شد — {status['processed']}/{len(rows)} پردازش شد، "
                  f"{remaining} ردیف باقی مانده. فایل را بعد از تمدید سهمیه دوباره آپلود کنید تا ادامه یابد "
                  f"(ردیف‌های تمام‌شده تکرار نمی‌شوند).")
        else:
            status.update(status="done", processed=len(rows), result_filename="result.xlsx")
            save_status(job_dir, status)
            print(f"✅ کامل شد → {result_path}")

    except Exception as e:
        status.update(status="error", error=str(e))
        save_status(job_dir, status)
        print(f"❌ خطا: {e}")
        raise


if __name__ == "__main__":
    main()
