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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import OPENAI_API_KEY
from common.textnorm import tokenize
from search.ambiguity import compute_ambiguity, representative_per_group
from search.engine import load_engine
from search.taxonomy_labels import label_for

from pipeline.bulk_shared import (
    COL_BRAND, COL_CATEGORY_HINT, COL_DESC, COL_NOTES, COL_OUT_CATEGORY,
    COL_OUT_CONFIDENCE, COL_OUT_ID, COL_OUT_OFFICIAL_DESC, COL_OUT_OTHER_OPTIONS,
    COL_OUT_STATUS, COL_OUT_USER_CHOICE, COL_TYPE, STATUS_FOUND,
    STATUS_NOT_FOUND, STATUS_REVIEW, TYPE_DOMESTIC, TYPE_IMPORTED, VALID_TYPES,
    output_header_order, read_rows, write_result_xlsx,
)

MAX_WORKERS = 8
RETRIEVE_LIMIT = 200
OPTION_SEP = " | "


def load_status(job_dir):
    with open(job_dir / "status.json", encoding="utf-8") as f:
        return json.load(f)


def save_status(job_dir, status):
    tmp = job_dir / "status.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    tmp.replace(job_dir / "status.json")


def build_query_text(row):
    """شرح + برند-زدایی قطعی (نه حدسی) + دسته/توضیحات تکمیلی برای رفع ابهام."""
    desc = (row.get(COL_DESC) or "").strip()
    brand = (row.get(COL_BRAND) or "").strip()
    if brand:
        desc = desc.replace(brand, " ")
        desc = re.sub(r"\s+", " ", desc).strip()

    extra = " ".join(
        (row.get(col) or "").strip()
        for col in (COL_CATEGORY_HINT, COL_NOTES)
        if (row.get(col) or "").strip()
    )
    return f"{desc} {extra}".strip()


def row_type(row):
    t = (row.get(COL_TYPE) or "").strip()
    return t if t in VALID_TYPES else ""


def pick_ids(item, rtype):
    if rtype == TYPE_IMPORTED:
        return list(item.imported_ids)
    if rtype == TYPE_DOMESTIC:
        return list(item.domestic_ids)
    return list(item.domestic_ids) + list(item.imported_ids)


def confidence_pct(raw_results):
    """هرچه فاصله‌ی امتیاز رتبه‌ی ۱ با رتبه‌ی ۲ بیشتر باشد، اطمینان بالاتر —
    بدون فراخوانی مدل، فقط از شکاف امتیاز بازیابی که از قبل محاسبه شده."""
    if not raw_results:
        return 0
    top_score = raw_results[0][1]
    if top_score <= 0:
        return 0
    second_score = raw_results[1][1] if len(raw_results) > 1 else 0.0
    margin = max(0.0, 1 - (second_score / top_score))
    return round(margin * 100)


def process_row(engine, mode, row):
    """جستجوی واقعی یک ردیف (۱ یا ۲ فراخوانی API در حالت هوشمند). فقط برای
    ردیف‌هایی صدا زده می‌شود که قبلاً نتیجه‌ی قطعی نداشته‌اند."""
    query_text = build_query_text(row)
    rtype = row_type(row)
    raw_tokens = set(tokenize(query_text))

    if not raw_tokens:
        return {
            COL_OUT_ID: "", COL_OUT_OFFICIAL_DESC: "", COL_OUT_CATEGORY: "",
            COL_OUT_CONFIDENCE: "", COL_OUT_OTHER_OPTIONS: "", COL_OUT_STATUS: STATUS_NOT_FOUND,
        }

    if mode == "smart" and engine.hybrid is not None:
        lexical_q = engine.understander.lexical_query(query_text)
        semantic_q = engine.understander.semantic_query(query_text)
        raw_results = engine.hybrid.search_split(lexical_q, semantic_q, k=RETRIEVE_LIMIT)
        query_tokens = set(tokenize(semantic_q))
    else:
        raw_results = engine.lexical.search(query_text, k=RETRIEVE_LIMIT)
        query_tokens = raw_tokens

    if not raw_results:
        return {
            COL_OUT_ID: "", COL_OUT_OFFICIAL_DESC: "", COL_OUT_CATEGORY: "",
            COL_OUT_CONFIDENCE: "", COL_OUT_OTHER_OPTIONS: "", COL_OUT_STATUS: STATUS_NOT_FOUND,
        }

    ranked_items = [item for item, _ in raw_results]
    ambiguous, groups, top = compute_ambiguity(ranked_items, query_tokens, engine.item_tokens)

    if ambiguous:
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
            COL_OUT_CONFIDENCE: "", COL_OUT_OTHER_OPTIONS: "\n".join(lines), COL_OUT_STATUS: STATUS_REVIEW,
        }

    top_item = ranked_items[0]
    return {
        COL_OUT_ID: "، ".join(pick_ids(top_item, rtype)),
        COL_OUT_OFFICIAL_DESC: top_item.description,
        COL_OUT_CATEGORY: label_for(top_item.level1),
        COL_OUT_CONFIDENCE: confidence_pct(raw_results),
        COL_OUT_OTHER_OPTIONS: "",
        COL_OUT_STATUS: STATUS_FOUND,
    }


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
        COL_OUT_CONFIDENCE: 100,
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

        already_done, resolved_locally, needs_search = classify_rows(rows)

        status.update(status="processing", processed=len(already_done) + len(resolved_locally),
                       total=len(rows), total_to_search=len(needs_search))
        save_status(job_dir, status)

        print(f"📦 {len(rows)} ردیف — {len(already_done)} قبلاً نهایی، "
              f"{len(resolved_locally)} با انتخاب کاربر حل شد، {len(needs_search)} نیاز به جستجو.")

        for row, resolution in resolved_locally:
            row.update(resolution)

        if needs_search:
            engine = load_engine()
            lock = threading.Lock()
            done_count = status["processed"]

            def worker(row):
                return row, process_row(engine, mode, row)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(worker, row) for row in needs_search]
                for future in as_completed(futures):
                    row, result = future.result()
                    row.update(result)
                    with lock:
                        done_count += 1
                        status["processed"] = done_count
                        save_status(job_dir, status)
                    print(f"  ✓ {done_count}/{len(rows)} — {result[COL_OUT_STATUS]}")

        result_path = job_dir / "result.xlsx"
        write_result_xlsx(result_path, header_order, rows)

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
