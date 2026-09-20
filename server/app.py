#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرور جستجوی شناسه کالا — نسخه هیبرید (لایه بازیابی جدید)
------------------------------------
دو حالت جستجو:
  - "fast"  (جستجوی سریع)  : فقط کانال واژگانی BM25، بدون فراخوانی OpenAI، فوری و رایگان.
  - "smart" (جستجوی هوشمند): فهم پرس‌وجو (search/understand.py) + بازیابی هیبرید
                              (search/hybrid.py؛ BM25 + معنایی، ترکیب با RRF).

مرحله‌ی انتخاب نهایی (search/select.py) عمداً استفاده نمی‌شود — طبق محک،
دقت آن از گرفتن رتبه‌ی اول بازیابی بهتر نبود.

کاتالوگ، ایندکس واژگانی و ایندکس معنایی همگی در سطح ماژول (یک‌بار، هنگام
import) بارگذاری می‌شوند تا با gunicorn --preload بین همه‌ی workerها به
اشتراک گذاشته شوند.

اجرا (توسعه):
    python server/app.py
اجرا (تولید): از طریق systemd + gunicorn، نگاه کنید به README.md
"""
import datetime
import json
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request, send_file

from common.config import LOGS_DIR, OPENAI_API_KEY, PORT, UPLOAD_DIR
from common.textnorm import tokenize
from pipeline.bulk_shared import RETENTION_DAYS, ValidationError, read_rows, validate_rows
from search.ambiguity import compute_ambiguity
from search.engine import load_engine

MODE_FAST = "fast"
MODE_SMART = "smart"

BASE_DIR = Path(__file__).resolve().parent.parent
BULK_SCRIPT = BASE_DIR / "pipeline" / "bulk_lookup.py"

# عمق بازیابی داخلی (هر دو حالت) در برابر عمق نمایش اولیه در کلاینت:
# قبلاً هر دو ۸ بودند، پس یک تطابق در رتبه‌ی ۲۸ (مثل «شیر لبنیات» زیر ۷۶
# شیرآلات صنعتی) هرگز دیده نمی‌شد. حالا ۲۰۰ مورد بازیابی و به کلاینت
# برگردانده می‌شود؛ کلاینت فقط ۲۰ تای اول را نشان می‌دهد و بقیه را با دکمه‌ی
# «نمایش بیشتر» بدون فراخوانی دوباره‌ی API آشکار می‌کند.
RETRIEVE_LIMIT = 200
DISPLAY_LIMIT = 20  # فقط برای برش لاگ؛ کلاینت مقدار خودش را دارد

SEARCH_LOG_PATH = LOGS_DIR / "search_log.jsonl"

app = Flask(__name__)
_log_lock = threading.Lock()

print("⏳ در حال بارگذاری کاتالوگ و ایندکس‌ها ...")

ENGINE = load_engine()
ITEMS = ENGINE.items
LEXICAL = ENGINE.lexical
SEMANTIC = ENGINE.semantic
HYBRID = ENGINE.hybrid
UNDERSTANDER = ENGINE.understander
ITEM_TOKENS = ENGINE.item_tokens

if HYBRID is None:
    print("⚠️  ایندکس معنایی از پیش‌ساخته پیدا نشد. اجرا کنید: python pipeline/build_search_index.py")

print(f"✅ آماده — {len(ITEMS):,} شرح یکتا بارگذاری شد.")


def _to_scored_percent(results):
    """تبدیل [(item, خام‌امتیاز)] به [(item, درصد نسبی به بهترین نتیجه‌ی همین فهرست)]."""
    if not results:
        return []
    top = results[0][1]
    if top <= 0:
        return [(item, 0) for item, _ in results]
    return [(item, round(max(0.0, min(1.0, score / top)) * 100)) for item, score in results]


def make_record(item, score):
    return {
        "description": item.description,
        "domestic_ids": list(item.domestic_ids),
        "imported_ids": list(item.imported_ids),
        "score": score,
        # فقط برای گروه‌بندی کلاینت هنگام ابهام؛ در نتایج عادی نمایش داده نمی‌شود.
        "level1": item.level1,
    }


def _log_search(query, mode, results):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "results": [
            {
                "description": r["description"],
                "domestic_ids": r["domestic_ids"],
                "imported_ids": r["imported_ids"],
            }
            for r in results[:DISPLAY_LIMIT]
        ],
    }
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _log_lock:
            with open(SEARCH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        print(f"⚠️  خطا در ثبت لاگ جستجو: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "record_count": len(ITEMS),
        "has_smart": HYBRID is not None and bool(OPENAI_API_KEY),
        "has_api_key": bool(OPENAI_API_KEY),
    })


@app.route("/api/search", methods=["POST"])
def search():
    payload = request.get_json(force=True, silent=True) or {}
    query = (payload.get("query") or "").strip()
    mode = payload.get("mode") or MODE_FAST

    if not query:
        return jsonify({"results": []})

    if mode == MODE_SMART:
        if not OPENAI_API_KEY:
            return jsonify({"error": "برای «جستجوی هوشمند»، کلید OpenAI در .env لازم است."})
        if HYBRID is None:
            return jsonify({"error": "ایندکس معنایی آماده نیست. ابتدا python pipeline/build_search_index.py را اجرا کنید."})
        try:
            lexical_query = UNDERSTANDER.lexical_query(query)
            semantic_query = UNDERSTANDER.semantic_query(query)
            raw_results = HYBRID.search_split(lexical_query, semantic_query, k=RETRIEVE_LIMIT)
            # عبارت پاک‌شده (بدون برند/کلیدواژه‌های گسترش‌یافته) برای فیلتر AND
            # ابهام؛ از کش UNDERSTANDER می‌آید، فراخوانی اضافه‌ای ندارد.
            query_tokens = set(tokenize(semantic_query))
        except Exception as e:
            return jsonify({"error": f"خطا در ارتباط با OpenAI: {e}"})
    else:
        mode = MODE_FAST
        raw_results = LEXICAL.search(query, k=RETRIEVE_LIMIT)
        query_tokens = set(tokenize(query))

    ambiguous, groups, _top = compute_ambiguity([item for item, _ in raw_results], query_tokens, ITEM_TOKENS)
    results = [make_record(item, pct) for item, pct in _to_scored_percent(raw_results)]
    _log_search(query, mode, results)

    return jsonify({"results": results, "ambiguous": ambiguous, "groups": groups})


# ------------------------------------------------------------ جستجوی گروهی

def _cleanup_old_jobs():
    """پاکسازی فرصت‌طلبانه: هر job قدیمی‌تر از RETENTION_DAYS روز حذف می‌شود.
    هنگام هر آپلود تازه اجرا می‌شود؛ بدون نیاز به cron یا سرویس جدا."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    if not UPLOAD_DIR.exists():
        return
    for job_dir in UPLOAD_DIR.iterdir():
        status_path = job_dir / "status.json"
        try:
            with open(status_path, encoding="utf-8") as f:
                created_at = datetime.datetime.fromisoformat(json.load(f)["created_at"])
            if created_at < cutoff:
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            continue


def _job_dir(job_id):
    # جلوگیری از path traversal: job_id همیشه یک uuid.hex ساده است.
    if not job_id or not job_id.isalnum():
        return None
    d = UPLOAD_DIR / job_id
    return d if d.is_dir() else None


@app.route("/api/bulk/template")
def bulk_template():
    from pipeline.bulk_template import build_template_workbook
    buf = build_template_workbook()
    return send_file(
        buf,
        as_attachment=True,
        download_name="قالب_جستجوی_گروهی.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/bulk/prepare", methods=["POST"])
def bulk_prepare():
    _cleanup_old_jobs()

    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "فایلی انتخاب نشده است."}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in (".xlsx", ".csv"):
        return jsonify({"error": f"پسوند «{ext}» پشتیبانی نمی‌شود؛ فقط xlsx. و csv. مجاز است."}), 400

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    input_filename = f"input{ext}"
    f.save(job_dir / input_filename)

    try:
        headers, rows = read_rows(job_dir / input_filename)
        validate_rows(headers, rows)
    except ValidationError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": f"خطا در خواندن فایل: {e}"}), 400

    status = {
        "job_id": job_id,
        "original_filename": f.filename,
        "input_filename": input_filename,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "staged",
        "row_count": len(rows),
        "processed": 0,
        "mode": None,
        "error": None,
        "result_filename": None,
    }
    with open(job_dir / "status.json", "w", encoding="utf-8") as fp:
        json.dump(status, fp, ensure_ascii=False, indent=2)

    return jsonify({"job_id": job_id, "row_count": len(rows)})


@app.route("/api/bulk/start", methods=["POST"])
def bulk_start():
    payload = request.get_json(force=True, silent=True) or {}
    job_dir = _job_dir(payload.get("job_id"))
    mode = payload.get("mode") or MODE_SMART

    if job_dir is None:
        return jsonify({"error": "این job پیدا نشد؛ شاید فایل را دوباره باید آپلود کنید."}), 404
    if mode not in (MODE_FAST, MODE_SMART):
        return jsonify({"error": "حالت نامعتبر است."}), 400
    if mode == MODE_SMART and not OPENAI_API_KEY:
        return jsonify({"error": "برای «هوشمند»، کلید OpenAI در .env لازم است."}), 400

    status_path = job_dir / "status.json"
    with open(status_path, encoding="utf-8") as fp:
        status = json.load(fp)
    if status["status"] != "staged":
        return jsonify({"error": f"این job قبلاً شروع شده (وضعیت: {status['status']})."}), 400

    status["mode"] = mode
    status["status"] = "queued"
    with open(status_path, "w", encoding="utf-8") as fp:
        json.dump(status, fp, ensure_ascii=False, indent=2)

    log_path = job_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.Popen(
            [sys.executable, str(BULK_SCRIPT), str(job_dir)],
            cwd=str(BASE_DIR), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return jsonify({"status": "queued"})


@app.route("/api/bulk/status/<job_id>")
def bulk_status(job_id):
    job_dir = _job_dir(job_id)
    if job_dir is None:
        return jsonify({"error": "این job پیدا نشد."}), 404
    with open(job_dir / "status.json", encoding="utf-8") as fp:
        return jsonify(json.load(fp))


@app.route("/api/bulk/result/<job_id>")
def bulk_result(job_id):
    job_dir = _job_dir(job_id)
    if job_dir is None:
        return jsonify({"error": "این job پیدا نشد."}), 404
    with open(job_dir / "status.json", encoding="utf-8") as fp:
        status = json.load(fp)
    if status["status"] != "done":
        return jsonify({"error": f"هنوز آماده نیست (وضعیت: {status['status']})."}), 400

    original_stem = Path(status["original_filename"]).stem
    download_name = f"نتیجه_{original_stem}.xlsx"
    return send_file(
        job_dir / status["result_filename"],
        as_attachment=True,
        download_name=download_name,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
