"""
ابزارهای مشترک پایپ‌لاین
------------------------------------
شامل:
  - load_records: خواندن CSV/XLSX با پشتیبانی از هر دو ساختار ستونی موجود در پروژه
  - JsonlCheckpoint: ذخیره‌ی افزایشی نتایج برای قابلیت Resume
  - batched: تقسیم لیست به دسته‌های کوچک
  - with_retry: تلاش مجدد در صورت خطای شبکه/API با تأخیر فزاینده
"""
import csv
import json
import os
import time
from pathlib import Path


def load_records(path):
    """خواندن رکوردهای کالا از CSV یا XLSX.

    دو نوع ساختار ستونی که در این پروژه دیده شده پشتیبانی می‌شود:
      - ID, DescriptionOfID, Type  (نمونه شناسه‌های اختصاصی)
      - StuffCod, Description, type (شناسه‌های عمومی)
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    rows = []

    def pick(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return v
        return ""

    if ext in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(c).strip() if c else "" for c in next(rows_iter)]
        for row in rows_iter:
            rec = dict(zip(headers, row))
            rid = pick(rec, "StuffCod", "ID", "id")
            desc = pick(rec, "Description", "DescriptionOfID", "desc")
            typ = pick(rec, "type", "Type")
            if rid and desc:
                rows.append({"id": str(rid).strip(), "desc": str(desc).strip(), "type": str(typ).strip()})
    else:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = pick(row, "StuffCod", "ID", "id")
                desc = pick(row, "Description", "DescriptionOfID", "desc")
                typ = pick(row, "type", "Type")
                if rid and desc:
                    rows.append({"id": str(rid).strip(), "desc": str(desc).strip(), "type": str(typ).strip()})

    return rows


class JsonlCheckpoint:
    """ذخیره‌ی افزایشی نتایج برای قابلیت Resume.

    هر رکورد پردازش‌شده به‌صورت یک خط JSON به فایل اضافه می‌شود. اگر برنامه
    به هر دلیلی (قطع اینترنت، Timeout، بسته‌شدن، ری‌استارت سیستم) متوقف شود،
    با اجرای دوباره‌ی همان اسکریپت، رکوردهایی که کلیدشان از قبل در فایل است
    دوباره پردازش نمی‌شوند.
    """

    def __init__(self, path, key_field="id"):
        self.path = Path(path)
        self.key_field = key_field
        self.done = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.done.add(rec[self.key_field])
                    except Exception:
                        continue

    def is_done(self, key):
        return key in self.done

    def append(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.done.add(record[self.key_field])

    def load_all(self):
        records = []
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def with_retry(fn, retries=5, base_delay=2):
    """اجرای fn با تلاش مجدد در صورت بروز خطا (شبکه، Timeout، Rate limit و ...)."""
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"    ⚠️  خطا: {e} — تلاش مجدد در {wait} ثانیه... ({attempt + 1}/{retries})")
            time.sleep(wait)
    raise last_err
