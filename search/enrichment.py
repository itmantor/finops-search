"""
بارگذاری غنی‌سازی کاتالوگ (pipeline/enrich_catalog.py) و ساخت متن ایندکس
------------------------------------
دو خروجی پایپ‌لاین غنی‌سازی را با هم ترکیب می‌کند:
  checkpoints/catalog_enrichment.jsonl  — canonical, synonyms, uses (پاس ۱)
  checkpoints/catalog_examples.jsonl    — examples نهایی، پس از بازبینی (پاس ۲)

و برای هر کانال یک متن متفاوت می‌سازد — این تفاوت عمدی است:
  متن واژگانی (BM25)  = شرح + مسیر تاکسونومی + canonical + synonyms + [examples] + uses
                         (شبکه‌ی توکن گسترده؛ examples اختیاری تا اثرش جدا سنجیده شود)
  متن معنایی (Embedding) = شرح + canonical فقط
                         (یک عبارت یکپارچه؛ سند طولانی بردار را رقیق می‌کند)
"""
import hashlib
import json

from common.catalog import load_catalog
from common.config import CHECKPOINT_DIR

PASS1_PATH = CHECKPOINT_DIR / "catalog_enrichment.jsonl"
PASS2_PATH = CHECKPOINT_DIR / "catalog_examples.jsonl"
MARKET_PATH = CHECKPOINT_DIR / "catalog_enrichment_cosmetics.jsonl"  # نگاه کنید به pipeline/reenrich_cosmetics.py


def desc_hash(description):
    return hashlib.sha1(description.encode("utf-8")).hexdigest()


def taxonomy_path(item):
    parts = [p for p in (item.level1, item.level2, item.level3, item.level4) if p]
    return " > ".join(parts)


def load_enrichment(pass1_path=None, pass2_path=None, examples_source="reviewed", market_path=None):
    """نگاشت desc_hash -> {"canonical", "synonyms", "uses", "examples"}.

    examples_source="reviewed" (پیش‌فرض): examples از پاس ۲ (catalog_examples.jsonl،
    بازبینی‌شده) می‌آید؛ برای شرح‌هایی که هنوز پاس ۲ روی آن‌ها اجرا نشده، خالی می‌ماند.
    examples_source="raw": پاس ۲ اصلاً خوانده نمی‌شود؛ examples همان examples_raw
    پاس ۱ (بدون بازبینی) است — برای سنجش اینکه آیا examples اصلاً کمک می‌کند،
    بدون نیاز به تکمیل پاس ۲.

    market_path (پیش‌فرض checkpoints/catalog_enrichment_cosmetics.jsonl، اگر
    وجود داشته باشد): واژگان بازاری غنی‌سازی هدفمندِ یک شاخه‌ی خاص (نگاه کنید
    به pipeline/reenrich_cosmetics.py) — به‌جای جای‌گزینی، به synonyms پاس ۱
    همان شرح افزوده می‌شود (اضافه، نه حذف؛ فقط برای شرح‌هایی که در این فایل
    هستند، بقیه‌ی کاتالوگ دست‌نخورده می‌ماند)."""
    pass1_path = pass1_path or PASS1_PATH
    pass2_path = pass2_path or PASS2_PATH
    market_path = market_path or MARKET_PATH

    enrichment = {}
    if pass1_path.exists():
        with open(pass1_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                enrichment[rec["desc_hash"]] = {
                    "canonical": rec.get("canonical", ""),
                    "synonyms": rec.get("synonyms", []),
                    "uses": rec.get("uses", []),
                    "examples": rec.get("examples_raw", []) if examples_source == "raw" else [],
                }

    if examples_source == "reviewed" and pass2_path.exists():
        with open(pass2_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = rec["desc_hash"]
                enrichment.setdefault(h, {"canonical": "", "synonyms": [], "uses": [], "examples": []})
                enrichment[h]["examples"] = rec.get("examples", [])

    if market_path.exists():
        with open(market_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = rec["desc_hash"]
                if h not in enrichment:
                    continue
                existing = set(enrichment[h]["synonyms"])
                for s in rec.get("synonyms_market", []):
                    if s not in existing:
                        enrichment[h]["synonyms"].append(s)
                        existing.add(s)

    return enrichment


def lexical_text(item, rec, include_examples=True):
    """متن ایندکس واژگانی: شرح + مسیر + canonical + synonyms + [examples] + uses."""
    parts = [item.description, taxonomy_path(item), rec.get("canonical", "")]
    parts.extend(rec.get("synonyms", []))
    if include_examples:
        parts.extend(rec.get("examples", []))
    parts.extend(rec.get("uses", []))
    return " ".join(p for p in parts if p)


def semantic_text(item, rec):
    """متن ایندکس معنایی: شرح + canonical فقط (بدون رقیق‌شدن بردار)."""
    canonical = rec.get("canonical", "")
    return f"{item.description} {canonical}".strip() if canonical else item.description


def make_lexical_text_fn(enrichment, include_examples):
    def text_fn(item):
        rec = enrichment.get(desc_hash(item.description))
        if not rec:
            return item.description
        return lexical_text(item, rec, include_examples=include_examples)
    return text_fn


def make_semantic_text_fn(enrichment):
    def text_fn(item):
        rec = enrichment.get(desc_hash(item.description))
        if not rec:
            return item.description
        return semantic_text(item, rec)
    return text_fn


if __name__ == "__main__":
    enrichment = load_enrichment()
    items = load_catalog()
    covered = sum(1 for item in items if desc_hash(item.description) in enrichment)
    print(f"{covered:,}/{len(items):,} شرح دارای غنی‌سازی است.")
