"""
کاتالوگ شرح‌های یکتای کالا
------------------------------------
داده‌ی خام data/taxonomy.xlsx (شیت Sheet2) شامل ~50700 سطر است، اما همان
شرح می‌تواند زیر چند شناسه (تولید داخل ۲۷۲۰ / وارداتی ۲۷۱۰) تکرار شده
باشد. جستجو باید روی شرح‌های یکتا (~23700 مورد) انجام شود، نه سطرها.

نتیجه‌ی پردازش در checkpoints/catalog.pkl کش می‌شود و فقط وقتی xlsx از
کش جدیدتر باشد دوباره ساخته می‌شود.
"""
import pickle
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import openpyxl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

XLSX_PATH = DATA_DIR / "taxonomy.xlsx"
SHEET_NAME = "Sheet2"
CACHE_PATH = CHECKPOINT_DIR / "catalog.pkl"

DOMESTIC_PREFIX = "2720"
IMPORTED_PREFIX = "2710"

CatalogItem = namedtuple(
    "CatalogItem",
    [
        "description",
        "level1",
        "level2",
        "level3",
        "level4",
        "domestic_id",
        "imported_id",
        "all_ids",
        "type_strings",
    ],
)


def _clean(v):
    if v is None:
        return ""
    return str(v).strip()


def _build_catalog(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME]
    rows = ws.iter_rows(values_only=True)
    headers = [_clean(c) for c in next(rows)]

    by_desc = defaultdict(list)
    for row in rows:
        rec = dict(zip(headers, row))
        desc = _clean(rec.get("DescriptionOfID"))
        rid = _clean(rec.get("ID"))
        if not desc or not rid:
            continue
        by_desc[desc].append(rec)

    items = []
    for desc, recs in by_desc.items():
        path_counts = Counter(
            (_clean(r.get("level 1")), _clean(r.get("level 2")),
             _clean(r.get("level 3")), _clean(r.get("level 4")))
            for r in recs
        )
        level1, level2, level3, level4 = path_counts.most_common(1)[0][0]

        all_ids = sorted({_clean(r.get("ID")) for r in recs})
        domestic_id = next((i for i in all_ids if i.startswith(DOMESTIC_PREFIX)), None)
        imported_id = next((i for i in all_ids if i.startswith(IMPORTED_PREFIX)), None)
        type_strings = tuple(sorted({_clean(r.get("Type")) for r in recs if _clean(r.get("Type"))}))

        items.append(CatalogItem(
            description=desc,
            level1=level1,
            level2=level2,
            level3=level3,
            level4=level4,
            domestic_id=domestic_id,
            imported_id=imported_id,
            all_ids=tuple(all_ids),
            type_strings=type_strings,
        ))

    return items


def load_catalog(xlsx_path=None, cache_path=None, force_rebuild=False):
    """بارگذاری کاتالوگ شرح‌های یکتا؛ در صورت امکان از کش پیکل استفاده می‌شود."""
    xlsx_path = Path(xlsx_path) if xlsx_path else XLSX_PATH
    cache_path = Path(cache_path) if cache_path else CACHE_PATH

    if not force_rebuild and cache_path.exists():
        if cache_path.stat().st_mtime >= xlsx_path.stat().st_mtime:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    items = _build_catalog(xlsx_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(items, f)

    return items
