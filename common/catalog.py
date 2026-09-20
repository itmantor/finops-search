"""
کاتالوگ شرح‌های یکتای کالا
------------------------------------
داده‌ی خام data/taxonomy.xlsx (شیت Sheet2) شامل ~50700 سطر است، اما همان
شرح می‌تواند زیر چند شناسه (تولید داخل ۲۷۲۰ / وارداتی ۲۷۱۰) تکرار شده
باشد. جستجو باید روی شرح‌های یکتا (~23700 مورد) انجام شود، نه سطرها.

یک شرح می‌تواند بیش از یک شناسه‌ی تولید داخل یا وارداتی داشته باشد؛ همه‌ی
آن‌ها در domestic_ids/imported_ids نگه داشته می‌شوند (نه فقط اولی)، چون
این شناسه‌ها روی فاکتور مالیاتی درج می‌شوند و جا انداختن‌شان ریسک واقعی
دارد. طبقه‌بندی هر شناسه بر اساس ستون Type انجام می‌شود (مرجع اصلی)؛
پیشوند عددی (۲۷۲۰/۲۷۱۰) فقط وقتی Type خالی/نامعتبر باشد به‌عنوان جایگزین
استفاده می‌شود.

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


class CatalogItem(namedtuple(
    "CatalogItem",
    [
        "description",
        "level1",
        "level2",
        "level3",
        "level4",
        "domestic_ids",
        "imported_ids",
        "all_ids",
        "type_strings",
    ],
)):
    __slots__ = ()

    @property
    def domestic_id(self):
        """شناسه‌ی تولید داخل اول، برای سازگاری با کدهای قدیمی؛ کد جدید باید domestic_ids را بخواند."""
        return self.domestic_ids[0] if self.domestic_ids else None

    @property
    def imported_id(self):
        """شناسه‌ی وارداتی اول، برای سازگاری با کدهای قدیمی؛ کد جدید باید imported_ids را بخواند."""
        return self.imported_ids[0] if self.imported_ids else None


def _clean(v):
    if v is None:
        return ""
    return str(v).strip()


def _classify_ids(recs):
    """طبقه‌بندی شناسه‌های یک شرح به تولید داخل/وارداتی، بر اساس ستون Type
    (مرجع اصلی و معتبر). فقط وقتی Type برای یک سطر خالی یا ناشناخته باشد،
    به پیشوند عددی شناسه (۲۷۲۰/۲۷۱۰) به‌عنوان جایگزین رجوع می‌شود. شناسه‌های
    خدماتی (Type = «شناسه عمومی خدمت») در هیچ‌کدام قرار نمی‌گیرند."""
    domestic, imported = set(), set()
    for r in recs:
        rid = _clean(r.get("ID"))
        if not rid:
            continue
        typ = _clean(r.get("Type"))
        if "وارداتی" in typ:
            imported.add(rid)
        elif "تولید داخل" in typ:
            domestic.add(rid)
        elif rid.startswith(DOMESTIC_PREFIX):
            domestic.add(rid)
        elif rid.startswith(IMPORTED_PREFIX):
            imported.add(rid)
    return tuple(sorted(domestic)), tuple(sorted(imported))


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

        all_ids = tuple(sorted({_clean(r.get("ID")) for r in recs}))
        domestic_ids, imported_ids = _classify_ids(recs)
        type_strings = tuple(sorted({_clean(r.get("Type")) for r in recs if _clean(r.get("Type"))}))

        items.append(CatalogItem(
            description=desc,
            level1=level1,
            level2=level2,
            level3=level3,
            level4=level4,
            domestic_ids=domestic_ids,
            imported_ids=imported_ids,
            all_ids=all_ids,
            type_strings=type_strings,
        ))

    return items


def load_catalog(xlsx_path=None, cache_path=None, force_rebuild=False):
    """بارگذاری کاتالوگ شرح‌های یکتا؛ در صورت امکان از کش پیکل استفاده می‌شود."""
    xlsx_path = Path(xlsx_path) if xlsx_path else XLSX_PATH
    cache_path = Path(cache_path) if cache_path else CACHE_PATH

    if not force_rebuild and cache_path.exists():
        if not xlsx_path.exists() or cache_path.stat().st_mtime >= xlsx_path.stat().st_mtime:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    items = _build_catalog(xlsx_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(items, f)

    return items
