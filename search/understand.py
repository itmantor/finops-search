"""
فهم پرس‌وجو (Query Understanding)
------------------------------------
پرس‌وجوی خام و نویزدار کاربر را با یک فراخوانی CHAT_MODEL به یک شیء
ساخت‌یافته تبدیل می‌کند: برند، عبارت پاک‌شده (بدون برند/عدد/واحد/رنگ/طعم/
پرانتز) و مجموعه‌ای از کلمات کلیدی بازار فارسی.

نتیجه هم در حافظه و هم در checkpoints/query_understanding.jsonl (کلید شده
با متن خام پرس‌وجو) کش می‌شود تا اجراهای تکراری محک هزینه‌ی دوباره نداشته
باشند.
"""
import json

from common.config import CHAT_MODEL, CHECKPOINT_DIR, OPENAI_API_KEY
from common.utils import with_retry

CACHE_PATH = CHECKPOINT_DIR / "query_understanding.jsonl"

SYSTEM_PROMPT = """تو یک دستیار فهم پرس‌وجوی جستجوی کالا در یک کاتالوگ فارسی هستی.
پرس‌وجوی خام کاربر را بگیر و آن را به یک شیء JSON با دقیقاً این ساختار تبدیل کن:

{"brand": "...", "clean": "...", "keywords": ["...", "..."]}

قوانین (همه مهم هستند):
- خروجی باید کاملاً فارسی باشد. کاتالوگ فارسی است؛ خروجی انگلیسی همه چیز را خراب می‌کند.
- "brand": نام برند شناسایی‌شده در پرس‌وجو، یا رشته‌ی خالی "" اگر برندی نبود.
- "clean": نام عمومی محصول، فقط فارسی، بدون برند، بدون عدد/واحد/تعداد/سایز/رنگ/طعم و بدون هر چیزی داخل پرانتز.
  نمونه‌ی نویزهایی که باید حذف شوند: "رویوال"، "۲۴ ساعته"، "سایز اسمال"، "۷۵ سانتی متر"، "۳ عددی"، "( وانیل )"، "( فاقد پودر )".
- اگر کاربر به‌جای نام محصول، عملکرد آن را توصیف کرده (مثلاً "یه چیزی که میخوریم و سردرد را خوب میکند")،
  دسته‌ی محصول و نام استاندارد فارسی آن را استنباط کن.
- "keywords": ۵ تا ۱۵ عبارت رایج بازار فارسی شامل مترادف‌ها، کلمات دسته‌بندی و نام‌های متداول.
- برای داروها، حوزه‌ی درمانی و شکل دارویی را هم در keywords بیاور (مثلاً: مسکن، ضد درد، سیستم عصبی، قرص، خوراکی).
- محافظه‌کار باش: عبارت گسترده‌تر و درست را به عبارت اختراعی و بیش‌ازحد خاص ترجیح بده.

فقط شیء JSON را برگردان، بدون هیچ توضیح یا متن اضافه."""


class QueryUnderstander:
    def __init__(self, cache_path=None):
        self.cache_path = cache_path or CACHE_PATH
        self._cache = {}
        self._client = None
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._cache[rec["query"]] = rec["result"]

    def _append_cache(self, query, result):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"query": query, "result": result}, ensure_ascii=False) + "\n")

    def understand(self, query):
        """پرس‌وجوی خام را به {"brand", "clean", "keywords"} تبدیل می‌کند؛ نتیجه کش می‌شود."""
        if query in self._cache:
            return self._cache[query]

        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)

        def call():
            return self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

        resp = with_retry(call)
        try:
            result = json.loads(resp.choices[0].message.content)
        except Exception:
            result = {"brand": "", "clean": query, "keywords": []}

        result.setdefault("brand", "")
        result.setdefault("clean", query)
        result.setdefault("keywords", [])

        self._cache[query] = result
        self._append_cache(query, result)
        return result

    def lexical_query(self, query):
        """پرس‌وجوی مناسب کانال واژگانی: clean + همه‌ی کلیدواژه‌ها (شبکه‌ی توکن گسترده برای BM25)."""
        u = self.understand(query)
        parts = [u["clean"]] + list(u["keywords"])
        return " ".join(p for p in parts if p)

    def semantic_query(self, query):
        """پرس‌وجوی مناسب کانال معنایی: فقط عبارت پاک‌شده (بردار یکپارچه، بدون رقیق‌شدن)."""
        u = self.understand(query)
        return u["clean"] or query
