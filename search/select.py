"""
انتخاب نهایی از میان کاندیدهای بازیابی‌شده
------------------------------------
مرحله‌ی آخر زنجیره: فهم پرس‌وجو -> بازیابی هیبرید (تا ۴۰ کاندید) -> این
مرحله. یک فراخوانی CHAT_MODEL می‌گیرد که فقط از میان کاندیدهای داده‌شده
(با اندیس) انتخاب می‌کند؛ هرگز شرح یا شناسه‌ی خودش را نمی‌سازد — این تنها
راهی است که کاتالوگ ۲۳٬۷۴۲ موردی بدون ریسک هالوسینیشن شناسه‌ی مالیاتی
جواب بدهد (شناسه‌ها همیشه از خود CatalogItem کاندید انتخاب‌شده گرفته
می‌شوند، نه از متن تولیدشده‌ی مدل).

اگر هیچ کاندیدی درست نباشد، index باید null برگردد (نه حدس زدن). alternates
فقط وقتی چند کاندید واقعاً نزدیک/مبهم‌اند پر می‌شود، نه به‌صورت پیش‌فرض.

در صورت شکست فراخوانی (بعد از تلاش مجدد) یا پاسخ نامعتبر/غیرقابل‌تجزیه،
به‌جای کرش، رتبه‌ی اول بازیابی با confidence="low" و fallback=True
برگردانده می‌شود.
"""
import hashlib
import json

from common.config import CHAT_MODEL, CHECKPOINT_DIR, OPENAI_API_KEY
from common.utils import with_retry

CACHE_PATH = CHECKPOINT_DIR / "selection_cache.jsonl"
MAX_CANDIDATES = 40
CONFIDENCE_LEVELS = ("high", "medium", "low")

SYSTEM_PROMPT = """تو یک دستیار انتخاب شناسه‌ی کالا از یک کاتالوگ مالیاتی فارسی هستی.
به تو پرس‌وجوی خام کاربر و فهرستی شماره‌گذاری‌شده از حداکثر ۴۰ شرح کاندید
(خروجی یک موتور بازیابی) داده می‌شود.

وظیفه‌ی تو فقط انتخاب است، نه تولید:
- هرگز شرح یا شناسه‌ی جدید نساز. فقط از میان اندیس‌های داده‌شده انتخاب کن.
- بهترین تطبیق را با شماره‌ی همان گزینه در فهرست برگردان.
- اگر هیچ‌کدام از کاندیدها واقعاً با پرس‌وجو تطبیق ندارد، "index" را null
  بگذار — حدس‌زدن از میان گزینه‌های نامرتبط بدتر از اعتراف به نبود جواب است.
- "confidence": "high" اگر مطمئنی، "medium" اگر قابل‌قبول ولی با کمی
  عدم‌قطعیت، "low" اگر صرفاً حدس بهترین گزینه‌ی موجود است.
- "alternates": حداکثر ۲ شماره‌ی دیگر، فقط وقتی واقعاً بین چند کاندید نزدیک
  مردد هستی (مثلاً دو شرح تقریباً یکسان با تفاوت جزئی). در غیر این صورت
  آرایه‌ی خالی بده؛ alternates پرکردنِ همیشگی نیست.

خروجی را دقیقاً به این شکل JSON برگردان، بدون هیچ توضیح اضافه:
{"index": <عدد یا null>, "confidence": "high|medium|low", "alternates": [<عدد>, ...]}"""


def _cache_key(query, candidates):
    desc_key = "|".join(item.description for item, _score in candidates)
    raw = query + "␟" + desc_key
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _candidate_payload(item, index):
    return {
        "index": index,
        "rank": index + 1,
        "description": item.description,
        "domestic_ids": list(item.domestic_ids),
        "imported_ids": list(item.imported_ids),
    }


def _no_match(confidence, alternates=None):
    return {
        "index": None, "rank": None,
        "description": None, "domestic_ids": [], "imported_ids": [],
        "confidence": confidence, "alternates": alternates or [], "fallback": False,
    }


def _fallback(candidates, error=None):
    """در صورت شکست کامل، رتبه‌ی اول بازیابی را با اطمینان پایین برمی‌گرداند."""
    if not candidates:
        result = _no_match("low")
    else:
        result = _candidate_payload(candidates[0][0], 0)
        result["confidence"] = "low"
        result["alternates"] = []
    result["fallback"] = True
    result["error"] = str(error) if error else None
    return result


class Selector:
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
                    self._cache[rec["key"]] = rec["result"]

    def _append_cache(self, key, result):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")

    def select(self, query, candidates):
        """candidates: [(CatalogItem, score), ...] به ترتیب رتبه (رتبه‌ی ۱ اول).

        دیکشنری برمی‌گرداند: index/rank/description/domestic_ids/imported_ids
        برای گزینه‌ی نهایی (یا None اگر index=null)، به‌همراه confidence،
        alternates (لیستی با همان ساختار) و fallback (True اگر به‌خاطر شکست
        فراخوانی/پاسخ نامعتبر، رتبه‌ی اول بازیابی برگردانده شده).
        """
        candidates = list(candidates)[:MAX_CANDIDATES]
        if not candidates:
            return _fallback(candidates)

        key = _cache_key(query, candidates)
        if key in self._cache:
            return self._cache[key]

        lines = [f"{i + 1}. {item.description}" for i, (item, _score) in enumerate(candidates)]
        user_content = f"پرس‌وجو: {query}\n\nکاندیدها:\n" + "\n".join(lines)

        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)

        def call():
            return self._client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

        try:
            resp = with_retry(call)
            parsed = json.loads(resp.choices[0].message.content)
            result = self._normalize(parsed, candidates)
        except Exception as e:
            result = _fallback(candidates, error=e)

        self._cache[key] = result
        self._append_cache(key, result)
        return result

    @staticmethod
    def _normalize(parsed, candidates):
        n = len(candidates)

        def to_zero_based(v):
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            return v - 1 if 1 <= v <= n else None

        confidence = parsed.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            confidence = "low"

        idx = to_zero_based(parsed.get("index"))

        alt_indices = []
        for v in (parsed.get("alternates") or [])[:2]:
            zi = to_zero_based(v)
            if zi is not None and zi != idx and zi not in alt_indices:
                alt_indices.append(zi)
        alternates = [_candidate_payload(candidates[zi][0], zi) for zi in alt_indices]

        if idx is None:
            return _no_match(confidence, alternates)

        result = _candidate_payload(candidates[idx][0], idx)
        result["confidence"] = confidence
        result["alternates"] = alternates
        result["fallback"] = False
        return result
