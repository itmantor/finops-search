#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سرور جستجوی شناسه کالا — نسخه V2
------------------------------------
- جستجوی متنی رایگان (substring)
- جستجوی معنایی با FAISS
- جستجوی ترکیبی: 70% امتیاز معنایی + 30% امتیاز کلیدواژه
- GPT ReRank اختیاری روی ۵۰ نتیجه برتر برای انتخاب و مرتب‌سازی ۱۰ مورد نهایی

اجرا:
    python server/app.py
سپس مرورگر را به http://localhost:8000 ببرید (یا پورت تنظیم‌شده در .env).
"""
import json
import os
import pickle
import re
import sys
import webbrowser
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

from common.config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY, OUTPUT_DIR, PORT

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INDEX_PATH = OUTPUT_DIR / "faiss.index"
META_PATH = OUTPUT_DIR / "metadata.pkl"

INDEX = None
META = None

EXPAND_CACHE = {}

EXPAND_PROMPT = """شما یک دستیار متخصص در شناسایی کالاهای صنعتی و تجاری در بازار ایران هستید.
عبارت جستجوی کوتاه زیر را به قالب زیر گسترش بده تا با ساختار متنی که برای ایندکس‌کردن کالاها استفاده شده هم‌خوان باشد:

نام کالا: <نام کامل و رایج کالا>

دسته اصلی: <دسته‌بندی سطح بالا>
زیرگروه: <زیرگروه تخصصی‌تر>
رده تخصصی: <رده دقیق‌تر، در صورت وجود>

کلمات مرتبط:
<چند مترادف یا اصطلاح رایج بازار ایران برای همین کالا، هرکدام در یک خط>

قوانین مهم (حتماً رعایت شود):
- فقط از مترادف‌ها، نام‌های رایج و دسته‌بندی‌های واقعی و متداول در بازار ایران استفاده کن.
- هیچ توضیح خلاقانه، جزئیات ساختگی یا حدس دور از واقعیت اضافه نکن.
- اگر کالا را با اطمینان نمی‌شناسی، یک دسته‌بندی کلی‌تر و امن‌تر انتخاب کن، نه جزئیات نامطمئن.
- خروجی باید کاملاً به زبان فارسی و دقیقاً با همین قالب (بدون توضیح اضافه، بدون مقدمه) باشد.

عبارت جستجو: {query}"""


def expand_query(query):
    """عبارت کوتاه کاربر را با یک فراخوانی GPT به قالب متن غنی‌شده اسناد (مرحله ۵ پایپ‌لاین) تبدیل می‌کند."""
    if query in EXPAND_CACHE:
        return EXPAND_CACHE[query]
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": EXPAND_PROMPT.format(query=query)}],
        )
        expanded = (resp.choices[0].message.content or "").strip()
        if not expanded:
            return None
    except Exception as e:
        print(f"⚠️  خطا در گسترش عبارت جستجو: {e}")
        return None
    EXPAND_CACHE[query] = expanded
    return expanded


def load_index():
    global INDEX, META
    if INDEX_PATH.exists() and META_PATH.exists():
        INDEX = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            META = pickle.load(f)
        print(f"✅ ایندکس با {len(META['ids']):,} رکورد بارگذاری شد.")
    else:
        print("⚠️  ایندکس FAISS پیدا نشد. ابتدا کل پایپ‌لاین را اجرا کنید: python pipeline/run_all.py")


load_index()


def keyword_score(query, desc):
    query = (query or "").strip()
    desc = desc or ""
    if not query:
        return 0.0
    if query in desc:
        return 1.0 if desc.startswith(query) else 0.7
    q_words = set(query.split())
    d_words = set(desc.split())
    if not q_words:
        return 0.0
    return len(q_words & d_words) / len(q_words)


def passes_filter(i, type_filter):
    if type_filter == "all":
        return True
    typ = META["types"][i] if META.get("types") else ""
    if type_filter == "general":
        return "عمومی" in typ
    if type_filter == "specific":
        return "اختصاصی" in typ
    return True


def make_record(i):
    return {
        "id": META["ids"][i],
        "desc": META["descs"][i],
        "type": META["types"][i] if META.get("types") else "",
        "level1": META["level1"][i] if META.get("level1") else "",
    }


def gpt_rerank(query, candidates):
    """از میان نامزدها، GPT بهترین ۱۰ مورد را انتخاب و بر اساس ارتباط مرتب می‌کند."""
    items = [{"idx": n, "desc": META["descs"][i]} for n, (_, i, _) in enumerate(candidates)]
    prompt = (
        f"عبارت جستجو: {query}\n\n"
        "کالاهای زیر را بر اساس ارتباط معنایی واقعی با عبارت جستجو ارزیابی کن، "
        "بهترین ۱۰ مورد را انتخاب و به ترتیب ارتباط مرتب کن.\n"
        "فقط یک آرایه JSON از عدد idx ها برگردان، مثل: [3, 7, 1]\n\n"
        + "\n".join(f"{it['idx']}: {it['desc']}" for it in items)
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
        match = re.search(r"\[[\d,\s]+\]", text)
        order = json.loads(match.group(0)) if match else []
        reranked = [candidates[n] for n in order if 0 <= n < len(candidates)]
        return reranked or candidates[:10]
    except Exception as e:
        print(f"⚠️  خطا در GPT ReRank: {e}")
        return candidates[:10]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "record_count": len(META["ids"]) if META else 0,
        "has_index": INDEX is not None,
        "has_api_key": bool(OPENAI_API_KEY),
    })


@app.route("/api/search", methods=["POST"])
def search():
    payload = request.get_json(force=True, silent=True) or {}
    query = (payload.get("query") or "").strip()
    mode = payload.get("mode", "text")
    type_filter = payload.get("type_filter", "all")
    rerank = bool(payload.get("rerank", False))

    if not query:
        return jsonify({"results": []})

    if META is None or INDEX is None:
        return jsonify({"error": "ایندکس آماده نیست. ابتدا پایپ‌لاین را اجرا کنید (python pipeline/run_all.py)."})

    if mode == "text":
        results = []
        for i, desc in enumerate(META["descs"]):
            if query in desc and passes_filter(i, type_filter):
                results.append(make_record(i))
        return jsonify({"results": results[:50]})

    # mode == semantic : ابتدا تطابق متنی دقیق، سپس تکمیل با جستجوی معنایی/ترکیبی (۷۰٪ معنایی + ۳۰٪ کلیدواژه)
    if client is None:
        return jsonify({"error": "برای جستجوی معنایی، کلید OpenAI در .env لازم است."})

    exact_starts, exact_rest, exact_idxs = [], [], set()
    for i, desc in enumerate(META["descs"]):
        if query in desc and passes_filter(i, type_filter):
            exact_idxs.add(i)
            (exact_starts if desc.startswith(query) else exact_rest).append(i)

    exact_results = []
    for i in (exact_starts + exact_rest)[:20]:
        rec = make_record(i)
        rec["score"] = 100
        rec["match_type"] = "exact"
        exact_results.append(rec)

    remaining = 20 - len(exact_results)
    expanded_query = None
    semantic_results = []

    if remaining > 0:
        expanded_query = expand_query(query)
        embed_input = expanded_query or query

        try:
            qvec = np.array(
                client.embeddings.create(model=EMBED_MODEL, input=[embed_input]).data[0].embedding,
                dtype="float32",
            ).reshape(1, -1)
        except Exception as e:
            return jsonify({"error": f"خطا در ارتباط با OpenAI: {e}"})

        faiss.normalize_L2(qvec)
        scores, idxs = INDEX.search(qvec, 50)

        combined = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or i in exact_idxs or not passes_filter(i, type_filter):
                continue
            semantic = float(score)
            kw = keyword_score(query, META["descs"][i])
            final = 0.7 * semantic + 0.3 * kw
            combined.append((final, i, semantic))

        combined.sort(key=lambda x: x[0], reverse=True)
        top = combined[:20]

        if rerank and top:
            top = gpt_rerank(query, top)

        for final, i, semantic in top:
            rec = make_record(i)
            rec["score"] = round(final * 100)
            rec["match_type"] = "semantic"
            semantic_results.append(rec)

    results = exact_results + semantic_results[:remaining]

    return jsonify({"results": results, "expanded_query": expanded_query or ""})


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", PORT)), debug=False)
