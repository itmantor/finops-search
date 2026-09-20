# راهنمای ابزار جستجوی شناسه کالا و خدمت

این ابزار شرح آزاد کاربر را می‌گیرد و شناسه‌های تولید داخل و وارداتی
کاتالوگ مالیاتی را پیدا می‌کند. دو حالت جستجو دارد:

- **جستجوی سریع** — فقط کانال واژگانی (BM25)، بدون فراخوانی OpenAI، فوری
  و رایگان. حالت پیش‌فرض.
- **جستجوی هوشمند** — ابتدا فهم پرس‌وجو (برند/عبارت پاک‌شده/کلیدواژه‌ها)،
  سپس بازیابی هیبرید (BM25 + جستجوی معنایی با FAISS، ترکیب با RRF). چند
  ثانیه طول می‌کشد و نیاز به کلید OpenAI دارد.

هر جستجو (پرس‌وجوی خام، حالت، و نتایج نمایش‌داده‌شده) در `logs/search_log.jsonl`
ثبت می‌شود — این داده‌ی واقعی کاربران است و منبع اصلی برای بهبود جستجو در
آینده خواهد بود (خیلی معتبرتر از ۱۷ مورد دستی `benchmark.json`).

> نکته: مرحله‌ی انتخاب نهایی با GPT (`search/select.py`) عمداً در محصول
> استفاده نمی‌شود — طبق محک، دقتش از نمایش مستقیم رتبه‌ی اول بازیابی بهتر
> نبود. فایل در مخزن نگه داشته شده برای بازبینی بعدی.

---

## ⚠️ نکته امنیتی

کلید OpenAI فقط در فایل `.env` نگهداری می‌شود که در `.gitignore` قرار دارد
و هرگز نباید commit یا در جایی فرستاده شود.

---

## معماری فعلی

```
data/taxonomy.xlsx (Sheet2)
        │
        ▼
common/catalog.py            ← شرح‌های یکتا + همه‌ی شناسه‌های تولید داخل/وارداتی هرکدام
        │
        ├─► pipeline/embed_catalog.py    ─► checkpoints/catalog_embeddings.jsonl        (بردار خام، فعلاً استفاده نمی‌شود)
        ├─► pipeline/enrich_catalog.py   ─► checkpoints/catalog_enrichment.jsonl        (پاس ۱: canonical/synonyms/examples_raw/uses)
        │                                 checkpoints/catalog_examples.jsonl            (پاس ۲: examples بازبینی‌شده)
        ├─► pipeline/embed_enriched.py   ─► checkpoints/catalog_embeddings_enriched.jsonl (بردار شرح+canonical؛ ۷۴۲ مگابایت)
        └─► pipeline/build_search_index.py ─► checkpoints/catalog_semantic.index         (FAISS، سبک)
                                              checkpoints/catalog_semantic_positions.npy  (نگاشت به کاتالوگ)
```

سرور (`server/app.py`) فقط از این‌ها می‌خواند، نه از jsonl خام ۷۴۲ مگابایتی:
- `common/catalog.py` → کاتالوگ
- `search/enrichment.py` + `checkpoints/catalog_enrichment.jsonl` + `catalog_examples.jsonl` → متن ایندکس واژگانی غنی‌شده
- `search/lexical.py` (BM25، در حافظه ساخته می‌شود)
- `search/semantic.py` → `SemanticIndex.from_prebuilt(...)` روی `catalog_semantic.index` + `catalog_semantic_positions.npy`
- `search/hybrid.py` (ترکیب با RRF) + `search/understand.py` (فقط در حالت هوشمند)

پایپ‌لاین قدیمی (خوشه‌بندی + Taxonomy خودکار + FAISS تخت روی شرح خام) به
`pipeline/legacy/` منتقل شده و دیگر توسط سرور استفاده نمی‌شود — نگاه کنید
به `pipeline/legacy/README.md`.

---

## نصب

```
pip install -r requirements.txt
```

`.env.example` را کپی و به `.env` تغییر نام دهید، سپس `OPENAI_API_KEY` را پر کنید.
(کلید فقط برای «جستجوی هوشمند» و برای اجرای مجدد پایپ‌لاین غنی‌سازی لازم است؛
«جستجوی سریع» بدون کلید هم کار می‌کند.)

---

## اجرا برای توسعه (لوکال)

```
python server/app.py
```
به `http://localhost:8000` بروید.

## اجرا در تولید

سرویس با `gunicorn --preload` و حداکثر ۲ worker اجرا می‌شود تا کاتالوگ و
ایندکس‌ها فقط یک‌بار در پردازه‌ی master بارگذاری شوند و بین workerها به
اشتراک بروند (نه چند بار در حافظه تکرار شوند). این کار از طریق systemd
انجام می‌شود — به بخش «سرویس systemd» زیر نگاه کنید.

nginx روی پورت ۸۰۸۰ با Basic Auth به `127.0.0.1:8000` proxy می‌کند
(تنظیمات آن جدا از این پروژه است و تغییری در آن لازم نیست).

---

## به‌روزرسانی کاتالوگ یا غنی‌سازی

اگر `data/taxonomy.xlsx` عوض شد، یا خواستید غنی‌سازی را دوباره اجرا کنید:

```
# ۱. کش کاتالوگ را دوباره می‌سازد (خودکار، چون xlsx جدیدتر از cache است)
python -c "from common.catalog import load_catalog; load_catalog(force_rebuild=True)"

# ۲. غنی‌سازی (پاس ۱ + پاس ۲) — Resume دارد، فقط شرح‌های جدید/ناتمام را پردازش می‌کند
python pipeline/enrich_catalog.py

# ۳. Embedding معنایی روی شرح+canonical — Resume دارد
python pipeline/embed_enriched.py

# ۴. تبدیل به فایل‌های سبک FAISS + numpy که سرور می‌خواند (همیشه بعد از ۲ و ۳ لازم است)
python pipeline/build_search_index.py

# ۵. سرویس را ری‌استارت کنید تا کاتالوگ/ایندکس تازه بارگذاری شود
sudo systemctl restart finops-search
```

برای سنجش کیفیت قبل از استقرار، از `bench.py` استفاده کنید (نگاه کنید به
docstring بالای خودش برای پرچم‌های `--mode`, `--understand`, `--enriched`, `--select`).

---

## سرویس systemd

فایل واحد در `deploy/finops-search.service` نگه داشته می‌شود. نصب/فعال‌سازی
(یک‌بار، نیاز به sudo):

```
sudo cp deploy/finops-search.service /etc/systemd/system/finops-search.service
sudo systemctl daemon-reload
sudo systemctl enable --now finops-search
```

بررسی وضعیت:
```
sudo systemctl status finops-search
```

ری‌استارت (مثلاً بعد از به‌روزرسانی کد یا ایندکس):
```
sudo systemctl restart finops-search
```

خواندن لاگ‌ها:
```
sudo journalctl -u finops-search -f          # زنده
sudo journalctl -u finops-search -n 200      # ۲۰۰ خط آخر
```

سرویس با `Restart=always` اجرا می‌شود و در بوت سیستم هم خودکار بالا می‌آید
(به‌خاطر `enable`)، پس قطع SSH یا ری‌استارت سرور آن را متوقف نمی‌کند.

---

## ساختار پوشه‌ها

```
finops_search_v2/
├── .env                        ← تنظیمات محرمانه (در گیت نیست)
├── requirements.txt
├── deploy/
│   └── finops-search.service   ← واحد systemd
├── data/                       ← taxonomy.xlsx
├── checkpoints/                ← فایل‌های میانی (غنی‌سازی، Embedding، ایندکس FAISS)
├── logs/
│   └── search_log.jsonl        ← لاگ هر جستجوی واقعی (پرس‌وجو، حالت، نتایج نمایش‌داده‌شده)
├── common/
│   ├── catalog.py               ← شرح‌های یکتا + شناسه‌های تولید داخل/وارداتی
│   ├── config.py                ← بارگذاری تنظیمات از .env
│   ├── textnorm.py              ← نرمال‌سازی/توکنایز فارسی
│   └── utils.py
├── search/
│   ├── lexical.py                ← ایندکس BM25
│   ├── semantic.py               ← ایندکس FAISS (+ from_prebuilt برای سرور)
│   ├── hybrid.py                 ← ترکیب با RRF
│   ├── understand.py             ← فهم پرس‌وجو (فقط حالت هوشمند)
│   ├── enrichment.py             ← ساخت متن ایندکس از خروجی غنی‌سازی
│   └── select.py                 ← انتخاب نهایی با GPT — در محصول استفاده نمی‌شود
├── pipeline/
│   ├── embed_catalog.py
│   ├── enrich_catalog.py
│   ├── embed_enriched.py
│   ├── build_search_index.py    ← ساخت ایندکس سبک برای سرور (بعد از هر تغییر کاتالوگ/غنی‌سازی اجرا شود)
│   └── legacy/                  ← پایپ‌لاین قدیمی، بازنشسته (نگاه کنید به legacy/README.md)
├── bench.py                     ← اجراکننده‌ی محک روی benchmark.json
└── server/
    ├── app.py
    └── templates/index.html
```
