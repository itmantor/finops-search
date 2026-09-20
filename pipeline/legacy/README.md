# پایپ‌لاین قدیمی — بازنشسته

فایل‌های این پوشه (۰۱ تا ۰۷ و `run_all.py`) مسیر اول پروژه بودند: دسته‌بندی
خودکار با خوشه‌بندی (UMAP + HDBSCAN)، ساخت Taxonomy با GPT، و در نهایت یک
ایندکس FAISS تخت (`outputs/faiss.index` + `outputs/metadata.pkl`) که سرور
قدیمی مستقیماً از رویش جستجو می‌کرد.

این مسیر دیگر توسط سرور استفاده نمی‌شود. مسیر زنده‌ی فعلی این است:

| قدیم (این پوشه) | جدید (زنده) |
|---|---|
| `01_embed_initial.py` .. `07_build_faiss.py`, `run_all.py` | `pipeline/embed_catalog.py`, `pipeline/enrich_catalog.py`, `pipeline/embed_enriched.py`, `pipeline/build_search_index.py` |
| `outputs/faiss.index` + `outputs/metadata.pkl` | `checkpoints/catalog_semantic.index` + `checkpoints/catalog_semantic_positions.npy` |
| جستجوی متنی substring + جستجوی معنایی ۷۰٪/۳۰٪ + GPT ReRank اختیاری | جستجوی سریع (BM25) + جستجوی هوشمند (فهم پرس‌وجو + بازیابی هیبرید BM25/معنایی با RRF) |

دلیل تغییر: کاتالوگ جدید بر پایه‌ی شرح‌های یکتا (`common/catalog.py`) و
غنی‌سازی LLM (`pipeline/enrich_catalog.py`: canonical/synonyms/examples/uses)
ساخته می‌شود، نه خوشه‌بندی. نتیجه در محک (`bench.py`) بهتر بود.

این فایل‌ها فقط برای مرجع/تاریخچه نگه داشته شده‌اند و اجرا نمی‌شوند. برای
معماری فعلی به `README.md` اصلی پروژه نگاه کنید.
