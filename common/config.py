"""
بارگذاری تنظیمات از فایل .env
------------------------------------
هیچ کلید یا مسیر حساسی داخل کد نوشته نمی‌شود؛ همه چیز از .env خوانده می‌شود.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

for _d in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small").strip()
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini").strip()
INPUT_FILE = os.environ.get("INPUT_FILE", "").strip()
PORT = int(os.environ.get("PORT", "8000"))

if not OPENAI_API_KEY:
    print("⚠️  هشدار: OPENAI_API_KEY در فایل .env تنظیم نشده است. "
          "فایل .env.example را کپی کرده و به .env تغییر نام دهید، سپس کلید را وارد کنید.")
