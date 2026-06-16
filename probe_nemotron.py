"""Quick live probe: time a call to nvidia/nemotron-nano-9b-v2:free.

Run with:  .venv\Scripts\python.exe probe_nemotron.py
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from app.openrouter import chat  # noqa: E402

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "List three common Windows vulnerabilities in one sentence each."},
]
model = "nvidia/nemotron-nano-9b-v2:free"

t0 = time.time()
try:
    reply = chat(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=512,
    )
    dt = time.time() - t0
    print(f"OK   in {dt:.1f}s   chars={len(reply)}")
    print("---")
    print(reply[:600])
except Exception as e:
    dt = time.time() - t0
    print(f"FAIL in {dt:.1f}s   type={type(e).__name__}   msg={e}")
