"""Sanity-check that the post-fix view file contains the right shape.

This is the same pattern as verify_changes.py from earlier turns:
read the served file as text and assert the key structural changes
are present. A green run means the live code matches what the tests
pinned.
"""
import sys

src = open(
    r"d:\puku projects\stage1\web\streamlit_app.py",
    "r",
    encoding="utf-8",
).read()

# Slice the cache-hit branch: from "content\": cached}" to ~600 chars
# after. A rerun() must be inside that window.
cache_anchor = 'content": cached}'
cache_idx = src.find(cache_anchor)
cache_window = src[cache_idx : cache_idx + 600]

# Slice the success path: from the assistant-reply append through the
# next 400 chars.
success_anchor = '"role": "assistant", "content": reply'
success_idx = src.find(success_anchor)
success_window = src[success_idx : success_idx + 400]

checks = {
    "cache-hit rerun() present":            "st.rerun()" in cache_window,
    "pass-2 success rerun() present":       "st.rerun()" in success_window,
    "_friendly_error_message imported":     "_friendly_error_message" in src,
    "_render_friendly_error view wrapper":  "_render_friendly_error" in src,
    "error block calls the wrapper":        "_render_friendly_error(exc, model)" in src,
    "st.toast still present":               'st.toast(f"Calling {model}' in src,
    "Thinking... placeholder still present": "Thinking…" in src,
}

width = max(len(k) for k in checks)
all_ok = True
for name, ok in checks.items():
    flag = "OK " if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {flag}  {name.ljust(width)}")

sys.exit(0 if all_ok else 1)
