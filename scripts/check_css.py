"""Lint web/styles.css (the single source of truth for the Streamlit UI).

Catches:
  * Unbalanced CSS braces
  * A regression: re-introduction of `*` selectors with `color:`
    AND `!important` in the same body (the original bug).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "web" / "styles.css"


def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def parse_rules(text: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    header_buf: list[str] = []
    body_buf: list[str] = []
    depth = 0
    in_body = False
    header = ""
    for ch in text:
        if not in_body:
            if ch == "{":
                header = "".join(header_buf).strip()
                header_buf = []
                in_body = True
                depth = 1
            else:
                header_buf.append(ch)
        else:
            if ch == "{":
                depth += 1
                body_buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body = "".join(body_buf)
                    rules.append((header, body))
                    body_buf = []
                    in_body = False
                else:
                    body_buf.append(ch)
            else:
                body_buf.append(ch)
    return rules


raw = TARGET.read_text(encoding="utf-8")
css = strip_comments(raw)
opens = css.count("{")
closes = css.count("}")
print(f"CSS braces: {opens} open / {closes} close")
if opens != closes:
    print("CSS braces: UNBALANCED")
    sys.exit(1)

rules = parse_rules(css)
print(f"CSS rules: {len(rules)}")

bad: list[tuple[str, str]] = []
# Selectors we author the contents of and therefore trust to use `*`.
# Anything else with `*` + `color: ... !important` is the smoking gun
# of the original bug (`section[data-testid="stSidebar"] *`,
# `.main .block-container *`).
OWNED_PREFIXES = (
    ".hero",
    ".bubble-",
    ".status",
    ".empty-state",
    ".sm-",          # sidebar card / pill / row
    ".stAlert",      # notification colour is inherited safely
    ".main details",  # our expander body in main column
)

for header, body in rules:
    if header.startswith("@"):
        continue
    selectors = [p.strip() for p in header.split(",")]
    hit = False
    owned = False
    for sel in selectors:
        if "*" in sel.split():
            hit = True
            if any(sel.startswith(pre) for pre in OWNED_PREFIXES):
                owned = True
                break
    if not hit or owned:
        continue
    if re.search(r"color\s*:\s*[^;]+!important", body):
        bad.append((header, body))

if bad:
    print("CSS: FAIL - bare-`*` color rules with !important (outside owned scopes):")
    for header, body in bad:
        print(f"  selector: {header[:120]}")
        print(f"  body:     {body.strip()[:200]}")
    sys.exit(1)
print(f"CSS: no bare-`*` color rules with !important ({len(rules)} rules scanned)")