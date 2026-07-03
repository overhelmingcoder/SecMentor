"""Inventory any selector whose token list still contains a bare `*`.

Useful for an audit pass after the CSS cleanup. Prints each surviving
`*`-bearing selector tagged with `[color !important]` if its body
recolors with `!important`, or `[safe]` otherwise.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "web" / "styles.css"

src = TARGET.read_text(encoding="utf-8")
css = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)

rules: list[tuple[str, str]] = []
header: list[str] = []
body: list[str] = []
depth = 0
in_body = False
current_header = ""
for ch in css:
    if not in_body:
        if ch == "{":
            current_header = "".join(header).strip()
            header = []
            in_body = True
            depth = 1
        else:
            header.append(ch)
    else:
        if ch == "{":
            depth += 1
            body.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rules.append((current_header, "".join(body).strip()))
                body = []
                in_body = False
            else:
                body.append(ch)
        else:
            body.append(ch)

print("--- selectors whose token list contains a bare `*` ---")
for hd, b in rules:
    if hd.startswith("@"):
        continue
    for sel in (s.strip() for s in hd.split(",")):
        toks = sel.split()
        if "*" in toks:
            has_color_imp = bool(re.search(r"color\s*:\s*[^;]+!important", b))
            tag = "[color !important]" if has_color_imp else "[safe]"
            print(f"  {tag:<22} {sel}")
