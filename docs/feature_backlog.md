# SecMentor — Feature Backlog

Candidate list for everything we could ship after Phase 8 (and the agreed Phase 9 hygiene pass). Each item has an effort tag, an impact rating, a one-line description, an effective implementation, and a dependency on what must land first.

## Scoring legend

- **Effort** — S = ≤ 1 hr, M = ≤ ½ day, L = 1–2 days, XL = 3+ days
- **Impact** — ★ to ★★★★★ (1 = nice to have, 5 = transformative)
- **Depends on** — feature must be present before this one is useful

---

## Tier 1 — Quick wins (no auth, no DB, no schema)

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 1 | Streaming LLM responses | S | ★★★★★ | Removes the "blank screen" feel; perceived 3–5× faster even when wall time is identical | Set `stream=True` in `app/openrouter.py`; yield tokens or words; consume with `st.write_stream` in the view | nothing |
| 4 | Copy-to-clipboard on assistant bubble | S | ★★ | One tap, removes the drag-to-select friction on code blocks | A small `st.button("📋")` next to each bubble; use a one-line `st.components.v1.html` snippet with `navigator.clipboard` | nothing |
| 5 | Markdown rendering | S | ★★★ | The bot already returns Markdown; today you see raw backticks | Replace `st.write(msg)` with `st.markdown(msg)`; keep `unsafe_allow_html=False`; add a small CSS override so code blocks use the active theme | nothing |
| 7 | Conversation export | S | ★★ | Power users want to take chats elsewhere | The `_serialize_for_download()` helper already exists in `web/chat_helpers.py`; wire it to a sidebar `st.download_button` that emits `.md` | nothing |
| 8 | Dark/light theme toggle | S | ★ | Removes a complaint before it happens | `.streamlit/config.toml` with `theme.base = "dark"`; expose a `?theme=` query param or a settings expander | nothing |

**Batch 1 ships in one PR.** Combined: ~80–120 LOC, all unit-testable with a mock LLM, zero schema work.

---

## Tier 2 — Trust + visibility

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 6 | Token + cost counter | S | ★★★ | Free-tier users hit 200/day silently; a visible counter builds trust | Parse `usage.prompt_tokens` / `completion_tokens` from the OpenRouter response; render a one-liner under each bubble; compute $ from a static `MODEL_PRICING` dict keyed by model id | streaming (1) |
| 11 | Model health dot | M | ★★★ | Users cannot tell which slot they are on; a green/yellow/red dot makes the rotation visible | New `ModelRouter.health_snapshot()` returning `[{slot, status, last_ok, last_error}]`; cache for 30 s; render as colored circles next to the model name in the dropdown | cache layer (perf) |
| 12 | Auto-failover toast | S | ★★ | Quietly rotating slots is invisible; a one-time toast on first failover is reassuring | Catch the exception in `_ask()`; identify the recovered `slot_id` from `ModelRouter.last_attempt`; fire `st.toast("Recovered on slot N", icon="🔁")` | nothing |
| 23 | Structured (JSON) logging | M | ★★★ | Streamlit Cloud logs are unreadable; JSON makes incidents greppable | New `app/logging_setup.py` with a JSON formatter; one `logger.info("chat_turn", extra={...})` per phase (key_validate, model_pick, prompt_build, request, response, fail) | nothing |
| 24 | Rate-limit + rotation dashboard | M | ★★ | Surfaces 429s and rotation without forcing the user to read logs | Sidebar expander reading from the same in-memory counters that drive the health dot; show last 5 events with timestamps | structured logging (23) |
| 25 | Prompt-injection guard | M | ★★★★ | A defensive-mode bot must not be hijacked; a cheap regex layer buys a lot of safety | New `app/guards.py` with `sanitize_user_input()` and `sanitize_rag_chunk()`; strip "ignore previous instructions" patterns, cap length, drop control characters; reject *before* prompt assembly | nothing |

---

## Tier 3 — Memory + RAG

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 9 | RAG over uploaded files | L | ★★★★★ | Today the file is sent to the LLM once; later turns forget it. RAG gives recall. | Chunk file (sliding window, 512 tokens, 64 overlap) → embed with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → store in FAISS keyed by `(user_id, chat_id)` → on each turn, top-k=4 retrieval prepended to the messages list | auth, DB, FAISS |
| 10 | Pinned "context" panel | M | ★★ | Users forget what they uploaded; a visible "grounded on" list is the missing UI | Sidebar widget reading from the `artifacts` table: filename, size, chunk count, remove button; on remove, drop from FAISS and mark `artifacts.deleted_at` | auth, DB, RAG (9) |
| 14 | Saved system prompts | M | ★★★ | Power users want their own persona; named presets beat free-form edit | New `app/prompts_registry.py` with a dict of named prompts; a dropdown that swaps `active_system_prompt`; user-editable prompts stored in `users.preferences` (JSON column) | auth, DB |
| 17 | RAG over OWASP / CWE / MITRE corpus | XL | ★★★★ | Citations beat vibes for a defensive tutor; a "CWE-79" link in the answer is gold | One-time ingest script `scripts/ingest_corpus.py`; embeddings live in a *separate* FAISS index; retrieval is two-vector: user docs (k=4) + corpus (k=2) | RAG (9), embed model |

---

## Tier 4 — Reports

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 18 | PDF report from logs / nmap / pcap | XL | ★★★★★ | The headline feature; "upload nmap, get a PDF" is what you pitch | `app/parsers.py` (nmap, syslog, auth.log, pcap text-export) → `app/reports.py` builds a structured finding list with severity + remediation → WeasyPrint renders an HTML template with print CSS → `st.download_button` ships the PDF | auth, DB, RAG (9), parsers |

**Note**: parsers and renderer are independent. Build parsers first (testable in isolation with fixture files), then the renderer (testable with fake findings), then wire to the UI last.

---

## Tier 5 — MCP

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 15 | MCP server (SecMentor as a tool provider) | L | ★★★★ | Lets Claude Desktop / Cursor / Cline call SecMentor's parsers as MCP tools | New `app/mcp_server.py` with the official `mcp` Python SDK; expose `parse_nmap`, `parse_auth_log`, `extract_iocs`, `lookup_cve` as MCP tools over **stdio** transport; run as a sidecar process | parsers (18 base) |
| 16 | MCP client inside Streamlit | XL | ★★★★ | The assistant invokes our *own* tools — the "agent" arc you flagged in `docs/my_first_ai_journey.md` | `app/mcp_client.py` wraps the same SDK; chat loop sees a tool-call response, surfaces it to the LLM, and continues the turn (multi-step) | MCP server (15), streaming (1) |
| 20 | NVD web search | M | ★★★ | "What is CVE-2024-3400?" should pull the NVD entry, not hallucinate | Either an MCP tool wrapping `services.nvd.nist.gov`, or a direct `httpx` call; cache by CVE id for 24 h; rate-limit per the NVD API key guidance | nothing (cheap path) |

---

## Cross-cutting — Architectural

| # | Feature | Effort | Impact | Why | Effective implementation | Depends on |
|---|---------|--------|--------|-----|--------------------------|------------|
| 19 | Multi-user + shared chats | M | ★★★ | "Share chat" turns a private tutor into a collaboration tool | New `chats.visibility` enum (`private`, `unlisted`, `public`) + a token-based read-only URL stored in `chats.share_token` | auth, DB |
| 21 | Async I/O in `app/openrouter.py` | M | ★★★ | 5 serial key checks at startup is 5× slower than 5 parallel | `httpx.AsyncClient` + `asyncio.gather` for `validate_all_keys()`; keep Streamlit sync, but call `asyncio.run` once at session start | nothing |
| 22 | Response cache | M | ★★ | Repeat questions on free tier = free cycles | `functools.lru_cache(maxsize=512)` keyed on `(model, sha256(messages))`; TTL via a small wrapper; **warning**: stale answers, document it in the UI | nothing |
| 2 | Chat history sidebar | M | ★★★★ | Power users have 30+ chats; "where was that log analysis?" | Sidebar list of last 10 chats per user; click loads into view; rename + delete actions | auth, DB |
| 3 | Prompt template gallery | M | ★★★ | Empty box is the #1 cause of "I don't know what to ask" | `app/prompts_gallery.py` with a list of `(name, prompt, tags)`; one click inserts into the input box | nothing |

---

## Phase mapping (how the tiers land)

| Phase | Scope | Features touched |
|-------|-------|------------------|
| **9 — hygiene** | Markers, pinned deps, scripts/, `docs/architecture.md` | none user-visible |
| **9.1 — perf (new)** | Caching, streaming, async, logging | 1, 4, 5, 6, 11, 12, 21, 23 |
| **10 — parsers** | nmap, syslog, auth.log, pcap text | 18 base |
| **11 — RAG** | Embeddings, FAISS, retrieval loop | 9, 10 |
| **12 — reports** | WeasyPrint + findings schema | 18 top |
| **13 — corpus** | OWASP/CWE/MITRE ingest | 17 |
| **14 — auth** | bcrypt, signed cookies, `session_version` | enables 2, 14, 19 |
| **15 — history** | Sidebar list, rename, delete | 2 |
| **16 — tie together** | Saved prompts, sharing, caching, guard, dashboard | 3, 14, 19, 22, 24, 25 |
| **Post-16 — MCP** | Server + client + NVD | 15, 16, 20 |

---

## Recommended next two to ship

1. **Phase 9.1 — perf PR** (items 1, 4, 5, 6, 11, 12, 21, 23). Half a day of work, ~200 LOC, ships as a single PR. No auth, no DB, no schema. All unit-testable. The streaming + health dot + token counter together will change how the deployed app *feels*, which is the main pain point right now.
2. **Prompt template gallery** (item 3). Low effort, high learning value, and it sets up the registry pattern that Phase 14's "saved prompts" will reuse.

If you confirm, the next step is `docs/phase_09_1_perf.md` describing the perf PR — items in scope, test plan, and what is explicitly *out* of scope — before any code lands.
