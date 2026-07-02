# 🛡 SecMentor

> **AI-Powered Cybersecurity Learning & Analysis Platform**
> Learn · Analyze · Defend · Research

A Streamlit chatbot that answers cybersecurity questions using free LLMs routed through [OpenRouter](https://openrouter.ai). Built as a personal learning project across 7 phases plus a Session-4 post-hoc pass, a Session-7 multimodal pass, a Session-8 UI fix pass, a Session-9 RAG + chat-history pass, and a Phase-15 OSINT pass:
plain request/response → memory → prompt engineering → web UI → refactor → multi-model selector, two-pass pattern, friendly-error classifier → file & image upload (multimodal) → copy-to-clipboard bubble buttons (no inline JS, idempotent delegated listener) → **persistent chat history sidebar + per-chat RAG over uploaded files + curated global security corpus (OWASP / MITRE / CWE / GTFOBins / Sigma) streamed into the prompt before the user turn → scoped OSINT reconnaissance (`/recon` slash command — DNS / URL / IP / WHOIS / crt.sh) gated by a "lab-only" scope token so it can never reach the public internet on a production target**.

The web UI ships in **CTF / Lab mentor** mode by default (a lab-scoped teaching persona that can produce runnable exploit snippets *for your own lab*), with a one-click switch to the conservative **Defensive (4-pillar)** prompt. The CLI keeps the defensive default so headless / scripted use stays in the tighter scope. Both modes share the same hard refusals — see the [Safety](#safety) section below.

Layout note: the `phase3_cli/` and `phase6_web/` folder names from earlier phases were renamed to `cli/` and `web/` in Phase 7 once the teaching scaffold was no longer useful. The engines behind them are unchanged.

See `docs/technical_write_up.md` for the technical reference and
`docs/my_first_ai_journey.md` for the personal learning log.

---

## Quick start

### 1. Get an OpenRouter key (free)

1. Go to <https://openrouter.ai> and sign up.
2. Open <https://openrouter.ai/keys> and click **Create Key**.
3. Copy the key (starts with `sk-or-v1-...`). Treat it like a password.

### 2. Clone and configure

```bash
git clone <your-repo-url> secmentor
cd secmentor
copy .env.example .env        # Windows
# or:  cp .env.example .env   # macOS / Linux
```

Open `.env` in any text editor and fill in:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
OPENROUTER_MODEL=provider/model-name:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_APP_NAME=SecMentor
```

> **Free-model heads-up.** The `:free` tier is rate-limited per provider
> (rough ballpark: 20 req/min, 50–200/day depending on the upstream
> provider). The default model `google/gemma-4-31b-it:free` is the
> most reliable on the day of writing. If you see a friendly
> `⏳ … is rate-limited upstream.` banner in the UI, either wait ~30s
> or pick a different model from the sidebar.

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or:  source .venv/bin/activate   # macOS / Linux
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run it

CLI chatbot:

```bash
python cli/chatbot.py
```

Web interface (recommended):

```bash
streamlit run web/streamlit_app.py
```

The web UI opens on `http://localhost:8501` by default. Type a cybersecurity
question, press Enter, and the assistant reply appears in the chat.

### 6. Run the test suite

```bash
python -m unittest discover -v
```

Expected: `Ran N tests in …` → `OK`. The breakdown by file:

| File | Tests | Coverage |
|---|---|---|
| `tests/test_smoke.py` | 21 classes | Engine, helpers, view contracts, two-pass pattern, model selector, friendly-error path, multi-key router, prompt boundary, `sys.path` bootstrap, copy-button contract (20 tests) |
| `tests/test_files.py` | 14 classes | File processor (PDF text extract, image bytes, MIME, 4 MB cap, 8 typed errors) + multimodal bridge (vision auto-upgrade, stub-block, fallback model, base64) |
| `tests/test_storage.py` | — | Chat + message + artifact + chunk + recon audit row CRUD, cascade deletes |
| `tests/test_rag.py` | 8 classes | Per-chat RAG store: add, search, lazy index build, version-counter invalidation, cross-chat isolation |
| `tests/test_rag_global.py` | 7 classes | Global corpus store: add, search, source-filter, dedup by sha256, degraded mode |
| `tests/test_rag_corpus.py` | — | OWASP / MITRE / CWE / GTFOBins / Sigma parsers + dispatch + registry + validation |
| `tests/test_recon.py` | 17 classes | Phase 15 OSINT: `/recon` grammar, refang, normalize, scope-token safety, 5 transports, orchestrator, two renderers, audit log |
| `tests/test_streaming.py` | — | `stream_chat` SSE consumer + round-robin router wrapper |

Run any one file with `python -m unittest tests.test_recon -v`.

> **Optional — pre-flight check before running.** If you are mid-development and you have edited `app/config.py` (added or renamed a function), a long-lived Streamlit worker may still be holding the *old* module object. The pre-flight script catches that before you start a new server:
>
> ```powershell
> .\.venv\Scripts\python.exe verify_changes.py
> # READY  -> streamlit run web/streamlit_app.py --server.port 8765
> # STALE_WORKER: kill PIDs 1234, 5678  -> run the printed Stop-Process command, then re-run
> # IMPORT_BROKEN  -> the traceback tells you which symbol is missing from the new code
> ```
>
> The script never echoes secret values. It only inspects running processes, the project port, and the importable symbol table.

---

## Project layout

```
app/              # Core library: config, OpenRouter client, router, prompts, file_processor (multimodal)
  recon/          # Phase 15 OSINT subsystem: scope-token gate, normalize, 5 transports, orchestrator, two renderers
cli/              # Terminal interface (defensive prompt by default)
web/              # Streamlit web interface + pure UI-logic helpers
tests/            # Automated checks (see "Running the tests" below for the breakdown by file)
docs/             # Technical and personal study logs
  phases/         # One design doc per phase (phase_12_rag_and_history.md is the most recent)
.streamlit/
  config.toml     # Theme pin: light base, dark text, enterprise-blue accent
verify_changes.py # Pre-flight check (catches stale-worker import errors)
run.py            # One-command runner: venv + deps + .env bootstrap + streamlit
improvements.md   # Change-log + tuning guide for offensive / defensive behavior
```

Key files:

- `app/openrouter.py` — HTTP client to OpenRouter's `/chat/completions` endpoint. Raises `OpenRouterError` (and four subclasses) on any HTTP non-200; 30-second timeout.
- `app/config.py` — `iter_api_keys()` (5-slot rotation pool), `iter_models()` (multi-model list), `HTTP_TIMEOUT_SECONDS=30`. Reads from `.env` via `python-dotenv`.
- `app/router.py` — `ModelRouter` class: round-robin slot rotation, 401-disables-slot, 429-backoff, 5xx-retry, key redaction (`****abcd`).
- `app/prompts.py` — The two system prompts: `CYBERSECURITY_SYSTEM_PROMPT` (defensive, four pillars) and `OFFENSIVE_MENTOR_SYSTEM_PROMPT` ("SecMentor", lab-scope). The web UI can swap between them from the sidebar.
- `app/file_processor.py` — Multimodal file pipeline: `process_upload(uploaded_file)` returns `ProcessedFile(text, image_b64, mime)`. Handles PDF (text extract via pymupdf, lazy-imported), PNG/JPG/GIF/WebP images (base64, 4 MB cap), raises 8 typed `FileProcessingError` kinds. No Streamlit dependency.
- `cli/chatbot.py` — REPL-style terminal chat. Always uses the defensive prompt.
- `web/streamlit_app.py` — The Streamlit view. `sys.path` bootstrap, sidebar (model selector + **teaching mode** radio + **Layout mode** radio + file uploader), `_init_state`, two-pass `_ask` driver (`pending_request` in `session_state`), `_render_friendly_error`. Theme is pinned via `.streamlit/config.toml`; the design-token CSS in this file adds light-mode guards so OS / browser dark-mode UA stylesheets cannot override text colour.
- `web/chat_helpers.py` — Pure UI-logic helpers: `_active_system_prompt(state)` (the teaching-mode → prompt bridge, fail-closed), `_is_rate_limit_error`, `_friendly_error_message`, `_build_messages`, `_truncate_history`, `_serialize_for_download`, `_count_chars`, `_bubble_alignment`, **and the multimodal bridge**: `_is_image_mime`, `_is_pdf_mime`, `build_user_turn_content` (returns `str | list[dict]` — text-only OR OpenAI vision content array), `select_model_for_request` (auto-upgrades to a vision-capable model when the turn contains an image), and `_DEFAULT_FREE_VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"`. No Streamlit calls in any of these.
- `tests/test_smoke.py` — 147 tests across 22 classes. Covers the engine, the helpers, the view contracts, the two-pass pattern, the model selector, the friendly-error path, the multi-key router, the prompt boundary, the `sys.path` bootstrap, the `key="teaching_mode"` radio windowed-search guard, and the copy-to-clipboard button contract (`CopyButtonHtmlTests` + `CopyButtonInitScriptTests`, 20 tests).
- `tests/test_files.py` — 28 tests across 14 classes. Covers `app/file_processor.py` (PDF text extraction, image bytes, MIME detection, size cap, error taxonomy) and the multimodal bridge in `web/chat_helpers.py` (vision auto-upgrade, stub-block path, fallback model, base64 encoding).
- `verify_changes.py` — Pre-flight script: `READY` / `STALE_WORKER: kill PIDs …` / `IMPORT_BROKEN`. Uses `Get-CimInstance` and `Get-NetTCPConnection` to find stale workers before they bite.
- `run.py` — One-command runner. Locates a Python 3.11+ interpreter, creates `.venv/`, installs `requirements.txt`, copies `.env.example → .env` if missing, runs `verify_changes.py`, then launches `streamlit run web/streamlit_app.py` in the foreground (or detached with `--detach`).
- `improvements.md` — The change-log. Section A documents the offensive behavior (the SecMentor prompt, the swap, how to tune it). Section B documents the requirement-issue solutions (5 keys, multi-model, friendly errors, two-pass, pre-flight, light-mode pin). Section C lists future improvement suggestions.

---

## How to run it on your own

```bash
python run.py
```

That's it. On first run it creates the venv, installs deps, and copies `.env.example` → `.env` (you'll be prompted to paste your real `sk-or-v1-...` key). On every run it does a pre-flight check, then starts the server. Open **http://localhost:8765** and press **Ctrl-C** to stop.

```bash
python run.py --port 9000     # different port
python run.py --no-preflight  # skip the pre-flight check
python run.py --detach        # run in the background (CI / scripts)
```

---

## How the web app works (60-second tour)

1. The sidebar lets you pick a model from 5 curated `:free` options (Gemma 4 31B, Llama 3.3 70B, Mistral Small 24B, Qwen 2.5 72B, Nemotron 30B) or paste a custom model ID.
2. Type a question into `st.chat_input` and press Enter.
3. **Or drag a file** into the chat input area (`st.chat_input` accepts a file attachment). Supported: PDF, PNG, JPG, GIF, WebP. Limits: 4 MB images, ~200 K characters of extracted PDF text. See **File & image upload** below.
4. The view uses a **two-pass pattern**: pass 1 stores the question (and any file) in `session_state["pending_request"]` and calls `st.rerun()`; pass 2 calls the OpenRouter engine, appends the assistant reply, and reruns again so the history loop at the top of the script paints the new bubble.
5. A `Thinking…` placeholder appears in the spinner window; the assistant bubble replaces it when the model returns.
6. Errors are classified at the helper layer (`_is_rate_limit_error`) and rendered at the view layer (`_render_friendly_error`). The raw exception is kept in a collapsed `st.expander` for debugging.
7. The same question twice in one session hits an in-memory cache (no second upstream call).
8. `/clear` (or the sidebar Reset) wipes the transcript without a server restart.

### Web UI features

The view file (`web/streamlit_app.py`) renders a **production-grade cybersecurity aesthetic** inspired by Microsoft Security Copilot, CrowdStrike Falcon, Palo Alto Cortex, Datadog, and GitHub Enterprise — not a generic ChatGPT look. Specifically:

- **Hero card** — eyebrow + 🛡 **SecMentor** wordmark + new subtitle + tagline + 5 capability badges (Cybersecurity Mentor · Security Research · File Analysis · CTF & Lab Guidance · Multi-Model AI).
- **Layout mode** (sidebar radio) — Compact / Standard / Wide / Full width. The choice is persisted in `localStorage` and applied to `<body>` via a small inline `<script>`, so the density survives page reloads. CSS variables under each `.layout-*` body class control bubble width, padding, and chat-input width.
- **Empty-state card** — icon + heading + description + four suggestion chips (`Explain a recent CVE`, `Walk me through a web attack chain`, `Harden a Linux server`, `Review a log excerpt for IOCs`).
- **Light-mode theme pin** — `.streamlit/config.toml` sets `base = "light"`, `textColor = "#0f172a"`, `backgroundColor = "#f4f6fa"`, `primaryColor = "#1d4ed8"`. The CSS adds `color-scheme: light !important` on `:root` and `body, .stApp, .stApp *` plus high-specificity `!important` rules on every text element in the main column and inside assistant bubbles. This is the **fix for the white-on-white text bug** reported during development — Streamlit's theme follows the OS / browser dark-mode preference, and the unguarded CSS lost the cascade war to the UA stylesheet.
- **Status pill** — model, model id, slot health, and a thinking-spinner modifier on the right, all in a single styled bar with separator dots.
- **Chat input** — framed as a card with shadow, accent-blue caret, and a Send button styled to match the rest of the UI.
- **Dark code blocks** in assistant bubbles — `#0f172a` background, `#e2e8f0` text, so fenced code blocks stay readable inside a light bubble.

### Teaching mode (sidebar)

The sidebar has a **Teaching mode** radio:

- **Defensive (4 pillars)** — the original `CYBERSECURITY_SYSTEM_PROMPT`. Concept-level only. Best for threat modeling, IR, DevSecOps, AI-security, and structural understanding of attacks.
- **CTF / Lab mentor** *(default)* — the `OFFENSIVE_MENTOR_SYSTEM_PROMPT` ("SecMentor" persona). Unlocks lab scope on HackTheBox, TryHackMe, PortSwigger Academy, DVWA, WebGoat, picoCTF, your own home lab, or a sanctioned pentest. May produce **runnable** exploit snippets *for your lab*, always labelled (e.g. `# for HackTheBox 'Lame' on 10.10.10.3`) and always paired with the defensive countermeasure. The CLI keeps the defensive default — the wider mentor authorization is opt-in there — but the web UI ships in mentor mode so a learner who opens the app gets the lab-capable persona first.

Switching the radio swaps the system prompt in place; the transcript is preserved. A new chat (or `/clear`) re-seeds the system message from the current mode. The "About / scope" expander at the bottom of the main pane documents the boundary in-app.

Both modes share the same hard refusals: real production targets, named-vendor WAF/EDR/MFA bypasses, brand-new malware, and critical-infrastructure targets stay out of scope in either mode. Mentor mode is a wider *lab-only* authorization, never a license to target real systems.

### File & image upload (multimodal)

The web UI's `st.chat_input` accepts file attachments. Drop a file into the input area (or click the paperclip icon) and ask a question about it in the same turn.

**Supported formats and limits:**

| Type | Limit | What happens |
|---|---|---|
| `application/pdf` | 200 K characters of extracted text | `pymupdf` extracts the text; the first chunk is sent inline, the rest is truncated with a note. |
| `image/png`, `image/jpeg`, `image/gif`, `image/webp` | 4 MB | The bytes are base64-encoded and sent as an OpenAI-style `image_url` content block. |
| Anything else | — | A friendly error banner: `Unsupported file type: <ext>. Supported: PDF, PNG, JPG, GIF, WebP.` |

**How the model is chosen (auto-upgrade):** the helper `select_model_for_request(text, model, has_image)` checks whether the turn carries an image. If you picked a text-only model (Gemma, Llama, Mistral, Qwen), it silently swaps to the hardcoded fallback vision model `nvidia/nemotron-nano-12b-vl:free` so the request actually works. The status line in the UI tells you when the swap happened (`🔄 Auto-upgraded to a vision-capable model for this turn.`). If you already picked Nemotron 30B or pasted a custom model that supports images, the helper leaves your choice alone.

**The two layers of the pipeline:**

1. **`app/file_processor.py`** — pure, no-Streamlit. One public function: `process_upload(uploaded_file) -> ProcessedFile`. It returns `(text, image_b64, mime)` — exactly one of `text` / `image_b64` is populated per call. Eight typed `FileProcessingError` kinds so the view can render a clean banner for each failure mode (oversize, unsupported type, corrupt PDF, etc.).
2. **`web/chat_helpers.py`** — multimodal bridge. `build_user_turn_content(text, processed_file)` returns either a plain `str` (text-only turn) or a `list[dict]` in the OpenAI vision content shape (`[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`). The engine in `app/openrouter.py` doesn't need to know which is which — it serializes the messages to JSON and the model provider handles either shape.

**What this is NOT (limitations to be aware of):**

- **No persistence.** Uploads live in `session_state` for the duration of the browser tab. Close the tab, the bytes are gone. (See `improvements.md` row B.9 / §8.1 of the technical write-up for the Phase 12 plan.)
- **No OCR for scanned PDFs.** `pymupdf` extracts the text layer. A scanned PDF with no text layer returns an empty string — the helper raises `FileProcessingError("pdf_no_text", ...)` and the UI tells you to try OCR separately. Tesseract integration is on the Phase 12 backlog.
- **No DOCX / XLSX / PPTX.** Only PDF and the four image formats above. Office docs are on the Phase 12 backlog.
- **Vision model is hardcoded to one provider.** `nvidia/nemotron-nano-12b-vl:free` is the only free-tier vision model that has been empirically verified against the OpenRouter `:free` pool at the time of writing. If that model is rotated out, the auto-upgrade will return a 404 and you'll see a friendly error — the fix is to update the constant in `web/chat_helpers.py` (line ~704) to the next working vision model.
- **No streaming for image-heavy messages.** Vision responses are awaited as a single chunk; long-running image Q&A will block the spinner. Streaming for vision is on the Phase 12 backlog.
- **One file per turn.** Multi-file turns ("compare these two PDFs") are not yet supported.

**Testing:** the multimodal pipeline is covered by 28 tests in `tests/test_files.py` (14 classes) — the file processor, the MIME sniffer, the size cap, every error kind, the vision auto-upgrade, the stub-block path, the hardcoded fallback model, and the OpenAI content-array shape. Run them with `python -m unittest tests.test_files -v`.

See row 16 of `docs/technical_write_up.md` for the build log, Decision 11 for the docs-lag-code rule, and `improvements.md` §B.9 for the cross-cutting summary.

---

### Copy-to-clipboard bubble buttons

Every assistant reply renders a small **`📋 Copy`** button next to the bubble. Click it and the full reply text lands on your clipboard — nothing leaks, no extra permissions prompt, no broken HTML. The button is part of the assistant row, not the user row, so the transcript still scrolls naturally on both sides.

**What you see, in order of feedback after you click:**

1. **Idle** — the button reads `📋 Copy` with a faint border, sized to match the bubble's cap-height.
2. **Click** — the helper pulls the bubble's full text from the `data-text` attribute on the button, calls the modern `navigator.clipboard.writeText` API, and (in restricted contexts where the modern API is blocked) falls back to a hidden `<textarea>` + `document.execCommand('copy')`.
3. **Success** — the label briefly flips to `✅ Copied!` for ~1.5 s, then restores to `📋 Copy`. The restore is driven by a per-element `__copyBtnBusy` guard so rapid double-clicks cannot stack overlapping timers.
4. **Failure** — the label flips to `⚠ Copy failed` instead. The error is swallowed (clipboard is best-effort UX), and the original label restores on the next interaction.

**Why this is not the obvious `onclick="…"` implementation.** Inline `onclick="var text='…'"` inside an `unsafe_allow_html` stream looks harmless until the model returns a string that itself contains both an apostrophe and a double quote. The HTML parser then terminates the attribute at the inner quote and **leaks the rest of the handler as visible text inside the bubble** — the bug that triggered this whole pass. The fix is to:

- **Never put the payload in an attribute that needs to round-trip through HTML** — instead, store it in a `data-*` attribute and read it back through the browser's automatic `dataset.text` decoding.
- **HTML-escape the attribute value once on the Python side** with `html.escape(value, quote=True)`, which escapes `&`, `<`, `>`, `"`, *and* `'` (Python 3.13).
- **Register the click listener exactly once, on `document`**, with `event.target.closest('.bubble-copy-btn')` matching, instead of attaching one listener per button. The init script is **idempotent** on both sides: a Python module-level `_COPY_BUTTON_INIT_EMITTED` flag stops re-emission across Streamlit reruns, and a JS `window.__secMentorCopyBtnWired` flag stops re-registration if the script somehow runs twice.
- **Guard the click with `btn.__copyBtnBusy`** so a frantic user pressing the button three times only fires one copy.

**Where the code lives:**

| File | Role |
|---|---|
| `web/chat_helpers.py` — `_copy_button_html(text)` | Pure helper. Returns `<button class="bubble-copy-btn" data-label="…" data-text="<html-escaped>">📋 Copy</button>`. No JS in any attribute. Returns a single-line string — safe to pass straight into `st.markdown(..., unsafe_allow_html=True)`. |
| `web/chat_helpers.py` — `_copy_button_init_script()` | Pure helper. Returns the one-time `<script>` block with the delegated `document.addEventListener('click', …)` listener, modern + legacy clipboard paths, busy guard, label-restore, and `window.__secMentorCopyBtnWired` guard. Returns `""` on the second call within a single process. |
| `web/streamlit_app.py` — line 631 | `st.markdown(_copy_button_init_script(), unsafe_allow_html=True)` after the CSS render. Idempotent. |
| `web/streamlit_app.py` — line 1246 | `f'</div>{_copy_button_html(content)}</div></div>'` — copy button rendered next to each assistant bubble. |
| `web/streamlit_app.py` — `_CUSTOM_CSS` (~line 478) | `.bubble-copy-btn` base + `:hover`, `:active`, `:focus-visible` rules. |
| `tests/test_smoke.py` | `CopyButtonHtmlTests` (10 tests) and `CopyButtonInitScriptTests` (10 tests) pin the contract. |

**Testing the contract:** 20 unit tests in `tests/test_smoke.py` cover both halves of the split — `CopyButtonHtmlTests` pins the HTML shape (button tag, no `onclick`, four-entity escaping for `& < > "`, apostrophe escaping to `&#x27;`, Unicode, 4 KB payload, label-restore wiring) and `CopyButtonInitScriptTests` pins the init script (script-tag wrapper, idempotent re-call, delegated listener, `closest('.bubble-copy-btn')` matching, modern API, legacy fallback, busy guard, label restore, `data-text` payload read, `window.__secMentorCopyBtnWired` window guard). The full suite is 365 tests (see the project layout above for the breakdown); runs in ~10 s on a warm venv.

**XSS analysis.** A user prompt that returns an assistant reply containing `<script>alert(1)</script>` is handled cleanly: the reply text is run through `html.escape(..., quote=True)` before being dropped into the `data-text` attribute, so the payload is stored as the four-entity string `&lt;script&gt;alert(1)&lt;/script&gt;`. The browser decodes the `data-*` attribute back to the literal text on the JS side, and the only `innerHTML` write in the lifecycle is to set `btn.textContent` (label-restore), which never interprets HTML. No `eval`, no `Function()`, no `innerHTML +=`.

See `docs/copy_button.md` for the full design doc, and row 17 of `docs/technical_write_up.md` for the build log of this pass.

### Persistent chat history sidebar

Every conversation you start lives in a sidebar list on the left, ordered by **most-recently-touched first**. Click any title to reopen it; click the **+ New chat** button at the top of the sidebar to start a fresh one. The list survives page refreshes, browser restarts, and Streamlit reruns because every chat is a row in SQLite (`~/.stage1/secmentor.db`), not a slot in `session_state`.

**What the sidebar shows for each row:**

- A **title** (the first user prompt, trimmed to ~40 chars) with a **pencil icon** beside it for inline rename.
- A **relative timestamp**: `just now` (under 30 s), `5m ago`, `2h ago`, `3d ago`, or `Mar 14`. The format is computed at render time from the chat's `updated_at` column, so it stays correct as time passes without any background job.
- A **trash icon** for soft delete. Click it once: the row is struck through in the UI and greyed out, but the data is still on disk. Click the trash again (now showing ✕) to confirm. After one rerun with no confirmation, the row is hard-deleted along with its messages, artifacts, chunks, and source citations.

**Why this is not just `st.session_state["chats"]`.** Streamlit's session state is per-tab, per-process. A refresh wipes it; a server restart wipes it; opening a second tab in the same browser gives you a *different* copy of the same chats. None of those are acceptable for a learner who wants to come back tomorrow and find their SQL-injection walkthrough still in slot #2. The fix is to make the chat a row in SQLite, render the sidebar by querying `chats` ordered by `updated_at DESC`, and call `touch_chat(chat_id)` from `_ask` the moment a turn is appended. `session_state` still holds the *currently open* chat id; everything else lives on disk.

**Where the code lives:**

- `app/storage.py` — `create_chat`, `list_chats`, `get_chat`, `touch_chat`, `update_chat_title`, `soft_delete_chat`, `hard_delete_chat` (cascade to messages, artifacts, chat_chunks, sources).
- `web/streamlit_app.py` — the `_render_sidebar` block (`st.button("+ New chat")`, `st.text_input` rename inline, `st.button("🗑")` two-step confirm) and the `_ask` driver calling `touch_chat(chat_id)` after every successful turn.
- `web/chat_helpers.py` — `_relative_time(updated_at)` is a pure helper; no Streamlit import.
- `tests/test_chat_view.py` — `SidebarChatsViewTests` (sidebar list ordering, rename, soft-delete confirm) and `PersistOnAskTests` (a fake `Storage` + `RagStore` is wired into a real `_ask` to confirm the chat is `touch_chat`-ed and the message is `append_message`-ed on every send).

### RAG over uploaded files (Phase 12)

When you upload a PDF, log file, or text dump to the sidebar uploader, the artifact is not just attached to the next prompt — it is **chunked, embedded, and indexed in a per-chat FAISS store**, and the next prompt you send in that chat can retrieve from it. The retrieved chunks are dropped into the system prompt as a numbered context block (`[CTX-1] … [CTX-N]`), and the model answers against them. A **Sources** row is rendered below the assistant bubble with clickable links to the original artifact and the matched line ranges.

**The pipeline, end to end:**

1. **Upload** — `app/file_processor.process_upload(file)` returns `ProcessedFile(text, image_b64, mime)`. Text is extracted (pymupdf for PDFs, lazy-imported so it is not a hard dep for the rest of the test suite), bytes for images.
2. **Chunk** — `app/rag_chunker.chunk(text, chars_per_chunk=800, overlap=120)` splits on word boundaries. Returns `list[Chunk(chunk_id, text, start, end)]`.
3. **Embed** — `app/rag_embedder.embed(texts)` returns L2-normalized float32 vectors (`shape=(n, 384)`). The embedder is the sentence-transformers `all-MiniLM-L6-v2` model, **lazily downloaded on first use** and **cached** at `~/.cache/torch/sentence_transformers/`. A degraded mode is available: if `PUKU_RAG_OFFLINE=1` is set in `.env`, the embedder returns a zero vector and the store short-circuits to an identity search (so RAG never breaks the test suite on a machine that cannot download models).
4. **Store** — `app/rag_store.RagStore(chat_id).add(chunks)` persists each chunk to the `chat_chunks` table (`add_chunks_returning_ids` returns the row ids) and increments the chat's `_index_version` counter. The FAISS `IndexFlatIP` is **not** built on add; it is built lazily on the first `search()` call after a version bump, and rebuilt only if `_index_version` has changed.
5. **Retrieve** — at `_ask` time, `retrieve_for_chat(chat_id, query, k=5)` embeds the query, runs `index.search(query_vec, k)`, and filters results with `score > DEFAULT_SCORE_THRESHOLD = 0.30`. Cosine similarity after L2 normalization is bounded in `[0, 1]`, so 0.30 is a tight threshold — a chunk only gets in if it is *actually* related to the query, not a sloppy keyword match.
6. **Inject** — the surviving chunks are formatted as `[CTX-1] <text>\n[CTX-2] <text>\n…` and prepended to the system prompt for that turn only. The chat's `messages` history is unchanged; the context is **not** stored as a turn. (It is re-derived at every turn, so a later edit to the source artifact re-ranks the results automatically.)
7. **Cite** — each cited chunk carries `artifact_id` and `start/end` offsets. After the assistant reply renders, `web/streamlit_app.py` walks the citations and renders a `Sources:` row of clickable links.

**Why a per-chat index, not one global one.** A learner uploading a `pcap.txt` for chat A and a `sudoers.conf` for chat B should *never* see results from chat B's index when asking a follow-up in chat A. The test `tests/test_rag.py::CrossChatIsolationTests` pins this: the two stores are constructed with separate `RagStore` instances, the query is run on both, and the result sets must be disjoint. The version-counter pattern (`_index_version` on the `chats` row) is the cheap invalidation: after `add()`, the counter increments, and the next `search()` sees the new version and rebuilds — no need for an explicit "rebuild" call, no race between two reruns trying to build the same index.

**Where the code lives:**

- `app/rag_chunker.py` — `chunk(text, chars_per_chunk, overlap)` returns `list[Chunk]`. Respects word boundaries; no mid-word cuts.
- `app/rag_embedder.py` — `EmbedderProtocol` (duck-typed; `embed(texts) -> np.ndarray`), `l2_normalize(vec)`, `EMBEDDING_DIM = 384`, `DEFAULT_SCORE_THRESHOLD = 0.30`, **degraded mode** via `PUKU_RAG_OFFLINE=1`.
- `app/rag_store.py` — `RagStore(chat_id)` with `add(chunks)`, `search(query, k)`, `invalidate(chat_id)`, `rebuild(chat_id)`. Helpers `default_store()`, `add_to_chat()`, `retrieve_for_chat()`. Lazy `import faiss`; thread-safe with `RLock`.
- `app/storage.py` — `add_chunks_returning_ids`, `list_chunks_for_chat`, `get_chat_id_for_artifact`. `_index_version` column on `chats`.
- `web/chat_helpers.py` — `_build_messages` accepts an optional `context_chunks` list and injects the `[CTX-…]` block. The chunk-injection code path does not know whether the chunks came from the per-chat store or the global corpus — it just receives a `list[(chunk_id, text, score)]` and serializes.
- `tests/test_rag.py` — `RagStoreTests`, `RagSentinelTests`, `EmbedderTests`, `RagStoreAddTests`, `RagStoreSearchTests`, `RagStoreConfigTests`, `CrossChatIsolationTests`, `BuildMessagesWithRagTests`, `RagInjectionTests` — 23 tests, all green.

### Global security corpus (Phase 12 PR-E)

Beyond the per-chat index, the engine ships with a **second, independent FAISS index** over a curated security corpus: the **OWASP Top 10**, **MITRE ATT&CK** techniques, the **CWE** catalog, **GTFOBins** (legitimate-binary-abuse catalog), and **Sigma** detection rules. This index is built at first startup from JSON files in `data/corpus/`, is **deduplicated by SHA-256** so re-ingests are idempotent, and is queried at the same time as the per-chat store. The two result sets are merged (per-chat first, then corpus), filtered by `score > 0.30`, and the corpus hits are tagged with a `source_kind` (`owasp` / `mitre` / `cwe` / `gtfobins` / `sigma`) so the UI can show them in their own group with a **shield icon** instead of a generic **paperclip icon**.

**Why a second index, not just a flag on the per-chat one.** Three reasons:

1. **Different lifecycle.** Per-chat chunks are short-lived (the chat can be deleted tomorrow); the corpus is permanent and shared.
2. **Different write pattern.** Per-chat: many small adds as artifacts are uploaded. Corpus: a few large bulk loads at startup, then read-only for the rest of the run.
3. **Different trust model.** Per-chat chunks come from user uploads — they might be a log file with attacker-controlled content. Corpus chunks come from a curated, version-pinned dataset. The citations in the UI reflect this: corpus hits show their source kind and a stable external link, per-chat hits show the artifact filename and a line range.

**Query-time merge.** `app/rag_global.search(query, k=5, source_filter=None)` returns its own ranked list, and the per-chat `retrieve_for_chat` returns its own. The engine does the merge in the helper layer (`web/chat_helpers.py`) — the two `RagStore` instances are siblings, not parent-and-child. A `source_filter=("owasp", "mitre")` argument lets the UI offer a "search the corpus, ignore my uploads" toggle, useful for a learner who wants to ask "what is the canonical description of SQL injection?" without their own files muddying the result.

**Where the code lives:**

- `app/rag_global.py` — `GlobalCorpusStore` with `add(source_kind, items)`, `search(query, k, source_filter)`, `clear()`, `stats()`. Persists chunks to the `global_chunks` table keyed by `(source_kind, sha256)` for dedup. Lazy `import faiss`.
- `scripts/ingest_security_corpus.py` — one-off script: walks `data/corpus/*.json`, parses each into `(chunk_id, text, source_kind, external_url)`, and calls `GlobalCorpusStore.add()`. Re-runnable; safe to re-run after a corpus update.
- `web/streamlit_app.py` — at startup, if the `global_chunks` table is empty, calls the ingest script (or skips with a clear warning if the corpus files are missing). Renders corpus hits with a `🛡️` icon and the source kind as the link text (`OWASP A03:2021 — Injection`).
- `tests/test_rag_global.py` — `GlobalIndexConstructionTests`, `AddAndSearchTests`, `SourceFilterTests`, `ClearAndReAddTests`, `DegradedModeTests`, `CrossSourceMergeTests`, `EdgeCaseTests` — 23 tests, all green.
- `tests/test_rag_corpus.py` — parser coverage: `OwaspParserTests`, `MitreParserTests`, `CweParserTests`, `GtfoBinsParserTests`, `SigmaParserTests`, `ParseSourceDispatchTests`, `FetchSourceTests`, `CorpusRegistryTests`, `CorpusSourceValidationTests` — 22 tests, all green.

---

### Recon (OSINT) — Phase 15

The web UI also exposes a **scoped OSINT reconnaissance subsystem** that gathers public, passive information about a target domain or IP and renders the result inside the same chat as a structured report. The whole subsystem is invoked from a single slash command — type **`/recon <target> [scope:engagement]`** into the chat input and the engine fans out across five data sources in parallel, normalizes everything, and renders both a **Markdown report** and a **JSON dump** for downstream tooling.

**The reason this exists as a slash command and not a free-form chat turn.** Recon is a *tool invocation*, not a *conversation*. Treating it as a slash command makes the contract obvious to the user (you typed `/recon`, you got a recon report — no LLM rewrites your target, no chat model is invoked, no prompt injection can swap your domain for an attacker-controlled one), keeps the cost out of the LLM token budget, and gives the audit log a clean trigger to record.

**The scope-token gate — the only thing standing between `/recon` and the public internet.** The orchestrator will not dispatch *any* transport unless the target is wrapped in a `scope:` token that names a sanctioned engagement:

| Scope token | Meaning |
|---|---|
| `engagement` | A specific paid engagement, listed in the operator's own tracking doc |
| `ctf` | A CTF challenge (HackTheBox, TryHackMe, PortSwigger, DVWA, WebGoat, picoCTF) |
| `labs` | A general self-study lab (your own VM, your own subnet, your own throwaway domain) |
| `redteam` | An authorized red-team exercise with a written rules-of-engagement doc |
| `personal-lab` | A home / personal sandbox — anything you own and can re-image |
| `bugbounty` | A bug-bounty program on a platform you have an active researcher account with |

Without one of those tokens, `assert_target_allowed(target)` raises `TargetBlockedError` and the orchestrator short-circuits before any socket opens. The list lives in `app/config.py` as `_RECON_SCOPE_TOKENS = frozenset({...})` so it is greppable, single-source, and unit-testable.

**The five transports (all passive, all public-data only):**

| Transport | Data source | Bound by | Failure mode |
|---|---|---|---|
| **DNS** (`app/recon/dns_lookup.py`) | `socket.getaddrinfo` over the system resolver | `RECON_HTTP_TIMEOUT_SECONDS=15` | Returns `(ok, addresses_or_error)`; never raises out of the function |
| **URL info** (`app/recon/urlinfo.py`) | HEAD request to the URL with a custom User-Agent | `RECON_HTTP_TIMEOUT_SECONDS=15` | Returns status, final URL after redirects, server header, content-type — *not* body |
| **IP info** (`app/recon/ipinfo.py`) | `ipapi.co` then `ipinfo.io` (token if `IPINFO_TOKEN` is set) | per-provider timeout | Both providers tried in order; first one that answers wins |
| **WHOIS** (`app/recon/whois.py`) | Two-hop TCP/43 — query the TLD's WHOIS server first, follow the `whois:` referral | socket timeout | Returns registrar + a small set of dates; raw text is *not* persisted |
| **crt.sh** (`app/recon/crt_sh.py`) | `https://crt.sh?q=<domain>&output=json` (CT log subdomain enumeration) | `RECON_CRT_SH_TIMEOUT_SECONDS=20` | **Kill-switchable** via `RECON_CRT_SH_ENABLED=False` — defaults to True |

**`RECON_CRT_SH_ENABLED` is a kill-switch, not a preference.** The CT-log lookup is the one transport with a real-world quota cost (crt.sh is a free public service; hammering it gets the lab IP throttled). The env var lets an operator turn it off globally without touching code or tests, and the test suite exercises both branches (`CrtShDisabledTests`, `CrtShConfigParserTests`, `CrtShOrchestratorDisabledTests`). When disabled, the orchestrator emits a `crt_sh=disabled` field in the report (not an error, not an empty list — a labeled no-op) so the absence is visible in the audit log.

**Why refang + normalize before dispatch.** A user typing `/recon exämple[.]com` or `/recon hxxps://evil[.]example[.]com` should not bypass the safety rail, nor should the orchestrator accidentally treat the fanged form as a different target. `app/recon/normalize.py` runs `refang()` first (rewrites `[.]` → `.`, `hxxp` → `http`, punycode-decodes IDN, strips URL-unsafe whitespace), then `normalize_target()` parses the result into a `NormalizedTarget` discriminated union of `{domain, url, ip}` — the orchestrator then dispatches the right transports for the right shape (no WHOIS on a raw IP, no URL probe on a bare domain). The view layer also runs `assert_target_allowed` on the *fanged* input first so a `[.]`-obfuscated attacker target cannot sneak through.

**Parallel fan-out + per-tool budgets.** The orchestrator (`app/recon/orchestrator.py`) dispatches all five transports with `concurrent.futures.ThreadPoolExecutor(max_workers=5)` and a wall-clock budget of `RECON_HTTP_TIMEOUT_SECONDS=15` per transport. No transport can block the rest; the slowest one caps the total run at ~15s for HTTP transports and `RECON_CRT_SH_TIMEOUT_SECONDS=20` for crt.sh. Each transport returns a typed `ReconSection(status, summary, data, error)` so a single failure (e.g. crt.sh rate-limited) does not blank the rest of the report.

**Two renderers — Markdown for humans, JSON for tools.** `app/recon/report.py` exports `render_report_markdown(report)` (the bubble content) and `render_report_json(report)` (the downloadable artifact, dropped into `app/storage.py` as an `artifact` row of `kind="recon_json"` alongside the chat message). A recon chat turn therefore persists three things: the user slash command, the assistant markdown bubble, and the structured JSON artifact that other tools can re-load.

**Audit log.** Every recon invocation writes one row to `recon_audit_log` in `~/.stage1/secmentor.db`: `(timestamp, target, scope_token, tool, status, duration_ms, result_excerpt)`. `status` is the narrow enum `("ok", "blocked", "error")` — a recon call is a small structured event, not a document. `result_excerpt` is the first 500 chars of the rendered output so an investigator can scan the audit without re-running the tool; the full payload is *not* persisted (privacy + storage cost). `duration_ms` is wall-clock at the call site, so it includes the orchestrator's own overhead, not just the socket time. The audit retention is `RECON_AUDIT_RETENTION_DAYS=90` (a `VACUUM` pass runs on app startup).

**Fallback subdomains.** If DNS returns NXDOMAIN on the bare domain, the orchestrator appends the configured `RECON_FALLBACK_SUBDOMAINS` (a comma-separated list — `www,mail,api` is a sensible default) and re-queries. Each candidate is a separate `ReconSection` so a successful `www.example.com` resolve sits next to four `nxdomain` failures, not folded into one ambiguous row.

**What this is NOT (limitations to be aware of):**

- **Not active scanning.** No port scans, no banner grabs beyond the HTTP HEAD response, no nmap, no vulnerability checks. The five transports are passive / public-data only.
- **Not a substitute for Burp / nmap / Amass.** Those are the right tools when you need them; `/recon` is a 15-second "what does the public internet know about this domain" smoke test for a learner mid-ctf.
- **No DNS-over-HTTPS.** Resolves go through the system resolver (`getaddrinfo`). If the system DNS is being intercepted by the network, the recon picks that up too — *that is by design*, do not paper over it with DoH.
- **No proxy / Tor.** Outbound traffic uses the system's default route. If the operator needs anonymity, the right answer is to run the whole app inside a lab VM on a network you control.
- **One scope token per turn.** You cannot pass `engagement,redteam` and mean "either is fine"; the gate requires exactly one named token.
- **No scheduling / no cron.** Recon runs on demand. A nightly-diff mode ("what changed on this domain since yesterday?") is on the post-16 backlog.

**Where the code lives:**

- `app/recon/__init__.py` — public surface re-exports: `run_recon`, `stream_recon`, `ReconReport`, `render_report_markdown`, `render_report_json`, `normalize_target`, `refang`, `NormalizedTarget`, `TargetBlockedError`, `assert_target_allowed`.
- `app/recon/normalize.py` — `refang()`, `normalize_target()`, `NormalizedTarget`.
- `app/recon/safety.py` — `TargetBlockedError`, `assert_target_allowed()`, `_RECON_SCOPE_TOKENS` is *not* here (single-source lives in `app/config.py`).
- `app/recon/dns_lookup.py`, `urlinfo.py`, `ipinfo.py`, `whois.py`, `crt_sh.py` — one transport each, all returning `ReconSection`.
- `app/recon/orchestrator.py` — `run_recon(target, scope_token)` and `stream_recon(target, scope_token)` (yields sections as they complete for the spinner UI).
- `app/recon/report.py` — two renderers.
- `app/storage.py` — `recon_audit_log` table + `record_recon_call(...)` + `RECON_AUDIT_RETENTION_DAYS` vacuum.
- `web/streamlit_app.py` — `/recon` slash-command detection in `_ask`, wire-up of `stream_recon` to a status-spinner, JSON artifact download button.

**Testing.** 17 test classes in `tests/test_recon.py` cover every layer: `ParseReconCommandTests`, `RefangTests`, `NormalizeTargetTests`, `SafetyTests`, `DNSResolveTests`, `UrlInfoProbeTests`, `IpInfoLookupTests`, `WhoisLookupTests`, `CrtShLookupTests`, `CrtShDisabledTests`, `CrtShConfigParserTests`, `CrtShOrchestratorDisabledTests`, `CrtShReportSoftEmptyTests`, `OrchestratorTests`, `RenderMarkdownTests`, `RenderJsonTests`, `ReconStorageTests`. Network access is stubbed at the stdlib boundary (`socket`, `urllib.request`) so the suite is hermetic and runs in <1s.

See `docs/phase_15_recon.md` for the design doc (status, scope, orchestrator contract, sub-component breakdown, test plan, known limitations, rollout), row 18 of `docs/technical_write_up.md` for the build log, and `improvements.md` §B.10 for the cross-cutting summary.

---

## Safety

- The `.env` file contains your real API key. It is git-ignored. **Never** paste the real key in chat, screenshots, or source files. If you leak it, go to <https://openrouter.ai/keys> and rotate immediately.
- The web UI defaults to **mentor** mode (lab-capable persona). The CLI defaults to **defensive** mode (concept-only). Both modes share the same hard refusals (real targets, named-vendor bypasses, brand-new malware, critical infrastructure). Mentor mode adds a lab-only authorization for runnable snippets on platforms the user owns or has written authorization to test. See `docs/technical_write_up.md` Decision 6 for the boundary reasoning.
- The chatbot does not browse the web, does not call tools, and does not execute code. It is a single-turn (with session memory) text interface to a hosted LLM.
- **Five-key rotation.** The free tier is rate-limited per (key, model) pair. If you have more than one OpenRouter account, you can rotate across them. Add the extra keys to `.env`:
  ```dotenv
  OPENROUTER_API_KEY=sk-or-v1-PRIMARY...
  OPENROUTER_API_KEY_2=sk-or-v1-SECONDARY...
  OPENROUTER_API_KEY_3=sk-or-v1-TERTIARY...
  OPENROUTER_API_KEY_4=sk-or-v1-QUATERNARY...
  OPENROUTER_API_KEY_5=sk-or-v1-QUINARY...
  ```
  The engine builds a `5 keys × N models` slot pool, marks a slot disabled on 401/403, backs off briefly on 429, and round-robins the rest. The view's "Slot health" expander shows the live status. The full key never leaves the `app/` package — it is redacted to `****abcd` in every user-facing message.

---

## Deployment guide

The project is a small Streamlit app. There is no database, no auth, no background workers. Any host that can run `streamlit run` and read your `.env` is enough. The most beginner-friendly path is **Streamlit Community Cloud** (free, takes about 5 minutes once you have a GitHub repo). Two more options are listed below for when you outgrow it.

### Option A — Streamlit Community Cloud (recommended for beginners, free)

**What it is:** Streamlit's official free hosting. Push to GitHub, click a few buttons, get a public URL.

**Step 1 — put the code on GitHub**

```bash
git init
git add .
git commit -m "Initial commit — SecMentor (Stage 1)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Before you push, double-check `git status` does **not** list `.env`. The `.gitignore` already excludes it; if it shows up, **stop** and check you did not create `.env` inside the repo.

**Step 2 — connect Streamlit Cloud**

1. Go to <https://share.streamlit.io> and sign in with the same GitHub account.
2. Click **New app**.
3. Fill in:
   - **Repository:** `your-username/your-repo`
   - **Branch:** `main`
   - **Main file path:** `web/streamlit_app.py`
4. Open the **Advanced settings** section:
   - **Python version:** `3.11` (or whatever your local `.venv` uses; check with `python --version`).
5. Click **Deploy**. The first build takes 2–4 minutes. Watch the logs for `Your app is live`.

**Step 3 — add your secrets (do this BEFORE the first user hits the app)**

Streamlit Cloud does not read your `.env` file. It uses its own secrets manager.

1. In the app's dashboard, click **⋮ → Settings → Secrets**.
2. Paste this (replace the placeholder with your real key):

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-PASTE-YOUR-KEY-HERE"
   OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
   OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
   OPENROUTER_APP_NAME = "SecMentor"
   ```

3. Click **Save**. The app reruns automatically.

**Step 4 — verify**

Open the public URL (looks like `https://your-repo-name.streamlit.app`). Type a question, confirm the assistant bubble appears, then open a second tab and try a second model from the sidebar to confirm the secrets are wired.

**Optional — make the repo private.** On the Streamlit Cloud dashboard, the app inherits the repo's visibility. If the repo is private, only you and your invited collaborators can see the app.

**Troubleshooting:**

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` on first build | `requirements.txt` missing or empty | Verify the file is committed; trigger a reboot from the dashboard |
| `OPENROUTER_API_KEY not set` in the UI | Secrets not saved | Settings → Secrets → paste the TOML block above → Save |
| App boots, but every reply is `OpenRouterError: HTTP 401` | Wrong or rotated key | Generate a new key at <https://openrouter.ai/keys> and update Secrets |
| App boots, replies are `HTTP 429` banners | Free-tier rate limit | Wait ~30s, or pick a different `:free` model from the sidebar |

---

### Option B — Run on your own server (VPS, home box, or a Docker host)

Use this when you want the URL to live on a domain you own, or when you need it on a private network. The pattern is: run Streamlit behind a reverse proxy, in a detached process, with the `.env` file in the project root.

**Local detached run (what the dev session uses):**

```powershell
# Windows PowerShell — from the project root
Start-Process -FilePath ".venv\Scripts\streamlit.exe" `
  -ArgumentList "run","web/streamlit_app.py",`
              "--server.port=8765",`
              "--server.address=0.0.0.0",`
              "--server.headless=true" `
  -RedirectStandardOutput "streamlit.out.log" `
  -RedirectStandardError  "streamlit.err.log" `
  -WindowStyle Hidden -PassThru | Select-Object Id
```

```bash
# macOS / Linux
nohup .venv/bin/streamlit run web/streamlit_app.py \
  --server.port=8765 --server.address=0.0.0.0 --server.headless=true \
  > streamlit.out.log 2> streamlit.err.log &
```

Health check:

```bash
curl http://127.0.0.1:8765/_stcore/health
# expected: {"status": "ok"}
```

**Docker (recommended for VPS):**

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for the build, then Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/
COPY cli/  ./cli/
COPY web/  ./web/

EXPOSE 8501

# Streamlit's recommended runtime settings for containers
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "web/streamlit_app.py"]
```

Build and run:

```bash
docker build -t ai-security-chatbot .
docker run -d --name chatbot -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  ai-security-chatbot
```

> **Secrets in Docker:** `--env-file .env` reads the same file you use locally. Make sure `.env` is **not** copied into the image (the `.gitignore` already excludes it; the `Dockerfile` above does not `COPY` it).

**Reverse proxy (nginx) in front of Streamlit:**

```nginx
server {
    listen 80;
    server_name chatbot.example.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Streamlit WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Add HTTPS with `certbot --nginx -d chatbot.example.com`.

---

### Option C — Hugging Face Spaces (free, good alternative to Streamlit Cloud)

If you prefer to keep the project off GitHub, HF Spaces will run Streamlit apps.

1. Create a new Space at <https://huggingface.co/new-space>, SDK = **Streamlit**.
2. Upload the contents of `app/`, `cli/`, `web/`, plus `requirements.txt` and `packages.txt` (empty file is fine).
3. In the Space's **Settings → Variables and secrets**, add the same four `OPENROUTER_*` variables.
4. The Space builds automatically; the URL is `https://huggingface.co/spaces/<your-username>/<space-name>`.

---

### Things to check before any deploy

Run this checklist on a clean clone. If any item fails, the deploy will fail.

```bash
# 1. .env exists, has the key, key is not the placeholder
test -f .env && grep -q "sk-or-v1-" .env && ! grep -q "PASTE" .env

# 2. requirements.txt is committed
git ls-files requirements.txt

# 3. .env is NOT committed
git ls-files .env   # must be empty

# 4. The test suite is green
python -m unittest discover -v

# 5. The view file compiles
python -m py_compile web/streamlit_app.py web/chat_helpers.py

# 6. The app boots and answers a question
streamlit run web/streamlit_app.py --server.headless=true &
sleep 5
curl -sf http://127.0.0.1:8501/_stcore/health
```

---

### What this project is **not** ready for (yet)

- **Multi-user load.** Streamlit is single-process; a few concurrent users is fine, hundreds is not. The `app/` engine itself is stateless, so the upgrade path is to wrap it in a FastAPI service and put a real WSGI/ASGI server in front.
- **Authenticated access.** There is no login. Streamlit Community Cloud has email-gating built in; for your own server, put the app behind a reverse-proxy auth (e.g. oauth2-proxy) or a Cloudflare Access policy.
- **Browser-tested integration tests.** The unit suite covers the engine, helpers, view contracts, RAG, corpus parsers, and the recon subsystem; it does not exercise `chat_input` → rerun → bubble render end-to-end in a real browser. A `tests/test_streamlit_integration.py` using `streamlit.testing.v1.AppTest` is the owed next slice — see Decision 8 in `docs/technical_write_up.md`.

---

## License & credits

MIT — see `LICENSE`.

This is a personal learning project. The OpenRouter API is used under their terms of service. The system prompts in `app/prompts.py` are original, written for defensive cybersecurity education and lab-scoped offensive mentoring.
