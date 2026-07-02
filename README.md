# 🛡 SecMentor

> **AI-Powered Cybersecurity Learning & Analysis Platform**
> Learn · Analyze · Defend · Research

A Streamlit chatbot that answers cybersecurity questions using free LLMs routed through [OpenRouter](https://openrouter.ai). Built as a personal learning project across 7 phases plus a Session-4 post-hoc pass, a Session-7 multimodal pass, and a Session-8 UI fix pass:
plain request/response → memory → prompt engineering → web UI → refactor → multi-model selector, two-pass pattern, friendly-error classifier → file & image upload (multimodal) → **copy-to-clipboard bubble buttons (no inline JS, idempotent delegated listener)**.

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

Expected: `Ran 175 tests in …` → `OK` (28 file-processor + 147 smoke, including 20 copy-button contract tests).

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
cli/              # Terminal interface (defensive prompt by default)
web/              # Streamlit web interface + pure UI-logic helpers
tests/            # Automated checks (pytest, 175 tests: 28 file + 147 smoke)
docs/             # Technical and personal study logs
.streamlit/
  config.toml     # Theme pin: light base, dark text, enterprise-blue accent
verify_changes.py # Pre-flight check (catches stale-worker import errors)
run.py            # One-command runner: venv + deps + .env bootstrap + streamlit
improvements.md   # Change-log + tuning guide for offensive / defensive behavior
probe_nemotron.py # One-off script used during multimodal bring-up (kept for reference)
probe_post_fix.py # One-off script used after the white-on-white fix (kept for reference)
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

**Testing the contract:** 20 unit tests in `tests/test_smoke.py` cover both halves of the split — `CopyButtonHtmlTests` pins the HTML shape (button tag, no `onclick`, four-entity escaping for `& < > "`, apostrophe escaping to `&#x27;`, Unicode, 4 KB payload, label-restore wiring) and `CopyButtonInitScriptTests` pins the init script (script-tag wrapper, idempotent re-call, delegated listener, `closest('.bubble-copy-btn')` matching, modern API, legacy fallback, busy guard, label restore, `data-text` payload read, `window.__secMentorCopyBtnWired` window guard). All 175 tests pass in ~5 s.

**XSS analysis.** A user prompt that returns an assistant reply containing `<script>alert(1)</script>` is handled cleanly: the reply text is run through `html.escape(..., quote=True)` before being dropped into the `data-text` attribute, so the payload is stored as the four-entity string `&lt;script&gt;alert(1)&lt;/script&gt;`. The browser decodes the `data-*` attribute back to the literal text on the JS side, and the only `innerHTML` write in the lifecycle is to set `btn.textContent` (label-restore), which never interprets HTML. No `eval`, no `Function()`, no `innerHTML +=`.

See `docs/copy_button.md` for the full design doc, and row 17 of `docs/technical_write_up.md` for the build log of this pass.

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
- **Persistent history.** Conversation history lives in `session_state`, which is per-browser-tab. A database (SQLite is enough) is the next click when that matters.
- **Authenticated access.** There is no login. Streamlit Community Cloud has email-gating built in; for your own server, put the app behind a reverse-proxy auth (e.g. oauth2-proxy) or a Cloudflare Access policy.
- **Browser-tested integration tests.** The 175-test suite covers the units; it does not exercise `chat_input` → rerun → bubble render end-to-end. A `tests/test_streamlit_integration.py` using `streamlit.testing.v1.AppTest` is the owed next slice — see Decision 8 in `docs/technical_write_up.md`.

---

## License & credits

MIT — see `LICENSE`.

This is a personal learning project. The OpenRouter API is used under their terms of service. The system prompts in `app/prompts.py` are original, written for defensive cybersecurity education and lab-scoped offensive mentoring.
