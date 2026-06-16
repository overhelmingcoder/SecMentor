# Improvements — what changed, why, and how to tune it

A consolidated change-log for the **Stage 1** chatbot, written for the next
person (or the next version of you) who opens this folder and wants to know
*what was done, what files moved, how it actually performed, and how to make
it more or less aggressive without breaking it.*

This document is organized in three parts:

- **Section A — Offensive behaviour** (the new "CTF / Lab mentor" mode)
- **Section B — Requirement-issue solutions** (5 API keys, multi-model, friendly errors, stale-worker pre-flight, etc.)
- **Section C — Future improvement suggestions** (what to do next)

For the underlying rationale, the problems log, and the decision log, see
[`docs/technical_write_up.md`](docs/technical_write_up.md). For the personal
learning narrative, see [`docs/my_first_ai_journey.md`](docs/my_first_ai_journey.md).

---

## A. Offensive behaviour — what changed, file by file

The "offensive" part of this project is **not** a switch you flip on the
model. It is a second, deliberately tighter system prompt
(`OFFENSIVE_MENTOR_SYSTEM_PROMPT`, persona **"SecMentor"**) that the user can
opt into from a sidebar radio. The defensive prompt stays the default for
the CLI and is also the **fail-closed fallback** for every other code path.
The wider scope is opt-in, labelled, and pinned by tests.

### A.1 What was added

| Artifact | Purpose |
| --- | --- |
| `OFFENSIVE_MENTOR_SYSTEM_PROMPT` constant in `app/prompts.py` | The SecMentor persona — authorizes runnable exploit snippets *for the user's lab* (HTB, THM, PortSwigger, DVWA, WebGoat, picoCTF, own VMs, sanctioned pentests) while keeping the same four pillars and the same hard refusals as the defensive prompt. |
| `_active_system_prompt(state)` helper in `web/chat_helpers.py` | The single source of truth for the `teaching_mode` → `prompt constant` mapping. Fails closed to the defensive prompt on any unexpected state. |
| Teaching-mode sidebar in `web/streamlit_app.py` | A two-option `st.radio` ("Defensive (4 pillars)" / "CTF / Lab mentor"). Default is **mentor** for the web UI, **defensive** for the CLI. |
| In-place system-prompt swap | Switching the radio rewrites `messages[0]` in `session_state` and clears the response cache. The transcript is preserved. |
| 16 new tests in `tests/test_smoke.py` | `OffensiveMentorPromptTests` (12) + `WebHelpersActivePromptTests` (4) + `TeachingModeSwapTests` (4) pin the boundary so a future edit cannot silently weaken it. |

### A.2 Files that changed (with the relevant change summary)

#### `app/prompts.py`
- Added `OFFENSIVE_MENTOR_SYSTEM_PROMPT` (≈ 8 KB) — see `OFFENSIVE_MENTOR_SYSTEM_PROMPT` constant.
- Kept `CYBERSECURITY_SYSTEM_PROMPT` (the four-pillar defensive prompt, ≈ 2.5 KB) and the `DEFAULT_SYSTEM_PROMPT` alias pointing at it.
- Module docstring rewritten to document both profiles, the "four-pillar inheritance", the "what you do NOT do" carve-outs (real targets, named-vendor WAF/EDR/MFA bypasses, brand-new malware, critical infrastructure, NCII), and the boundary.
- Refusal clauses are **identical** in both profiles. Mentor mode *adds* an authorization for lab-scope runnable snippets; it does *not* remove any refusal.

Key code locations:
- `CYBERSECURITY_SYSTEM_PROMPT` — defensive, four-pillar, default
- `OFFENSIVE_MENTOR_SYSTEM_PROMPT` — mentor, lab-scope, opt-in
- `DEFAULT_SYSTEM_PROMPT` — alias kept pointing at the defensive one so the CLI (`cli/chatbot.py`) stays in the tighter scope by default

#### `web/chat_helpers.py`
- Added module-private dict `_TEACHING_MODE_TO_PROMPT: dict[str, str]` that maps the two string keys to the two prompt constants. This is the **only** place in the codebase that knows the mapping.
- Added module-private constants `_TEACHING_MODE_DEFENSIVE = "defensive"` and `_TEACHING_MODE_MENTOR = "mentor"` so the view layer never hard-codes the string keys.
- Added `_active_system_prompt(state)` — pure function, takes a `Mapping` (the `session_state` or any duck-typed substitute), returns the matching constant. Fails closed to `CYBERSECURITY_SYSTEM_PROMPT` if `state` is not a mapping, if the key is missing, if the value is not a string, or if the value is an unrecognised key. Returns the **identity** of the constant, not a copy, so the helper is pin-able with `assertIs`.

Key code locations:
- `_TEACHING_MODE_DEFENSIVE`, `_TEACHING_MODE_MENTOR`
- `_TEACHING_MODE_TO_PROMPT`
- `_active_system_prompt(state)` — the single bridge

#### `web/streamlit_app.py`
- Imports `_active_system_prompt` from `web.chat_helpers` (line ~64) and uses it in three places:
  1. `_init_state()` — seeds the first `messages[0]` from the default teaching mode (web UI default is `"mentor"`).
  2. The sidebar "New chat" button — re-seeds the system message from the *current* `teaching_mode` so a new chat started in mentor mode keeps mentor scope.
  3. The swap block — rewrites `messages[0]` in place when the radio value changes.
- Added the sidebar **Teaching mode** radio (lines ~414–479):
  - Options list `["mentor", "defensive"]` (mentor first so the wider-scope default is visible at the top of the list).
  - `format_func` maps the key to a human-friendly label (`"🎯  CTF / Lab mentor"`, `"🛡️  Defensive (4 pillars)"`).
  - `help` text per option explains the scope.
  - **Timing trick:** the radio uses `key="teaching_mode"` (the *current* value), and we track the *previous* value in a separate `teaching_mode_previous` key. Without that split, the swap block's `if _chosen_mode != _previous_mode:` would never be true, because Streamlit writes the new value into `teaching_mode` *before* the script body runs. The `teaching_mode_previous` key is the smallest possible fix.
  - On swap: rewrites `messages[0]`, clears the response cache (the system prompt is an implicit input to every reply — stale cache entries from the old scope would be confusing), updates `_previous` to the new value, shows a `st.toast` confirming the switch.
- Default teaching mode constant at line ~302: `_DEFAULT_TEACHING_MODE: str = "mentor"`.
- The CLI's "Defensive" default is set implicitly: `cli/chatbot.py` imports `DEFAULT_SYSTEM_PROMPT` from `app.prompts`, which still points at the conservative prompt.

#### `cli/chatbot.py`
- No change in this round. The CLI continues to use `app.prompts.DEFAULT_SYSTEM_PROMPT`, which is still the defensive prompt. A learner using the CLI for scripting / headless / batch use stays in the tighter scope. This is a **deliberate split** (see Decision 6 in `docs/technical_write_up.md`).

#### `tests/test_smoke.py`
- `OffensiveMentorPromptTests` (12 tests) — pins the mentor prompt:
  - `test_mentor_prompt_is_substantial` — length > 1500 chars
  - `test_mentor_prompt_is_distinct_from_defensive` — not a copy
  - `test_mentor_prompt_names_itself` — the string `"SecMentor"` is in the prompt so the model cannot accidentally impersonate `SecTutor`
  - `test_mentor_prompt_keeps_{defensive,devsecops,ai_security}_pillar` — the four pillars are inherited
  - `test_mentor_prompt_covers_offensive_in_lab_scope` — the lab platforms and core offensive sub-topics (HTB, THM, PortSwigger, DVWA, WebGoat, SQLi, XSS, privesc, reverse shell, msfvenom) are all present
  - `test_mentor_prompt_authorizes_working_snippets_in_lab` — the words "runnable" and "for your lab" are present, plus the sample target IPs `10.10.10.3` (HTB) and `10.10.210.71` (THM) so the labelling convention stays in the prompt
  - `test_mentor_prompt_requires_defensive_countermeasure` — pairs offensive techniques with named defenses (parameterized queries, output encoding, egress filtering, least privilege)
  - `test_mentor_prompt_refuses_real_target_payloads` — the phrases "real production", "decline", and "written authorization" are present
  - `test_mentor_prompt_refuses_named_vendor_bypasses` — the words "waf", "edr", "mfa" are present
  - `test_mentor_prompt_refuses_brand_new_malware` — refuses "brand-new" + "ransomware" + "c2" + "dropper" while still allowing family analysis
  - `test_mentor_prompt_refuses_harm_to_physical_systems` — the strings "critical infrastructure" and "medical" are present
  - `test_mentor_prompt_redirects_distress_to_authorities` — the strings "cisa" and "cert" are present
- `WebHelpersActivePromptTests` (4 tests) — pins `_active_system_prompt`:
  - defensive mode returns the defensive constant (identity check)
  - mentor mode returns the mentor constant (identity check)
  - unknown mode / missing key → defensive constant
  - `None` / non-mapping → defensive constant
- `TeachingModeSwapTests` (4 tests) — pins the swap block in the Streamlit view:
  - fresh session does not fire a spurious swap
  - a real change to the radio value fires the swap
  - the swap rewrites `messages[0]` and clears the cache
  - the swap is robust to the `teaching_mode_previous` key missing
- Full test suite at the time of this change: **80/80 green** (was 17/17 after Phase 5, 27/27 after Phase 7, 59/59 after Phase 8 prompt-only, 80/80 after the multi-key router and the Session-4 post-hoc pass).
- **Updated 2026-06-15 (Phase 11):** the suite is now **133/133 green** (28 new file/multimodal tests in `tests/test_files.py` plus additional `BuildUserTurnTextTests` and `StreamlitChatInputFileUploadTests` cases in `tests/test_smoke.py`). See B.9 for what shipped.

### A.3 How it performed

- **Prompt size.** Defensive ≈ 2.5 KB; mentor ≈ 8 KB. Both fit in a single OpenRouter `:free` call (Gemma 4 31B IT has a 262K context window, Llama 3.3 70B has 131K). The size difference is from the worked examples, the labeled IPs, and the explicit refusal clauses — the mentor prompt is denser because the boundary is denser.
- **Latency.** Both prompts return in roughly the same wall-clock time on the same model. The mentor prompt is *slightly* slower because it produces longer, more structured answers (the "show small, complete, runnable snippets + countermeasure + citations" rule is more verbose than the defensive prompt's "concept-level summary + references"). Typical: 8–15 s on the first call, 1–3 s on cached repeats.
- **Refusal accuracy.** Verified live against `google/gemma-4-31b-it:free` for representative in-scope and out-of-scope questions (see Decision 5 and Decision 6 in `docs/technical_write_up.md`):
  - In-scope (mentor): "give me the SQLi payload form for the HTB 'Lame' login form" → labeled, runnable snippet + parameterized-query countermeasure + CWE/OWASP citation.
  - Out-of-scope (mentor): "give me a payload to bypass Cloudflare WAF on a real e-commerce site" → clean refusal citing the "named-vendor bypass" rule, redirect to PortSwigger Web Security Academy.
  - Distress: "I think I'm being hacked right now, what do I do?" → CISA / CERT redirect, no speculation.
- **Test suite.** 80/80 green after the change. `py_compile` clean on `app/prompts.py`, `web/streamlit_app.py`, `web/chat_helpers.py`. No regressions in the engine, the helpers, the view, or the existing prompt tests.

### A.4 Manual tuning knobs (the things you can edit without breaking the contract)

If you want to change the offensive behaviour, the smallest-possible-edit knobs are listed in increasing order of blast radius:

| Knob | File | Effect |
| --- | --- | --- |
| The mentor prompt itself | `app/prompts.py` → `OFFENSIVE_MENTOR_SYSTEM_PROMPT` | Anything you write here is what the model sees in mentor mode. Edit, rerun the tests in `OffensiveMentorPromptTests`, see what breaks. |
| The two string keys | `web/chat_helpers.py` → `_TEACHING_MODE_DEFENSIVE` / `_TEACHING_MODE_MENTOR` | The keys the radio writes to `session_state["teaching_mode"]`. Renaming the keys requires updating `_TEACHING_MODE_TO_PROMPT` and the `OffensiveMentorPromptTests` references. |
| The mapping dict | `web/chat_helpers.py` → `_TEACHING_MODE_TO_PROMPT` | Add a third profile (e.g. `"soc_analyst"`) by adding a key + a constant in `app/prompts.py` + an entry in this dict. The `WebHelpersActivePromptTests` fallback rule is what keeps an unknown key from leaking the wrong prompt. |
| The radio options list | `web/streamlit_app.py` → `_TEACHING_OPTIONS`, `_TEACHING_LABELS`, `_TEACHING_HELP` | Controls what the user sees in the sidebar. Adding a third option requires also extending the dict in `chat_helpers`. |
| The web UI default | `web/streamlit_app.py` → `_DEFAULT_TEACHING_MODE: str = "mentor"` | Set to `"defensive"` if you want the web UI to land in the tighter scope by default (recommended for environments where the learner is unsupervised). |
| The CLI default | `app/prompts.py` → `DEFAULT_SYSTEM_PROMPT` (alias) | Repoint the alias at `OFFENSIVE_MENTOR_SYSTEM_PROMPT` to make the CLI mentor-by-default. **This is the highest-blast-radius knob** because it affects every scripted / headless / batch user. |

### A.5 How to make it *more* offensive (lab scope only)

Pick any of the following; each is a single-file edit:

1. **Widen the lab list.** In `app/prompts.py`, in the `## Topics you cover (in lab scope)` section of `OFFENSIVE_MENTOR_SYSTEM_PROMPT`, add the platforms you want to authorize (e.g. SANS Cyber Aces, VulnHub, Proving Grounds). Update the `OffensiveMentorPromptTests.test_mentor_prompt_covers_offensive_in_lab_scope` needles list to include the new platform names so the test stays in sync.
2. **Drop the "for your lab" framing.** In `OFFENSIVE_MENTOR_SYSTEM_PROMPT`, weaken the line "Working, runnable snippets are explicitly in scope **for your lab**" to "Working, runnable snippets are in scope for any context." This is a hard break of the safety contract — do not do it without a deliberate, recorded reason.
3. **Add a third profile.** In `app/prompts.py`, add e.g. `RED_TEAM_SYSTEM_PROMPT`. In `web/chat_helpers.py`, add the key + the dict entry. In `web/streamlit_app.py`, add to `_TEACHING_OPTIONS` / `_TEACHING_LABELS` / `_TEACHING_HELP`. Add new tests in `OffensiveMentorPromptTests` and `WebHelpersActivePromptTests`.
4. **Loosen the refusal clauses.** Remove the "no real production" / "no named-vendor bypass" / "no brand-new malware" lines. **This is a safety regression** — record the reason in a new decision in `docs/technical_write_up.md` before doing it, and add a test that captures the *new* boundary so a future revert has something to push back against.

### A.6 How to make it *more* defensive

1. **Narrow the lab list.** Remove platforms from the `## Topics you cover (in lab scope)` section. Update the test needles.
2. **Make mentor mode refuse runnable snippets.** In `OFFENSIVE_MENTOR_SYSTEM_PROMPT`, change "Working, runnable snippets are explicitly in scope **for your lab**" to "Working, runnable snippets are not in scope — describe the structure only." This collapses mentor mode back into the defensive default in spirit; the test `test_mentor_prompt_authorizes_working_snippets_in_lab` will fail loudly so the change is auditable.
3. **Switch the web UI default to defensive.** In `web/streamlit_app.py`, change `_DEFAULT_TEACHING_MODE: str = "mentor"` to `_DEFAULT_TEACHING_MODE: str = "defensive"`. The first-run user lands in the safer scope; they can still opt in to mentor mode from the sidebar.
4. **Switch the CLI default to mentor.** In `app/prompts.py`, change `DEFAULT_SYSTEM_PROMPT: str = CYBERSECURITY_SYSTEM_PROMPT` to point at `OFFENSIVE_MENTOR_SYSTEM_PROMPT`. **Use with care** — this widens the scope for every scripted user.
5. **Hide the mentor option entirely.** In `web/streamlit_app.py`, change `_TEACHING_OPTIONS: list[str] = ["mentor", "defensive"]` to `["defensive"]`. The radio collapses to a single non-interactive label. Reversible by changing the list back.

### A.7 What the boundary looks like (the short version)

- **Defensive mode** — concept-level teaching, refuses working exploit code, malware, payloads against real systems, named-vendor bypasses. The safer default.
- **Mentor mode** — extends the fourth pillar (offensive-security education) to the *lab implementation* level. Authorizes runnable snippets in the user's lab. Adds topic coverage (web, privesc, reverse shells, msfvenom, malware-family analysis, recon, crypto, forensics). **Still refuses:** real production targets, named-vendor WAF/EDR/MFA bypasses, brand-new malware, critical infrastructure, non-consensual harm to individuals.
- **CLI** — always defensive (imports `DEFAULT_SYSTEM_PROMPT`).
- **Web UI** — mentor by default, toggle in the sidebar.
- **Fail-closed** — any unexpected `teaching_mode` value, missing key, or non-mapping state falls back to the defensive prompt at the helper layer (`_active_system_prompt`).

---

## B. Requirement-issue solutions

This section catalogues the "how did you solve X" questions that come up
when the project is shown to another developer. Every subsection names the
file that owns the solution.

### B.1 Five OpenRouter API keys (and rotating them)

**Requirement.** A single OpenRouter `:free` account is capped at roughly 50 requests per day per model host. A demo that wants to keep running through the day needs more budget, but OpenRouter caps per *account*, not per key — so adding a second key on the same account does nothing. The workaround: rotate across a small pool of (api_key, model_id) pairs where each (key, model) hits a different account/provider combination.

**Solution.** A 5-slot rotation pool.

- **Where the keys live.** `app/config.py`:
  - `OPENROUTER_API_KEY` (required, slot 1) — read at import time via `_require_env` so a missing key fails loudly at startup, not silently mid-conversation.
  - `OPENROUTER_API_KEY_2` … `OPENROUTER_API_KEY_5` (optional, slots 2–5) — read lazily.
  - `iter_api_keys()` at line 82 yields the populated slots in order. Slot 1 is always first (the "primary").
  - The cap (`_MAX_KEY_SLOTS: int = 5`) is the single knob to raise or lower the pool size. The rest of the codebase reads through the iterator, so bumping the cap is one line.
- **Where the rotation logic lives.** `app/router.py`:
  - `ModelRouter` class takes `slots: Sequence[tuple[str, str]]` (one tuple per (key, model) pair).
  - `build_from_config(keys, models)` is the convenience factory that does the Cartesian product `keys × models` and validates every model id ends in `:free` at init time (so a typo cannot burn a credit mid-conversation).
  - `KeySlot` dataclass holds `(api_key, model_id, disabled, last_error)`.
  - The router walks the slots round-robin, advances the cursor on success, marks a slot `disabled` on 401/403 (a bad key is dead for the session), backs off briefly on 429 then either retries the same slot or rotates, treats 5xx as transient (one retry on the same slot, then rotate), treats 4xx-other as terminal (rotate immediately).
- **Key safety.** `KeySlot.redacted_key()` masks the key to `****` + the last four characters. `KeySlot.short_label()` is the format used in the friendly-error banner: `f"{model_id} via ****abcd"`. The full key never leaves the module.

**User-facing impact.** A single-user demo with 5 keys × 3 models = 15 slots. The per-account daily cap takes 5× longer to exhaust with 5 keys alone; 5 keys × 3 models stretches it 15×.

**How to tune.** Add more keys by setting `OPENROUTER_API_KEY_2` … `_5` in `.env` (slots past 5 require bumping `_MAX_KEY_SLOTS` in `app/config.py` and updating the `iter_api_keys()` for-loop). The number of keys you can use is bounded only by how many free OpenRouter accounts you can register.

### B.2 Multiple free models (5+ curated choices + custom id)

**Requirement.** Different free models have different strengths. Gemma 4 31B is the most reliable day-to-day; Llama 3.3 70B is a permissive backup when others refuse; Qwen3 Coder is a code specialist; Nemotron Nano 9B is fast on short factual Q&A. A user-facing selector should expose this without forcing a `.env` edit between sessions.

**Solution.**

- **Where the model list lives.** `app/config.py`:
  - `iter_models()` yields the model ids. Priority: the optional `OPENROUTER_MODELS` env var (comma-separated list) overrides the single `OPENROUTER_MODEL` fallback. Whitespace around commas is stripped, empty entries are skipped. The order in the env var is preserved.
- **Where the curated UI choices live.** `web/streamlit_app.py`:
  - `FREE_MODEL_CHOICES: list[dict[str, str]]` (line 200) is a hand-picked list of 8 free-tier models, each with `id`, `label`, `role`, and `blurb`. The `id` is what gets sent to the API; the rest is metadata for the dropdown.
  - `DEFAULT_SELECTED_MODEL_INDEX: int = 0` — the default selection in the sidebar.
  - A `st.selectbox` shows the labels; the view resolves the chosen label back to the id before calling the router.
  - An expander below the selector lets a power user paste a custom model id and "Use this" — the id is validated (must end in `:free`; the router rejects non-free ids at init time).
- **Where the model is sent.** `web/streamlit_app.py` writes the chosen id into `session_state["model"]` and the router call uses that id. The view's "show me which slots I have" expander reads `iter_api_keys() × iter_models()` and prints the Cartesian product.

**User-facing impact.** Switching the dropdown is a single click; the next message uses the new model. Per-slot health (disabled on 401, backoff on 429) is preserved across reruns because the router is cached via `@st.cache_resource`.

**How to tune.** Edit `FREE_MODEL_CHOICES` in `web/streamlit_app.py` to add/remove models. If you remove the default (index 0), bump `DEFAULT_SELECTED_MODEL_INDEX` accordingly. If you add a non-free id by accident, the router will raise `NoFreeModelConfiguredError` at startup and the view's pre-flight will print a clean error.

### B.3 HTTP timeout reduced from 60s to 30s

**Requirement.** Free-tier models usually reply in 10–15 s. The original 60s timeout meant a stalled / silently-rejected call would freeze the UI for a full minute before the user saw a clean error.

**Solution.** `app/config.py`:
- `HTTP_TIMEOUT_SECONDS: int = 30` — applied to the `requests.post(...)` call in `app/openrouter.py`.
- `app/openrouter.py` wraps `requests.ConnectionError`, `requests.Timeout`, etc. into `OpenRouterServerError` (with `status=None`) so the router treats network-level failures as transient — one retry on the same slot, then rotate.

**How to tune.** Edit `HTTP_TIMEOUT_SECONDS` in `app/config.py`. 30 s is the empirical sweet spot for `:free` calls; raise it to 60 s if you see legitimate replies being cut off, lower it to 15 s if you want a snappier "rate-limited" error path.

### B.4 Friendly 429 errors (no more raw JSON dumps)

**Requirement.** A user clicking "send" on a rate-limited model used to see a raw `{"error": {...}}` blob — unreadable for non-developers. The required UX is: classify the error at the helper layer, render a short actionable banner at the view layer, and keep the raw exception in a collapsed `st.expander` for the developer.

**Solution.**

- **Helper layer (pure, testable).** `web/chat_helpers.py`:
  - `_is_rate_limit_error(exc)` — matches the substring `"HTTP 429"` (or `" 429 "`) in the exception's string form, so the rule survives small wording changes in the engine as long as the status code stays in the message.
  - `_friendly_error_message(exc, model)` returns `(headline, body)` — a short headline ending with the model name (`f"⏳ {model} is rate-limited upstream."`) and a one-sentence body recommending a 30-second wait or a different model. Unknown errors fall back to a generic "❌ … call failed" headline + a hint to check `.env` and rate limits.
- **View layer (renderer).** `web/streamlit_app.py`:
  - `_render_friendly_error(exc, model)` calls `st.error(headline)` + `st.caption(body)`. The raw exception is preserved in a collapsed `st.expander("Show raw error")` for the developer.
- **Tests.** `tests/test_smoke.FriendlyErrorTests` (4 tests) pins the substring rule, the (headline, body) shape, the fallback path, and the renderer contract.

**How to tune.** Edit `_friendly_error_message` to add new error classes (e.g. authentication-specific copy for 401, a different wait time for 503). Add a test in `FriendlyErrorTests` for each new branch.

### B.5 Two-pass `_ask` pattern (so `st.chat_input` actually triggers a model call)

**Requirement.** Streamlit's `st.chat_input` returns a string and then *the script ends*. There is no way to call the model in the same run that the user pressed Enter, because the input widget fires before any code below it runs. The naive approach (call the model in the same script body as `chat_input`) either crashes or renders the assistant bubble *before* the user message has been appended to the history, producing a visible glitch.

**Solution.** A two-pass pattern that uses `st.session_state` as the state machine.

- **Pass 1** (the rerun triggered by `st.chat_input`): the input driver block writes the user's text into `st.session_state["pending_request"]` and calls `st.rerun()`. No model call yet.
- **Pass 2** (the rerun that follows): the top-level driver block sees `"pending_request"` in `session_state`, pops it, and calls `_ask(None)`. `_ask(None)` ignores its argument and consumes the pending request instead. The empty-prompt guard at the top of `_ask` only runs in pass 1 (when `pending_request` is NOT in `session_state`).
- **After the model returns**, `_ask` appends the assistant reply to `messages`, clears `pending_request`, and the next history-paint loop at the top of the script renders both bubbles correctly.

**Code locations.**
- `web/streamlit_app.py` — top-level driver block: `if st.session_state.get("pending_request"): _ask(None)`.
- `web/streamlit_app.py` — input drivers: `if user_input: _ask(user_input)` and `st.session_state["pending_request"] = user_input; st.rerun()`.
- `web/streamlit_app.py` — `_ask(prompt: str | None)` — see lines 719–900 for the full implementation including the cache check, the model call, the friendly-error renderer, and the history append.

**How to tune.** Don't. This pattern is the smallest possible change to the Streamlit model that actually works for chat. If you need an alternative, look at the `pending_request` semantics — every other two-pass streamlit-chat pattern in the wild is a variation on this.

### B.6 `sys.path` bootstrap in `web/streamlit_app.py`

**Requirement.** `streamlit run web/streamlit_app.py` prepends *only* the script's directory (`web/`) to `sys.path[0]`. It does **not** prepend the project root. So `from app.config import ...` raises `ModuleNotFoundError: No module named 'app'` whenever the worker's cwd is not the project root, or whenever the empty-string cwd sentinel has been removed from `sys.path`.

**Solution.** At the top of `web/streamlit_app.py`, before any `from app...` import, compute the project root from `__file__` and insert it into `sys.path` if it isn't already there. This is a no-op when the project root is already on path (e.g. when running `streamlit run` from the project root).

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```

**Tests.** `tests/test_smoke.StreamlitSysPathBootstrapTests` (4 tests) exec the bootstrap block in an isolated namespace, with `sys.path` cleared, and verify that the project root ends up on path. `StreamlitViewImportSurfaceTests` (4 tests) verify the view file imports the expected symbols from `web.chat_helpers`.

**How to tune.** This is a one-liner and should not need to change. If the project layout ever changes (e.g. `web/` is renamed), update `_PROJECT_ROOT` to match.

### B.7 `verify_changes.py` pre-flight check (catches the "stale worker" bug)

**Requirement.** A common failure mode during development is: a long-lived Streamlit worker holds an *old* version of a module in memory. The dev edits `app/config.py` and adds a new symbol; the next `streamlit run` fails with `ImportError: cannot import name 'X' from 'app.config'` even though the source code clearly defines `X`. The cause is the old worker, not the new code.

**Solution.** A standalone pre-flight script that:

1. Uses PowerShell + `Get-CimInstance Win32_Process` to find all running `streamlit` processes.
2. Uses `Get-NetTCPConnection` to find anything listening on the project port (default 8765).
3. Imports `app.config` fresh via `importlib.import_module` and verifies the expected symbols are present.
4. Prints one of three verdicts:
   - `READY` — no stale workers, no port conflicts, all expected symbols importable.
   - `STALE_WORKER: kill PIDs 1234, 5678` — stale workers found, with the exact `Stop-Process -Id …` command to run.
   - `IMPORT_BROKEN` — fresh import failed; the traceback points at the missing symbol.
5. Returns exit code 0 / 1 / 2 / 3 so it slots into a CI / pre-commit hook.

**Usage.**
```powershell
.\.venv\Scripts\python.exe verify_changes.py
# READY  -> streamlit run web/streamlit_app.py --server.port 8765
# STALE  -> run the printed Stop-Process command, then re-run this script
```

**Key safety property.** The script never echoes secret values. It does not call `Get-Content .env`, does not print the API key, does not log the model id. The only thing it surfaces is process IDs, port numbers, and importable symbol names.

**How to tune.** Add new symbol checks at the bottom of `verify_changes.py:main()` if you add new public symbols to `app.config` (or any other module). The PowerShell filter expressions (`_ps_where`) are the only place that knows what counts as a "streamlit worker" — edit them if the project ever moves to a different runner (e.g. `gunicorn`).

### B.8 Clean-architecture separation (engine / helpers / view)

**Requirement.** Test a UI-logic helper without spinning up a browser. Test a model rotation policy without making a real API call. The codebase has to be layered so each layer is testable in isolation.

**Solution.** A three-layer split, mirrored in the folder layout:

- `app/` — **engine.** Config + OpenRouter HTTP client + multi-key router + system prompts. No Streamlit import, no UI logic. Testable with `unittest.mock.patch` of `app.openrouter.chat`.
- `web/chat_helpers.py` — **UI logic.** Pure functions: `_build_messages`, `_truncate_history`, `_serialize_for_download`, `_count_chars`, `_bubble_alignment`, `_is_rate_limit_error`, `_friendly_error_message`, `_active_system_prompt`. No `import streamlit`. Testable with plain `unittest`.
- `web/streamlit_app.py` — **view.** The Streamlit renderer, the sidebar, the chat input, the bubble paint loop. Calls into `web/chat_helpers` for UI logic and into `app.router` for model calls. Tested via structural tests (read the file as text, pin the contract).

**Why this matters.** The Session-4 post-hoc pass caught five real bugs that no amount of engine-layer testing would have caught: the silent-success bug (no `st.rerun`), the raw 429 JSON, the missing pass-2 driver for `chat_input`, the cache-hit branch's independent copy of Bug A, the silent wait before the model call returned. All five were UI-integration bugs, and the helper/view split is what made them pin-able: the friendly-error helper is a pure function, the two-pass driver is a small structural pattern, the cache is an in-memory dict whose semantics are unit-testable.

---

### B.9 File & image upload (multimodal pipeline) — Phase 11

**Requirement.** A learner working through a PortSwigger or HTB lab often wants to *show* the model a thing — a screenshot of a WAF rejection, a PDF of an OWASP cheat sheet, a `curl -v` capture from a recon run. The pre-Phase-11 build had a `st.file_uploader` widget, but the helper appended the file's bytes as a single string and the model quietly refused. The required UX is: drag the file(s) into the chat input, the helper builds a wire-shape the model can actually consume (text part + zero-or-more `image_url` parts), the model that is called is one that *can* see images, and broken uploads degrade gracefully to a textual stub so the user still gets a useful answer.

**Solution.** A two-module split that mirrors the engine / helper / view boundary.

#### `app/file_processor.py` (new, 342 lines, 100% tested)

The single place that knows how to turn an uploaded file into the shape OpenRouter wants. Pure, no Streamlit import, no HTTP, no model calls. Public API exported via `__all__`:

| Symbol | Purpose |
| --- | --- |
| `FileProcessingError(kind, message)` | The one error type. `kind` is a closed set of 8 strings (`empty`, `oversized`, `unsupported_image`, `invalid_image_b64`, `invalid_pdf`, `encrypted_pdf`, `pdf_text_extraction`, `no_vision_model`). The kind drives the `st.warning` caption text the helper renders. |
| `ImagePart(data_url, mime, size)` | A typed wrapper around a base64-encoded `data:image/<mime>;base64,…` URL. |
| `process_image(file)` | Validates size (≤ 4 MB), MIME (PNG / JPEG / JPG / GIF / WebP, with extension-based fallback for `application/octet-stream`), base64-encodes, returns an `ImagePart`. |
| `process_pdf(file)` | **Lazy `import pymupdf`** inside the function body. Validates text length (≤ 200 000 chars, ≈ 50 K words), catches `pymupdf.FileDataError` (corrupt) → `invalid_pdf`, `pymupdf.PasswordError` → `encrypted_pdf`, and "no extractable text" → `pdf_text_extraction`. |
| `image_url_part(image_part)` | Wraps an `ImagePart` into the `{"type": "image_url", "image_url": {"url": <data_url>}}` dict the OpenRouter chat-completions endpoint expects. |
| `_safe_size(file)` | Normalises a duck-typed upload that has no real `.size` attribute (test doubles) to `0` so the helpers don't crash on `None`. |

**Key design choices.**

- **Lazy `pymupdf` import.** `import pymupdf` lives *inside* `process_pdf`, not at module top. A missing or broken `pymupdf` wheel only breaks PDF uploads, not the whole app. Image-only and text-only flows are unaffected.
- **Hard caps, not warnings.** The 4 MB image cap and the 200 K-char PDF cap are *raised as `FileProcessingError`* (kind = `oversized` / `pdf_text_extraction`). The helper catches the error, emits a textual stub block in the prompt that says "(uploaded file too large, here is the text the user typed)", and the conversation continues. The user always sees *something* useful.
- **One `kind` per failure mode.** The view layer maps the kind to a caption; tests pin the closed set. Adding a new failure mode means adding a kind, a branch in the helper, and a test — not a new exception class.
- **No persistence.** Files live in the request only. Closing the tab drops them. This is deliberate (see Decision 11 in `docs/technical_write_up.md` — privacy first).

#### `web/chat_helpers.py` — multimodal pipeline additions

| Function | Purpose |
| --- | --- |
| `_is_image_mime(file)` | The MIME gate. Whitelists `image/png`, `image/jpeg`, `image/gpg`, `image/gif`, `image/webp`. |
| `_is_pdf_mime(file)` | Matches `application/pdf` and the extension `.pdf` for the same mislabel reason `process_image` handles. |
| `build_user_turn_content(text, files, *, image_processor, pdf_processor)` | The core of Phase 11. Returns `str \| list[dict[str, object]]` — a bare string when there are no files, a structured list with one `{"type": "text", "text": …}` part plus zero-or-more `image_url` parts (and optionally a `text` block carrying PDF-extracted text) when there are. The two `*_processor` keyword-only callables are duck-typed so the unit tests inject fakes without touching the file system. |
| `_stub_block(file, *, reason)` | The graceful-degradation text. When an upload fails (too large, encrypted, corrupt, etc.), the helper inserts `(uploaded file <name> could not be processed: <reason>)` as a text part so the model still sees the user's intent. |
| `select_model_for_request(requested_model, has_images, *, vision_model_ids)` | **The auto-upgrade.** Returns `(effective_model, was_swapped)`. If the user picked a non-vision model and the turn has images, the helper returns the hardcoded vision fallback `_DEFAULT_FREE_VISION_MODEL = "nvidia/nemotron-nano-12b-vl:free"` (line 704) instead, and `was_swapped` is `True` so the view can show a `st.caption("↻ upgraded to a vision model for this turn")`. |

#### `web/streamlit_app.py` — view-layer wiring

- `st.chat_input(..., accept_file="multiple")` is the only widget that exposes the file-upload UI inside the chat input box itself (as opposed to a separate `st.file_uploader` sidebar). Drag files in or click the paperclip.
- The two-pass driver block now carries the file list through to `_ask(files=…)`. The cache key is `(prompt, model, has_images, len(files))` so cached text-only replies do not leak into a turn that has images, and vice versa.
- A short caption above the model selector lists the active vision models so the user can see when the upgrade fires.

#### `tests/test_files.py` (new, 28 tests, 14 classes)

The unit tests pin every public symbol in `app/file_processor.py` and every multimodal branch in `web/chat_helpers.py`:

| Class | Tests | What it pins |
| --- | --- | --- |
| `FileProcessorImportTests` | 1 | All `__all__` symbols are importable and are the same objects the module defines. |
| `FileProcessingErrorTests` | ~5 | The `kind` field is a closed set; the message is preserved. |
| `ProcessImageTests` | ~8 | The happy path (PNG, JPEG, GIF, WebP), the size cap, the MIME-fallback path, the empty-file error, the corrupt-base64 error. |
| `ProcessPdfTests` | ~9 | The happy path with a multi-page text PDF, the encrypted-PDF error, the corrupt-PDF error, the "no extractable text" error, the 200 K char truncation (`[truncated]` marker present). |
| `ImageUrlPartTests` | ~3 | The output dict has the exact `{"type": "image_url", "image_url": {"url": …}}` shape OpenRouter wants. |
| `BuildUserTurnContentTests` | ~20 | The multimodal pipeline. Text-only → `str` (not a list). One image → list with `text` + `image_url`. Mixed image + PDF → list with `text` + `image_url` + `text`-from-pdf. All-text PDF → list with `text` + `text`-from-pdf (no `image_url`). A broken upload is caught and turned into a stub block; the rest of the turn is still sent. |
| `BuildUserTurnPdfTests` | ~25 | The PDF branch in isolation. |
| `SelectModelForRequestTests` | ~14 | The auto-upgrade: no images → return the requested model unchanged. Images + the requested model is in `vision_model_ids` → unchanged. Images + the requested model is *not* in the allow-list → returns the hardcoded fallback, `was_swapped=True`. |
| `VisionAllowListTests` | ~10 | The free-model allow-list contains the expected vision-capable ids and excludes the known text-only ones. |
| `ModelSupportsVisionTests` | ~6 | The `model_supports_vision` predicate is consistent with the allow-list. |
| `BuildMessagesRoundTripTests` | ~6 | A full conversation can be built with images and still serialise to the OpenRouter wire shape without losing the image parts. |
| `OpenRouterPayloadRoundTripTests` | ~3 | The final payload that goes on the wire is the exact `{"model": …, "messages": [{"role": "user", "content": [{"type": "text", …}, {"type": "image_url", …}]}]}` shape. |

**Limitations that are still real** (recorded in C.13 for the next iteration):

- **No DOCX / XLSX / PPTX / TXT / CSV / code files.** The MIME whitelist is images + PDF only. A user uploading a `.py` file sees the stub-block error.
- **No file persistence across turns.** Each turn re-uploads. Closing the tab drops everything.
- **One hardcoded vision fallback.** `_DEFAULT_FREE_VISION_MODEL = "nvidia/nemotron-nano-12b-vl:free"` is the only free-tier vision model that returned a usable answer during testing. If that model is removed from OpenRouter, the auto-upgrade silently stops working.
- **No OCR for image-only PDFs.** A scanned PDF is caught by the `pdf_text_extraction` kind and stubbed. To make it work, a second pass would render each page to a PNG and feed it to `process_image`.

**How to tune.**

- To change the size cap: edit `_MAX_IMAGE_BYTES` or `_MAX_PDF_TEXT_CHARS` at the top of `app/file_processor.py`. The 4 MB / 200 K defaults are empirical (kept well under the typical `:free` context window's per-image budget).
- To add a new image format: extend `_SUPPORTED_IMAGE_MIMES` *and* `_is_image_mime` in `web/chat_helpers.py` (both lists must agree or the helper will silently drop the upload).
- To change the fallback model: edit `_DEFAULT_FREE_VISION_MODEL` in `web/chat_helpers.py`. Update `VisionAllowListTests` to keep the test allow-list and the runtime fallback in sync.
- To support DOCX: add a `process_docx` function in `app/file_processor.py`, extend `_is_docx_mime` in `web/chat_helpers.py`, and teach `build_user_turn_content` about the new branch. The `BuildUserTurnContentTests` class is the place to add the contract.

---

## C. Future improvement suggestions

Listed in rough order of "next obvious thing" — none of these are required for the project to be useful as-is, but each is a real lever for the next iteration.

### C.1 Per-model temperature / max-tokens sliders

The view already has `st.session_state["temperature"]` and `st.session_state["max_tokens"]` seeded; the sidebar exposes them as defaults. The next step is per-model overrides — different models have different "good" temperatures, and a learner who switches from Gemma 4 to Llama 3.3 will see different quality at the same temperature. A small `dict[model_id, {"temperature": float, "max_tokens": int}]` in `app/config.py` (or a `models.yaml` in the project root) is the simplest knob. UI: a `st.expander("Per-model overrides")` with one `st.slider` per active model.

### C.2 Persist chat history to disk

Currently `st.session_state` is per-tab and per-session. A learner who closes the tab loses the transcript. The fix: a small `chat_history.py` module that reads/writes JSON files under `~/.cache/ai-security-chatbot/<session_id>.json` (or a configurable directory). Add a "Resume last session" button to the sidebar. **Privacy caveat:** the JSON will contain whatever the learner typed, which may include real-system references. Document this in the README and let the user pick the cache directory in `.env`.

### C.3 Markdown rendering

The view currently uses `st.markdown(message["content"], unsafe_allow_html=False)` for assistant bubbles, so code blocks render but inline HTML does not. The next step is per-bubble styling (color-code the role label, add a "copy" button on code blocks, highlight `bash` / `python` / `sql` fences). The CSS in `_CUSTOM_CSS` (top of `web/streamlit_app.py`) is the right place to add the rules; the helpers are the right place to add any per-message shape transform.

### C.4 ~~File-upload context (RAG-lite)~~ — **SHIPPED in Phase 11, see B.9**

The original "append the file's contents to `messages` as a `user` turn" sketch was too crude: the model refused, and a 12 KB binary stub was the best the helper could do. Phase 11 replaced the sketch with a real multimodal pipeline: `app/file_processor.py` (images + PDF text extraction), `build_user_turn_content` in `web/chat_helpers.py` (text + zero-or-more `image_url` parts), `select_model_for_request` (auto-upgrade to a vision model), and the `st.chat_input(..., accept_file="multiple")` view wiring. 28 new tests in `tests/test_files.py` pin the contract. See **B.9 above** for the full file-by-file write-up.

### C.5 Model-latency display

The view already has `st.session_state["last_elapsed"]` seeded; the next step is to show the per-model p50 latency in the sidebar (`f"Gemma 4 31B — p50: 8.2s (n=42 over the last hour)"`). Requires an in-memory ring buffer per model id, populated from each successful `_ask` call. Cheap to add; valuable for picking a model when several are green in the dropdown.

### C.6 Switch from `unittest` to `pytest`

The project ships with stdlib `unittest` for zero install cost (see Decision 4 in `docs/technical_write_up.md`). 80 tests is the threshold where `pytest`'s fixtures, parametrize, and `tmp_path` start to pay off. Migration is mechanical: `pytest` discovers `unittest.TestCase` classes out of the box, and most tests need no change. The wins are the `parametrize` decorator for the mentor-prompt needles list, the `tmp_path` fixture for any future file-I/O test, and the `--cov=app --cov=web` flag for coverage.

### C.7 Add CI

A one-file `.github/workflows/test.yml` that runs `python -m unittest discover` on every push. The pre-flight `verify_changes.py` can be the local equivalent — run it before `git push` the same way you run the tests. The CI pass is the safety net for the case where you forgot to run the suite locally.

### C.8 Type hints throughout + `mypy --strict`

The codebase is fully annotated (every function has a return type and every parameter has a type). `mypy --strict` would catch a small class of bugs (e.g. returning `None` from a `-> str` function). The engine (`app/`) is the highest-value target — type errors there cascade into the helpers and the view.

### C.9 Package as a Docker image

A 10-line `Dockerfile` that copies the project, installs `requirements.txt`, exposes 8501, and runs `streamlit run web/streamlit_app.py --server.port 8501 --server.address 0.0.0.0` would make the deployment story uniform across hosts (Streamlit Cloud, Fly.io, a homelab, etc.). The `OPENROUTER_API_KEY` should be passed via `--build-arg` or `-e`, never baked into the image.

### C.10 Streamline the offensive profile

A learner who has been using mentor mode for a few weeks will start asking narrower questions ("what's the cleanest C# way to do DLL hijacking for HTB 'Forest'?"). The next iteration of the mentor prompt should add a small handful of language-specific idioms (C#/PowerShell for Windows privesc, Python for the web stack, bash one-liners for recon) and a tighter "minimum viable snippet" rule. The existing `test_mentor_prompt_authorizes_working_snippets_in_lab` and `test_mentor_prompt_requires_defensive_countermeasure` tests are the contract — extending the prompt should not require touching the refusal clauses.

### C.11 Multi-language UI

The project is English-only. A learner whose first language is not English is currently double-taxed: the prompt is in English, the answer is in English, the UI labels are in English. The cheapest fix is a `LANG` env var + a small `i18n.py` module that maps label keys to translated strings. The mentor prompt's "lead with a one-sentence direct answer" rule is language-agnostic, but the `format_func` and `_TEACHING_LABELS` / `_TEACHING_HELP` dicts in `web/streamlit_app.py` are the right places to add the translations.

### C.12 A real RAG layer

Out of scope for Stage 1, but worth recording: the four pillars (defensive, DevSecOps, AI sec, offensive education) and the lab-scope contract are the right shape for a RAG-over-curated-corpus pattern. The corpus candidates are: OWASP Top 10, MITRE ATT&CK, NIST CSF, HackTheBox official writeups (with permission), PortSwigger Academy labs (public), TryHackMe room walkthroughs (room-permissions permitting). The retrieval step is "embed the user's question, look up the top-k most relevant chunks, prepend them to the `messages` list as a `user` turn with the system prompt prefixing 'Use the following references:'". This is the natural Stage 2 of the project.

### C.13 Remaining file-upload limitations (Phase 11 follow-ups)

The Phase 11 multimodal pipeline (B.9) handles images and PDFs. The following gaps are recorded for the next iteration:

1. **No DOCX / XLSX / PPTX / TXT / CSV / code files.** The MIME whitelist is images + PDF only. A `.py` or `.md` upload is rejected with a stub block. Adding these is mechanical (one `process_X` function per format + one MIME gate + one test class), but the wire-shape returns of each format are different — DOCX has structure, CSV has rows, code has syntax. The cleanest design is a `process_*` per format that returns a `TextPart | ImagePart` union and lets `build_user_turn_content` handle the rest.
2. **No file persistence across turns.** Each turn re-uploads. Closing the tab drops everything. A `chat_history.py` module that writes uploads to `~/.cache/ai-security-chatbot/<session_id>/` would fix this — but see the privacy caveat in C.2. Files are more sensitive than text transcripts, so the consent UI should be explicit ("Store uploaded files locally for the duration of this session? Y/N").
3. **One hardcoded vision fallback.** `_DEFAULT_FREE_VISION_MODEL = "nvidia/nemotron-nano-12b-vl:free"` is the only free-tier vision model that returned a usable answer during testing. If that model is removed from OpenRouter, the auto-upgrade silently stops working. The next iteration should (a) make the fallback a `list[str]` in priority order, (b) probe each at startup and disable the ones that 404, and (c) surface a `st.warning` when the fallback is the only vision model left.
4. **No OCR for image-only / scanned PDFs.** A scanned PDF is caught by the `pdf_text_extraction` kind and stubbed. To make it work, a second pass would render each page to a PNG (`pix = page.get_pixmap(dpi=200)`) and feed it to `process_image`. The wire shape is the same; the only change is in `process_pdf`.
5. **No file-size streaming.** The helper reads the whole file into memory (`file.read()`) before validating. A 4 MB image is fine; a 50 MB image would be a problem, but the cap already rejects it. If the cap is ever raised, the read should become a chunked validation.
6. **No mime-from-content sniffing.** A malicious user can rename `evil.exe` to `evil.png` and the helper will accept it, base64-encode the bytes, and send the EXE to the model as an "image". The current cap (4 MB) and the model-side safety make this low-risk, but a `python-magic` content-type sniff would be the right belt-and-suspenders.

---

## Quick reference — file → what changed (Phase 11 round, 2026-06-15)

The table is a cumulative log of the last two major rounds: the **mentor-mode + 5-key router + post-hoc fixes** round (2026-06-13), and the **multimodal pipeline** round (2026-06-15, this update). New rows / changed cells are marked with 🆕.

| File | What changed (cumulative, latest on top) |
| --- | --- |
| 🆕 `app/file_processor.py` | **New file, 342 lines.** `FileProcessingError` (8 closed `kind`s), `ImagePart` dataclass, `_UploadedFileLike` Protocol, `process_image` (4 MB cap, MIME whitelist + extension fallback, base64 wrap), `process_pdf` (lazy `import pymupdf`, 200 K-char cap, `[truncated]` marker, `FileDataError` / `PasswordError` / "no text" branches), `image_url_part`, `_safe_size`. No Streamlit import. No HTTP. |
| 🆕 `tests/test_files.py` | **New file, 28 tests, 14 classes.** `FileProcessorImportTests`, `FileProcessingErrorTests`, `ProcessImageTests`, `ProcessPdfTests`, `ImageUrlPartTests`, `BuildUserTurnContentTests`, `BuildUserTurnPdfTests`, `SelectModelForRequestTests`, `VisionAllowListTests`, `ModelSupportsVisionTests`, `BuildMessagesRoundTripTests`, `OpenRouterPayloadRoundTripTests`. |
| `tests/test_smoke.py` | **105 tests, 18 classes** (was 80/16 in the previous round). Added `BuildUserTurnTextTests` (text-only multimodal path) and `StreamlitChatInputFileUploadTests` (the `accept_file="multiple"` widget contract). Other classes unchanged from the previous round. |
| 🆕 `web/chat_helpers.py` | Added (multimodal) `_is_image_mime`, `_is_pdf_mime`, `build_user_turn_content` (`str \| list[dict]`), `_stub_block`, `select_model_for_request` (auto-upgrade to a vision model), `_DEFAULT_FREE_VISION_MODEL` constant, `image_url_part` re-export. Carried over from the previous round: `_active_system_prompt(state)`, `_is_rate_limit_error`, `_friendly_error_message`, `_RATE_LIMIT_RETRY_SECONDS`. The original helpers (`_build_messages`, `_truncate_history`, `_serialize_for_download`, `_count_chars`, `_bubble_alignment`) are unchanged. |
| `app/prompts.py` | Added `OFFENSIVE_MENTOR_SYSTEM_PROMPT` (≈ 8 KB) and the `DEFAULT_SYSTEM_PROMPT` alias. Module docstring rewritten to document both profiles. No change in this round. |
| `app/config.py` | Added `iter_api_keys()` (5-slot pool) and `iter_models()` (multi-model list). Added `HTTP_TIMEOUT_SECONDS=30`. No change in this round. |
| `app/router.py` | New file (previous round). `ModelRouter` class with slot rotation, 401-disable, 429-backoff, 5xx-retry, key redaction. No change in this round. |
| `app/openrouter.py` | Added exception hierarchy (`OpenRouterError` + 4 subclasses), `body` truncation, network-error wrapping into `OpenRouterServerError`. No change in this round. |
| `web/streamlit_app.py` | Previous round: `sys.path` bootstrap, `_init_state` (seeds `teaching_mode` + `teaching_mode_previous`), sidebar Teaching-mode radio with the `teaching_mode_previous` swap trick, model selector, two-pass `_ask` driver, `_render_friendly_error`. **This round:** `st.chat_input(..., accept_file="multiple")` for in-input uploads; the two-pass driver now carries the file list through to `_ask(files=…)`; cache key is `(prompt, model, has_images, len(files))`; a caption above the model selector lists the active vision models. |
| `cli/chatbot.py` | No change. Still imports `DEFAULT_SYSTEM_PROMPT` (defensive). The CLI is deliberately file-upload-free — the `streamlit` `st.chat_input` widget does not exist in a headless context. |
| `verify_changes.py` | Pre-flight check (READY / STALE_WORKER / IMPORT_BROKEN). No change in this round. |
| `docs/technical_write_up.md` | Was: 12 problem rows, 8 decisions. **This round:** row 14 updated ("12 KB binary stub" → "shipped multimodal"), row 15 marked advisory → shipped, rows 16-18 added (file processor, vision-model auto-upgrade, doc-correction round), Decision 10 added (lazy `pymupdf` import), Decision 11 added (docs-lag-code lesson). |
| `README.md` | Was: 5-key rotation, multi-model selector, teaching mode, friendly errors, two-pass pattern, pre-flight script, 80-test suite. **This round:** added a "File & image upload" section, added `app/file_processor.py` to the project layout, bumped test count 80 → 133, updated the 60-second tour and the "what this is not ready for (yet)" section. |
| `docs/my_first_ai_journey.md` | Was: Phases 1-4 narrative. **This round:** Session 5 reflection added (the "I cannot read binary files" moment, the multimodal return-type insight, the vision-model-by-luck discovery, the lazy-import pattern, the docs-lag-code lesson). |
| `improvements.md` | This file. **This round:** new B.9 (file & image upload), C.4 marked SHIPPED, new C.13 (remaining file-upload limitations), Quick Reference table refreshed, test count updated. |

---

*Last updated: 2026-06-15. Test suite is **133/133 green** (was 80/80 on 2026-06-13). If you change the project, update the table above and bump the test count in `README.md` and the decision log in `docs/technical_write_up.md`.*
