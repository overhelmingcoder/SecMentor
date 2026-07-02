# Copy-to-Clipboard Bubble Buttons — Design Doc

> The architecture, the failure mode that triggered the rewrite, and the
> contract the new implementation is pinned to. This is the long-form
> companion to the **Copy-to-clipboard bubble buttons** section of the
> README. It exists so the design decision is not lost the next time
> someone (probably future-me) is tempted to "just use `onclick`" again.

---

## 1. The feature

Every assistant reply in the Streamlit web UI renders a small
`📋 Copy` button next to the bubble. Click it and the full reply text
lands on the user's clipboard, with a brief `✅ Copied!` confirmation
that auto-restores to the idle label. The button is part of the
assistant row only — user bubbles do not get one.

The feature is *technically* simple. The reason it has a design doc is
that the **obvious implementation is broken** in a way that is invisible
until the model returns a string that happens to contain both
apostrophes and double quotes.

---

## 2. The failure mode that triggered the rewrite

The first cut of the helper looked like this:

```html
<button class="bubble-copy-btn"
        onclick="(function(btn){var text=&quot;PAYLOAD&quot;;…})()">
  📋 Copy
</button>
```

with the assistant reply HTML-escaped into the `PAYLOAD` slot, rendered
via `st.markdown(..., unsafe_allow_html=True)`.

The trap is that the *outer* attribute is double-quoted, and the *inner*
JS literal was also double-quoted. The browser's HTML parser does not
know about JavaScript — it terminates the attribute at the first
unescaped `"` it sees, which is the one inside `var text="…"`. The
rest of the handler, including the closing `})()` and `">`, gets
appended to the *element text* of the button and is rendered as
**visible text inside the bubble** after every assistant reply.

The user reported the bug with a screenshot showing the leaked
JavaScript after the response. The leaked snippet was a working
copy handler — which is what made the bug subtle: **the copy
button itself was still functional**, but the user saw raw code
in the assistant bubble.

The fix is structural, not cosmetic. See §3.

---

## 3. The two-helper split

The new implementation lives in `web/chat_helpers.py` as two pure
helpers. Both are unit-tested independently of Streamlit.

### 3.1 `_copy_button_html(text) -> str`

Returns a **single-line, self-contained** `<button>` element. No
JavaScript in any attribute. The payload rides in a `data-text`
attribute that is HTML-escaped once on the Python side.

```python
import html as _html

_COPY_BUTTON_LABEL = "📋 Copy"
_COPY_BUTTON_LABEL_COPIED = "✅ Copied!"
_COPY_BUTTON_LABEL_FAILED = "⚠ Copy failed"

def _copy_button_html(text: str) -> str:
    safe = _html.escape(text, quote=True)   # & < > " ' all become entities
    return (
        f'<button class="bubble-copy-btn"'
        f' data-label="{_COPY_BUTTON_LABEL}"'
        f' data-label-copied="{_COPY_BUTTON_LABEL_COPIED}"'
        f' data-label-failed="{_COPY_BUTTON_LABEL_FAILED}"'
        f' data-text="{safe}">'
        f'{_COPY_BUTTON_LABEL}</button>'
    )
```

Key invariants:

- **The only HTML-shaped string the browser ever parses is the
  `data-text` attribute body**, and it is escaped against both
  `& < > " '` (Python 3.13's `html.escape(..., quote=True)`). The
  browser's own `dataset.text` decoder reverses the escaping at the
  JS boundary, so the value seen by `navigator.clipboard.writeText`
  is the exact original `text` argument.
- **The visible label and the three states are also stored in
  `data-*` attributes**, so the JS side never has to ship a literal
  "📋 Copy" string of its own.
- **The button has no inline event handler.** All behaviour is wired
  by the delegated listener described in §3.2.

### 3.2 `_copy_button_init_script() -> str`

Returns the **one-time** `<script>` block that wires up the delegated
click listener. The function is **idempotent at the Python layer**:
a module-level `_COPY_BUTTON_INIT_EMITTED` flag short-circuits the
second-and-later call to the empty string. It is also idempotent at
the JS layer: a `window.__secMentorCopyBtnWired` flag stops a second
script block (e.g. one injected by a different Streamlit render
pass) from attaching a second listener.

```python
_COPY_BUTTON_INIT_EMITTED: bool = False

def _copy_button_init_script() -> str:
    global _COPY_BUTTON_INIT_EMITTED
    if _COPY_BUTTON_INIT_EMITTED:
        return ""
    _COPY_BUTTON_INIT_EMITTED = True
    return """
<script>
(function () {
    if (window.__secMentorCopyBtnWired) return;
    window.__secMentorCopyBtnWired = true;
    document.addEventListener('click', async function (ev) {
        var btn = ev.target && ev.target.closest && ev.target.closest('.bubble-copy-btn');
        if (!btn) return;
        if (btn.__copyBtnBusy) return;
        btn.__copyBtnBusy = true;
        var text = btn.dataset.text || '';
        var original = btn.dataset.label || btn.textContent;
        var copied = btn.dataset.labelCopied || '✅ Copied!';
        var failed = btn.dataset.labelFailed || '⚠ Copy failed';
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.position = 'absolute';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
            btn.textContent = copied;
        } catch (_) {
            btn.textContent = failed;
        }
        setTimeout(function () {
            btn.textContent = original;
            btn.__copyBtnBusy = false;
        }, 1500);
    });
})();
</script>
"""
```

Key invariants:

- **`document.addEventListener('click', ...)`**, not per-button
  `addEventListener`. The handler matches via
  `ev.target.closest('.bubble-copy-btn')`, so a single listener
  serves every copy button currently in the DOM and every copy
  button that will ever be added by a future Streamlit rerun.
- **`btn.__copyBtnBusy`** is a per-element debounce. Three rapid
  clicks only fire one copy.
- **Modern clipboard API first**, with a synchronous
  `document.execCommand('copy')` fallback for restricted contexts
  (e.g. Streamlit Cloud preview iframes, where the modern API is
  blocked).
- **Label restore from `dataset.label`** — the helper does not
  hardcode the idle label into the script. The script is data-driven.
- **No `innerHTML` writes.** All label changes go through
  `textContent`, which is XSS-safe by construction.
- **`__secMentorCopyBtnWired` window guard** — if a second script
  block ever leaks in (e.g. from a sidebar rerender that re-renders
  the head), it returns early instead of stacking listeners.

### 3.3 Why the split?

The split is not arbitrary. It is the shape that makes both halves
unit-testable and idempotent:

- `_copy_button_html(text)` is **pure**: input text in, button HTML
  out, no side effects. It is the part that has to handle the
  XSS-sensitive escaping, and it is the part that gets called once
  per assistant bubble. Testing it is just string-equality.
- `_copy_button_init_script()` is **idempotent at the process
  level**: a Python module-level flag stops the second-and-later
  call. This matters because Streamlit rerenders the page on every
  interaction, and rerunning `st.markdown(_copy_button_init_script(), ...)`
  on every rerun is the only way to guarantee the script is
  present after, say, a model swap in the sidebar. The Python-side
  flag means we can call it freely without re-allocating the same
  script body 200 times.
- The init script is **idempotent at the JS level** too, so even
  if the script block somehow gets re-injected (e.g. by a
  `setComponents` reload), the window guard prevents a second
  listener from stacking on `document`.

---

## 4. The CSS

The button is styled in `web/streamlit_app.py`'s `_CUSTOM_CSS` string
(around line 478). The full rule block:

```css
.bubble-copy-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    line-height: 1;
    color: var(--bubble-copy-color, #1d4ed8);
    background: var(--bubble-copy-bg, rgba(29, 78, 216, 0.08));
    border: 1px solid var(--bubble-copy-border, rgba(29, 78, 216, 0.28));
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.05s ease;
}
.bubble-copy-btn:hover  { background: rgba(29, 78, 216, 0.16); border-color: rgba(29, 78, 216, 0.45); }
.bubble-copy-btn:active { transform: translateY(1px); }
.bubble-copy-btn:focus-visible { outline: 2px solid rgba(29, 78, 216, 0.55); outline-offset: 2px; }
```

The button is a low-saturation pill that sits below the assistant
bubble, sized to match the bubble's cap-height. The `focus-visible`
rule means keyboard users see a focus ring — the button is reachable
via Tab.

---

## 5. XSS analysis

The reply text is the model's output. It is the most XSS-sensitive
piece of state in the whole app, and the copy button sits inside the
same `unsafe_allow_html=True` render call as the bubble. The contract
the new implementation has to defend is:

> An assistant reply that contains `<script>alert(1)</script>` (or
> any other HTML-shaped string) must (a) render safely as visible
> text inside the bubble, and (b) round-trip exactly the same string
> when the user clicks Copy.

Defence in depth, layer by layer:

1. **Reply text is HTML-escaped once on the Python side** before
   being dropped into the `data-text` attribute. The escape covers
   `& < > " '`. After this, the model payload is a string of
   `&lt;`, `&gt;`, `&#x27;`, etc. — no HTML-meaningful bytes.
2. **The browser's `dataset.text` getter decodes the attribute**
   back to the original string at the JS boundary. This is a
   browser-level operation, not a string-decode we wrote, and it
   is what lets us ship the payload through an HTML attribute
   without writing a string-decoder.
3. **`navigator.clipboard.writeText(text)` takes a JS string**, not
   HTML. The string is written to the clipboard as plain text.
4. **The only `innerHTML`-shaped write in the lifecycle is to
   `btn.textContent`** (label restore). `textContent` is XSS-safe
   by construction — the browser treats the assigned value as text,
   not as HTML. There is no `innerHTML +=`, no `document.write`,
   no `eval`, no `Function()` constructor anywhere in the helper
   or the init script.
5. **The init script is loaded once**, via `st.markdown(..., unsafe_allow_html=True)`
   on the view's head render. Streamlit's `unsafe_allow_html` does
   not sandbox the script — it injects it raw — but the script
   body is the constant string in §3.2, not user input. There is
   no path by which a model reply can alter the script body.

The pipeline is "data in, data out" — the user sees a literal
`<script>alert(1)</script>` in the bubble, the same literal in the
clipboard, and no script ever runs.

---

## 6. Idempotency in detail

Streamlit rerenders the page on every interaction: every model
swap, every sidebar click, every chat input. The init script
render at line 631 of `web/streamlit_app.py` is **not** inside a
`if` block — it is called on every rerun. Without idempotency
that would mean 200+ `document.addEventListener('click', ...)`
calls by the end of a normal session.

The defence is two-sided:

| Layer | Guard | What it stops |
|---|---|---|
| Python | `_COPY_BUTTON_INIT_EMITTED` module flag | Re-emitting the `<script>` block body after the first call within a process. The second call returns `""` and the second `st.markdown` is a no-op. |
| JS | `window.__secMentorCopyBtnWired` | Re-attaching the delegated listener if the script body somehow leaks in twice (e.g. via a `components.html` call or a future code path). |
| JS | `btn.__copyBtnBusy` | A rapid double-click firing two copies back-to-back. |

The two idempotency guards are not redundant. The Python guard
protects against accidental re-renders (the normal case). The
JS guard protects against the *init script body itself* being
injected twice (an edge case the Python guard does not cover,
because by the time the second `addEventListener` call would run,
the Python guard has already done its job and the second
`st.markdown` is empty).

---

## 7. The test contract

`tests/test_smoke.py` has two test classes for this feature. Together
they pin the contract at the unit level.

### `CopyButtonHtmlTests` — 10 tests

Pins the *output shape* of `_copy_button_html(text)`. Sample tests:

- `test_returns_a_button_element` — output starts with
  `<button class="bubble-copy-btn"` and ends with `</button>`.
- `test_no_inline_onclick_attribute` — the rendered HTML must not
  contain `onclick=` anywhere. This is the regression test for the
  original bug.
- `test_embeds_payload_as_html_escaped_attribute` — `text="<x>"`
  round-trips through the `data-text` attribute, with `<` and `>`
  escaped to `&lt;` and `&gt;`.
- `test_escapes_apostrophes` — `text="it's"` becomes
  `data-text="it&#x27;s"` (Python 3.13's `html.escape(..., quote=True)`
  escapes apostrophes — we rely on this).
- `test_escapes_double_quotes` — `text='a"b'` becomes
  `data-text="a&quot;b"`.
- `test_escapes_angle_brackets_and_ampersand` — `<x>&y</x>` becomes
  `&lt;x&gt;&amp;y&lt;/x&gt;`.
- `test_handles_unicode` — `text="日本語 🛡"` round-trips intact.
- `test_handles_long_strings` — a 4 KB payload still round-trips
  through the attribute decoder.
- `test_stores_original_label_for_restore` — `data-label="📋 Copy"`
  is present so the JS can read it back.
- `test_view_wires_helper` — `web/streamlit_app.py` calls
  `_copy_button_html(...)` in the assistant bubble render.

### `CopyButtonInitScriptTests` — 10 tests

Pins the *output shape* of `_copy_button_init_script()`. Sample
tests:

- `test_returns_a_script_block` — the body is wrapped in
  `<script>...</script>`.
- `test_second_call_returns_empty` — idempotency at the Python
  layer.
- `test_uses_delegated_document_listener` — the body contains
  `document.addEventListener('click'`.
- `test_matches_via_closest_selector` — the body contains
  `closest('.bubble-copy-btn')`.
- `test_uses_modern_clipboard_api` — the body contains
  `navigator.clipboard.writeText`.
- `test_uses_legacy_fallback` — the body contains
  `document.execCommand('copy')` for restricted contexts.
- `test_uses_busy_guard` — the body references
  `__copyBtnBusy` to debounce double-clicks.
- `test_restores_label_from_dataset` — the body reads
  `btn.dataset.label`.
- `test_reads_payload_from_dataset_text` — the body reads
  `btn.dataset.text`.
- `test_inner_window_guard_for_duplicate_wiring` — the body
  references `window.__secMentorCopyBtnWired`.

Both classes are pure-string assertions. The init script body is
treated as a black box — we don't run it, we just assert that the
right pieces are present. This is the right level of test for a
delegated listener whose behaviour depends on a live DOM: the JS
engine test would be a `tests/test_streamlit_integration.py` job
(see Decision 8 in the technical write-up), not a smoke test.

---

## 8. The bug the new design does not have

The original failure mode was: **inline `onclick` + double-quoted
inner JS literal = HTML parser truncates the attribute early, leaks
the rest as visible text.** The new design does not have this bug
because:

- There is no JS in any HTML attribute. The only attribute that
  carries user-derived content is `data-text`, and it carries it
  HTML-escaped. The browser's parser cannot terminate a `data-*`
  attribute early because the helper is the one writing the
  attribute, and the helper always writes it as `data-text="<one
  entity-escaped string>"`.
- The handler is wired once on `document`, not per-button. There
  is no per-button JS to leak.
- The Python-side idempotency flag means the script body is only
  *emitted* once. The JS-side window guard means the script body
  is only *registered* once. Even if a future code path tried to
  re-emit the script, both guards would catch it.

The regression test for this is `test_no_inline_onclick_attribute`
in `CopyButtonHtmlTests`. If a future refactor reintroduces inline
JS into the button HTML, that test fails.

---

## 9. What this design does *not* cover

- **Selection-based copy.** The button copies the *full reply
  text*, not the user's selection inside the bubble. Adding
  `getSelection()` support would be a clean follow-up — the helper
  already round-trips arbitrary strings.
- **Right-click → Copy.** Some users prefer the right-click
  context menu. The button does not suppress it; users can still
  use it. The button is additive.
- **Multi-bubble selection copy.** Copying multiple replies in
  one go is not supported.
- **Per-bubble disable for streaming replies.** During a streaming
  reply, the button is rendered immediately with the *partial*
  text, and clicking it copies the partial text. This is the
  intended behaviour, but it is worth noting: if a user clicks
  Copy mid-stream, they get what was visible at that instant.

---

## 10. The line counts

| File | Function | Lines |
|---|---|---|
| `web/chat_helpers.py` | `_copy_button_html` | ~20 |
| `web/chat_helpers.py` | `_copy_button_init_script` | ~50 |
| `web/streamlit_app.py` | CSS for `.bubble-copy-btn` | ~25 |
| `web/streamlit_app.py` | init script render | 1 line (with comment) |
| `web/streamlit_app.py` | per-bubble render | 1 line (in the assistant branch) |
| `tests/test_smoke.py` | `CopyButtonHtmlTests` | ~150 |
| `tests/test_smoke.py` | `CopyButtonInitScriptTests` | ~150 |

The runtime surface area added by this feature is one CSS rule
block + one `<script>` tag + one `<button>` per assistant bubble.
Nothing else.
