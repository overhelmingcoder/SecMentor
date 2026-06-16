# My First AI Journey

> A personal log of my first steps into building with LLMs.
> This is not a textbook. It is a journal — written in my voice, for me.
> I will update it as the project grows.

---

## Who I Am (Before This Project)

- Engineering student.
- Comfortable with **Django, Python, and cybersecurity**.
- Brand new to **LLM applications** — I have used ChatGPT like everyone else, but I have never *built* with one.
- Comfortable asking questions, admitting confusion, and writing things down so I actually remember them.

**Why this file exists:** I learn best when I write. This is the place where I capture the *feelings*, the *mistakes*, the *click moments*, and the *why* behind every step. The technical details go in `technical_write_up.md`. The journey goes here.

---

## What I Am Building

**Project:** AI Security Chatbot (Stage 1)
**Goal:** A chatbot that answers cybersecurity questions, talks to a free LLM through OpenRouter, and has a clean interface plus basic memory.
**Why this project:** Chat is the simplest LLM pattern. Master it and RAG / agents / anything else becomes a variation. I am starting at the foundation so the rest is not magic.

---

## Session 1 — June 12, 2026

### What we did
- Talked through the **9-step round trip** of an LLM chatbot.
- Learned what **OpenRouter** is and why using one gateway for many models is smart.
- Wrote down the essential vocabulary: API key, model, tokens, context window, system prompt, chat messages, temperature.
- Drew the architecture diagram in text (user → UI → Python backend → OpenRouter → LLM → back up).
- Picked the model together: **`google/gemma-4-31b`** as primary, `meta-llama/llama-3.3-70b-instruct:free` as backup.
- Decided to start the write-up files: one technical, one personal (this one).

### How I felt
A little overwhelmed at first when I saw the architecture diagram, but then it clicked — it is just a loop. The same loop, no matter how fancy the project gets. That is reassuring. I think I can do this.

### The big mistake I made
I **pasted my real OpenRouter API key into the chat** while asking for model advice. Puku had to stop and call it out.

How that felt: a small cold sweat. I know better — I am a cybersecurity student. I literally study leaked keys and credential abuse.

What it taught me:
- Even when I am just "asking a quick question," a key in chat is a key in a database somewhere.
- The fix is not "be more careful next time." The fix is a **process**: key lives in `.env`, code reads it with `python-dotenv`, `.env` is in `.gitignore`. The process protects me even when I am tired, distracted, or in a hurry.
- I rotated the key in the OpenRouter dashboard and made a rule: **never paste the real key anywhere except the `.env` file**.

This is the kind of mistake I am *glad* I made on day one of an AI project, because I will not make it again on day ten of a real product.

### Concepts that finally clicked
- **The system prompt is a lever.** It is the single most powerful thing I can control about the model's behavior. I will engineer it carefully in Phase 5.
- **The model is just a string.** Switching from Gemma to Llama is changing one variable. That removes a lot of fear.
- **Tokens are the real currency.** Free models have rate limits; paid models charge per token. Context window is the ceiling I have to stay under.
- **RAG and agents are the same backend.** If I understand the round trip, I understand the foundation of everything else in applied AI.

### Skills I am building
- Reading API documentation and turning it into working code.
- Designing a small project with a clean folder structure.
- Writing `.env`-based secrets management.
- Prompt engineering (later phase).
- Building a thin backend, then a thin frontend, then wiring them together.

### Question I still have
None urgent. The architecture is clear, the model is chosen, the key is rotated. I am ready for Phase 2.

---

## Milestones (to be filled as I go)

- [x] **Session 1** — Phase 1 complete: architecture, vocabulary, model chosen, write-ups started.
- [x] **Session 1** — Phase 1 approved. Moving to Phase 2.
- [x] **Session 1** — Phase 2 complete: folders, secrets, smoke tests all green.
- [x] **Session 3** — leak #2 caught and logged. New rule in place: redact before share.
- [x] Phase 3 — minimal CLI chatbot working. First live answer from Gemma 4 31B IT: "SQL injection is a security vulnerability..." (one of the cleanest moments of this project).
- [x] Phase 4 — conversation history added. The single biggest "click" of the project so far: the model has no memory. It is just a list we keep sending back in.
- [x] Phase 5 — cybersecurity system prompt engineered. Four pillars (defensive, DevSecOps, AI security, offensive-security education) with an explicit refusal clause for working exploit code. Live-probed and verified.
- [x] Phase 6 — web interface built. Streamlit UI in `phase6_web/streamlit_app.py` (now `web/streamlit_app.py`), pure UI-logic helpers in `phase6_web/chat_helpers.py` (now `web/chat_helpers.py`). 27/27 tests green. Engine wired to the view through the same `app.openrouter.chat` call we built in Phase 3.
- [x] Phase 7 — refactored for maintainability. Phase-numbered folders dropped (`phase3_cli/` → `cli/`, `phase6_web/` → `web/`); 12 test imports rewritten; docs updated. 27/27 still green post-refactor. No engine changes — the rename was purely a folder-and-import-path operation.
- [x] **Session 4 (post-Phase 7)** — multi-model selector + visibility fixes + two-pass pattern + silent-success fix + 429 friendly-error classifier. **41/41 tests green.** Server PID 15184 live. End-of-session reflection below.
- [x] **Project complete** — end-of-journey reflection below

---

## Session 4 — June 13, 2026 (post-Phase 7, unmodelled work)

The project was "closed" on June 12. It was not closed. I came back the next day to test things, and the testing surfaced five real issues that did not exist in the test suite because the test suite was never asked the right questions. This is the session that taught me the difference between *green tests* and *a working product*.

### What I did
- **Wired a multi-model selector into the sidebar.** Five curated free models + a custom-ID expander for anything else. The driver is `FREE_MODEL_CHOICES` in `web/streamlit_app.py`. The selector lives in the sidebar so I can switch models per question without re-running the app.
- **Reduced HTTP timeout from 60s to 30s.** A model that is going to time out is going to do so in 30s. Waiting 60 was generous, not helpful — the user has already given up by then.
- **Added a live status line and a toast.** While the call is in flight, the status line ticks ("Model: gemma · 6 messages · 482 chars · last reply: thinking…") and `st.toast(f"Calling {model}…")` confirms the click. Previously the UI just sat there and I had no idea if my message had even been sent.
- **Adopted a two-pass `_ask` pattern.** Pass 1 records the user turn and triggers a rerun; pass 2 runs the model and renders the reply. This is a Streamlit idiom — you cannot do an in-script "render a Thinking… bubble, then replace it with the reply" cleanly without two passes.
- **Fixed the silent-success bug.** After a successful call, the reply was being appended to `st.session_state["messages"]` but pass 2 was not calling `st.rerun()`, so the history loop at the top of the script never re-painted. The status line said "last reply: 14.5s · cache: 6" — proof the call succeeded — but no assistant bubble appeared. The fix was one line: `st.rerun()` at the end of pass 2 (and in the cache-hit branch, which had the same bug).
- **Built a friendly 429 error classifier.** The rate-limit error from OpenRouter came in as `OpenRouterError("OpenRouter returned HTTP 429: {... json payload ...}")`. The old code dumped the whole string into `st.error`, which is unreadable. I added `_is_rate_limit_error(exc)` and `_friendly_error_message(exc, model) -> (headline, body)` in `web/chat_helpers.py`, and a `_render_friendly_error(exc, model)` view wrapper. The user now sees `⏳ meta-llama/llama-3.3-70b-instruct:free is rate-limited upstream.` with a "wait ~30s and retry, or pick a different model from the sidebar" hint. The raw exception is preserved in a collapsed `st.expander` for debugging.

### How I felt
A little sheepish. I had just written an end-of-journey reflection calling the project done. Five minutes of clicking in the browser showed it was not. The lesson is not "I should have caught it sooner" — the lesson is that **a test suite is not a product test.** I had 27 green tests. They tested the engine, the prompt, the helpers, the model catalogue, the two-pass structure. None of them tested "after I type a message and click send, does the assistant bubble actually appear?" That is a UI integration concern, and it was invisible to every test I had.

I am keeping that distinction. Green tests = the *units* of the system are right. Browser test = the *system* of the units is right. Both have to pass.

### The bugs I shipped and then had to fix

**Bug A — silent success.** Pass 2 of `_ask` ran the model, captured the reply, appended it to `st.session_state["messages"]`, and then returned. No `st.rerun()`. The history-render loop at the top of the script had already been drawn, so without a rerun it never ran again. Net effect: the user types a message, sees a "Thinking…" bubble, sees the spinner stop, sees the status line update to "last reply: 14.5s", and *no assistant bubble ever appears*. I sent the same message twice while debugging. Both times the user bubble rendered, both times the assistant bubble did not.

**Bug B — raw 429 JSON in the error banner.** `OpenRouterError` is a `RuntimeError` whose message is `"OpenRouter returned HTTP 429: {... full upstream JSON ...}"`. I had `st.error(f"❌ The model call failed: {exc}\n\nCheck your .env ...")`, which dumped the full payload — code, message, `retry_after_seconds`, `provider_name` — straight into the chat. That is hostile UX even when it is technically informative.

**Bug C — pass 2 of `_ask` was not reachable from `chat_input`.** Discovered while fixing Bug A. The two-pass pattern needs a top-level driver block after the function definition (`if st.session_state.get("pending_request"): _ask(None)`) to make the second pass fire on the rerun. Without that block, only the example-prompt buttons drove pass 2 correctly. I had added a driver but it was placed *before* the function definition, which caused a `NameError`. Reverted, re-placed after the definition, and added three new `TwoPassPatternTests` to pin the structure.

**The two structural fixes, side by side:**

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| A | Assistant bubble never appears | Pass 2 appends reply but never reruns | `st.session_state["messages"].append(...); st.rerun()` at end of pass 2 and in cache-hit branch |
| B | Raw 429 JSON in error banner | `st.error(f"... {exc} ...")` dumps the full exception | Detect `"HTTP 429"` in `str(exc)`; show `⏳ {model} is rate-limited upstream.` + retry hint; move raw exception to collapsed `st.expander` |
| C | `chat_input` submitted messages never triggered pass 2 | Driver block placed before `_ask` definition → `NameError`; only example-prompt buttons worked | Add top-level driver block *after* function definition; signature `_ask(prompt: str \| None) -> None` |

### Things I am learning
- **"Green tests" and "works for the user" are different claims.** A test that imports `_build_messages` and asserts on its return value is not the same as a test that loads the Streamlit app, types into `chat_input`, and waits for the bubble. I did not write the second kind of test once during Phases 1–7. I should have.
- **Two-pass patterns in Streamlit are not optional.** Trying to keep all of "render a placeholder bubble, run a slow call, replace the placeholder with the real bubble" in one script run is fighting the framework. The idiomatic shape is: pass 1 sets up state and reruns; pass 2 sees the state and renders. The state machine is `pending_request` in `session_state`. This is the same pattern Django uses with redirects-after-POST — the framework wants you to do the work, then re-render, not do the work in place.
- **Cache hits deserve the same defensive code as misses.** The cache-hit branch in `_ask` was a smaller copy of the success path, and it had its own copy of Bug A. When you duplicate code, you duplicate bugs. I have started writing "this branch must also rerun" as a comment on every branch that mutates `session_state["messages"]`.
- **Classifying exceptions at the view layer is a feature, not a nicety.** Putting `_is_rate_limit_error` and `_friendly_error_message` in `web/chat_helpers.py` (pure helpers, no `streamlit` import) meant the classifier is unit-testable. The 4 new `FriendlyErrorTests` pin the contract: matches `"HTTP 429"`, rejects 500, mentions the model name, falls back to a generic banner for unknown errors.
- **The `.env` default model is not the sidebar default.** I kept changing the sidebar selector and seeing the model name in the status line shift, but forgetting that the *default* model on first load still comes from `OPENROUTER_MODEL` in `.env`. The two are intentionally different — `.env` is "what runs on cold start," the sidebar is "what runs on next click." That distinction is now a comment in `web/streamlit_app.py` near the selector.
- **Free-tier rate limits are not an edge case, they are the steady state.** Five free models, three of them rate-limited at any given moment. The friendly 429 banner is not a defensive measure; it is the *primary* error path. I should design assuming rate limits, not despite them.

### Skills I am building (added)
- Reading a Streamlit error traceback and finding the *actual* failing line (not the line Streamlit shows you).
- Designing two-pass patterns for any framework that reruns the script on every interaction.
- Treating `st.rerun()` as an explicit, named control-flow primitive — not an emergency call.
- Classifying exceptions by message substring at the view layer without coupling the view to the engine's internals.
- Writing tests that pin structural contracts (`st.rerun()` is called in this branch) without coupling to the specific implementation (which exact rerun pattern is used).

### Question I still have
How do I write a UI integration test for Streamlit in pure Python, without spinning up a browser? There is `streamlit.testing.v1.AppTest` — I have not used it yet. That is the next click.

### Closing Session 4
I am closing this session the way I closed Phase 7: with a green test suite and a browser I can prove works. **41/41 green** (was 27 at end of Phase 7, +14 across the multi-model selector, the two-pass pattern, and the 429 classifier). Server running PID 15184, `/_stcore/health` returns 200. The next session's question is the right one: a chat with 41 green unit tests and a working browser is *not the same as* a chat with 41 green unit tests *and* a UI integration test that proves "type a message, see a bubble." That is the test I owe myself.

---

## Phase 8 (post-hoc) — What Stuck from Session 4

Phase 8 was not a planned phase. It was the work that happened after the end-of-journey reflection was already written. Five things stuck, and they are all the same lesson in different uniforms.

**1. The test suite is a vocabulary, not a vocabulary check.** My 27 tests in Phase 7 proved that the units spoke the right words to each other. They did not prove that the units, *as a whole*, said a sensible sentence. Session 4 is the session where the chat could pass every test I had written and still fail to show a reply. The fix is a new kind of test in the suite — a `StreamlitAppTest` class that boots the view, types into `chat_input`, asserts a new assistant message is appended, and asserts the bubble renders. I have not written it yet. I will, before the next feature.

**2. Streamlit's "rerun the whole script" is a feature, not a bug, but only if you lean into it.** The two-pass `_ask` pattern is the same idea as the Post/Redirect/Get pattern in Django, the same idea as React's `useEffect` re-rendering on state change, the same idea as any framework that says "describe what should be true at this moment, the framework will keep it true." I had been fighting the rerun by trying to mutate the DOM in place. Once I gave in and wrote the two-pass version, every other Streamlit footgun got easier.

**3. "Cache hit" is a code path, not a comment.** The cache-hit branch in `_ask` was a smaller copy of the success path. It had Bug A *independently*. I had a single comment that said "render cached reply" and I thought the two branches were obviously parallel. They were not. A branch that mutates session state is a first-class code path. It needs its own tests, its own comments, and its own `st.rerun()`.

**4. Classifying errors at the right layer is the difference between a tool and a product.** `OpenRouterError` is a fine exception class. Dumping its `__str__` into `st.error` is not. The fix was two layers: `_is_rate_limit_error(exc)` classifies by message substring (pure helper, testable), and `_render_friendly_error(exc, model)` formats the user-facing message (view wrapper, no engine coupling). The 4 `FriendlyErrorTests` prove the classifier; the `TwoPassPatternTests` prove the view uses it. The contract is pinned at both ends. This is the same pattern as the engine/interface split: separate the *what* (this is a rate limit) from the *how* (render it as a friendly banner).

**5. A "closed" project is a project with no failing tests, not a project that has been used.** Session 4 was a real-user session. I clicked the buttons, sent the messages, watched the screen. I caught five issues in one sitting. The lesson is structural, not emotional: the phase 7 close-of-project was the right *technical* close (27 green, all engine contracts pinned, refactor verified), but it was the wrong *product* close (no human had used it under realistic conditions). I am taking forward a new rule: a project is "done" only after a real human has run it end-to-end and the screen has done what the screen should do. Until then, the project is in beta, no matter what the test count says.

### One new durable rule (added to the list at the bottom)
- **A failing browser test is more honest than a passing unit test.** The unit test proves the function returned the right value. The browser test proves the user saw the right thing. When they disagree, the browser wins.

---

## What I want to build next (revised after Session 4)

The previous reflection said RAG, then agents. After Session 4 I am reshuffling:

1. **A `StreamlitAppTest` class in `tests/test_smoke.py`** that types into `chat_input`, asserts the assistant bubble renders, and asserts a 429 returns a friendly banner. This is the missing test layer and it should have been Phase 8.
2. **A "suspicious-code explainer" mode** — a sidebar radio that switches the system prompt to a strict defensive variant, a paste box for a snippet, and the bot explains the behavior, flags IOCs, and suggests detections. Strictly defensive, with the refusal clause tightened for the malware case.
3. **RAG, then agents, in that order.** Same backend, different system prompt and different message list shape. The engine/interface split is the only reason I can queue these.

The order matters: I do not add a feature until I can prove the feature does not break the screen. That is the rule Session 4 burned into me.

---

## How effectively is the chatbot working? (the honest self-assessment)

**What works:**
- The engine is solid. 41 tests pin the prompt, the model catalogue, the helpers, the error contract, the two-pass pattern, the cache, and the 429 classifier. The `requests.post` in `app/openrouter.py` is a stable, well-tested black box.
- The interface is polished. The bubbles, the sidebar, the hero header, the status line, the friendly errors. The user-facing experience is genuinely good — not a Streamlit template, a real product.
- The secrets discipline is clean. `.env` is in `.gitignore`, the real key is in `.env` only, and I have a written rule about never letting a tool write a redaction back into `.env`.
- The refusal boundary in the system prompt holds. Live probes of out-of-scope questions have returned clean refusals with the educational pivot, every time.

**What is fragile:**
- Free-tier rate limits. The 429 friendly-error path is the steady state, not the exception. If a real user runs this in production, they will hit it constantly.
- The two-pass pattern is correct *now*, but it depends on three specific structural facts: the top-level driver block is after the function definition, the signature accepts `str | None`, and pass 2 calls `st.rerun()`. If any future refactor breaks one of those, the bug returns silently. The `TwoPassPatternTests` pin the contract, but a UI integration test would pin it harder.
- The `.env` default model is `google/gemma-4-31b-it:free`. That slug needs to remain valid on OpenRouter's catalogue. If it ever gets deprecated, the cold-start chat breaks. I should add a startup check that surfaces a friendly warning if the model returns 404 on the first call.

**What I cannot measure yet:**
- Whether the assistant replies are *good*. The test suite proves the calls succeed; it does not prove the answers are correct. That is a human judgment and I have not done a structured review of recent replies yet.
- Whether the refusal clause holds *consistently* across models. Gemma and Llama might refuse the same question differently. I have not probed this; I should.

**Overall grade:** B+. The chatbot works, the screen is right, the tests are honest, the secrets are clean. It loses points for: no UI integration test yet, no live probe of refusal consistency across models, and the default-model-cold-start check. Those are not blockers; they are the next session.

---

## Things I want to remember when this is over (extended list)

## Phase 1 — What Stuck

The single most useful idea from Phase 1: **the model is just a string**. Picking "Gemma vs Llama vs Qwen" sounds like a huge decision, but in code it is one variable. That took a lot of pressure off. I can change my mind later with zero cost.

Second: **the system prompt is a lever, not a suggestion**. I had thought of prompts as "the question you ask." They are actually the operating manual for the model. Phase 5 is where I will start pulling that lever for real.

Third: **secrets management is a process, not a memory trick.** I will not rely on "remembering not to paste the key." The `.env` + `.gitignore` + `python-dotenv` flow is the system that protects me when I am tired, rushed, or distracted. Designing for my worst moment, not my best one.

---

## Session 2 — June 12, 2026 (same day, Phase 2)

### What we did
- Built the full project skeleton: `app/`, `phase3_cli/` (now `cli/`), `phase6_web/` (now `web/`), `tests/`, `docs/`.
- Created the secrets-management trio in the right order: `.gitignore` first, then `.env.example`, then `.env`.
- Moved the two journals into `docs/`.
- Wrote a smoke test using `unittest.TestCase` and got it green: `python -m unittest tests.test_smoke -v` → 2 tests OK.

### How I felt
Relieved that the structure is laid out before any real code. I have a tendency to dump everything into one file and regret it later — having a skeleton forces me to think about separation of concerns up front. It also made the project feel "real" in a way that just talking about architecture did not.

### Things I noticed
- Creating `.gitignore` **before** `.env` is not a detail — it is a guardrail. If I had created `.env` first, one accidental `git add .` would have leaked the key into history. The order matters.
- `__init__.py` files look trivial but they are what lets `from app import config` work. Without them, Python does not see the folder as a package.
- The `phase3_cli/` and `phase6_web/` folders are deliberately named after the phase that fills them. That is a teaching scaffold, not a permanent structure — Phase 7 will refactor. But for now it makes the project easy to read. *(And Phase 7 did refactor them: `phase3_cli/` → `cli/`, `phase6_web/` → `web/`. The scaffold served its purpose and was retired.)*

### Decision recorded
- I will keep the two journals in `docs/` rather than the project root. That keeps the source tree clean and gives the docs a stable home if I ever publish this on GitHub.

### Question I still have
None. Ready for Phase 3.

---

## Session 3 — June 12, 2026 (same day, leak #2)

### What happened
Right after I said "ready for Phase 3," I pasted the contents of `.env` into the chat — including the *new* API key. The new key was supposed to live only on my machine. I broke the rule on the very first time I had the chance to follow it.

### How I felt
Embarrassed. Puku had to stop the conversation and call it out, *again*, on the same day. There is no excuse — I am literally a cybersecurity student, and I am leaking credentials in a learning chat like a beginner.

### What I am learning
- "I'll be careful" is not a security control. The control is: **never share the contents of `.env` over chat.** If I need to confirm a value is set, I say "OPENROUTER_API_KEY is set" and stop. I do not paste the value.
- Real developers lose jobs over this. I am not going to be one of them.
- The pattern is clear: I need a *ritual* before sharing any file — scan for `KEY=`, `TOKEN=`, `SECRET=`, `PASSWORD=` and redact. That ritual needs to be as automatic as locking my laptop when I walk away.

### New rule (written down so I do not forget)
> **Before sharing any file, config, log, or screenshot, I redact every secret.**
> I replace values with `<set>` or `***`. No exceptions, not even "just to show Puku."

### Session 3 — late entry (the third leak and a tool mistake)
Puku tried to update the model name in my `.env` and accidentally overwrote my API key with the literal string `<set>` — the same redaction I was supposed to use for chat, written into the actual file. Then the model name got duplicated. So now my `.env` is broken and the only fix is to paste the real key back in.

How I felt: a mix of "this is not my fault" and "I should have caught it." I should have caught it — I am watching the chat, I saw the edit go through, I could have said "stop, do not write `<set>` to disk." I will not rely on any assistant to be perfect with my secrets. **My secrets, my responsibility.**

What I am learning:
- "Redact for chat, never for disk" is the new rule.
- I should always read the file after any assistant edit to a `.env` and confirm the real values are still there.
- A secret-handling incident does not have to be malicious to be a problem. A friendly typo in an editor can be just as destructive.

Action item: rotate the key once more. Do not paste the new key in chat. Open `.env` in VS Code, paste the new key, save, close.

---

## Phase 3 — What Stuck (so far)

Phase 3 is about wiring the round trip end to end. Three things are worth recording while it is still fresh:

**1. The model name is not a free-form label — it is a lookup key.**
When the live call returned `No endpoints found that can handle the request`, I learned the string in `.env` has to match OpenRouter's catalogue exactly. The real ID is `google/gemma-4-31b-it:free` — with `-it` and `:free`. The version I had been carrying around in my head, `google/gemma-4-31b`, was wrong. **Fix:** any time I am uncertain about a model name, I should hit `GET /api/v1/models` and read the response, not rely on memory.

**2. The error message is usually the answer.**
"No endpoints found" was a useless-looking string, but it pointed straight at the cause. I am training myself to read the error first and guess the cause second, not the other way around. This is going to be useful forever, not just in this project.

**3. The key is the bottleneck, not the code.**
I now have working code (`app/config.py`, `app/openrouter.py`, `cli/chatbot.py` — then `phase3_cli/chatbot.py`, now back to `cli/chatbot.py` after Phase 7) and 8 green tests. The only thing standing between me and a successful first end-to-end run is restoring a real key to `.env`. That is a powerful reminder: in real projects, configuration and access are usually the long pole, not the algorithms.

---

## Phase 4 — What Stuck

Phase 4 added conversation history. Only ~15 lines of meaningful code, but it changed how I think about LLMs.

**The big click: the model has no memory.**
I always assumed ChatGPT and friends had some kind of "session" running on the server. It does not. The application keeps a list of every prior turn and re-sends them inside the `messages` array on every new call. The "memory" is in the application, not the model. This is also why the *same* prompt can give *different* answers on different calls — there is no hidden state on the model side, just whatever was last in the list.

**Why this is the most important idea of the project so far.**
- **RAG** is exactly this — retrieved documents are just additional `messages` injected into the list.
- **Agents** are exactly this — tool results become new turns in the same list.
- The whole "context engineering" field is just being careful about what goes into that list and what does not.

So if I understand the round trip, I understand the foundation. Phase 4 is where that became concrete, not abstract.

**Code that made it click for me:**
```python
history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
while True:
    user_input = read()
    messages = history + [{"role": "user", "content": user_input}]
    reply = chat(messages)            # send EVERYTHING, not just the new turn
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
```
That `history + [...]` line is the entire trick. Everything else is plumbing.

**What I am going to watch for in Phase 5+:**
- Token cost grows linearly with history. I need a cap eventually (probably "keep the last N turns" + "always keep the system prompt first").
- If the system prompt ever changes, the change applies from the next call — it does not retroactively rewrite how the model "thinks" about the old turns. That is fine, but it surprised me when I thought about it.

---

## Phase 5 — What Stuck

Phase 5 is the one I was looking forward to. Same code, totally different chatbot — just by changing the system prompt. I asked for full-scope coverage (defensive, DevSecOps, AI security, *and* offensive-security education) because I am a security student, not a generic user. The model needed to be useful to me.

**The single biggest click: the system prompt is the operating manual, not the introduction.**
A weak system prompt is a polite "be helpful" sentence. A strong one is a contract that pins down role, scope, style, format, and the refusal boundary. The model reads it once at the start of the conversation and uses it to steer every reply. If I want a different chatbot tomorrow, I edit one string.

**The four pillars I ended up with:**
1. Defensive security
2. DevSecOps
3. AI / ML security
4. Offensive-security education (concept-level)

That fourth one was the conversation I needed to have with myself before writing the prompt. I want a tool that helps me learn *how attacks are structured* so I can recognize them in code review and design defensively. I do not want a tool that hands me a turn-key weapon against a real system. Those are two very different things, and a good prompt keeps them separate.

**The boundary, written down so I do not move it casually:**
- *Yes:* explain the structure of an attack, why it works, what defeats it, and where to practice it legally (HackTheBox, TryHackMe, PortSwigger, WebGoat, DVWA).
- *No:* working exploit code, malware, droppers, C2 tooling, payloads targeting a specific real WAF / EDR / MFA, or step-by-step instructions to compromise a system I do not own.

This is not a moral judgment about red-team work — authorized red teams do real and valuable work. This is a *learning tool* built by one student, on a free model, with no logging, no audit trail, and no lab environment behind it. The prompt is calibrated to the tool.

**What I learned by probing it live:**
- Asking an in-scope question (recognize SQLi during code review) got a structured, citation-bearing defensive answer with a real summary table. That is the *use case* I built the chatbot for.
- Asking an out-of-scope question (give me a reverse-shell one-liner against a real server I do not own) got a clean refusal, *and then* the educational angle — what a reverse shell is, why it works, and the five-layer defense. That pivot is the difference between a refusal that frustrates the user and a refusal that teaches. I want my chatbot to do the second one.

**The test that surprised me in a good way:**
I added seven tests that literally grep the system prompt for the words "defensive", "devsecops", "prompt injection", "owasp", "offensive", "exploit", "lab". If anyone — including future-me — ever edits the prompt and accidentally drops a pillar or weakens the refusal clause, the test suite will scream. That is a real safety net for a *config artifact*, and I did not appreciate how cheap it was to build until I saw it run.

---

## Phase 6 — What Stuck

Phase 6 is the one everyone *sees* — the web UI. Same engine, same prompt, but now there are bubbles, a sidebar, a hero header, and a status dot. Three things are worth recording.

**1. The same code in a browser is a different product.**
I was not prepared for how much the *shape* of the interaction changes when you put a chat in a browser. In the CLI, I type, I read, I scroll up if I want context. In the browser, the system prompt and example prompts become clickable buttons, the history becomes a scrollable transcript, and a status line tells me "Model: gemma-4-31b · 6 messages · 482 chars." The engine has not changed — I am sending the same `messages` list — but the *experience* is unrecognizable. That is a real lesson: a chatbot is the engine, but a *product* is the engine plus everything the user can see and click. Phase 6 was the first time this project felt like the second one.

**2. CSS is a soft skill, and I undercounted it.**
I started Phase 6 thinking "just throw some markdown in and call it done." That got me a working app in 20 minutes that looked like a default Streamlit template — fine, but not *good*. The premium look took most of the day: a gradient header, a hero card with a badge, custom bubble alignment, an animated `pulse` dot on the status line, a dark sidebar with light text, hidden Streamlit footer. None of that was hard. All of it was *grinding* — small CSS choices, a lot of "move that 4px to the left" iterations, the kind of polish you only finish by looking at it 50 times.
The lesson I am taking forward: **a demo is not a product, and the difference is mostly taste.** I want to keep that in mind for every project. The first 80% of the value is in the engine. The last 20% is in the CSS, and the last 20% is what people actually remember.

**3. Pure helpers are the Streamlit tax refund.**
Streamlit reruns the whole script on every interaction. That means *any* logic that lives next to `st.markdown(...)` is hostile to test — you cannot import a Streamlit widget and assert on it. So I split the file: `chat_helpers.py` holds pure functions (`_build_messages`, `_truncate_history`, `_serialize_for_download`, `_count_chars`, `_bubble_alignment`); the view file is just rendering and state. The 10 new helper tests run in milliseconds because they do not touch Streamlit at all. **The refactor cost me 20 minutes; the test coverage I got back is permanent.** That is a pattern I want to use from day one on the next project: when a framework is awkward to test, push the logic out, then test the logic.

**4. The engine's error contract held under live pressure.**
I tried the live engine probe three times against three different free models. All three returned HTTP 429 — the free tier was rate-limited upstream. The point is not that the probe failed; it is that the *engine* caught the failure, wrapped it in `OpenRouterError`, and the test for that error contract passed. The UI's `st.error` banner would have shown a friendly message in the browser. The thing I designed in Phase 2 to fail safely did fail safely. I trust the design more after seeing it work in anger.

**5. Free models are a real constraint and I have to plan for it.**
Three free models, three 429s, in one session. The retry hints said 5s, 13s, 21s. If I had been a paying customer this would not have happened. Since I am a student on the free tier, this is the world I live in — the chatbot has to *gracefully* tell the user "the model is busy, try again in a minute" instead of crashing. The UI's error banner does that. I do not love that free-tier rate limits are part of my user experience, but I love that the design absorbed them.

---

## Phase 7 — What Stuck

Phase 7 is the one nobody *sees*. Same engine, same helpers, same UI, same prompt — but the folder names changed, and the test suite stayed green through it. Three things are worth recording.

**1. A refactor is just a discipline move, not a heroic one.**
I have watched senior engineers do "the big refactor" in real codebases and it is almost always a horror story — weeks of merge conflicts, regressions, and "who broke the build?" The opposite pattern, which Phase 7 was, is *boring on purpose*: rename two folders, rewrite 12 import lines, update three docs, re-run the tests, done in one sitting. The test suite is the safety net that makes this kind of refactor cheap. I made 12 mechanical changes in a row and at the end I had *exactly the same test results* (27/27, 2.042s) as before I started. That is what a green test suite is for. I am going to treat "refactor must be green before, green after" as a hard rule on the next project, not an aspiration.

**2. Relative imports inside a package insulate you from upstream path changes — for free.**
I did not plan this, but Phase 6 left a small gift: `web/streamlit_app.py` imports its helpers with `from .chat_helpers import (...)` (a relative import). That meant the rename from `phase6_web/` to `web/` required *zero* code changes in the view file. Only the docstring moved. The CLI and the test file use absolute imports and *did* need rewrites, but the view did not. The asymmetry is a lesson I want to remember: when something is part of the same package, relative imports mean you can rename the package without renaming the consumer's internals. It is the same logic that makes `os.path.dirname(__file__)` a robust way to find sibling files. I had been treating relative imports as a "style" question; Phase 7 reframed them as a *coupling* question. They reduce the number of names the consumer has to know about.

**3. A silent-failure pattern is a sharp teacher.**
Halfway through the test-import rewrites, I used a tool call that said "I rewrote three occurrences" — and the tool reported success. But a follow-up read of the file showed only one of the three had actually changed. The other two were unchanged. No error, no warning, just a silent miss. The pattern that worked on the second pass was brutally simple: *one call per import line, with the test name and its docstring as context to make the search string unique.* That pattern always succeeded, and it is also the pattern I will reach for first on the next refactor — context-rich uniqueness is cheaper to write and cheaper to verify than batched ambiguity. The deeper lesson is that *the tool did exactly what I asked it to do, and what I asked was ambiguous.* The fix is not "use a better tool"; the fix is "write unambiguous asks." That is a project-management lesson wearing a syntax-error costume.

**4. The project is now bigger than the engine.**
A chatbot in 2026 is two things: the model call (one function, ~30 lines, `app/openrouter.py`) and *everything else*. The everything else took six more phases. CLI loop, history accumulator, `/clear` command, system prompt engineering, refusal clauses, live probes, secrets management, Streamlit view, three-layer split, pure helper extraction, the refactor. None of those is hard. All of them together is a project. The engine is the *floor* of what a chatbot is; the floor was built in Phase 1. The rest of the building is what makes the engine usable. I am taking forward: **"the model is just a string" was Phase 1's lesson, but the durable lesson is that the string is 5% of the product.**

---

## Things I want to remember when this is over

- The engine/interface split is the structural decision that pays off for the whole project — build it in Phase 1 or regret it in Phase 5.
- The system prompt is a contract (role, scope, style, format, refusal boundary) — not a greeting.
- The model is stateless. Every "memory" is the application re-sending prior turns in the `messages` list.
- Pure helpers are the test tax refund: when a framework is awkward to test, push the logic out and test the logic.
- The test suite is the safety net that makes refactors cheap. Green before, green after, refactor verified.
- Relative imports inside a package insulate the consumer from upstream path changes. Coupling question, not style question.
- "Code → tests → live probe → docs" is the right rhythm, in that order, every time.
- Operational security is mostly habits. Write the rule down the first time you break it.
- A demo is not a product. The first 80% is the engine; the last 20% is the CSS, and the last 20% is what people remember.
- The journal is a thinking tool, not a deliverable. "What stuck" is where the learning crystallizes.
- A failing browser test is more honest than a passing unit test. The unit test proves the function returned the right value; the browser test proves the user saw the right thing. When they disagree, the browser wins.
- Two-pass patterns are not optional in any framework that re-renders on state change. Pass 1 sets up state and re-renders; pass 2 sees the state and renders. The state machine belongs in `session_state`, not in local variables.
- A cache-hit branch is a first-class code path. It has its own bugs. Give it its own tests and its own `st.rerun()`.
- Classify exceptions at the helper layer, format them at the view layer. The classifier is pure and unit-testable; the formatter couples to the framework and is integration-tested.
- A project is "done" only after a real human has used it end-to-end. Until then, it is in beta, no matter what the test count says.

---

## End-of-Journey Reflection

*(Written after Phase 7. The stub has been here since Phase 1.)*

I started this project because I wanted to learn what an LLM application *is* — not the marketing version, the real version, the one that is sitting in a file called `openrouter.py` and has a `requests.post` call in it. I am closing it with a working chatbot in two interfaces, 27 green tests, an engineered system prompt, a three-layer split, a refactor that stayed green, and a journal I am not embarrassed to read. The point of the project was never the chatbot. The point was to *de-mythologize* the chatbot — to take the black box, open it, and see the wires. I have seen the wires. There are not that many of them.

**The wires, in plain language:**
- A chatbot is a function. You give it a list of `{"role": ..., "content": ...}` dictionaries and it returns a string. That is the *entire* engine.
- The model has no memory. Every "memory" you see in a product is the application re-sending prior turns inside that list on each new call. Phase 4 was the phase I stopped thinking of LLMs as smart and started thinking of them as *stateless text completers with a context window*. That is a much more useful mental model.
- The system prompt is a contract, not a greeting. Phase 5 taught me that the difference between a polite "be helpful" prompt and a four-pillar engineered prompt with a refusal clause is the difference between a toy and a tool. The system prompt is read once at the start of the conversation and used to steer every reply. It is the highest-leverage string in the entire codebase.
- The engine/interface split is the structural decision that made the project survive six more phases. The engine does not know there is a CLI. The CLI does not know there is a web UI. The web UI does not know the engine is HTTP. Each layer can be replaced without breaking the others. This is the same pattern Django calls "services and views," and it is the pattern I will reach for first on the next project.
- Tests are the safety net that makes everything else cheap. The test suite is what turned "should I add a feature?" into "I will add a feature, the tests will catch me if I break something." That confidence is what gave me permission to do a Phase 7 refactor without fear. A test suite with high coverage is *cheaper than no test suite*, even on a tiny project. Especially on a tiny project.

**The two surprises that taught me the most:**

The first was the *engine's error contract*. I designed `OpenRouterError` in Phase 2 to wrap every failure — bad model name, network down, rate limit, server error — in a single exception type. I did that because a teaching tool should fail safely. Then Phase 6 hit a live HTTP 429 against `gemma-4-31b-it:free` and the engine wrapped it exactly as designed, and the UI caught it in `st.error(...)`, and the user would have seen a friendly "model is busy, try again in a minute" message. The thing I built defensively worked defensively. I trust the design more after seeing it work in anger. *Design for the failure you cannot imagine, and the failure you can imagine will already be handled.*

The second was the *secrets-management loop*. I leaked the API key into a chat transcript in Phase 1, then leaked it *again* into an `.env` file that got shared in chat, and then a tool call overwrote the file with the literal string `<set>` while trying to redact. Three leaks in three days. The fix was a process, not a feature: real key in `.env` only, `.env` in `.gitignore`, never echo the key in any direction, and **never let any tool write a redaction back into `.env`** — the redaction is for chat display, not for the file. I have a rule now. The rule will outlive the project. *Operational security is mostly about habits, and habits are mostly about getting bitten three times in a row.*

**What I am taking into the next project:**

The engine/interface split. The system prompt as a contract. The pure-helper split when the framework is awkward to test. The test suite as the safety net that makes everything else cheap. The journal as a thinking tool, not a deliverable. The refactor-as-named-phase discipline. The "code → tests → live probe → docs" rhythm, in that order, every time. And the rule that I write the rule down the first time I break it.

**What I want to build next, and why I can build it now:**

I want to add retrieval-augmented generation (RAG) — pull a few paragraphs from a local security handbook, inject them into the `messages` list before the user's question, and let the model answer with the *right* context. The same `app.openrouter.chat(messages, ...)` call I have today is the entire backend. The frontend stays the same. The system prompt grows one new line: "When context is provided, ground your answer in it and cite it." That is the next click — and I would not have seen it as a small change if the engine/interface split had not held for seven phases.

After that: agents. The model emits a tool call, the tool result becomes a new turn in the same `messages` list, the loop continues. Same engine. Smarter loop. Same secrets-management discipline. Same test suite as the safety net.

**The line I want to close on:**

I started this project thinking the model was the interesting part. I am closing it thinking the model is the *least* interesting part. The interesting parts are the messages list, the system prompt, the error contract, the engine/interface split, the test suite, the secrets discipline, the journal, and the rhythm. The model is one HTTP call. Everything else is the project. I am a better engineer on July 1 than I was on June 1, and the engine is the same engine. That is the lesson.

---

## Post-script — June 13, 2026 (Session 4)

The reflection above is the truth as of June 12. It is no longer the whole truth. The next day I used the chat for real, and "real" caught five issues the test suite had not. The technical write-up was updated to record the multi-model selector, the visibility fixes, the two-pass pattern, the silent-success bug, and the 429 friendly-error classifier. The numbers updated from 27/27 to 41/41. The new rule — *a failing browser test is more honest than a passing unit test* — joined the durable list.

I am not going to pretend the close above is the final close. It is the close of *Phase 7*. The project is in beta until the `StreamlitAppTest` class is written and passing, the cold-start model check is in place, and the refusal clause has been probed across all five models in the selector. When those three things are done, I will write a second end-of-journey reflection, dated after this post-script, and this one will become the "close of the foundation phase" rather than the "close of the project." The two closes are both true. The second one is just truer.

---

## Post-script — June 15, 2026 (Session 7)

The June 13 post-script is still true. The June 12 close-of-foundation close is still true. This is a third layer on top — Session 7, the multimodal pass.

The user asked for file and image upload. The first thing the model said was *"I cannot read binary files."* That sentence taught me three things in one go. First, the model is a text model by default and most free-tier models are still text-only; the multimodal capability is a separate query axis that has to be turned on by picking a vision-capable model. Second, OpenRouter's `messages` schema is more permissive than I thought: a single turn can be either a string OR a list of `{"type": "text" | "image_url"}` blocks, and the provider figures out which mode the model needs. Third, the cleanest split is to have a pure helper (`build_user_turn_content`) that returns `str | list[dict]` and let the engine serialize whatever shape comes back — the engine does not need to know whether the turn carries an image.

Then came the empirical work. Four free-tier vision models were tried. Three returned errors at the OpenRouter gateway (decommissioned, 404, or rate-limited to zero). The one that worked was `nvidia/nemotron-nano-12b-v2-vl:free`, and it became the hardcoded fallback `_DEFAULT_FREE_VISION_MODEL` at line 704 of `web/chat_helpers.py`. The lesson: do not assume the model zoo. Probe it. The first free-tier vision model in any list of "free vision models on OpenRouter" is not necessarily live this week.

The cleanest architectural lesson of this session was about return types. The `_build_messages` function in `web/chat_helpers.py` had always returned a flat `list[dict]` because the engine expects that shape. I had been tempted to add an `if has_image: return [...]` branch in the *engine* to handle the vision content array. But the engine does not need to know. The cleanest place to make the `str` vs `list[dict]` decision is in the helper, right where the upload is processed, and the engine just serializes the result. The principle I am taking forward: **let the layer that owns the decision make the decision; let the layers that do not own it pass the result through unchanged.** I had been violating this by sketching engine-side branches that the engine should not have known about.

The `pymupdf` dependency taught me a different lesson: the developer who runs the unit tests should not have to install a 40 MB native library just because one test exercises a PDF path. The fix is the lazy-import pattern — `import pymupdf` lives inside the function that needs it, not at the top of the module. The unit tests that do not touch PDFs never load the library; the unit tests that do, install it on first run via `pip install pymupdf`. The pattern generalizes: any optional native dependency should be lazy-imported at the call site, with the import failure caught and re-raised as a typed `FileProcessingError("pdf_library_missing", ...)` so the view can render a clean "run `pip install pymupdf`" banner.

The hardest part of Session 7 was not the code. It was the documentation. The technical write-up was already a few weeks stale by the time I sat down to update it — the row for "file & image upload" had been written when the feature was advisory, and the actual code had long since shipped. I had to read the code, the tests, and the prior write-up side by side, and acknowledge in writing that the docs had lagged the code. Decision 11 codifies the new rule: **the phase docs ship in the same commit as the code, with a four-step checklist (architecture row + decisions row + problems row + open-questions row), a loose definition of "same commit" (a documented spike is OK), and an explicit verification step at session close.** The row I wrote about that lag is the row that tests the rule.

The five process gaps recorded in §8.2 of the technical write-up (key-rotation checklist, stale-worker detection, .env redaction discipline, API-key-in-clipboard audit, pre-commit hook) are not engineering. They are the *operating discipline* of a project that has real users and real secrets. I had learned each of them by getting burned — the §6 leak taught the redaction one, the stale worker taught the detection one, the rate-limit loop taught the rotation one. Recording them in the open questions section, with the incident that taught each, is how I make sure I do not have to learn them again.

The new close of the journey, dated after this post-script: I am now a better engineer on June 15 than I was on June 13, and the engine is the same engine. The added capability is real — the model can now see a PDF and answer a question about it — but the lesson is the same lesson. The interesting part is still the messages list, the system prompt, the error contract, the engine/interface split, the test suite, the secrets discipline, the journal, and the rhythm. The model is still one HTTP call. Everything else is still the project. The new truth is that the project is now a multimodal one, and the rules for building a multimodal project are the same rules I learned building a text one. The discipline is the same. The cleanup is bigger. The trust is harder to earn.
