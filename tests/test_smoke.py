"""Smoke + structural tests for Phase 3 (with Phase 8 additions).

These tests do NOT call the live OpenRouter API. They verify that:
- The package layout is still valid (Phase 2 regression check).
- The config module loads from .env.
- The openrouter module is importable and exposes the expected surface.
- The error path of `chat()` is exercised by feeding it a bad base URL
  pointing at localhost, so we exercise the real requests code path
  without spending an API credit.
- The teaching-mode sidebar swap block in the Streamlit view fires on a
  real user click and does not spuriously fire on a fresh session, and
  the **web UI default is the CTF/lab mentor persona** ("SecMentor").

Run with:  python -m unittest tests.test_smoke -v
"""

import importlib
import os
import re
import sys
import unittest
from unittest import mock
from unittest.mock import patch

import pytest

from app import config, openrouter, prompts
from app.openrouter import OpenRouterError, chat
from web.chat_helpers import _active_system_prompt

pytestmark = pytest.mark.smoke

# Absolute path to the Streamlit view script. Used by the sys.path
# bootstrap tests below so they can exec the bootstrap block in an
# isolated namespace without spinning up a real Streamlit session.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_STREAMLIT_VIEW = os.path.join(_REPO, "web", "streamlit_app.py")


def _read_streamlit_view() -> str:
    """Read the Streamlit view source as text.

    Streamlit's renderer can't be unit-tested (it requires a browser), but
    the file itself is a Python module and we can read it like any other
    file. Structural tests below pin its contract so a bad edit to the
    two-pass pattern or the model selector surfaces in CI, not in a
    browser screenshot.
    """
    with open(_STREAMLIT_VIEW, "r", encoding="utf-8") as fh:
        return fh.read()


class SmokeTests(unittest.TestCase):
    """Basic structural checks (Phase 2 regression)."""

    def test_app_modules_import(self):
        self.assertIsNotNone(config)
        self.assertIsNotNone(openrouter)
        self.assertIsNotNone(prompts)

    def test_default_prompt_is_string(self):
        self.assertIsInstance(prompts.DEFAULT_SYSTEM_PROMPT, str)
        self.assertGreater(len(prompts.DEFAULT_SYSTEM_PROMPT), 0)


class PromptTests(unittest.TestCase):
    """Verify the Phase 5 engineered cybersecurity system prompt.

    These checks pin the prompt's scope and guardrails. If anyone ever
    edits `app/prompts.py` and accidentally drops a pillar or weakens the
    refusal clause, the test suite tells us immediately.
    """

    PROMPT = prompts.DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_is_substantial(self):
        """A real prompt should be a few hundred characters, not a stub."""
        self.assertGreater(len(prompts.DEFAULT_SYSTEM_PROMPT), 400)

    def test_prompt_covers_defensive_pillar(self):
        self.assertIn("defensive", self.PROMPT)
        self.assertIn("incident response", self.PROMPT)

    def test_prompt_covers_devsecops_pillar(self):
        self.assertIn("devsecops", self.PROMPT)
        self.assertIn("supply-chain", self.PROMPT)

    def test_prompt_covers_ai_security_pillar(self):
        self.assertIn("prompt injection", self.PROMPT)
        self.assertIn("owasp", self.PROMPT)

    def test_prompt_covers_offensive_education_pillar(self):
        self.assertIn("offensive", self.PROMPT)
        self.assertIn("attack", self.PROMPT)

    def test_prompt_has_explicit_refusal_clause(self):
        """The refusal to produce weaponized output must be present."""
        self.assertIn("you do not generate", self.PROMPT)
        self.assertIn("exploit", self.PROMPT)
        self.assertIn("malware", self.PROMPT)

    def test_prompt_points_users_to_legal_labs(self):
        """Educational redirect to authorized lab platforms must exist."""
        self.assertIn("lab", self.PROMPT)


class OffensiveMentorPromptTests(unittest.TestCase):
    """Pin the Phase 8 CTF / lab mentor prompt's scope and refusal clauses.

    The mentor prompt is a *deliberate* expansion of the defensive prompt:
    it authorizes working exploit snippets, payload one-liners, msfvenom
    recipes, and malware-family analysis, BUT only when the user is
    working in a lab they own (HTB, THM, PortSwigger, DVWA, WebGoat,
    picoCTF, or their own VMs).

    The tests in this class pin the boundary so a future edit cannot
    silently weaken the refusal clauses (real-target payloads, WAF/EDR
    bypasses, brand-new malware strains) or silently drop the
    authorization framing that lets a student actually use the mentor
    mode for its intended purpose.
    """

    PROMPT = prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT
    PROMPT_LOWER = PROMPT.lower()

    def test_mentor_prompt_is_substantial(self):
        """A real prompt should be several KB, not a stub."""
        self.assertGreater(len(self.PROMPT), 1500)

    def test_mentor_prompt_is_distinct_from_defensive(self):
        """The two profiles must be different prompts, not duplicates."""
        self.assertNotEqual(
            prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT,
            prompts.CYBERSECURITY_SYSTEM_PROMPT,
            "mentor prompt must not be a copy of the defensive prompt",
        )

    def test_mentor_prompt_names_itself(self):
        """The persona must be self-identifying so the model cannot
        accidentally impersonate the safer 'SecTutor' voice."""
        self.assertIn("SecMentor", self.PROMPT)

    def test_mentor_prompt_keeps_defensive_pillar(self):
        self.assertIn("defensive", self.PROMPT_LOWER)
        self.assertIn("threat modeling", self.PROMPT_LOWER)

    def test_mentor_prompt_keeps_devsecops_pillar(self):
        self.assertIn("devsecops", self.PROMPT_LOWER)
        self.assertIn("supply chain", self.PROMPT_LOWER)

    def test_mentor_prompt_keeps_ai_security_pillar(self):
        self.assertIn("prompt injection", self.PROMPT_LOWER)
        self.assertIn("owasp", self.PROMPT_LOWER)

    def test_mentor_prompt_covers_offensive_in_lab_scope(self):
        """The mentor prompt must explicitly mention the lab scope and
        the core offensive-security sub-topics a CTF learner needs."""
        for needle in (
            "hackthebox",
            "tryhackme",
            "portswigger",
            "dvwa",
            "webgoat",
            "sql injection",
            "xss",
            "privilege escalation",
            "reverse shell",
            "msfvenom",
        ):
            self.assertIn(
                needle, self.PROMPT_LOWER,
                f"mentor prompt is missing expected term {needle!r}",
            )

    def test_mentor_prompt_authorizes_working_snippets_in_lab(self):
        """The mentor prompt must explicitly authorize working snippets
        in the lab context — otherwise it collapses back into the
        defensive-only behavior the user is trying to escape."""
        self.assertIn("runnable", self.PROMPT_LOWER)
        # 'for your lab' framing is the policy that distinguishes mentor
        # mode from 'no working code ever' mode.
        self.assertIn("for your lab", self.PROMPT_LOWER)
        # Reference examples for the labeling convention the model is
        # expected to follow in every code block.
        self.assertIn("10.10.10.3", self.PROMPT)  # HackTheBox sample
        self.assertIn("10.10.210.71", self.PROMPT)  # TryHackMe sample

    def test_mentor_prompt_requires_defensive_countermeasure(self):
        """Every offensive answer in mentor mode must include the
        defensive countermeasure alongside the technique."""
        self.assertIn("defensive", self.PROMPT_LOWER)
        self.assertIn("countermeasure", self.PROMPT_LOWER)
        # A handful of named countermeasures that should appear as
        # worked examples the model is meant to mirror.
        for needle in (
            "parameterized queries",
            "output encoding",
            "egress filtering",
            "least privilege",
        ):
            self.assertIn(
                needle, self.PROMPT_LOWER,
                f"mentor prompt should cite {needle!r} as a worked "
                f"defensive countermeasure",
            )

    def test_mentor_prompt_refuses_real_target_payloads(self):
        """The headline boundary: no payloads against a specific named
        real system the user does not own."""
        # The phrase 'real production' is the boundary the prompt uses
        # internally; the model is told to redirect real targets to
        # the lab equivalent.
        self.assertIn("real production", self.PROMPT_LOWER)
        self.assertIn("decline", self.PROMPT_LOWER)
        # 'written authorization' is the legal framing for sanctioned
        # pentests, as opposed to 'I found a real server on Shodan'.
        self.assertIn("written authorization", self.PROMPT_LOWER)

    def test_mentor_prompt_refuses_named_vendor_bypasses(self):
        """The WAF/EDR/MFA bypass refusal must be present, otherwise
        mentor mode silently becomes a vendor-bypass assistant."""
        for needle in ("waf", "edr", "mfa"):
            self.assertIn(needle, self.PROMPT_LOWER)

    def test_mentor_prompt_refuses_brand_new_malware(self):
        """The model may analyze malware families but must not author
        a brand-new strain on demand."""
        self.assertIn("brand-new", self.PROMPT_LOWER)
        self.assertIn("malware", self.PROMPT_LOWER)
        # Specifically: the prompt should say it won't write a fresh
        # ransomware / C2 / dropper, while still allowing analysis.
        self.assertIn("ransomware", self.PROMPT_LOWER)
        self.assertIn("c2", self.PROMPT_LOWER)
        self.assertIn("dropper", self.PROMPT_LOWER)

    def test_mentor_prompt_refuses_harm_to_physical_systems(self):
        """Critical-infrastructure carve-out must be present."""
        self.assertIn("critical infrastructure", self.PROMPT_LOWER)
        self.assertIn("medical", self.PROMPT_LOWER)

    def test_mentor_prompt_redirects_distress_to_authorities(self):
        """If the user is in distress about a real incident, the prompt
        must direct them to CISA / CERT and stop speculating."""
        self.assertIn("cisa", self.PROMPT_LOWER)
        self.assertIn("cert", self.PROMPT_LOWER)


class WebHelpersActivePromptTests(unittest.TestCase):
    """Pin the helper that selects which system prompt is active.

    The web view stores `teaching_mode` in session_state. The helper
    `_active_system_prompt` returns the right constant for the mode so
    the view layer does not have to know the two prompt names directly.
    """

    def test_defensive_mode_returns_defensive_prompt(self):
        from web.chat_helpers import _active_system_prompt
        from app.prompts import CYBERSECURITY_SYSTEM_PROMPT

        self.assertIs(
            _active_system_prompt({"teaching_mode": "defensive"}),
            CYBERSECURITY_SYSTEM_PROMPT,
        )

    def test_mentor_mode_returns_mentor_prompt(self):
        from web.chat_helpers import _active_system_prompt
        from app.prompts import OFFENSIVE_MENTOR_SYSTEM_PROMPT

        self.assertIs(
            _active_system_prompt({"teaching_mode": "mentor"}),
            OFFENSIVE_MENTOR_SYSTEM_PROMPT,
        )

    def test_unknown_mode_falls_back_to_defensive(self):
        """An unknown mode must default to the safer prompt, never to
        the wider mentor scope — fail-closed on ambiguity."""
        from web.chat_helpers import _active_system_prompt
        from app.prompts import CYBERSECURITY_SYSTEM_PROMPT

        self.assertIs(
            _active_system_prompt({"teaching_mode": "experimental"}),
            CYBERSECURITY_SYSTEM_PROMPT,
        )
        self.assertIs(
            _active_system_prompt({}),  # missing key
            CYBERSECURITY_SYSTEM_PROMPT,
        )

    def test_non_dict_state_falls_back_to_defensive(self):
        """Defensive: a malformed state must not crash the helper."""
        from web.chat_helpers import _active_system_prompt
        from app.prompts import CYBERSECURITY_SYSTEM_PROMPT

        self.assertIs(
            _active_system_prompt(None),
            CYBERSECURITY_SYSTEM_PROMPT,
        )


class ConfigTests(unittest.TestCase):
    """Verify .env was loaded and the required values are present."""

    def test_api_key_is_loaded(self):
        # We do NOT assert the actual value (security) — only that it loaded.
        self.assertTrue(config.OPENROUTER_API_KEY)
        self.assertTrue(config.OPENROUTER_API_KEY.startswith("sk-or-v1-"))

    def test_model_is_loaded(self):
        # Either a single ``OPENROUTER_MODEL`` *or* a plural
        # ``OPENROUTER_MODELS`` list must yield at least one usable
        # free-tier id. This used to assert only the singular
        # (``config.OPENROUTER_MODEL`` non-empty), which broke the
        # deploy dashboard after the rotation-policy fix made the
        # plural list the supported primary form.
        from app.config import iter_models
        ids = list(iter_models())
        self.assertTrue(ids, "no model ids configured (set OPENROUTER_MODEL or OPENROUTER_MODELS)")
        for mid in ids:
            self.assertIn("/", mid)
            self.assertTrue(mid.endswith(":free"), f"{mid!r} must end with ':free'")

    def test_base_url_is_loaded(self):
        self.assertTrue(config.OPENROUTER_BASE_URL.startswith("https://"))
        self.assertTrue(config.OPENROUTER_BASE_URL.endswith("/chat/completions"))

    def test_numeric_defaults_are_sane(self):
        self.assertGreater(config.DEFAULT_TEMPERATURE, 0.0)
        self.assertLessEqual(config.DEFAULT_TEMPERATURE, 2.0)
        self.assertGreater(config.DEFAULT_MAX_TOKENS, 0)
        self.assertGreater(config.HTTP_TIMEOUT_SECONDS, 0)


class OpenRouterClientTests(unittest.TestCase):
    """Verify the openrouter.client surface and error paths."""

    def test_empty_messages_raises(self):
        with self.assertRaises(OpenRouterError):
            chat([])

    def test_network_failure_is_wrapped(self):
        """Pointing at a closed port must raise OpenRouterError, not crash."""
        # 127.0.0.1:1 is reserved and never listens -> connection refused.
        with patch.object(openrouter, "OPENROUTER_BASE_URL", "http://127.0.0.1:1"):
            with self.assertRaises(OpenRouterError):
                chat([{"role": "user", "content": "hello"}])


class ChatbotHistoryTests(unittest.TestCase):
    """Verify the Phase 4 history accumulator in the CLI chatbot.

    These tests are offline — they never call OpenRouter. They exercise the
    pure-Python message-builder logic so we can confirm the conversation
    memory is wired up correctly before the user sees it in the terminal.
    """

    def test_build_messages_includes_prior_turn(self):
        """A follow-up turn must carry the prior user/assistant exchange."""
        from cli.chatbot import _build_messages

        history = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "What is XSS?"},
            {"role": "assistant", "content": "Cross-site scripting."},
        ]
        messages = _build_messages(history, "Can you give an example?")

        # New payload should be exactly: history + new user turn.
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is XSS?")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Cross-site scripting.")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], "Can you give an example?")

    def test_build_messages_keeps_system_prompt_first(self):
        """No matter what the caller passes, the system message must lead."""
        from cli.chatbot import _build_messages

        # Even if a caller hands us garbage history, the system prompt is
        # whatever they put first — we trust the caller to maintain it.
        # This test pins the contract: history[0] is always the system role.
        history = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
        ]
        messages = _build_messages(history, "Hello again")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Hello again")


class ChatHelpersTests(unittest.TestCase):
    """Verify the Phase 6 pure helpers used by the Streamlit UI.

    The Streamlit file itself is not unit-tested (Streamlit is a view
    layer; its tests are "open the browser, click around"). But the
    helpers that shape state — building messages, truncating history,
    serializing a transcript — are pure functions and worth pinning
    down so a bad edit to the UI logic surfaces here, not in a
    browser screenshot.
    """

    def test_build_messages_appends_user_turn(self):
        from web.chat_helpers import _build_messages

        history = [
            {"role": "system", "content": "You are SecTutor."},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        messages = _build_messages(history, "Follow-up question")
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1], {"role": "user", "content": "Follow-up question"})
        # The original list must not be mutated.
        self.assertEqual(len(history), 3)

    def test_build_messages_rejects_empty_history(self):
        from web.chat_helpers import _build_messages

        with self.assertRaises(ValueError):
            _build_messages([], "hi")

    def test_build_messages_rejects_blank_user_input(self):
        from web.chat_helpers import _build_messages

        with self.assertRaises(ValueError):
            _build_messages(
                [{"role": "system", "content": "sys"}], "   \n  "
            )

    def test_truncate_history_keeps_system_prompt(self):
        from web.chat_helpers import _truncate_history

        history: list = [
            {"role": "system", "content": "You are SecTutor."},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        truncated = _truncate_history(history, max_messages=2)
        # System prompt preserved + last 2 turns only.
        self.assertEqual(truncated[0]["role"], "system")
        self.assertEqual(len(truncated), 3)
        self.assertEqual([m["content"] for m in truncated[1:]], ["u2", "a2"])

    def test_truncate_history_no_op_when_under_cap(self):
        from web.chat_helpers import _truncate_history

        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
        ]
        out = _truncate_history(history, max_messages=10)
        self.assertEqual(out, history)

    def test_truncate_history_rejects_zero(self):
        from web.chat_helpers import _truncate_history

        with self.assertRaises(ValueError):
            _truncate_history(
                [{"role": "system", "content": "s"}], max_messages=0
            )

    def test_serialize_for_download_includes_roles(self):
        from web.chat_helpers import _serialize_for_download

        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = _serialize_for_download(history, model="gemma/test:free")
        self.assertIn("Model: gemma/test:free", out)
        self.assertIn("--- USER ---", out)
        self.assertIn("hi", out)
        self.assertIn("--- ASSISTANT ---", out)
        self.assertIn("hello", out)

    def test_serialize_for_download_works_without_model(self):
        from web.chat_helpers import _serialize_for_download

        out = _serialize_for_download([{"role": "user", "content": "x"}])
        self.assertNotIn("Model:", out)
        self.assertIn("--- USER ---", out)

    def test_count_chars_sums_all_content(self):
        from web.chat_helpers import _count_chars

        history = [
            {"role": "system", "content": "abcd"},     # 4
            {"role": "user", "content": "hello"},     # 5
            {"role": "assistant", "content": "world!"},  # 6
        ]
        self.assertEqual(_count_chars(history), 15)

    def test_bubble_alignment_user_right_others_left(self):
        from web.chat_helpers import _bubble_alignment

        self.assertEqual(_bubble_alignment("user"), "right")
        self.assertEqual(_bubble_alignment("assistant"), "left")
        self.assertEqual(_bubble_alignment("system"), "left")
        self.assertEqual(_bubble_alignment("tool"), "left")


class StreamlitViewImportSurfaceTests(unittest.TestCase):
    """Pin the view's import surface against `web/chat_helpers.py`.

    The view is a top-level Streamlit script. It can only see helpers it
    explicitly imports from `web.chat_helpers`. If someone adds a new
    helper to `chat_helpers.py` and calls it from the view, the unit
    tests for that helper still pass (they import the helper directly).
    The `NameError` only fires when the browser loads the running app.

    This test parses the view source and the helper source, builds the
    set of names the view actually references, and asserts every
    referenced name from `chat_helpers.py` is in the view's import
    block. Bug family: the same family that produced the
    `NameError: name '_bubble_alignment' is not defined` and
    `NameError: name '_build_messages' is not defined` incidents.
    """

    @staticmethod
    def _defined_helpers_in_chat_helpers() -> set[str]:
        """Return the set of `_<name>` function names defined in
        `web/chat_helpers.py` (top-level def statements only)."""
        path = os.path.join(_REPO, "web", "chat_helpers.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        return set(re.findall(r"^def (_[a-z][a-z0-9_]*)\(", src, re.MULTILINE))

    @staticmethod
    def _view_imports_from_chat_helpers(view_src: str) -> set[str]:
        """Return the set of names listed in the view's
        `from web.chat_helpers import (...)` block. Tolerant of
        multi-line layouts, trailing commas, and blank lines. Returns
        an empty set if the import block is missing or uses a form
        other than the parenthesised one (e.g. one-name-per-line)."""
        match = re.search(
            r"from\s+web\.chat_helpers\s+import\s+\((.*?)\)",
            view_src,
            re.DOTALL,
        )
        if not match:
            return set()
        block = match.group(1)
        return {n.strip() for n in block.split(",") if n.strip()}

    @staticmethod
    def _view_references_to_chat_helpers(view_src: str) -> set[str]:
        """Return the set of `_<name>(` references in the view body.

        This is an over-approximation: it catches every call-shaped
        identifier that starts with an underscore. False positives are
        fine (an `_<name>(` inside a docstring or a comment is
        harmless); the goal is to surface *misses*, not to be exact.
        """
        # Strip docstrings so an `_<name>(...)` mentioned in a comment
        # block or module docstring doesn't trip the check. We use a
        # simple heuristic: a triple-quoted string anywhere in the
        # file is treated as documentation and skipped.
        scrubbed = re.sub(r'"""[\s\S]*?"""', "", view_src)
        scrubbed = re.sub(r"'''[\s\S]*?'''", "", scrubbed)
        return set(re.findall(r"\b(_[a-z][a-z0-9_]*)\s*\(", scrubbed))

    def test_view_imports_every_chat_helper_it_references(self):
        """The view's import block must cover every helper it calls.

        Pulls the list of helpers from `web/chat_helpers.py`,
        intersects with the view's actual call sites, and asserts the
        view's `from web.chat_helpers import (...)` block includes the
        full intersection. Failing here means the running app would
        hit `NameError` on first render.
        """
        defined = self._defined_helpers_in_chat_helpers()
        view_src = _read_streamlit_view()
        imported = self._view_imports_from_chat_helpers(view_src)
        referenced = self._view_references_to_chat_helpers(view_src)

        # What the view calls that lives in chat_helpers.
        called_from_helpers = referenced & defined
        # What is called but NOT imported. This is the failure case.
        missing = called_from_helpers - imported

        self.assertEqual(
            missing,
            set(),
            (
                "web/streamlit_app.py references chat_helpers names "
                "that are not in its import block. The running app "
                "will hit NameError on first render. "
                f"Missing: {sorted(missing)}. "
                f"Referenced from chat_helpers: {sorted(called_from_helpers)}. "
                f"Currently imported: {sorted(imported & defined)}."
            ),
        )


class TeachingModeSwapTests(unittest.TestCase):
    """Pin the sidebar's teaching-mode swap detection.

    Bug: when the user clicks the radio from "Defensive" to "CTF / Lab
    mentor", the widget with `key="teaching_mode"` writes the new value
    to `st.session_state["teaching_mode"]` *before* the script body runs
    on the rerun-after-click. The original view read `_previous_mode`
    from the same `st.session_state["teaching_mode"]` key, so it always
    saw the new value, the comparison `_chosen_mode != _previous_mode`
    was always false, and the swap block was silently skipped. The
    radio toggled visually but the system prompt never changed.

    Fix: track the *previous* mode in a separate
    `teaching_mode_previous` key that is only updated *after* the swap
    fires. On the rerun-after-click, `teaching_mode` (live, written by
    the widget) ≠ `teaching_mode_previous` (still the old value) and
    the swap runs.

    These tests pin (a) the helper correctly swaps the system prompt
    when teaching_mode changes (already covered elsewhere, but pinned
    here as a precondition), and (b) the view's swap-detection logic
    is structured to actually fire. The behavioral assertion uses a
    simulated session_state dict (the same shape the view uses); the
    structural assertion reads the view source.
    """

    @staticmethod
    def _simulate_swap(state: dict, chosen_mode: str) -> bool:
        """Replicate the view's swap-detection + swap block.

        Returns True if the swap would fire for `chosen_mode`. Mirrors
        the logic in `web/streamlit_app.py` exactly so a regression in
        the view code does not silently make this test pass.
        """
        previous_mode = state.get(
            "teaching_mode_previous",
            state.get("teaching_mode", "defensive"),
        )
        if chosen_mode != previous_mode:
            state["messages"][0] = {
                "role": "system",
                # The view uses the helper; we call it the same way.
                "content": _active_system_prompt({"teaching_mode": chosen_mode}),
            }
            state["response_cache"] = {}
            state["teaching_mode_previous"] = chosen_mode
            return True
        return False

    def test_swap_fires_when_user_clicks_mentor(self):
        """The first click from defensive → mentor must reseed the
        system message with the mentor prompt."""
        defensive_prompt = prompts.CYBERSECURITY_SYSTEM_PROMPT
        mentor_prompt = prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT
        state = {
            "teaching_mode": "mentor",  # what the widget just wrote
            "teaching_mode_previous": "defensive",  # what was there before
            "messages": [
                {"role": "system", "content": defensive_prompt},
                {"role": "user", "content": "hi"},
            ],
        }
        fired = self._simulate_swap(state, chosen_mode="mentor")
        self.assertTrue(
            fired,
            "Swap must fire on defensive → mentor click (pre-fix it "
            "was silently skipped because both the widget and the "
            "_previous_mode read targeted st.session_state['teaching_mode']).",
        )
        self.assertIs(
            state["messages"][0]["content"],
            mentor_prompt,
            "After swap, the system message must be the mentor prompt.",
        )
        self.assertEqual(
            state["teaching_mode_previous"],
            "mentor",
            "After swap, teaching_mode_previous must be updated so the "
            "next comparison is stable.",
        )

    def test_swap_does_not_fire_on_initial_render(self):
        """A fresh session must not spuriously swap.

        `_init_state` seeds `teaching_mode_previous` equal to the live
        `teaching_mode`. If that seed is missing, the helper's fallback
        (`st.session_state.get('teaching_mode', 'defensive')`) still
        produces a stable comparison, so the swap must not fire.
        """
        defensive_prompt = prompts.CYBERSECURITY_SYSTEM_PROMPT
        # Case A: seed present, comparison stable.
        state_seeded = {
            "teaching_mode": "defensive",
            "teaching_mode_previous": "defensive",
            "messages": [{"role": "system", "content": defensive_prompt}],
        }
        self.assertFalse(
            self._simulate_swap(state_seeded, chosen_mode="defensive"),
            "No swap on a no-op click.",
        )
        # Case B: seed missing (first run after upgrade), comparison
        # falls back to the live key, still stable.
        state_unseeded = {
            "teaching_mode": "defensive",
            "messages": [{"role": "system", "content": defensive_prompt}],
        }
        self.assertFalse(
            self._simulate_swap(state_unseeded, chosen_mode="defensive"),
            "No spurious swap when teaching_mode_previous is missing "
            "and the live key already matches.",
        )

    def test_swap_fires_back_to_defensive(self):
        """Mentor → defensive must also reseed."""
        defensive_prompt = prompts.CYBERSECURITY_SYSTEM_PROMPT
        mentor_prompt = prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT
        state = {
            "teaching_mode": "defensive",  # widget just wrote
            "teaching_mode_previous": "mentor",  # was mentor
            "messages": [
                {"role": "system", "content": mentor_prompt},
                {"role": "user", "content": "give me the SQLi payload"},
            ],
        }
        fired = self._simulate_swap(state, chosen_mode="defensive")
        self.assertTrue(fired, "Swap must fire on mentor → defensive click.")
        self.assertIs(
            state["messages"][0]["content"],
            defensive_prompt,
            "After swap back, the system message must be the defensive prompt.",
        )

    def test_view_uses_separate_previous_key(self):
        """Static check: the view source must track the previous
        mode in a separate key, not in the live `teaching_mode` key
        that the radio widget overwrites.

        Bug family: re-introducing `_previous_mode = st.session_state
        .get('teaching_mode', 'defensive')` would re-break the toggle
        and the user would see the radio flip without any system
        prompt change.
        """
        view_src = _read_streamlit_view()
        lines = view_src.splitlines()

        # The previous-mode read must reference the separate key.
        self.assertIn(
            "teaching_mode_previous",
            view_src,
            "The view must use a separate `teaching_mode_previous` key "
            "for the swap comparison, because the radio's `key='teaching_mode'` "
            "writes the new value into st.session_state['teaching_mode'] "
            "before the script body runs on the rerun-after-click.",
        )

        # Find the line where `_previous_mode = st.session_state.get(`
        # starts. The call is multi-line in the actual source (the
        # helper has a fallback default), so we look at the next ~6
        # lines to see if `teaching_mode_previous` appears in the call.
        previous_start = next(
            (i for i, line in enumerate(lines, start=1)
             if re.search(r"_previous_mode\s*=\s*st\.session_state\.get\(", line)),
            0,
        )
        self.assertGreater(
            previous_start, 0,
            "Could not find `_previous_mode = st.session_state.get(` "
            "in the view. The swap-detection logic must read the "
            "previous mode from session_state, not hard-code it.",
        )
        # Look at a window of 8 lines after `previous_start` (covers
        # the multi-line fallback in the actual source).
        window_end = min(previous_start + 8, len(lines))
        window = "\n".join(lines[previous_start - 1:window_end])
        self.assertIn(
            "teaching_mode_previous", window,
            "The `_previous_mode = st.session_state.get(...)` call must "
            "read the separate `teaching_mode_previous` key, not the "
            "live `teaching_mode` key (which the radio widget overwrites "
            "before the script body runs).",
        )

        # The previous-mode read must happen BEFORE the teaching-mode
        # ``st.radio`` call. The view may have other unrelated radios
        # (e.g. a layout-mode toggle) that would match a bare
        # ``st.radio(`` substring, so we anchor on the
        # ``key="teaching_mode"`` argument within a 12-line window after
        # the call (the actual call is multi-line in the source).
        radio_line = 0
        for i, line in enumerate(lines, start=1):
            if "st.radio(" not in line:
                continue
            window_end = min(i + 12, len(lines))
            window = "\n".join(lines[i - 1:window_end])
            if 'key="teaching_mode"' in window:
                radio_line = i
                break
        self.assertGreater(radio_line, 0,
                           "Could not find the teaching-mode st.radio( "
                           "(key=\"teaching_mode\") in the view.")
        self.assertLess(
            previous_start, radio_line,
            "The previous-mode read must happen BEFORE the teaching-mode "
            f"st.radio( call. Read at line {previous_start}, radio at line {radio_line}.",
        )

        # The swap block must update teaching_mode_previous after firing.
        # We assert the assignment appears inside the `if _chosen_mode !=
        # _previous_mode:` block, after the messages reseed.
        swap_block = re.search(
            r"if\s+_chosen_mode\s*!=\s*_previous_mode\s*:(.*?)(?=\n\s*st\.divider\(\)|\n\s*st\.markdown\(|\Z)",
            view_src,
            re.DOTALL,
        )
        self.assertIsNotNone(
            swap_block,
            "Could not find the `if _chosen_mode != _previous_mode:` "
            "swap block in the view.",
        )
        self.assertIn(
            "teaching_mode_previous",
            swap_block.group(1),
            "The swap block must update teaching_mode_previous after "
            "firing, otherwise the comparison is unstable.",
        )

    def test_view_seeds_mentor_as_default(self):
        """The web UI must default the teaching mode to the wider
        CTF/lab mentor persona (SecMentor), not the defensive one.

        Rationale: SecMentor is the product surface we built in
        Phase 8; learners landing on the page should get the lab-
        scoped teaching persona from the first turn. The defensive
        prompt stays one click away in the sidebar and is the
        default for the CLI (`app.prompts.DEFAULT_SYSTEM_PROMPT`).

        This is a structural test on the view source so the default
        can only change by also updating this pin (intentional
        breakage of the regression test = intentional doc change).
        """
        view_src = _read_streamlit_view()
        # 1) The fresh-session seed must be "mentor".
        self.assertRegex(
            view_src,
            r'_DEFAULT_TEACHING_MODE\s*:\s*str\s*=\s*"mentor"',
            "The fresh-session seed in `_init_state` must be the "
            "mentor mode (the wider CTF/lab scope).",
        )
        # 2) The sidebar radio options must list "mentor" first so
        # the visible default is the mentor persona (the radio's
        # index= fallback also lands on index 0).
        self.assertRegex(
            view_src,
            r'_TEACHING_OPTIONS\s*:\s*list\[str\]\s*=\s*\[\s*"mentor"\s*,\s*"defensive"\s*\]',
            'The sidebar radio must list ["mentor", "defensive"] so '
            "the visible default is the mentor persona.",
        )
        # 3) The `_previous_mode` fallback (used when both the
        # separate-key seed and the live key are missing on a
        # brand-new session) must agree with the seed default.
        self.assertRegex(
            view_src,
            r'st\.session_state\.get\(\s*"teaching_mode"\s*,\s*"mentor"\s*\)',
            "The `_previous_mode` fallback default must be "
            '"mentor" so a fresh session is self-consistent with '
            "the seed in `_init_state`.",
        )

    def test_view_swaps_to_mentor_prompt_on_first_render(self):
        """End-to-end (structural) check: on a fresh session the
        system message must be the mentor prompt, not the defensive
        one, and the swap block must NOT fire (because there is
        nothing to swap — the seed already matches the visible
        radio).
        """
        mentor_prompt = prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT
        defensive_prompt = prompts.CYBERSECURITY_SYSTEM_PROMPT
        state = {
            # _init_state seeds these on a fresh session.
            "teaching_mode": "mentor",
            "teaching_mode_previous": "mentor",
            "messages": [
                {"role": "system", "content": mentor_prompt},
            ],
        }
        # No swap on initial render (previous == chosen == "mentor").
        self.assertFalse(
            self._simulate_swap(state, chosen_mode="mentor"),
            "On a fresh session with the new default, no swap "
            "must fire — the seed already matches the radio.",
        )
        # And the system message is the mentor prompt, not defensive.
        self.assertIs(
            state["messages"][0]["content"],
            mentor_prompt,
            "Fresh-session default system message must be the "
            "OFFENSIVE_MENTOR_SYSTEM_PROMPT constant.",
        )
        # Sanity: the two prompts really are different objects —
        # otherwise the test would pass trivially.
        self.assertIsNot(
            mentor_prompt,
            defensive_prompt,
            "Sanity: the two prompt constants must be distinct objects.",
        )


class FriendlyErrorTests(unittest.TestCase):
    """Pin the rate-limit / generic-error mapping for the chat UI.

    The view used to dump the raw upstream JSON to the user. We now
    classify the error and show a short, actionable message. These
    tests guard the classification so a wording change in the engine
    ("HTTP 429" -> "status 429" for example) does not silently
    regress the UI to the raw-JSON banner.
    """

    def test_is_rate_limit_true_for_http_429_substring(self):
        from web.chat_helpers import _is_rate_limit_error

        class FakeExc(BaseException):
            pass

        self.assertTrue(
            _is_rate_limit_error(
                RuntimeError("OpenRouter returned HTTP 429: {...}")
            )
        )

    def test_is_rate_limit_false_for_500(self):
        from web.chat_helpers import _is_rate_limit_error

        self.assertFalse(
            _is_rate_limit_error(
                RuntimeError("OpenRouter returned HTTP 500: boom")
            )
        )

    def test_friendly_error_mentions_model_for_429(self):
        from web.chat_helpers import _friendly_error_message

        exc = RuntimeError(
            "OpenRouter returned HTTP 429: {retry_after_seconds: 29}"
        )
        headline, body = _friendly_error_message(exc, "meta-llama/llama-3.3-70b:free")
        self.assertIn("meta-llama/llama-3.3-70b:free", headline)
        self.assertIn("rate-limited", headline.lower())
        self.assertIn("sidebar", body.lower())

    def test_friendly_error_generic_for_unknown(self):
        from web.chat_helpers import _friendly_error_message

        exc = RuntimeError("OpenRouter returned HTTP 500: boom")
        headline, body = _friendly_error_message(exc, "some-model")
        self.assertIn("some-model", headline)
        self.assertIn(".env", body)


class FreeModelChoicesTests(unittest.TestCase):
    """Verify the curated free-model list shipped in the web view.

    The list is what the user sees in the sidebar dropdown. If anyone
    deletes all entries, adds a paid model id by mistake, or breaks the
    schema the view relies on, the UI breaks in a confusing way. These
    tests catch that at CI time.
    """

    @classmethod
    def setUpClass(cls):
        # Importing the view module spins up Streamlit's
        # DeltaGeneratorSingleton. Doing it once per class (instead of
        # once per test) keeps subsequent test methods from tripping
        # the "instance already exists" guard on a hot re-import.
        super().setUpClass()
        from web import streamlit_app as _view
        cls._view = _view

    def test_curated_list_imports_and_is_non_empty(self):
        FREE_MODEL_CHOICES = self._view.FREE_MODEL_CHOICES

        self.assertIsInstance(FREE_MODEL_CHOICES, list)
        self.assertGreater(len(FREE_MODEL_CHOICES), 0)

    def test_every_entry_has_required_keys(self):
        FREE_MODEL_CHOICES = self._view.FREE_MODEL_CHOICES

        required = {"id", "label", "role", "blurb"}
        for entry in FREE_MODEL_CHOICES:
            self.assertTrue(
                required.issubset(entry.keys()),
                f"entry {entry!r} is missing keys: {required - set(entry.keys())}",
            )
            for k in required:
                self.assertIsInstance(entry[k], str)
                self.assertGreater(
                    entry[k].strip(), "",
                    f"entry {entry!r} has empty {k!r}",
                )

    def test_default_index_constant_is_in_range(self):
        DEFAULT_SELECTED_MODEL_INDEX = self._view.DEFAULT_SELECTED_MODEL_INDEX
        FREE_MODEL_CHOICES = self._view.FREE_MODEL_CHOICES

        self.assertGreaterEqual(DEFAULT_SELECTED_MODEL_INDEX, 0)
        self.assertLess(DEFAULT_SELECTED_MODEL_INDEX, len(FREE_MODEL_CHOICES))

    def test_ids_are_unique(self):
        FREE_MODEL_CHOICES = self._view.FREE_MODEL_CHOICES

        ids = [m["id"] for m in FREE_MODEL_CHOICES]
        self.assertEqual(
            len(ids), len(set(ids)),
            f"duplicate model ids in curated list: {ids}",
        )


class TwoPassPatternTests(unittest.TestCase):
    """Pin the two-pass ``_ask`` pattern used to render the Thinking… bubble.

    Streamlit is single-pass per script run. To show the user bubble
    *before* the model finishes thinking, ``_ask`` writes the user turn
    to session_state and reruns (pass 1), then on the rerun runs the
    model and writes the assistant turn (pass 2). Pass 2 is the only
    place the placeholder, the toast, and the status-line marker are
    rendered.

    The first cut of this pattern worked for the example-prompt button
    (which set ``pending_prompt`` and the rerun found it), but it broke
    silently for the main ``st.chat_input`` path: chat_input returns ""
    on the rerun, ``pending_prompt`` is None, so pass 2 was never
    driven — the user just saw nothing.

    The fix is a top-level driver that calls ``_ask(None)`` when
    ``pending_request`` is set, *before* the input widgets run. These
    tests pin that contract so a future refactor doesn't silently
    regress the chat_input path.
    """

    def test_ask_signature_accepts_none(self):
        """``_ask(None)`` must be a valid call: it signals pass 2."""
        import ast

        tree = ast.parse(_read_streamlit_view())
        func_defs = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_ask"
        ]
        self.assertEqual(len(func_defs), 1, "_ask must be defined exactly once")
        arg = func_defs[0].args.args[0]
        self.assertIsInstance(arg.annotation, ast.BinOp)
        # BinOp with ast.BitOr (|) is the ast shape of `str | None`.
        self.assertIsInstance(arg.annotation.op, ast.BitOr)

    def test_pass2_driver_block_exists(self):
        """The top-level driver that fires pass 2 must be present."""
        src = _read_streamlit_view()
        # The block lives AFTER the function definition. We anchor on
        # the comment we added so a refactor that deletes the block
        # lights up the test rather than silently breaking the UI.
        self.assertIn(
            "Two-pass continuation",
            src,
            "missing the 'Two-pass continuation' driver block — pass 2 "
            "won't fire after a chat_input submission",
        )
        # And it must actually call _ask(None) under a guard on
        # pending_request.
        self.assertRegex(
            src,
            r'if\s+st\.session_state\.get\(\s*["\']pending_request["\']\s*\)\s*:\s*\n\s*_ask\(None\)',
            "pass-2 driver block is malformed; expected "
            "if st.session_state.get('pending_request'):\\n    _ask(None)",
        )

    def test_pass2_driver_appears_after_ask_definition(self):
        """The driver must come AFTER ``def _ask``, not before.

        We saw this mistake in an earlier turn: putting the driver
        block before the function definition causes NameError at run
        time because ``_ask`` is not yet bound. This test reads the
        source and asserts the byte offset of the driver is greater
        than the byte offset of the ``def _ask`` line.
        """
        src = _read_streamlit_view()
        def_offset = src.find("def _ask(")
        driver_offset = src.find("Two-pass continuation")
        self.assertGreater(def_offset, 0, "def _ask(...) not found in source")
        self.assertGreater(driver_offset, 0, "driver block not found in source")
        self.assertGreater(
            driver_offset, def_offset,
            "pass-2 driver block must come AFTER the _ask definition "
            "(otherwise the call to _ask(None) raises NameError)",
        )

    def test_pass2_success_path_calls_rerun(self):
        """After the reply is appended, pass 2 must rerun the script.

        Without the rerun, the assistant turn is appended to
        session_state but the history-render loop at the top of the
        script never re-paints — the user sees no reply until they
        trigger some other rerun (e.g. by sending a second message).
        Pin this so the bug cannot regress silently.
        """
        src = _read_streamlit_view()
        # Locate the success path. The reply is appended in this block.
        # The rerun must be inside the same indentation block as the
        # append so it actually fires on the happy path.
        # We anchor on a unique phrase from the success path.
        anchor = '"role": "assistant", "content": reply'
        idx = src.find(anchor)
        self.assertGreater(idx, 0, "success path append not found in source")
        # Look at the next ~200 chars after the append — the rerun
        # must be there. Window is generous so it can sit after the
        # persist helper call (which appends the assistant message and
        # touches the chat for activity sorting). Contract: rerun must
        # be on the same code path as the append, in the same
        # indentation block — not 2000 lines later in another function.
        window = src[idx : idx + 1500]
        self.assertIn(
            "st.rerun()", window,
            "pass-2 success path must call st.rerun() after appending "
            "the assistant turn, otherwise the user never sees the reply",
        )

    def test_cache_hit_path_calls_rerun(self):
        """The cache-hit branch must also rerun so the cached reply
        actually renders in the history loop."""
        src = _read_streamlit_view()
        anchor = "response_cache"
        idx = src.find(anchor)
        # Find every occurrence and inspect the one inside the cache-hit
        # block (the one that appends a `cached` value).
        cache_idx = src.find("append(\n            {\"role\": \"assistant\", \"content\": cached}", idx)
        self.assertGreater(cache_idx, 0, "cache-hit append not found in source")
        # Window is generous so the rerun can sit after the persist
        # helper call (which appends a message, touches the chat, and
        # updates the title if applicable). Contract: rerun must be on
        # the same code path as the append, in the same indentation
        # block — not 2000 lines later in another function.
        window = src[cache_idx : cache_idx + 1500]
        self.assertIn(
            "st.rerun()", window,
            "cache-hit branch must call st.rerun() after appending "
            "the cached reply, otherwise repeat questions are invisible",
        )

    def test_error_path_uses_friendly_helper(self):
        """The error block must call the friendly-error helper instead
        of dumping the raw exception into ``st.error``."""
        src = _read_streamlit_view()
        self.assertIn(
            "_render_friendly_error(",
            src,
            "error handler must call _render_friendly_error; the old "
            "raw st.error(f'... {exc}') path leaks upstream JSON to "
            "the user",
        )


class StreamlitSysPathBootstrapTests(unittest.TestCase):
    """Pin the ``sys.path`` bootstrap that lets ``streamlit run
    web/streamlit_app.py`` resolve ``from app.config import ...``.

    Streamlit's ``modified_sys_path`` only prepends the SCRIPT's
    directory (``web/``) to ``sys.path[0]``. The project root is NOT
    added. So unless the bootstrap is present, the app crashes on
    import with ``ModuleNotFoundError: No module named 'app'`` as soon
    as the worker process is launched from a cwd that is not the
    project root. These tests pin the contract so a future refactor
    can't silently drop the bootstrap.
    """

    def test_streamlit_app_uses_absolute_imports(self):
        """``web/streamlit_app.py`` must not use relative imports.
        Relative imports only work for modules inside the same package;
        ``streamlit_app`` is a top-level script, so a relative import
        like ``from . import app`` would raise ``ImportError`` even
        before the ``sys.path`` question matters.
        """
        src = _read_streamlit_view()
        # The only legal pattern is `from __future__ import ...`
        for line in src.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("from "):
                continue
            self.assertFalse(
                stripped.startswith("from ."),
                f"streamlit_app.py uses a relative import: {line!r}. "
                "Top-level Streamlit scripts must use absolute imports.",
            )

    def test_bootstrap_appears_before_streamlit_import(self):
        """The ``sys.path`` bootstrap must run BEFORE
        ``import streamlit as st`` and BEFORE any ``from app...``
        import, otherwise the ImportError has already been raised.
        """
        src = _read_streamlit_view()
        bootstrap_match = re.search(
            r"^# --- sys\.path bootstrap ---.*?^import streamlit",
            src,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(
            bootstrap_match,
            "expected a `sys.path bootstrap` comment block that ends "
            "right before `import streamlit`; if you renamed the "
            "comment, update this test too.",
        )
        # And the bootstrap must actually mutate sys.path
        self.assertIn(
            "sys.path.insert(0,",
            bootstrap_match.group(0),
            "bootstrap block must call sys.path.insert(0, ...) so the "
            "project root is found before any app.* import.",
        )

    def test_bootstrap_makes_app_importable_from_web_only_path(self):
        """Simulate Streamlit's ``modified_sys.path`` semantics: only
        the SCRIPT's directory is on ``sys.path[0]``. After running
        the bootstrap block, ``import app.config`` must succeed.
        """
        project_root = os.path.dirname(os.path.dirname(_STREAMLIT_VIEW))
        web_dir = os.path.dirname(_STREAMLIT_VIEW)

        # Start with a path that mirrors what a long-running worker
        # process might have: web/ on sys.path[0], project root absent.
        clean_path = [web_dir] + [
            p for p in sys.path
            if p not in ("", project_root, web_dir)
        ]

        # Execute just the bootstrap block (lines 1..35 approx) of
        # streamlit_app.py in an isolated namespace, then verify
        # app.config is importable.
        with open(_STREAMLIT_VIEW, "r", encoding="utf-8") as fh:
            src = fh.read()

        # Cut off at `import streamlit as st` so we don't actually
        # spin up a Streamlit session in the test process.
        cut = src.index("import streamlit as st")
        bootstrap_src = src[:cut]

        namespace: dict = {"__name__": "__not_streamlit__", "__file__": _STREAMLIT_VIEW}
        exec(bootstrap_src, namespace)

        # The bootstrap must have inserted the project root
        self.assertIn(
            project_root, sys.path,
            f"bootstrap failed to add project root {project_root!r} "
            f"to sys.path; sys.path is now: {sys.path[:5]!r}",
        )

        # And with that on sys.path, app.config must import cleanly
        # (importlib fresh-imports it so the test does not depend on
        # whether app.config is already loaded in this test process).
        importlib.invalidate_caches()
        try:
            importlib.import_module("app.config")
        except ModuleNotFoundError as exc:
            self.fail(
                f"even with the bootstrap, app.config failed to import: "
                f"{exc}. sys.path[0:3] = {sys.path[:3]!r}"
            )


class OpenRouterErrorHierarchyTests(unittest.TestCase):
    """Pin the typed error hierarchy used by the router.

    The router branches on ``isinstance`` to decide whether to retry,
    rotate, or disable a slot. If anyone renames a subclass or drops
    the ``status`` attribute, the routing logic falls back to string
    parsing — which is exactly what we want to prevent.
    """

    def test_base_error_carries_status_provider_model_body(self):
        from app.openrouter import OpenRouterError

        exc = OpenRouterError(
            "boom",
            status=429,
            provider="openrouter",
            model="google/gemma-4-31b-it:free",
            body='{"error":"rate limit"}',
        )
        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.provider, "openrouter")
        self.assertEqual(exc.model, "google/gemma-4-31b-it:free")
        self.assertEqual(exc.body, '{"error":"rate limit"}')
        # str(exc) must still work (used everywhere in the UI/CLI logs).
        self.assertIn("boom", str(exc))
        # Base class is still a RuntimeError (backwards compat).
        self.assertIsInstance(exc, RuntimeError)

    def test_default_attributes_are_none(self):
        from app.openrouter import OpenRouterError

        exc = OpenRouterError("simple message")
        self.assertIsNone(exc.status)
        self.assertIsNone(exc.provider)
        self.assertIsNone(exc.model)
        self.assertIsNone(exc.body)

    def test_four_typed_subclasses_exist_and_inherit_base(self):
        from app.openrouter import (
            OpenRouterAuthError,
            OpenRouterClientError,
            OpenRouterError,
            OpenRouterRateLimitError,
            OpenRouterServerError,
        )

        for cls in (
            OpenRouterAuthError,
            OpenRouterRateLimitError,
            OpenRouterServerError,
            OpenRouterClientError,
        ):
            with self.subTest(cls=cls):
                self.assertTrue(issubclass(cls, OpenRouterError))
                # Each subclass must accept the same kwargs as the base.
                instance = cls(
                    f"{cls.__name__} msg",
                    status=500,
                    provider="x",
                    model="y",
                    body="z",
                )
                self.assertEqual(instance.status, 500)
                self.assertIn(cls.__name__, str(instance))

    def test_view_uses_router_specific_exceptions_in_imports(self):
        """The view must import the router-specific exception types it
        catches in its `try/except` blocks. If a refactor drops one of
        these from the import block, the wiring silently breaks.
        """
        view_src = _read_streamlit_view()
        for symbol in (
            "AllSlotsExhaustedError",
            "ModelRouter",
            "NoFreeModelConfiguredError",
            "RouterError",
            "build_from_config",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(
                    "from app.router import",
                    view_src,
                    "view must import app.router",
                )
                self.assertIn(
                    symbol, view_src,
                    f"view must reference {symbol!r} (used in the router wiring)",
                )


class ModelRouterTests(unittest.TestCase):
    """Pin the routing behaviour of ``app.router.ModelRouter``.

    The router is the safety net that hides the per-account daily
    free-tier cap behind automatic rotation. These tests use
    ``unittest.mock`` to fake the network, so no real HTTP calls are
    made and no API credits are spent.
    """

    def _build(self, keys, models, **kwargs):
        """Build a router from short hand-crafted inputs.

        Keeps the call sites below readable.
        """
        from app.router import build_from_config

        return build_from_config(keys, models, **kwargs)

    def test_construction_rejects_non_free_model_id(self):
        """The router must reject any model id that doesn't end in
        ``:free`` — silent fallback to a paid model is a hard
        industry-standard anti-pattern we explicitly do not want.
        """
        from app.router import NoFreeModelConfiguredError

        with self.assertRaises(NoFreeModelConfiguredError) as ctx:
            self._build(["k1"], ["openai/gpt-4o"])  # not :free
        self.assertIn(":free", str(ctx.exception))

    def test_construction_rejects_empty_pool(self):
        from app.router import NoFreeModelConfiguredError

        with self.assertRaises(NoFreeModelConfiguredError):
            self._build([], [])

    def test_slot_label_redacts_key(self):
        """Slot labels must never include the raw API key.

        The labels are surfaced in the CLI banner and the Streamlit
        sidebar caption, both of which can appear in screenshots and
        log captures. A leak here is a credentials disclosure.
        """
        router = self._build(
            ["sk-or-v1-AAAAAAAABBBBBBBB"],
            ["m:free"],
        )
        label = router.slot_labels()[0]
        self.assertIn("m:free", label)
        self.assertIn("****", label)
        # The raw key must not appear anywhere in the label.
        self.assertNotIn("sk-or-v1-AAAAAAAABBBBBBBB", label)

    def test_healthy_slot_count_decreases_on_401(self):
        """A 401 must permanently disable the slot. The disabled slot
        should not be re-enabled on the next call — that is the whole
        point of the health state.
        """
        from app.openrouter import OpenRouterAuthError
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1", "k2"], ["m:free"], sleep=lambda _s: None)
        self.assertEqual(router.healthy_slot_count(), 2)

        # Both slots return 401 -> all disabled -> exhaustion.
        auth_exc = OpenRouterAuthError("unauthorized", status=401)
        with patch(
            "app.openrouter.chat",
            side_effect=auth_exc,
        ):
            with self.assertRaises(AllSlotsExhaustedError):
                router.chat([{"role": "user", "content": "hi"}])

        # All slots are now disabled; subsequent calls also exhaust
        # immediately without even reaching the network.
        self.assertEqual(router.healthy_slot_count(), 0)
        with self.assertRaises(AllSlotsExhaustedError):
            router.chat([{"role": "user", "content": "hi"}])

    def test_429_retries_then_rotates(self):
        """A 429 must be retried once on the same slot, and only then
        rotate to the next slot. If we rotated on the first 429 we
        would burn through the whole pool in a single user message.
        """
        from app.openrouter import OpenRouterRateLimitError
        from app.router import AllSlotsExhaustedError

        rate_limited = OpenRouterRateLimitError("rate limit", status=429)
        router = self._build(["k1", "k2"], ["m:free"], sleep=lambda _s: None)

        call_count = {"n": 0}

        def side_effect(*_a, **_k):
            call_count["n"] += 1
            raise rate_limited

        with patch("app.openrouter.chat", side_effect=side_effect):
            with self.assertRaises(AllSlotsExhaustedError):
                router.chat([{"role": "user", "content": "hi"}])

        # k1: first call + 1 retry = 2 calls, k2: first call + 1 retry = 2 calls
        # = 4 total. Anything else means we are over- or under-rotating.
        self.assertEqual(
            call_count["n"], 4,
            f"expected 4 total calls (2 slots × 2 attempts each), "
            f"got {call_count['n']}",
        )

    def test_success_advances_cursor(self):
        """After a successful call, the cursor must advance so the
        next call lands on a different slot position. Otherwise we
        keep hammering the same slot and never spread load.
        """
        router = self._build(
            ["k1", "k2"],
            ["m1:free", "m2:free"],
        )
        # 4 slots: k1/m1, k1/m2, k2/m1, k2/m2
        self.assertEqual(router.healthy_slot_count(), 4)

        # Stub the network so the real router.chat() runs to
        # completion. The stub records the (key, model) pair, which
        # is unique per slot, so we can identify the slot position
        # unambiguously. (Using just the model id is ambiguous:
        # k1/m1 and k2/m1 both look like model m1.)
        seen_pairs: list[tuple[str, str]] = []

        def fake_openrouter_chat(
            _messages, *, model, api_key=None, **_kwargs
):
            seen_pairs.append((api_key, model))
            return "ok"

        with patch("app.openrouter.chat", side_effect=fake_openrouter_chat):
            for _ in range(5):
                router.chat([{"role": "user", "content": "hi"}])

        # Translate each (key, model) pair to its slot index in
        # the router's pool, so the assertion talks about positions
        # (which the cursor advance code cares about) rather than
        # the raw pair.
        seen_positions: list[int] = []
        for key, model in seen_pairs:
            for i, slot in enumerate(router._slots):
                if slot.api_key == key and slot.model_id == model:
                    seen_positions.append(i)
                    break

        # 5 calls, 4 slot positions, cursor advances by 1 each
        # time -> positions 0, 1, 2, 3, 0. The 5th call must wrap
        # back to slot 0.
        self.assertEqual(len(seen_positions), 5, f"saw: {seen_positions}")
        self.assertEqual(
            seen_positions,
            [0, 1, 2, 3, 0],
            f"cursor did not advance + wrap as expected: {seen_positions}",
        )

    def test_all_slots_exhausted_error_lists_tried_slots(self):
        """The exhaustion error must expose which slots were tried
        (with redacted keys) so the CLI / web UI can show the user
        which slots were tried.
        """
        from app.openrouter import OpenRouterAuthError
        from app.router import AllSlotsExhaustedError

        router = self._build(
            ["sk-or-v1-XXXXXXXXXXXXXXXX"],
            ["m1:free", "m2:free"],
        )
        auth_exc = OpenRouterAuthError("nope", status=401)
        with patch("app.openrouter.chat", side_effect=auth_exc):
            with self.assertRaises(AllSlotsExhaustedError) as ctx:
                router.chat([{"role": "user", "content": "hi"}])

        exc = ctx.exception
        # Structural access: tried_slots must be a list of redacted
        # labels, one per slot that was tried.
        self.assertEqual(len(exc.tried_slots), 2, f"got: {exc.tried_slots}")
        joined = " ".join(exc.tried_slots)
        self.assertIn("m1:free", joined)
        self.assertIn("m2:free", joined)
        self.assertIn("****", joined)
        # The raw key must not appear in either the labels or the
        # human-readable message.
        self.assertNotIn("sk-or-v1-XXXXXXXXXXXXXXXX", joined)
        self.assertNotIn("sk-or-v1-XXXXXXXXXXXXXXXX", str(exc))


class BuildUserTurnTextTests(unittest.TestCase):
    """Pin ``web.chat_helpers.build_user_turn_text`` so the file-upload
    UX cannot regress.

    The helper collapses uploaded files (which the OpenRouter chat-
    completions endpoint cannot read) into a plain-text block. Tests
    use a duck-typed fake that matches the
    ``_UploadedFileLike`` Protocol (``name``, ``type``, ``size``,
    ``read()``) so the Streamlit runtime is not required.
    """

    def setUp(self):
        # Mirror the project-root-cd pattern used elsewhere in this file
        # so a `python -m unittest tests.test_smoke` invocation from
        # anywhere still resolves `from web.chat_helpers import ...`.
        self._orig_cwd = os.getcwd()
        self._orig_path = list(sys.path)
        project_root = os.path.dirname(os.path.dirname(_STREAMLIT_VIEW))
        os.chdir(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # Force a fresh import in case the helper changed during the
        # test session (a previous test could have cached an older
        # module).
        for mod in list(sys.modules):
            if mod == "web.chat_helpers" or mod.startswith("web.chat_helpers."):
                del sys.modules[mod]

    def tearDown(self):
        os.chdir(self._orig_cwd)
        sys.path[:] = self._orig_path

    @staticmethod
    def _fake_upload(name, mime, data):
        class _FakeUpload:
            def __init__(self, name, mime, data):
                self.name = name
                self.type = mime
                self.size = len(data)
                self._data = data

            def read(self, n=-1):
                if n is None or n < 0:
                    return self._data
                return self._data[:n]

            def seek(self, *_args, **_kwargs):
                return 0

        return _FakeUpload(name, mime, data)

    def test_text_only_passes_through_unchanged(self):
        """When there are no files, the user text is returned as-is
        (no trailing newlines, no header)."""
        from web.chat_helpers import build_user_turn_text

        out = build_user_turn_text("hello world", [])
        self.assertEqual(out, "hello world")

    def test_none_text_with_no_files_returns_empty_string(self):
        """The view passes ``getattr(user_chat, "text", None)`` which
        can legitimately be ``None`` (user attached files only). The
        helper must coerce that to an empty string instead of raising
        a ``TypeError`` on the ``+`` / join."""
        from web.chat_helpers import build_user_turn_text

        self.assertEqual(build_user_turn_text(None, []), "")
        self.assertEqual(build_user_turn_text("", []), "")

    def test_single_text_file_is_inlined_below_user_message(self):
        """A textual attachment should appear after a blank line,
        with a header that names the file and inlines the body."""
        from web.chat_helpers import build_user_turn_text

        fake = self._fake_upload(
            "evidence.log", "text/plain", b"2024-01-01 ALERT bad.exe\n"
        )
        out = build_user_turn_text("explain this log", [fake])
        self.assertTrue(
            out.startswith("explain this log\n\n"),
            f"expected text first then blank line; got: {out!r}",
        )
        self.assertIn("[Attached file: evidence.log", out)
        self.assertIn("2024-01-01 ALERT bad.exe", out)

    def test_binary_file_is_summarized_not_inlined(self):
        """A binary attachment (PNG) must NOT have its raw bytes
        inlined — the LLM would see garbage and the prompt would
        balloon. Instead the header must contain a compact
        ``[Attached image: …]`` stub. Images are *not* text paths
        here; they go through the multimodal ``image_url`` channel
        via :func:`build_user_turn_content`, so the display bubble
        is allowed to show nothing more than the stub."""
        from web.chat_helpers import build_user_turn_text

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
        fake = self._fake_upload("screenshot.png", "image/png", png_bytes)
        out = build_user_turn_text(None, [fake])

        self.assertIn("[Attached image: screenshot.png", out)
        # Raw PNG magic bytes must NOT appear in the prompt
        self.assertNotIn(b"\x89PNG\r\n\x1a\n".decode("latin-1"), out)

    def test_multiple_files_are_listed_in_order(self):
        """When the user attaches 3 files, the prompt must list
        them in the order they were attached and prefix with a
        header that mentions the count. Images are split out
        before the textual count, so the header reads ``2 files``
        (a.py + b.log) with the PNG appearing as its own
        ``[Attached image: …]`` stub."""
        from web.chat_helpers import build_user_turn_text

        files = [
            self._fake_upload("a.py", "text/x-python", b"print(1)"),
            self._fake_upload("b.log", "text/plain", b"line1\n"),
            self._fake_upload("c.png", "image/png", b"\x89PNG\r\n\x1a\n"),
        ]
        out = build_user_turn_text("triaging", files)

        self.assertIn("2 files", out)
        # Order must be preserved (a appears before b before c).
        idx_a = out.index("a.py")
        idx_b = out.index("b.log")
        idx_c = out.index("c.png")
        self.assertLess(idx_a, idx_b)
        self.assertLess(idx_b, idx_c)

    def test_files_only_no_text_still_produces_a_prompt(self):
        """If the user attaches files without typing anything, the
        helper must still produce a non-empty prompt (the view's
        `if prompt_text.strip():` guard would otherwise silently
        drop the turn)."""
        from web.chat_helpers import build_user_turn_text

        fake = self._fake_upload(
            "trace.json", "application/json", b'{"event":"x"}'
        )
        out = build_user_turn_text(None, [fake])
        self.assertTrue(out.strip(), "files-only prompt was empty")
        self.assertIn("trace.json", out)

    def test_oversized_text_file_is_truncated_with_marker(self):
        """A text file larger than the inline cap must be cut off
        with a 'truncated' marker so the LLM does not see megabytes
        of unrelated content."""
        from web.chat_helpers import build_user_turn_text, _MAX_INLINE_BYTES

        big = b"A" * (_MAX_INLINE_BYTES * 3)  # 3x the cap
        fake = self._fake_upload("huge.log", "text/plain", big)
        out = build_user_turn_text("review", [fake])

        # The header must include the original size so the model
        # knows how much was clipped.
        self.assertIn(str(len(big)), out)
        # And the body must NOT contain the full payload.
        self.assertLess(len(out), len(big) + 200)
        self.assertIn("truncat", out.lower())


class StreamlitChatInputFileUploadTests(unittest.TestCase):
    """Pin that ``st.chat_input`` is wired for file uploads.

    These are source-level checks (not behavioral) because the
    Streamlit widget cannot be invoked from a test process. The
    goal is to fail loudly if someone reverts the chat input to a
    plain text field.
    """

    def test_chat_input_uses_accept_file_multiple(self):
        """The chat input must enable multi-file attachments. If this
        regresses the user loses the 📎 button."""
        src = _read_streamlit_view()
        # Look for the chat_input call and verify the keyword is
        # present. A naive substring check is fine because the
        # widget is only called once in the view.
        self.assertIn(
            "accept_file=\"multiple\"",
            src,
            "st.chat_input must be called with accept_file=\"multiple\"; "
            "otherwise the user cannot attach files.",
        )

    def test_chat_input_declares_an_accepted_file_type_whitelist(self):
        """``file_type=`` must be set so Streamlit's browser-side
        file picker filters to extensions the helper can actually
        classify. An empty ``file_type=`` would let the user pick
        anything and the helper's binary-summarization path would
        swallow real content silently."""
        src = _read_streamlit_view()
        # Match `file_type=<something>` on the same logical line.
        match = re.search(r"file_type\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", src)
        self.assertIsNotNone(
            match,
            "st.chat_input must be called with file_type=<list-or-name>; "
            "an unrestricted picker will let users attach anything.",
        )
        # The variable must be non-empty at module level.
        var_name = match.group(1)
        with open(_STREAMLIT_VIEW, "r", encoding="utf-8") as fh:
            full_src = fh.read()
        decl_re = re.compile(
            rf"^{re.escape(var_name)}\s*(?::\s*[^=]+)?=\s*\[([^\]]*)\]",
            re.MULTILINE,
        )
        decl = decl_re.search(full_src)
        self.assertIsNotNone(
            decl,
            f"could not find a list assignment for {var_name!r}",
        )
        items = [s.strip().strip('"\'') for s in decl.group(1).split(",") if s.strip()]
        self.assertGreater(
            len(items), 4,
            f"{var_name} whitelist is too small ({len(items)} entries); "
            "users won't be able to attach the common file types.",
        )

    def test_view_routes_uploaded_files_through_helper(self):
        """The view must call ``build_user_turn_text`` with both
        ``text`` and ``files`` pulled off the chat return value,
        otherwise the uploaded files are silently dropped before
        reaching the LLM."""
        src = _read_streamlit_view()
        self.assertIn("build_user_turn_text(", src)
        # Both .text and .files must be extracted (so a refactor
        # that drops either one fails this test).
        self.assertIn(".text", src)
        self.assertIn(".files", src)
        # And the helper's output must gate the _ask() call so an
        # empty prompt is not sent.
        self.assertIn(
            "prompt_text and prompt_text.strip()",
            src,
            "view must guard against empty prompts (files-only with "
            "no recognised content would otherwise call _ask('')).",
        )


class StreamlitChatInputMediaGuardTests(unittest.TestCase):
    """Pin the per-session ``MediaFileStorageError`` guard.

    Streamlit's ``st.chat_input(accept_file="multiple")`` registers
    uploaded file bytes in a per-session in-memory store keyed by
    content hash, and ships the file's media ID to the browser. The
    browser echoes that ID back on every subsequent rerun so the
    input can re-display the file the user previously attached. If
    the server is restarted (Ctrl-C + relaunch, code change, hot
    reload) and the user keeps the tab open, the new server's media
    store is empty and Streamlit raises ``MediaFileStorageError``
    from inside the widget's render call. The user sees a blank page
    and a server-side traceback.

    The fix is a per-call ``try/except`` around the chat input
    render that (a) catches the error, (b) marks the session as
    "media-broken" so the next render swaps in a text-only fallback,
    (c) clears any in-flight pending request so a partial reply is
    not finalised against lost data, and (d) surfaces a one-time
    friendly banner explaining the situation. These tests pin those
    four behaviours structurally — they fail loudly if anyone
    regresses the guard or removes the fallback.
    """

    def test_view_imports_the_media_file_storage_exception(self):
        # The guard requires the exception class. If a future
        # Streamlit rename moves the class, this test points the
        # maintainer at the right path to update.
        src = _read_streamlit_view()
        self.assertIn(
            "from streamlit.runtime.memory_media_file_storage import",
            src,
            "view must import MediaFileStorageError from "
            "streamlit.runtime.memory_media_file_storage so the "
            "guard can catch a stale file ID from a previous "
            "server session.",
        )
        self.assertIn(
            "MediaFileStorageError",
            src,
            "view must bind the MediaFileStorageError class to a "
            "local alias (with a fallback class definition for "
            "older Streamlit versions) before the guard runs.",
        )

    def test_view_wraps_chat_input_in_media_storage_try_block(self):
        # The actual guard: the chat input call must sit inside a
        # try block whose except arm catches MediaFileStorageError.
        src = _read_streamlit_view()
        # We require the literal sequence: ``except _MediaFileStorageError``
        # appears at least once. The local alias is ``_MediaFileStorageError``
        # so the test does not have to know the exact import binding.
        self.assertRegex(
            src,
            r"except\s+_MediaFileStorageError\b",
            "st.chat_input(accept_file='multiple') must be wrapped "
            "in a try/except _MediaFileStorageError; without it a "
            "server restart with an open browser tab raises "
            "MediaFileStorageError from inside the widget render "
            "and blanks the page with a server-side traceback.",
        )

    def test_guard_marks_session_with_a_sticky_error_flag(self):
        # The first render that catches the error must flip a
        # per-session flag so the *next* render downgrades to a
        # text-only chat input (otherwise the same exception fires
        # on every rerun and the user is stuck).
        src = _read_streamlit_view()
        self.assertIn(
            "_SESSION_STATE_MEDIA_ERROR",
            src,
            "view must define a session-state key for the media "
            "error flag (e.g. _SESSION_STATE_MEDIA_ERROR) and write "
            "to it from the guard's except arm.",
        )
        # The flag must be set to True (not None / not deleted)
        # so a follow-up ``st.session_state.get(...)`` returns True.
        self.assertRegex(
            src,
            r"st\.session_state\[_SESSION_STATE_MEDIA_ERROR\]\s*=\s*True",
            "guard must set the media error flag to True; a None "
            "value or a deletion would not downgrade the chat "
            "input on the next render and the same exception "
            "would fire in a loop.",
        )

    def test_guard_drops_in_flight_pending_request(self):
        # Critical: a partial reply that was finalised against a
        # lost file would leave the assistant half-answering a
        # question the user can no longer reproduce. The guard must
        # drop the in-flight request before rerunning.
        src = _read_streamlit_view()
        self.assertIn(
            'st.session_state.pop("pending_request", None)',
            src,
            "guard must drop the in-flight 'pending_request' so a "
            "partial reply is not finalised against lost file data "
            "after the server restarts.",
        )
        self.assertIn(
            'st.session_state.pop("pending_started_at", None)',
            src,
            "guard must drop the matching 'pending_started_at' "
            "timestamp so a stale turn cannot be re-driven by the "
            "two-pass chat_input pattern after a media-store reset.",
        )

    def test_fallback_chat_input_omits_accept_file(self):
        # The text-only fallback must NOT use accept_file — if it
        # did, the same media-store lookup would fire on every
        # render and the guard would loop. A plain
        # ``st.chat_input("…")`` (no accept_file kwarg) is what
        # downgrades the input to text-only.
        src = _read_streamlit_view()
        chat_input_calls = list(
            re.finditer(r"st\.chat_input\s*\(", src)
        )
        self.assertGreaterEqual(
            len(chat_input_calls), 2,
            "view must have at least two st.chat_input calls — "
            "the guarded primary (with accept_file) and the "
            "text-only fallback (without).",
        )
        # Find each call's body via a bracket-balance scan, then
        # collect a {position: body} mapping so the test can pick
        # the text-only fallback (no accept_file) and the primary
        # (with accept_file) by their keyword shape rather than by
        # source order — the order depends on which branch the
        # maintainer puts the fallback in.
        def _body(call_match: "re.Match[str]") -> str:
            depth = 0
            end_idx = None
            for i in range(call_match.end() - 1, len(src)):
                c = src[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            self.assertIsNotNone(
                end_idx,
                "could not find the closing paren of one of the "
                "st.chat_input calls",
            )
            return src[call_match.end():end_idx]

        bodies = [_body(m) for m in chat_input_calls]
        has_primary = any("accept_file" in b for b in bodies)
        self.assertTrue(
            has_primary,
            "the primary st.chat_input call must still pass "
            "accept_file='multiple' so file attachments work in "
            "the normal (in-process) path",
        )
        text_only = [b for b in bodies if "accept_file" not in b]
        self.assertTrue(
            text_only,
            "view must have at least one st.chat_input call "
            "without accept_file — the text-only fallback used "
            "when MediaFileStorageError has been raised this "
            "session.",
        )

    def test_guard_logs_to_python_logger_not_st_error(self):
        # The raw ``MediaFileStorageError`` message embeds the lost
        # file's media ID — a useful breadcrumb in the server log,
        # but noise the user does not need to see. The guard must
        # log to the Python ``logging`` module (which Streamlit
        # routes to the server console) and not call ``st.error``.
        src = _read_streamlit_view()
        # Find the guard's except block.
        match = re.search(
            r"except\s+_MediaFileStorageError\b.*?(?=\n\S|\Z)",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "could not locate the guard's except arm in the view",
        )
        except_body = match.group(0)
        self.assertIn(
            "logging.getLogger(",
            except_body,
            "guard must log the MediaFileStorageError to the "
            "Python logging module (server console) so the lost "
            "file's media ID is preserved for debugging without "
            "leaking it into the chat UI.",
        )
        self.assertNotIn(
            "st.error(",
            except_body,
            "guard must not call st.error on the raw exception — "
            "the storage error message embeds the lost file's "
            "media ID and adds zero value to the user, who only "
            "needs to know 'the attachment is gone, re-attach'.",
        )

    def test_fallback_renders_a_recovery_banner_and_reset_button(self):
        # When the session is in the media-error state, the next
        # render must show (a) a st.warning explaining the
        # situation and (b) a button the user can click to
        # re-enable the file picker (in case they restarted the
        # server intentionally and want to attach again).
        src = _read_streamlit_view()
        # Both the flag check and the warning/button must be
        # present at module-level so they run on every rerun.
        self.assertIn(
            'st.session_state.get(_SESSION_STATE_MEDIA_ERROR)',
            src,
            "view must check the media-error flag at the top of "
            "every script run and render a recovery banner + "
            "reset button when it is set.",
        )
        self.assertIn(
            "st.warning(",
            src,
            "recovery branch must call st.warning so the user "
            "sees a clear, dismissable explanation that the "
            "attachment is no longer available.",
        )
        # The reset button must clear the flag and rerun.
        self.assertRegex(
            src,
            r"st\.session_state\.pop\(\s*_SESSION_STATE_MEDIA_ERROR\s*,\s*None\s*\)",
            "reset button must pop the media-error flag from "
            "session_state so the next render re-enables the "
            "file picker.",
        )


# --- Copy-to-clipboard helper (Tier 1 #4) -----------------------------------
# Verifies that ``web.chat_helpers._copy_button_html`` produces a tiny,
# self-contained HTML <button> whose payload cannot break out of the
# ``data-text`` HTML attribute — the one thing that can go wrong with
# a clipboard helper is XSS in the embedded reply text, so we test
# the escaping behaviour explicitly with payloads an LLM might
# realistically emit (apostrophes, double-quotes, angle brackets,
# ampersands, backticks, control chars, Unicode, very long strings).
#
# The companion init script (``_copy_button_init_script``) is tested
# separately by ``CopyButtonInitScriptTests`` further down.


class CopyButtonHtmlTests(unittest.TestCase):
    """Structural + safety tests for the copy-to-clipboard helper.

    The helper emits a tiny HTML ``<button>`` whose payload lives in a
    ``data-text="<html-escaped>"`` attribute. The browser parses the
    attribute for us, so by the time the delegated JS listener reads
    ``btn.dataset.text`` the original string is back, byte-for-byte —
    there is no XSS hole even if the assistant's reply contains
    ``<script>``, ``"xss"``, backticks, or Unicode.

    These tests pin that contract from the outside by extracting the
    raw ``data-text`` attribute body (entities intact) and then
    ``html.unescape``-ing it to check round-trip correctness.
    """

    def setUp(self) -> None:
        from web import chat_helpers  # local import: keeps the top-of-file
        # import list minimal and matches the lazy style used elsewhere.
        self._module = chat_helpers
        self._render = chat_helpers._copy_button_html

    def test_returns_a_button_element(self) -> None:
        html = self._render("hello world")
        self.assertTrue(html.lstrip().startswith("<button"))
        self.assertTrue(html.rstrip().endswith("</button>"))

    @staticmethod
    def _extract_data_text(html: str) -> str:
        """Return the HTML-decoded value of the ``data-text`` attribute.

        The helper embeds the assistant text as
        ``data-text="<html-escaped>"``. We extract the raw attribute
        body (so the safety tests can assert that dangerous raw chars
        never appear inside it) and then ``html.unescape`` it back to
        the original string for round-trip checks.
        """
        import html as _html_mod  # local import — only the tests need it
        match = re.search(r'data-text="([^"]*)"', html)
        if not match:
            raise AssertionError("`data-text=\"...\"` not found in rendered HTML")
        # The captured group is the raw attribute body (entities intact).
        # Decode it so round-trip checks see the original string.
        return _html_mod.unescape(match.group(1))

    def test_embeds_payload_as_html_escaped_attribute(self) -> None:
        # The new design puts the payload in a ``data-text`` HTML
        # attribute. The browser parses the attribute for us so JS can
        # just read ``btn.dataset.text`` and get the original string
        # back. We pin that:
        #   1. the attribute is present with a double-quoted body;
        #   2. the body round-trips to the original string via
        #      ``html.unescape`` (which is what the browser does).
        html = self._render("hello world")
        self.assertIn('data-text="', html)
        decoded = self._extract_data_text(html)
        self.assertEqual(decoded, "hello world")

    def test_no_inline_onclick_attribute(self) -> None:
        # Regression guard for the v1 bug: an inline ``onclick="..."``
        # attribute on a button whose payload is also double-quoted is
        # a quote-collision landmine (the HTML parser terminates the
        # attribute at the first inner ``"`` and the rest of the JS
        # leaks out as visible text). The new design intentionally has
        # NO ``onclick`` attribute — the JS lives in a delegated
        # listener registered once via ``_copy_button_init_script``.
        html = self._render("payload")
        self.assertNotIn("onclick=", html,
                         "button must not have an inline onclick attribute; "
                         "the delegated listener in _copy_button_init_script "
                         "handles all clicks")

    def test_escapes_apostrophes(self) -> None:
        # LLM copy is full of "don't", "user's input", "it's". On
        # CPython 3.13 ``html.escape(quote=True)`` actually escapes
        # apostrophes to ``&#x27;`` as well (defence-in-depth — even
        # though apostrophes cannot close a *double*-quoted HTML
        # attribute, an over-eager future refactor that flips the
        # attribute to single quotes must not silently break). We pin
        # that:
        #   1. Apostrophes appear as ``&#x27;`` in the raw body (so a
        #      naive switch to single-quoted attributes can't leak).
        #   2. The body round-trips to the original string via
        #      ``html.unescape`` (which is what the browser does).
        original = "Don't trust user input — it's adversarial."
        html = self._render(original)
        match = re.search(r'data-text="([^"]*)"', html)
        self.assertIsNotNone(match, "data-text attribute must be present")
        raw_body = match.group(1)
        # No raw apostrophes in the attribute body — they are all
        # entity-escaped to &#x27;.
        self.assertNotIn("'", raw_body)
        self.assertIn("&#x27;", raw_body)
        # Round-trip back to the original string.
        self.assertEqual(self._extract_data_text(html), original)

    def test_escapes_double_quotes(self) -> None:
        # An unescaped ``"`` inside the ``data-text="..."`` body would
        # close the HTML attribute early and let the rest of the reply
        # leak as visible text (the exact v1 bug we are guarding
        # against). ``html.escape(quote=True)`` MUST turn every inner
        # quote into ``&quot;``. We pin that:
        #   1. The escaped form ``&quot;`` actually appears in the
        #      body (proving escaping happened — not just deletion).
        #   2. The body has no raw ``"`` chars after the opening one
        #      that would close the attribute.
        original = 'say "hello" and <script>alert(1)</script>'
        html = self._render(original)
        match = re.search(r'data-text="([^"]*)"', html)
        self.assertIsNotNone(match, "data-text attribute must be present")
        raw_body = match.group(1)
        self.assertIn("&quot;", raw_body,
                      "inner double quotes must be entity-escaped so the "
                      "HTML attribute cannot terminate early")
        # The attribute body must not contain any raw angle brackets or
        # ampersands (all of which are HTML-significant).
        self.assertNotIn("<", raw_body)
        self.assertNotIn(">", raw_body)
        # Round-trip back to the original string.
        self.assertEqual(self._extract_data_text(html), original)

    def test_escapes_angle_brackets_and_ampersand(self) -> None:
        # Angle brackets and ``&`` are HTML-significant even inside an
        # attribute value: ``<`` could open a new tag, ``>`` is a tag
        # terminator in some parsers, and ``&`` starts a character
        # reference. ``html.escape(quote=True)`` MUST convert all
        # three to entities. We pin that here so a future refactor
        # that calls ``html.escape(value)`` without ``quote=True``
        # (which would skip ``"``) is caught — and also so a future
        # refactor that swaps the escape function entirely (e.g. to a
        # naive ``str.replace``) cannot silently drop one of the four
        # required substitutions.
        original = "<a href=\"x\">A & B</a>"
        html = self._render(original)
        match = re.search(r'data-text="([^"]*)"', html)
        self.assertIsNotNone(match, "data-text attribute must be present")
        raw_body = match.group(1)
        self.assertIn("&lt;", raw_body)
        self.assertIn("&gt;", raw_body)
        self.assertIn("&amp;", raw_body)
        # No raw dangerous chars in the attribute body.
        self.assertNotIn("<", raw_body)
        self.assertNotIn(">", raw_body)
        # Note: ``&quot;`` is also expected (from the inner quotes)
        # but the four-entity assertion above already covers it.
        # Round-trip back to the original.
        self.assertEqual(self._extract_data_text(html), original)

    def test_handles_unicode(self) -> None:
        # Security copy is heavy on em-dashes, curly quotes, and the
        # occasional CJK character. ``html.escape`` does NOT touch
        # non-ASCII codepoints by default — they are valid inside an
        # HTML attribute as raw UTF-8 bytes. We pin that em-dashes and
        # CJK chars round-trip through the attribute without being
        # turned into numeric entities (which would bloat the HTML
        # for no security gain).
        original = "Use \u2014 em-dash and \u4e2d\u6587 in CTF notes."
        html = self._render(original)
        match = re.search(r'data-text="([^"]*)"', html)
        self.assertIsNotNone(match, "data-text attribute must be present")
        raw_body = match.group(1)
        # Em-dash and CJK must appear as themselves, not as &#...;.
        self.assertIn("\u2014", raw_body)
        self.assertIn("\u4e2d\u6587", raw_body)
        self.assertEqual(self._extract_data_text(html), original)

    def test_handles_long_strings(self) -> None:
        # An assistant can dump a 4 KB code block. The helper must
        # accept it without truncation or stack-blowing recursion (the
        # implementation is one f-string concat, so there is no risk,
        # but we pin the behaviour so a future refactor does not
        # regress).
        original = "x" * 4096
        html = self._render(original)
        decoded = self._extract_data_text(html)
        self.assertEqual(len(decoded), 4096)
        self.assertEqual(decoded, original)

    def test_modern_and_legacy_clipboard_paths_live_in_init_script(self) -> None:
        # v1 inlined both the modern ``navigator.clipboard.writeText``
        # path AND the legacy ``document.execCommand('copy')`` fallback
        # into the button's onclick handler. v2 splits them out: the
        # button HTML has NEITHER (so it stays tiny and parseable), and
        # BOTH live in the init script. We pin the split here so a
        # future refactor doesn't quietly re-inline the JS into the
        # button (re-introducing the v1 quote-collision risk).
        html = self._render("payload")
        self.assertNotIn("navigator.clipboard", html,
                         "modern clipboard API must live in the init script, "
                         "not in the button HTML")
        self.assertNotIn("execCommand", html,
                         "legacy clipboard fallback must live in the init "
                         "script, not in the button HTML")

    def test_stores_original_label_for_restore(self) -> None:
        # The init script restores the button caption after the 1.4 s
        # confirmation window by reading ``btn.dataset.label`` and
        # writing it back. The label must therefore live in a
        # ``data-label`` attribute on the button. We match by regex so
        # we don't depend on whether the helper stores the emoji as a
        # surrogate pair or a single code unit (both forms are equal
        # under ``==`` but break a substring check).
        html = self._render("payload")
        self.assertRegex(
            html,
            r'data-label="[^"]*\bCopy\b[^"]*"',
            "data-label must carry the original button label so the "
            "init script can restore it after the 1.4 s confirmation",
        )
        # The init script (tested separately in
        # ``CopyButtonInitScriptTests``) must reference dataset.label;
        # we double-pin it here so the two ends stay in sync.
        init_script = self._module._copy_button_init_script()
        self.assertIn("dataset.label", init_script)

    def test_view_wires_helper(self) -> None:
        # The Streamlit view must import and call the helper from the
        # assistant branch of the history render. If a future refactor
        # drops the import or stops calling it, the button disappears
        # silently -- this test pins the wiring.
        src = _read_streamlit_view()
        self.assertIn("_render_copy_button_for_bubble", src,
                      "view must import _render_copy_button_for_bubble so "
                      "the assistant copy button is rendered as a per-bubble "
                      "iframe component")
        self.assertIn("web.chat_helpers", src)
        # And it must be rendered as a call to the iframe helper (the
        # whole point of the helper is to live inside a custom HTML
        # block, not a st.button that would lose the styling).
        self.assertRegex(
            src,
            r"_render_copy_button_for_bubble\(",
            "view must call _render_copy_button_for_bubble(...) from the "
            "assistant branch of the history render so each copy button "
            "becomes its own iframe component (the click handler lives "
            "inside the iframe, where navigator.clipboard works as a "
            "secure context -- a delegated listener in the parent window "
            "is blocked by Streamlit's component iframe sandbox)",
        )
        # And it must NOT route through the legacy helpers -- those
        # were broken by the same cross-origin sandbox and are no
        # longer the correct wiring.
        self.assertNotIn(
            "_copy_button_init_script", src,
            "view must NOT wire the legacy _copy_button_init_script "
            "helper -- that path is broken by Streamlit's cross-origin "
            "component sandbox and the button shows but does nothing",
        )
        self.assertNotIn(
            "_emit_copy_button_init_script", src,
            "view must NOT wire the legacy _emit_copy_button_init_script "
            "helper -- its parent.eval(...) bootstrapper is silently "
            "swallowed by the same-origin policy inside Streamlit's "
            "component iframe",
        )


class CopyButtonInitScriptTests(unittest.TestCase):
    """Structural + safety tests for the one-time delegated-listener script.

    The helper emits a single ``<script>`` block (idempotent: only the
    first call per process returns the block; later calls return ``""``).
    The block registers a click listener on ``document`` that handles
    every ``.bubble-copy-btn`` on the page. We pin:

      * the listener is registered on ``document`` (delegated, survives
        Streamlit re-renders);
      * it matches via ``.closest('.bubble-copy-btn')`` so clicks on
        child nodes (e.g. a future icon inside the button) still work;
      * the modern path uses ``navigator.clipboard.writeText`` and the
        legacy fallback uses ``document.execCommand('copy')``;
      * the busy-guard prevents double-click storms;
      * the original button label is restored from ``dataset.label``;
      * re-invoking the helper returns ``""`` (no duplicate listeners).
    """

    def setUp(self) -> None:
        # Force the helper to "re-emit" by resetting the module-level
        # guard. This makes the test independent of any earlier helper
        # call in the same process and lets us assert the idempotency
        # behaviour afterwards.
        from web import chat_helpers
        self._module = chat_helpers
        self._original_flag = chat_helpers._COPY_BUTTON_INIT_EMITTED
        chat_helpers._COPY_BUTTON_INIT_EMITTED = False

    def tearDown(self) -> None:
        # Restore whatever the rest of the suite expected so we do not
        # leak the test's override into other tests.
        self._module._COPY_BUTTON_INIT_EMITTED = self._original_flag

    def test_returns_a_script_block(self) -> None:
        html = self._module._copy_button_init_script()
        self.assertTrue(html.lstrip().startswith("<script"))
        self.assertTrue(html.rstrip().endswith("</script>"))

    def test_second_call_returns_empty(self) -> None:
        # The guard must flip on the first call so a second call in
        # the same process (e.g. after a Streamlit rerun) returns ""
        # and does not stack duplicate listeners.
        first = self._module._copy_button_init_script()
        self.assertTrue(first.startswith("<script"))
        second = self._module._copy_button_init_script()
        self.assertEqual(second, "",
                         "second call must return empty string to avoid "
                         "stacking duplicate document.addEventListener "
                         "registrations across Streamlit reruns")

    def test_uses_delegated_document_listener(self) -> None:
        html = self._module._copy_button_init_script()
        self.assertIn("document.addEventListener", html,
                      "must register a delegated listener on document so "
                      "buttons created by later Streamlit reruns still work")
        self.assertIn("'click'", html,
                      "must listen for click events on the delegated target")

    def test_matches_via_closest_selector(self) -> None:
        # The handler must find the .bubble-copy-btn via .closest(...) so
        # clicks on a future icon child of the button still resolve to
        # the button itself.
        html = self._module._copy_button_init_script()
        self.assertIn("closest('.bubble-copy-btn')", html)

    def test_uses_modern_clipboard_api(self) -> None:
        html = self._module._copy_button_init_script()
        self.assertIn("navigator.clipboard.writeText", html)

    def test_uses_legacy_fallback(self) -> None:
        # The fallback must be there so the button still works on older
        # browsers and inside the Streamlit Cloud preview iframe, where
        # navigator.clipboard is gated behind a user gesture and
        # sometimes blocked entirely.
        html = self._module._copy_button_init_script()
        self.assertIn("document.execCommand('copy')", html)

    def test_uses_busy_guard(self) -> None:
        # A per-element __copyBtnBusy flag must be set so rapid
        # double-clicks do not stack overlapping timeouts.
        html = self._module._copy_button_init_script()
        self.assertIn("__copyBtnBusy", html)

    def test_restores_label_from_dataset(self) -> None:
        # The handler must restore the original label from
        # btn.dataset.label so the "Copied" / "Failed" feedback clears
        # after the 1.4 s confirmation window.
        html = self._module._copy_button_init_script()
        self.assertIn("dataset.label", html)

    def test_reads_payload_from_dataset_text(self) -> None:
        # The handler must read the payload from btn.dataset.text
        # (the HTML-decoded original assistant reply) -- this is the
        # whole point of the new design.
        html = self._module._copy_button_init_script()
        self.assertIn("dataset.text", html)

    def test_inner_window_guard_for_duplicate_wiring(self) -> None:
        # Belt-and-braces: the script itself guards against being
        # evaluated twice on the same page (e.g. if a future refactor
        # accidentally calls the helper without the module-level
        # idempotency guard).
        html = self._module._copy_button_init_script()
        self.assertIn("__secMentorCopyBtnWired", html)


class CopyButtonEmitterTests(unittest.TestCase):
    """Pin ``_emit_copy_button_init_script``'s contract.

    The emitter routes the init script through
    ``streamlit.components.v1.html`` (which uses an iframe via
    ``srcdoc=`` so ``<script>`` tags survive), wrapped in a
    ``parent.eval(...)`` bootstrapper so the delegated listener runs in
    the **parent** document -- where the copy button actually lives.

    We assert:

      * the emitter calls ``st.components.v1.html`` exactly once,
      * the emitted HTML is wrapped in a ``<script>`` tag,
      * the bootstrapper calls ``parent.eval(...)`` (so the listener
        registers on the parent window, not the iframe),
      * the body's own ``</script>`` closer is escaped to ``<\\/script>``
        on the wire so the HTML parser does not terminate the wrapper
        prematurely (the same bug class that motivated the data-text
        rewrite, just one level up),
      * the original init-script body is preserved verbatim inside the
        JSON-encoded argument (the 20 ``CopyButtonInitScriptTests``
        still pin the body shape -- the emitter must not mutate it),
      * a second call is a no-op (idempotency guard),
      * the emitter returns ``None`` (it pushes to the page, it does
        not return the HTML to the caller).
    """

    def setUp(self) -> None:
        from web import chat_helpers
        self._module = chat_helpers
        self._original_flag = chat_helpers._COPY_BUTTON_INIT_EMITTED
        chat_helpers._COPY_BUTTON_INIT_EMITTED = False

    def tearDown(self) -> None:
        self._module._COPY_BUTTON_INIT_EMITTED = self._original_flag

    def _capture(self) -> list[dict]:
        """Run the emitter inside a real Streamlit script-run context.

        We patch ``st.components.v1.html`` with a recording mock and
        collect every call. ``streamlit.components.v1.html`` only works
        inside a script-run context, so we borrow one from ``AppTest``
        for the duration of the call. The ``AppTest`` body never runs
        (we never call ``.run()``) -- we just need the context object.
        """
        captured: list[dict] = []

        def _recorder(html, **kwargs):
            captured.append({"html": html, "kwargs": kwargs})

        # Stand up a real Streamlit script-run context so the helper
        # can call ``st.components.v1.html`` without raising.
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string("import streamlit as st\nst.write('')\n")
        at.run()
        try:
            with mock.patch(
                "streamlit.components.v1.html", side_effect=_recorder
            ):
                result = self._module._emit_copy_button_init_script()
        finally:
            # ``AppTest`` does not expose a teardown; the context object
            # is held on ``at`` and will be collected when the test exits.
            pass
        self.assertIsNone(
            result,
            "emitter must push the bootstrapper to the page, not return "
            "HTML to the caller -- callers do not want to st.markdown "
            "the result (that would route it back through the sanitizer "
            "we are trying to escape)",
        )
        return captured

    def test_emits_via_components_html(self) -> None:
        captured = self._capture()
        self.assertEqual(
            len(captured),
            1,
            "emitter must call st.components.v1.html exactly once on its "
            "first invocation -- multiple calls would stack duplicate "
            "iframes and the parent's __secMentorCopyBtnWired guard "
            "would still gate them, but it's wasteful",
        )
        kwargs = captured[0]["kwargs"]
        self.assertEqual(kwargs.get("height"), 0)
        self.assertEqual(kwargs.get("width"), 0)
        self.assertEqual(kwargs.get("scrolling"), False)

    def test_emits_a_script_block(self) -> None:
        captured = self._capture()
        payload = captured[0]["html"]
        self.assertTrue(payload.lstrip().startswith("<script>"))
        self.assertTrue(payload.rstrip().endswith("</script>"))

    def test_uses_parent_eval_bootstrapper(self) -> None:
        # The bootstrapper must call parent.eval(...) so the delegated
        # listener (which references bare ``window``/``document``/
        # ``navigator``) runs in the parent document where the copy
        # button lives. Without ``parent.eval`` the listener would
        # register inside the iframe's window and never catch parent
        # clicks.
        captured = self._capture()
        payload = captured[0]["html"]
        self.assertIn("parent.eval(", payload)
        self.assertIn("try {", payload)
        self.assertIn("catch (e)", payload)
        self.assertIn("console.error", payload)

    def test_escapes_inline_closer(self) -> None:
        # The init-script body has a literal ``</script>`` (its own
        # closing tag). When JSON-encoded and dropped into the wrapper
        # ``<script>...</script>`` block, that literal closer would
        # terminate the wrapper prematurely -- the HTML parser scans
        # for ``</script>`` as text, not as a JS token. The standard
        # fix is to replace ``</script>`` with ``<\/script>`` in the
        # wire bytes (which JSON then encodes as ``<\\/script>``).
        # Inside a JS string literal, the ``\/`` is just ``/`` -- the
        # runtime string is unchanged.
        captured = self._capture()
        payload = captured[0]["html"]
        self.assertIn(
            "<\\\\/script>",
            payload,
            "the body's own </script> closer must be escaped to <\\\\/script> "
            "on the wire so the HTML parser does not terminate the "
            "wrapper script tag at the body's own closer -- otherwise "
            "the body would be truncated and the delegated listener "
            "would never register.",
        )

    def test_preserves_original_body(self) -> None:
        # The 20 structural tests in ``CopyButtonInitScriptTests`` pin
        # the body shape -- the emitter must not mutate it. We confirm
        # the body is byte-identical (modulo the ``</script>`` ->
        # ``<\\/script>`` escape) by checking that every distinctive
        # marker is still present inside the JSON-encoded argument.
        captured = self._capture()
        payload = captured[0]["html"]
        body = self._module._copy_button_init_script()
        # The body is JSON-encoded inside ``parent.eval("...")`` so the
        # JSON-escape sequence for ``/`` is just ``/`` (no escape).
        # ``\n`` survives as a real newline, ``\"`` survives as ``\"``.
        body_on_wire = body.replace("</script>", "<\\/script>")
        self.assertIn(
            body_on_wire.replace("\n", "\\n").replace('"', '\\"'),
            payload,
            "the original init-script body (modulo the </script> -> "
            "<\\/script> wire escape and JSON string escaping) must be "
            "present verbatim inside the parent.eval argument so the 20 "
            "structural tests in CopyButtonInitScriptTests still apply "
            "to the runtime string",
        )
        # And the distinct markers are present directly.
        for marker in (
            "document.addEventListener",
            "'click'",
            "closest('.bubble-copy-btn')",
            "navigator.clipboard.writeText",
            "document.execCommand('copy')",
            "__copyBtnBusy",
            "dataset.label",
            "dataset.text",
            "__secMentorCopyBtnWired",
        ):
            self.assertIn(marker, payload, f"missing marker: {marker!r}")

    def test_second_call_is_noop(self) -> None:
        # The module-level idempotency guard must short-circuit the
        # second call. Without it, every Streamlit rerun would push a
        # new iframe and (worse) re-execute the body in the parent
        # window -- the body's own window guard catches it, but the
        # duplicated wire bytes are still wasted bandwidth.
        captured = self._capture()
        self.assertEqual(len(captured), 1)
        # Now capture again -- this time we expect zero calls because
        # the guard flipped on the first capture.
        second = []
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string("import streamlit as st\nst.write('')\n")
        at.run()
        with mock.patch(
            "streamlit.components.v1.html",
            side_effect=lambda html, **kw: second.append(html),
        ):
            self._module._emit_copy_button_init_script()
        self.assertEqual(
            len(second),
            0,
            "second call must be a no-op (module-level idempotency "
            "guard) -- otherwise every Streamlit rerun would push a "
            "duplicate iframe",
        )


class CopyButtonIframeHtmlTests(unittest.TestCase):
    """Structural + safety tests for ``_copy_button_iframe_html``.

    The helper builds a self-contained HTML document for a per-message
    ``st.components.v1.html`` iframe. The iframe hosts the assistant
    reply and a small "📋 Copy" button whose click handler runs inside
    the iframe's own window. This avoids the cross-origin parent/iframe
    dance that broke the previous delegated-listener design (where
    ``parent.eval(...)`` was blocked by the same-origin policy in
    Streamlit's component iframe).

    We pin:

      * the document is a valid HTML5 doctype,
      * the payload is rendered inside a ``<pre>`` so newlines and
        whitespace survive, and ``html.escape(quote=True)`` keeps any
        markup in the reply from being interpreted as HTML,
      * the click handler calls ``navigator.clipboard.writeText`` (the
        modern API -- the iframe is same-origin to the Streamlit
        server, so it counts as a secure context),
      * a legacy ``document.execCommand('copy')`` fallback is included
        for older browsers and sandboxes that gate ``navigator.clipboard``
        behind a user gesture,
      * the user gets visible feedback: the button label briefly
        becomes "✓ Copied" (or "⚠ Press Ctrl+C" on total failure) and
        then restores itself to the original label after a short delay,
      * the inline ``</script>`` closer (if any) is rewritten to
        ``<\\/script>`` on the wire so the HTML parser does not
        terminate the wrapper script tag at the body's own closer.
    """

    def setUp(self) -> None:
        from web import chat_helpers
        self._module = chat_helpers

    def _render(self, text: str) -> str:
        return self._module._copy_button_iframe_html(text)

    def test_returns_full_html_document(self) -> None:
        html = self._render("hello")
        self.assertTrue(html.lstrip().lower().startswith("<!doctype html>"),
                        "iframe srcdoc must be a complete HTML document so "
                        "the browser parses it as a real document (Streamlit "
                        "sets it as the ``srcdoc`` attribute, not as a "
                        "fragment)")
        self.assertIn("<html", html.lower())
        self.assertIn("<body", html.lower())

    def test_renders_payload_on_data_text(self) -> None:
        # The reply must NOT be rendered inside the iframe (the view
        # already renders it via ``st.markdown(content)``; rendering it
        # again would show the reply twice). Instead the payload lives
        # on a ``data-text`` attribute on the button, HTML-escaped once.
        html = self._render("line one\n  line two")
        self.assertNotIn(
            "<pre", html,
            "iframe must NOT contain a <pre> -- the view already renders "
            "the reply directly above the button; an extra <pre> would "
            "duplicate the reply in the bubble",
        )
        self.assertIn(
            "data-text=", html,
            "payload must live on a data-text attribute on the button "
            "(HTML-escaped) so the click handler can read it via "
            "btn.dataset.text without crossing the iframe boundary",
        )
        # The payload is in the attribute, escaped. We check the
        # ampersand escape because that is the simplest XSS guarantee.
        amp_html = self._render("Tom & Jerry")
        self.assertIn(
            "Tom &amp; Jerry", amp_html,
            "ampersands in the payload must be HTML-escaped inside the "
            "data-text attribute so they survive the round-trip through "
            "the browser's attribute decoder",
        )
        # Newlines survive too (the attribute value just contains a
        # literal \n; html.escape leaves it alone).
        nl_html = self._render("line one\nline two")
        self.assertIn("line one\nline two", nl_html,
                      "newlines in the payload must survive verbatim "
                      "(html.escape does not touch whitespace)")

    def test_renders_copy_button(self) -> None:
        html = self._render("hello")
        self.assertIn('id="copy-btn"', html,
                      "iframe must contain exactly one copy button so the "
                      "click handler can find it by id")
        self.assertIn("type=\"button\"", html)

    def test_includes_modern_clipboard_api(self) -> None:
        html = self._render("hello")
        self.assertIn("navigator.clipboard.writeText", html,
                      "the click handler must use the modern clipboard API; "
                      "the iframe is same-origin to the Streamlit server, "
                      "so this counts as a secure context and the API is "
                      "allowed")
        self.assertIn("navigator.clipboard && navigator.clipboard.writeText",
                      html,
                      "must guard the modern API behind a feature check so "
                      "the legacy fallback runs in older browsers")

    def test_includes_legacy_fallback(self) -> None:
        html = self._render("hello")
        self.assertIn("document.execCommand('copy')", html,
                      "must include the legacy execCommand fallback so the "
                      "button still works inside sandboxes that gate the "
                      "modern API behind a user gesture")

    def test_uses_dataset_text_for_payload(self) -> None:
        # The handler must read the payload from ``btn.dataset.text``,
        # which the browser populates by decoding the ``data-text``
        # attribute. The attribute is HTML-escaped once, so the JS side
        # gets the original bytes back without any further unescaping.
        html = self._render("hello")
        self.assertIn(
            "dataset.text", html,
            "must read the payload from btn.dataset.text (the browser's "
            "automatic decoding of the data-text attribute) -- this is "
            "the only safe way to round-trip a user-supplied string "
            "through an HTML attribute into JS without an XSS hole",
        )

    def test_shows_copied_feedback(self) -> None:
        html = self._render("hello")
        self.assertIn("Copied", html,
                      "button must show a 'Copied' confirmation on "
                      "success so the user knows the click registered")
        # And a failure label for the no-clipboard case.
        self.assertIn("Ctrl", html,
                      "button must show a 'Press Ctrl+C' hint on total "
                      "failure so the user has a manual path")

    def test_restores_label_after_timeout(self) -> None:
        html = self._render("hello")
        self.assertIn("setTimeout", html,
                      "handler must use setTimeout to restore the original "
                      "label after the confirmation window")

    def test_escapes_inline_closer(self) -> None:
        # Wire-escape any literal ``</script>`` inside the body so the
        # browser does not terminate the wrapper script tag at the
        # body's own closer.
        html = self._render("hello </script> world")
        self.assertIn("<\\/script>", html,
                      "any literal </script> in the body must be rewritten "
                      "to <\\/script> on the wire so the browser's HTML "
                      "parser does not terminate the wrapper script tag "
                      "prematurely (the same wire-escape the delegated "
                      "init script uses, one level down)")

    def test_includes_dataset_label(self) -> None:
        # The button's ``data-label`` carries the original label so the
        # handler can restore it (mirror of the legacy behaviour).
        html = self._render("hello")
        self.assertIn('data-label=', html)


class CopyButtonIframeRendererTests(unittest.TestCase):
    """Wire-level tests for ``_render_copy_button_for_bubble``.

    The renderer is the public entry point used by the Streamlit view:
    it takes an assistant message, converts markdown to plain text,
    builds the iframe srcdoc, and pushes it via
    ``st.components.v1.html``. We pin:

      * the renderer calls ``st.components.v1.html`` exactly once,
      * the kwargs route it through the same iframe-component path
        that ``_emit_copy_button_init_script`` uses (so the iframe
        inherits Streamlit's secure-context permissions),
      * the pushed HTML is the srcdoc from ``_copy_button_iframe_html``
        with the assistant message converted to plain text (not the
        raw markdown source -- otherwise the user pastes ``**bold**``
        into their email and sees asterisks),
      * the renderer returns ``None`` (it pushes to the page, like
        ``st.markdown``).
    """

    def setUp(self) -> None:
        from web import chat_helpers
        self._module = chat_helpers

    def _capture(self, content: str) -> list[dict]:
        captured: list[dict] = []

        def _recorder(html, **kwargs):
            captured.append({"html": html, "kwargs": kwargs})

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string("import streamlit as st\nst.write('')\n")
        at.run()
        with mock.patch(
            "streamlit.components.v1.html", side_effect=_recorder
        ):
            result = self._module._render_copy_button_for_bubble(content)
        self.assertIsNone(
            result,
            "renderer must push the iframe to the page, not return HTML "
            "to the caller (the caller must not st.markdown the result, "
            "which would route it back through the sanitizer)",
        )
        return captured

    def test_emits_via_components_html(self) -> None:
        captured = self._capture("hello world")
        self.assertEqual(
            len(captured), 1,
            "renderer must call st.components.v1.html exactly once per "
            "assistant message -- multiple calls would stack duplicate "
            "iframes per bubble",
        )

    def test_routes_through_iframe(self) -> None:
        captured = self._capture("hello world")
        kwargs = captured[0]["kwargs"]
        self.assertEqual(kwargs.get("scrolling"), False,
                         "must disable iframe scroll bars so the bubble "
                         "sizes itself to its content")

    def test_passes_plain_text_payload(self) -> None:
        # Markdown source -> plain text. Without the conversion the user
        # would paste ``**bold**`` into chat and see asterisks.
        captured = self._capture("**bold** and _italic_ and `code`")
        html = captured[0]["html"]
        # Markdown markers stripped.
        self.assertNotIn("**", html,
                         "renderer must strip markdown emphasis markers "
                         "from the payload (otherwise the user pastes "
                         "** into their email)")
        self.assertNotIn("`code`", html,
                         "renderer must strip backticks from inline code")
        # But the words survive.
        self.assertIn("bold", html)
        self.assertIn("italic", html)
        self.assertIn("code", html)

    def test_falls_back_to_raw_content(self) -> None:
        # Edge case: plain-text conversion returns empty string for
        # inputs like ``"\n"`` or only punctuation. The renderer must
        # fall back to the raw content so the user can still copy
        # something useful.
        captured = self._capture("\n")
        # If the fallback fired, the iframe contains the raw ``"\n"``
        # (escaped to the HTML entity). It might also be the empty
        # string, but it should not crash and must produce one iframe.
        self.assertEqual(len(captured), 1)
        self.assertIn("<pre", captured[0]["html"])


class SidebarChatsViewTests(unittest.TestCase):
    """Structural tests for the Phase 12 PR-C chat-history sidebar.

    These tests pin the *shape* of the Streamlit view's chat-history
    surface so the PR-C contract does not silently drift. The tests
    follow the same source-parse pattern as
    :class:`StreamlitViewImportSurfaceTests` (above) so the
    maintenance burden stays uniform across the suite.

    Concretely, PR-C requires that ``web/streamlit_app.py``:

    * Imports the chat-history functions from ``app.storage`` with
      the canonical aliases (``_create_chat``, ``_get_chat``,
      ``_init_db``, ``_list_chats``, ``_load_messages``,
      ``_soft_delete_chat``). The aliasing exists to keep the storage
      function names out of the view's top-level namespace *except*
      where the view explicitly wraps them in helpers like
      ``_new_chat``.
    * Defines three top-level helpers — ``_new_chat``,
      ``_open_chat``, ``_soft_delete_chat`` — that wrap the storage
      functions and manage the ``active_chat_id`` /
      ``chats_refresh_key`` session-state slots.
    * Imports ``_format_chat_timestamp`` from ``web.chat_helpers``
      (which provides the "just now / 5 min ago / Mon DD" formatting)
      and uses it in the sidebar list to render the per-row caption.
    * Wires the sidebar so a "➕  New chat" button calls
      ``_new_chat``; each row's title button calls ``_open_chat``
      with the row's chat id; each row's "🗑" button calls
      ``_soft_delete_chat`` with the row's chat id; and the list
      itself is fed by ``_list_chats(limit=...)`` (20 per the spec).
    """

    # ---- Static helpers ------------------------------------------------

    @staticmethod
    def _view_source() -> str:
        """Read the Streamlit view source as text.

        Centralised so a future rename / reorganisation only needs
        one update. Mirrors the convention used by
        :class:`StreamlitViewImportSurfaceTests`.
        """
        with open(_STREAMLIT_VIEW, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _storage_import_block(view_src: str) -> str:
        """Return the body of the ``from app.storage import (...)`` block.

        Returns an empty string if the block is missing, which the
        tests treat as a hard failure (an empty body cannot import
        any names).
        """
        match = re.search(
            r"from\s+app\.storage\s+import\s+\((.*?)\)",
            view_src,
            re.DOTALL,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _chat_helpers_import_block(view_src: str) -> str:
        """Return the body of the ``from web.chat_helpers import (...)`` block.

        Empty string if missing. Used to confirm
        ``_format_chat_timestamp`` is in the explicit allow-list of
        names the view is allowed to call directly.
        """
        match = re.search(
            r"from\s+web\.chat_helpers\s+import\s+\((.*?)\)",
            view_src,
            re.DOTALL,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _top_level_defs(view_src: str) -> set:
        """Return the set of top-level ``def`` names in the view.

        Top-level = anchored at start-of-line. This is intentionally
        strict: PR-C's helpers must be module-level callables (so
        ``st.button(..., on_click=_new_chat)`` resolves them by name)
        and not nested inside another function.
        """
        return set(re.findall(r"^def\s+(_[a-z][a-z0-9_]*)\s*\(", view_src, re.MULTILINE))

    # ---- Storage-import contract ---------------------------------------

    def test_storage_functions_are_imported_with_canonical_aliases(self) -> None:
        """The view must import every chat-history storage function.

        The aliases match the names the view uses throughout (and the
        spec's "convention over configuration" rule that the view
        only exposes storage functions through its own helpers).
        """
        body = self._storage_import_block(self._view_source())
        self.assertTrue(
            body,
            "view is missing the `from app.storage import (...)` block "
            "required by PR-C",
        )
        for alias in (
            "create_chat as _create_chat",
            "get_chat as _get_chat",
            "init_db as _init_db",
            "list_chats as _list_chats",
            "load_messages as _load_messages",
            "soft_delete_chat as _soft_delete_chat",
        ):
            self.assertIn(
                alias,
                body,
                f"storage import block must alias `{alias}` — PR-C needs "
                "all six chat-history functions reachable from the view",
            )

    # ---- Top-level helper contract -------------------------------------

    def test_three_chat_helpers_are_top_level_defs(self) -> None:
        """``_new_chat``, ``_open_chat``, ``_soft_delete_chat`` must be top-level.

        They are passed as ``on_click`` callbacks to ``st.button``,
        which Streamlit resolves by name from the script's module
        namespace. If any of them is nested inside another function,
        the button silently no-ops at runtime.
        """
        defs = self._top_level_defs(self._view_source())
        for name in ("_new_chat", "_open_chat", "_soft_delete_chat"):
            self.assertIn(
                name,
                defs,
                f"`{name}` must be a top-level def in the view so "
                "`st.button(..., on_click=...)` can resolve it",
            )

    # ---- chat_helpers import contract ----------------------------------

    def test_format_chat_timestamp_is_imported_from_chat_helpers(self) -> None:
        """``_format_chat_timestamp`` must be in the chat_helpers import block.

        The view uses it to render the per-row timestamp caption
        (e.g. "5 min ago"). If the name is missing from the
        import block, calling it would raise ``NameError`` at the
        first sidebar render.
        """
        body = self._chat_helpers_import_block(self._view_source())
        self.assertTrue(
            body,
            "view is missing the `from web.chat_helpers import (...)` block",
        )
        self.assertIn(
            "_format_chat_timestamp",
            body,
            "view must import `_format_chat_timestamp` from "
            "web.chat_helpers (PR-C sidebar uses it for the per-row "
            "timestamp caption)",
        )

    def test_view_references_format_chat_timestamp_in_body(self) -> None:
        """The view body must actually call ``_format_chat_timestamp``.

        An import without a call site is a dead import — the
        regression we want to catch is "the import survived but
        someone refactored the row to a plain ``str(chat)``".
        """
        self.assertRegex(
            self._view_source(),
            r"_format_chat_timestamp\s*\(",
            "view body must call `_format_chat_timestamp(...)` somewhere "
            "(the per-row caption in the sidebar list)",
        )

    # ---- Sidebar wiring contract ---------------------------------------

    def test_sidebar_has_new_chat_button_wired_to_helper(self) -> None:
        """The sidebar must contain a "➕  New chat" button calling ``_new_chat``.

        The label and the ``on_click`` callback must both appear in
        the view. We do not assert they are on the *same* line
        (Streamlit widgets commonly span lines), only that the
        literal label and the callback name are both present in the
        file.
        """
        src = self._view_source()
        self.assertIn(
            "➕",
            src,
            "view must contain a sidebar button labeled with the plus emoji "
            "(spec: '➕  New chat')",
        )
        self.assertIn(
            "New chat",
            src,
            "view must contain a sidebar button labeled 'New chat'",
        )
        self.assertIn(
            "on_click=_new_chat",
            src,
            "view must wire the 'New chat' button to `_new_chat` via "
            "`on_click=_new_chat` (Streamlit requires the callback to be "
            "passed as a kwarg)",
        )

    def test_sidebar_recent_list_uses_list_chats_with_limit(self) -> None:
        """The sidebar must feed the list from ``_list_chats(limit=...)``.

        Per the spec the limit is 20 (a UX choice — anything more
        makes the scrollable container slow). We assert the *call
        shape* rather than the literal number so a future
        refactor that hoists the limit into a named constant
        still passes.
        """
        self.assertRegex(
            self._view_source(),
            r"_list_chats\s*\(\s*limit\s*=",
            "view must call `_list_chats(limit=...)` to populate the "
            "sidebar recent-chats list (PR-C spec §4)",
        )

    def test_sidebar_row_delete_button_calls_soft_delete_helper(self) -> None:
        """Each row's 🗑 button must call ``_soft_delete_chat`` with the row's id.

        The 🗑 emoji is the spec's chosen affordance; the
        ``on_click=_soft_delete_chat`` wiring is what makes the
        click actually delete the row.
        """
        src = self._view_source()
        self.assertIn(
            "🗑",
            src,
            "view must contain a row-delete button labeled with the "
            "trash emoji (spec: '🗑')",
        )
        self.assertIn(
            "on_click=_soft_delete_chat",
            src,
            "view must wire the delete button to `_soft_delete_chat` "
            "via `on_click=_soft_delete_chat`",
        )

    def test_sidebar_row_title_button_calls_open_chat(self) -> None:
        """Each row's title button must call ``_open_chat`` with the row's id.

        The title is the clickable surface that loads a past chat
        into the main pane. Without ``on_click=_open_chat`` the
        button would do nothing — which is the most likely
        regression for a refactor that touches the row layout.
        """
        self.assertIn(
            "on_click=_open_chat",
            self._view_source(),
            "view must wire the row title button to `_open_chat` via "
            "`on_click=_open_chat`",
        )


class PersistOnAskTests(unittest.TestCase):
    """Structural tests for the Phase 12 PR-C turn-persistence wiring.

    The user reported that the chat-history sidebar showed the row
    ("New chat · 2 hr ago · 🗑") but clicking it never revealed any
    past messages — and the title never changed from the
    placeholder. The root cause was that the view's ``_ask``
    function appended every turn to ``st.session_state["messages"]``
    but never called the storage layer's ``append_message`` /
    ``touch_chat`` / ``update_chat_title`` functions, so the DB row
    stayed empty and the title stayed as the placeholder.

    These tests pin the *structural* fix so the regression cannot
    silently come back: the view must import the three storage
    aliases, define the three small persistence helpers, and call
    them from the right branches of ``_ask``. We deliberately do not
    spin up a real Streamlit session — the source-parse approach
    matches the rest of the suite (``SidebarChatsViewTests``,
    ``StreamlitViewImportSurfaceTests``) and avoids pulling in
    ``streamlit.testing.v1.AppTest`` just to check that a one-liner
    call exists.

    Concretely, the PR-C follow-up requires that ``web/streamlit_app.py``:

    * Imports ``append_message as _append_message``,
      ``touch_chat as _touch_chat`` and
      ``update_chat_title as _update_chat_title`` from
      ``app.storage``. The aliases match the names the helpers use.
    * Defines three top-level helpers — ``_persist_user_turn``,
      ``_persist_assistant_turn``,
      ``_persist_assistant_turn_partial`` — that wrap the storage
      functions and never let a storage exception escape into the
      ``_ask`` rendering loop.
    * Calls ``_persist_user_turn`` exactly once inside ``_ask``,
      after the user message is appended to
      ``st.session_state["messages"]`` and before the
      ``st.rerun()`` that triggers pass 2.
    * Calls ``_persist_assistant_turn`` in both the cache-hit
      branch (after the cached reply is appended to the
      transcript) and the streaming success branch (after
      ``st.write_stream`` returns).
    * Calls ``_persist_assistant_turn_partial`` in the
      partial-failure branch (where the trailing
      "_⚠️ Reply interrupted: …" marker is appended).
    """

    # ---- Static helpers ------------------------------------------------

    @staticmethod
    def _view_source() -> str:
        """Read the Streamlit view source as text.

        Mirrors the convention used by
        :class:`SidebarChatsViewTests` so the maintenance burden
        stays uniform across the suite.
        """
        with open(_STREAMLIT_VIEW, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _storage_import_block(view_src: str) -> str:
        """Return the body of the ``from app.storage import (...)`` block.

        Empty string if missing. The tests treat an empty body as
        a hard failure because no aliases can be imported through
        an empty parenthesised group.
        """
        match = re.search(
            r"from\s+app\.storage\s+import\s+\((.*?)\)",
            view_src,
            re.DOTALL,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _top_level_defs(view_src: str) -> set:
        """Return the set of top-level ``def`` names in the view.

        Top-level = anchored at start-of-line. Streamlit's
        ``st.button(..., on_click=...)`` resolves callbacks by
        name from the module namespace; a nested helper would
        silently no-op at runtime.
        """
        return set(
            re.findall(r"^def\s+(_[a-z][a-z0-9_]*)\s*\(", view_src, re.MULTILINE)
        )

    @staticmethod
    def _call_sites(view_src: str, callee: str) -> list:
        """Return the line numbers where ``callee(`` appears in the view.

        Filters out the ``def <callee>(...)`` definition line so the
        list is true *call sites* only. Used to assert the
        expected number of invocations for each helper.
        """
        out = []
        for i, line in enumerate(view_src.splitlines(), start=1):
            if callee in line and not line.lstrip().startswith("def "):
                out.append(i)
        return out

    @staticmethod
    def _function_body(view_src: str, name: str) -> str:
        """Return the body of the top-level ``def <name>(...) -> ...:``.

        Anchors on the *return-type annotation* (``) -> SomeType:``)
        rather than the bare ``):`` so the lazy match cannot run
        past the end of the signature when the function has
        typed parameters and a return annotation. The body is
        everything between the signature line and the next
        top-level ``def`` (or end of file).

        Raises ``AssertionError`` if the function is not found —
        every test using this helper has a clear story for why
        the function must exist.
        """
        # The signature regex requires a return-type annotation,
        # which every helper in this file has. Anchoring on it
        # prevents the previous regex (`\(.*?\):`) from
        # backtracking across the entire file when the function
        # has no return annotation (then no match is found and
        # the assert below fires — which is the correct outcome
        # for a missing helper).
        pattern = (
            r"^def\s+" + re.escape(name) + r"\s*\(.*?\)\s*->\s*[^:]+:\s*\n"
            r"(?P<body>.*?)(?=^def\s+|\Z)"
        )
        match = re.search(
            pattern,
            view_src,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, (
            f"could not locate top-level `def {name}(...) -> ...:` in the view"
        )
        return match.group("body")

    # ---- Storage-import contract ---------------------------------------

    def test_three_persistence_aliases_are_imported(self) -> None:
        """The view must import the three persistence storage aliases.

        Without these aliases the helpers would raise
        ``NameError`` on the first user turn, and the chat
        history would stay empty in the DB even after the user
        sent a dozen messages.
        """
        body = self._storage_import_block(self._view_source())
        self.assertTrue(
            body,
            "view is missing the `from app.storage import (...)` block "
            "required for persistence",
        )
        for alias in (
            "append_message as _append_message",
            "touch_chat as _touch_chat",
            "update_chat_title as _update_chat_title",
        ):
            self.assertIn(
                alias,
                body,
                f"storage import block must alias `{alias}` — the "
                f"persistence helpers in `_ask` call it by this name",
            )

    # ---- Top-level helper contract -------------------------------------

    def test_three_persist_helpers_are_top_level_defs(self) -> None:
        """``_persist_user_turn`` / ``_persist_assistant_turn`` /
        ``_persist_assistant_turn_partial`` must be top-level.

        The user-turn helper is the only one a unit test is
        likely to monkey-patch; the two assistant-turn helpers
        are top-level for symmetry and so a future refactor can
        unit-test the partial-failure branch the same way.
        """
        defs = self._top_level_defs(self._view_source())
        for name in (
            "_persist_user_turn",
            "_persist_assistant_turn",
            "_persist_assistant_turn_partial",
        ):
            self.assertIn(
                name,
                defs,
                f"`{name}` must be a top-level def in the view so "
                "tests can monkey-patch it and so its call sites "
                "in `_ask` resolve at runtime",
            )

    # ---- _ask call-site contract ---------------------------------------

    def test_ask_persists_user_turn_exactly_once(self) -> None:
        """``_ask`` must call ``_persist_user_turn`` exactly once.

        The single call site lives in pass 1 (after the user
        message is appended to the in-memory list, before
        ``st.rerun()``). Multiple call sites would mean the
        user message is appended to the DB twice per turn —
        visible as a duplicate in the conversation after a
        reload.
        """
        sites = self._call_sites(self._view_source(), "_persist_user_turn(")
        self.assertEqual(
            len(sites),
            1,
            f"view must call `_persist_user_turn(...)` exactly once "
            f"(found {len(sites)} call sites at lines {sites})",
        )

    def test_ask_persists_assistant_turn_in_both_success_branches(self) -> None:
        """``_ask`` must call ``_persist_assistant_turn`` in the cache-hit
        branch and the streaming success branch.

        Two call sites total: one after the cached reply is
        appended (so cache hits also land in the DB), and one
        after ``st.write_stream`` returns. Missing either one
        means a turn's assistant reply vanishes after a reload
        even though it is visible in the current session.
        """
        sites = self._call_sites(self._view_source(), "_persist_assistant_turn(")
        # 2 is the contract: cache-hit + streaming success.
        self.assertEqual(
            len(sites),
            2,
            f"view must call `_persist_assistant_turn(...)` exactly twice "
            f"(cache-hit branch + streaming success branch); found {len(sites)} "
            f"call sites at lines {sites}",
        )

    def test_ask_persists_partial_assistant_turn(self) -> None:
        """The partial-failure branch must call the dedicated partial helper.

        Using ``_persist_assistant_turn_partial`` rather than
        the success helper is what makes the call site
        self-documenting — the partial helper is allowed to
        persist the "Reply interrupted" suffix verbatim,
        whereas the success helper would still work but the
        call would read as a copy-paste of the success branch.
        """
        sites = self._call_sites(
            self._view_source(), "_persist_assistant_turn_partial("
        )
        self.assertEqual(
            len(sites),
            1,
            f"view must call `_persist_assistant_turn_partial(...)` "
            f"exactly once (partial-failure branch); found {len(sites)} "
            f"call sites at lines {sites}",
        )

    # ---- First-turn rename contract ------------------------------------

    def test_first_turn_rename_logic_is_in_helper(self) -> None:
        """The first-turn auto-rename must live in ``_persist_user_turn``.

        The rename rule ("if the chat's current title is the
        placeholder ``New chat`` or empty, overwrite it with
        the truncated user text") is the single piece of
        business logic that distinguishes a real chat history
        from a placeholder-only list. Pinning it inside
        ``_persist_user_turn`` keeps the call sites in
        ``_ask`` one-liners and ensures the rename fires
        exactly once per chat.
        """
        view_src = self._view_source()
        # Find the body of `_persist_user_turn` using the helper
        # that anchors on the return-type annotation, so the
        # body extraction cannot accidentally span across
        # subsequent top-level `def`s.
        body = self._function_body(view_src, "_persist_user_turn")
        # The body must call _update_chat_title (the rename) and
        # _touch_chat (the activity bump) and _append_message
        # (the user-turn persist). The placeholder check
        # (`"New chat"`) is also pinned to make sure the rule
        # does not silently broaden to e.g. "any non-empty
        # title" which would clobber legitimate titles.
        self.assertIn(
            "_update_chat_title(",
            body,
            "`_persist_user_turn` must call `_update_chat_title(...)` "
            "to apply the first-turn auto-rename",
        )
        self.assertIn(
            "_touch_chat(",
            body,
            "`_persist_user_turn` must call `_touch_chat(...)` to bump "
            "`updated_at` so the sidebar's 'x min ago' sort reflects "
            "activity",
        )
        self.assertIn(
            "_append_message(",
            body,
            "`_persist_user_turn` must call `_append_message(...)` to "
            "write the user turn to the DB",
        )
        self.assertIn(
            '"New chat"',
            body,
            "`_persist_user_turn` must gate the rename on the current "
            "title being the placeholder `\"New chat\"` (or empty) so "
            "later turns do not clobber a user-set title",
        )

    def test_title_truncation_helper_uses_60_char_cap(self) -> None:
        """The truncation helper must apply a 60-char cap with an ellipsis.

        The spec says the title is "the first 60 chars of the
        first user turn". Without the cap a long prompt
        produces an unreadable sidebar row; without the
        ellipsis suffix the user has no visual signal that
        the title was truncated. Both are pinned.
        """
        view_src = self._view_source()
        body = self._function_body(view_src, "_truncate_title")
        # Pin the cap constant.
        self.assertIn(
            "_TITLE_CHAR_CAP",
            body,
            "`_truncate_title` must reference `_TITLE_CHAR_CAP` so the "
            "60-char limit is enforced consistently",
        )
        # Pin the cap's value at its definition site.
        self.assertRegex(
            view_src,
            r"_TITLE_CHAR_CAP\s*=\s*60",
            "the title cap must be exactly 60 chars (spec: 'first 60 "
            "chars of the first user turn')",
        )
        # Pin the ellipsis suffix so the user sees the truncation.
        self.assertIn(
            "…",
            body,
            "`_truncate_title` must append a horizontal-ellipsis "
            "(`…`) so truncated titles are visually distinct from "
            "the full-text titles",
        )

    # ---- UI-safety contract --------------------------------------------

    def test_persist_helpers_swallow_storage_exceptions(self) -> None:
        """Each helper must catch storage exceptions so the UI does not
        crash on a transient DB error.

        Without the try/except, a locked-DB or disk-full error
        from ``_append_message`` would surface as an
        unhandled exception inside ``_ask``'s render path,
        blanking the page on every user turn until the user
        reloads. The contract is: the in-memory transcript
        still works, and a ``st.caption`` warning is shown
        so the regression is visible.
        """
        view_src = self._view_source()
        for name in (
            "_persist_user_turn",
            "_persist_assistant_turn",
            "_persist_assistant_turn_partial",
        ):
            body = self._function_body(view_src, name)
            self.assertIn(
                "try:",
                body,
                f"`{name}` must wrap storage calls in a `try:` block "
                "so a transient DB error does not break the UI",
            )
            self.assertIn(
                "except",
                body,
                f"`{name}` must have an `except` arm that catches the "
                "storage exception and surfaces a non-blocking warning",
            )
            self.assertIn(
                "st.caption(",
                body,
                f"`{name}` must surface the swallowed exception via "
                "`st.caption(...)` so the user knows the transcript is "
                "session-state-only this turn",
            )


# --- Multimodal message content coercion ------------------------------------
#
# The view's `_render_bubble` (web/streamlit_app.py) renders every past
# turn by replaying ``st.session_state["messages"]``. After the user
# uploads an image, the persisted ``content`` is a JSON-encoded
# ``list[dict]`` (a multimodal parts list), and ``app.storage.list_messages``
# decodes it back to a real ``list`` when the chat is reopened. The
# pre-fix renderer treated ``content`` as ``str`` and called
# ``content.replace(...)`` on it, which raised
# ``AttributeError: 'list' object has no attribute 'replace'`` and
# blanked the entire transcript on the very next rerun.
#
# The fix is a tiny pure helper in ``web.chat_helpers``,
# ``_coerce_message_text``, called once at the top of the render branch
# so every downstream method (``st.markdown``, ``.replace(...)``, the
# copy-button markdown-to-plain pipeline) sees a ``str``. These tests
# pin the contract: the helper must return a ``str`` for every input
# shape, must preserve text parts verbatim, must surface image parts
# as a short placeholder, must not crash on unknown shapes, and must
# not crash on ``None``. They run without Streamlit.


class CoerceMessageTextTests(unittest.TestCase):
    """Pin the ``_coerce_message_text(content) -> str`` contract.

    Defensive coverage for every shape ``app.storage.list_messages``
    can produce (or that a future caller could add): plain string,
    multimodal parts list, an unknown structured part, ``None``, and
    stray scalar shapes.
    """

    def setUp(self) -> None:
        from web import chat_helpers  # lazy import keeps the top-of-file
        # import list minimal and matches the rest of this module.
        self._fn = chat_helpers._coerce_message_text

    def test_plain_string_passes_through_verbatim(self) -> None:
        # The common text-only path. The helper must NOT strip, lowercase,
        # or otherwise mutate the string — the copy button then runs
        # ``_markdown_to_plain_text`` on the result, and any whitespace
        # loss here would mangle fenced code blocks.
        self.assertEqual(
            self._fn("hello world"), "hello world"
        )
        self.assertEqual(
            self._fn("  leading and trailing  "),
            "  leading and trailing  ",
        )
        self.assertEqual(self._fn(""), "")

    def test_list_with_text_part_returns_text(self) -> None:
        # A single text part. Must produce the literal text without
        # surrounding whitespace.
        self.assertEqual(
            self._fn([{"type": "text", "text": "why is SQL injection bad?"}]),
            "why is SQL injection bad?",
        )

    def test_list_with_text_part_missing_type_key(self) -> None:
        # Some OpenAI-compatible providers omit the ``type`` discriminator
        # on text parts and just send ``{"text": "..."}``. The helper must
        # accept the field rather than requiring ``type``.
        self.assertEqual(
            self._fn([{"text": "Explain CSRF."}]),
            "Explain CSRF.",
        )

    def test_list_with_image_url_string(self) -> None:
        # An image attached as a plain URL string. Must produce a
        # short placeholder rather than the URL itself, and must NOT
        # silently drop the text part.
        out = self._fn([
            {"type": "text", "text": "what does this say?"},
            {"type": "image_url", "image_url": "https://x.example/a.png"},
        ])
        self.assertIn("what does this say?", out)
        self.assertIn("[image:", out)
        self.assertIn("https://x.example/a.png", out)

    def test_list_with_image_url_dict_truncates_data_uris(self) -> None:
        # The common case is a base64 data URI; the value can run to
        # ~50 KB for a screenshot. Embedding the full URI in the
        # transcript would balloon ``st.session_state`` and slow every
        # rerun. The helper must keep just a short prefix.
        data_uri = (
            "data:image/png;base64,"
            + "Z" * 500  # well past the 32-char cutoff
        )
        out = self._fn([{
            "type": "image_url",
            "image_url": {"url": data_uri},
        }])
        self.assertIn("[image:", out)
        # A tail that lives only past the 32-char cutoff must NOT appear —
        # that would be the original bug, embedding 50 KB of base64 into
        # ``st.session_state`` on every rerun.
        self.assertNotIn("Z" * 50, out)
        self.assertIn("...", out)

    def test_list_with_unknown_part_type_renders_placeholder(self) -> None:
        # A future OpenRouter revision could add ``type: audio_url`` (or
        # anything else). The helper must NOT crash, and must surface
        # the unknown part as a small labelled placeholder rather than
        # dumping the dict into the bubble.
        out = self._fn([{"type": "audio_url", "audio_url": "x"}])
        self.assertIn("[audio_url]", out)

    def test_list_with_empty_text_part_is_skipped(self) -> None:
        # A text part with an empty string must NOT contribute a literal
        # "[]" or extra whitespace to the bubble.
        out = self._fn([
            {"type": "text", "text": ""},
            {"type": "text", "text": "real question"},
        ])
        self.assertEqual(out, "real question")

    def test_list_with_bare_dict_without_text_or_type_is_skipped(self) -> None:
        # ``[{"foo": "bar"}]`` has no field we know how to render.
        # The helper must skip it silently rather than stringifying the
        # whole dict (which would dump JSON into the bubble).
        out = self._fn([{"foo": "bar"}, {"type": "text", "text": "ok"}])
        self.assertEqual(out, "ok")

    def test_list_with_non_dict_item_stringifies(self) -> None:
        # Some providers send ``[{"text": "..."}, "stray scalar"]``.
        # Stringify the scalar so the bubble keeps something useful.
        out = self._fn(["just a string", {"type": "text", "text": "ok"}])
        self.assertIn("just a string", out)
        self.assertIn("ok", out)

    def test_none_returns_stringified_none(self) -> None:
        # The schema contract is "JSON blob" so a future caller could
        # fail to set ``content``. The helper must NOT crash; stringifying
        # ``None`` is the friendliest outcome (the bubble shows the
        # literal ``None`` and the user can scroll past it).
        self.assertEqual(self._fn(None), "None")

    def test_int_returns_stringified_int(self) -> None:
        # Defensive: stringify any unknown scalar rather than crashing
        # the whole rerun (the original bug). The value of "None" or
        # "42" in a bubble is bad UX, but a crashed rerun is worse.
        self.assertEqual(self._fn(42), "42")

    def test_never_returns_a_list(self) -> None:
        # The renderer's first line after the helper is
        # ``content = _coerce_message_text(content)``. The downstream
        # methods (``content.replace(...)``, ``st.markdown(content)``,
        # the copy button's markdown-to-plain pipeline) all assume
        # ``str``. If the helper ever returned a list, the original
        # bug would recur silently. Pin the return type as ``str`` for
        # every shape.
        for input_value in (
            "string",
            [{"type": "text", "text": "x"}],
            [{"type": "image_url", "image_url": "http://x"}],
            None,
            42,
            3.14,
            True,
            {"text": "bare dict"},
        ):
            with self.subTest(input=repr(input_value)):
                self.assertIsInstance(
                    self._fn(input_value), str,
                    f"_coerce_message_text returned non-str for {input_value!r}",
                )


# --- OpenRouter streaming delta list-shape ----------------------------------
#
# Companion regression for the engine-side fix in
# ``app.openrouter.stream_chat``. Some OpenRouter-compatible providers
# stream ``delta.content`` as a ``list[dict]`` of structured parts
# (typically one text part, occasionally text + a tool-call preamble)
# instead of a plain string. The pre-fix parser only yielded the chunk
# when ``content`` was a ``str``, so every list-shaped delta was
# silently dropped, the router's "stream returned no deltas" guard
# fired, and the user saw an empty bubble + a generic OpenRouter error
# banner.
#
# The fix is a four-line inline flatten in the SSE loop. We don't have
# a free-standing helper to unit-test (it's inside the streaming
# generator and depends on a real ``requests.Response``), so we pin
# the contract structurally: the source must contain the
# ``isinstance(content, list)`` branch that walks the parts and
# concatenates the text fields. If anyone refactors the SSE loop and
# regresses the list-delta path, this test fires.


class OpenRouterStreamListDeltaTests(unittest.TestCase):
    """Pin the ``delta.content`` list-shape handling in ``stream_chat``.

    The stream is wired into the ``st.write_stream`` consumer and the
    router's "no deltas" guard: any list-shaped delta that is silently
    dropped makes a healthy upstream look like a transient failure. The
    fix inlines a small flatten — these tests pin that the flatten
    (a) exists in the source, (b) walks both ``list[dict]`` and
    ``list[str]`` shapes, (c) only yields the result when it is
    non-empty (preserving the original ``and content`` truthiness).
    """

    def setUp(self) -> None:
        from app import openrouter as _openrouter
        self._module = openrouter

    def _stream_chat_source(self) -> str:
        import inspect
        return inspect.getsource(self._module.stream_chat)

    def test_stream_chat_handles_list_shaped_delta(self) -> None:
        src = self._stream_chat_source()
        # The flatten branch is anchored on an ``isinstance(content, list)``
        # check immediately after the ``delta.get("content")`` lookup. If
        # the SSE loop is refactored and this branch disappears, every
        # list-shaped upstream falls back to the silent-drop path and the
        # regression is invisible to the unit suite.
        self.assertIn(
            "isinstance(content, list)",
            src,
            "stream_chat must contain an isinstance(content, list) branch "
            "to flatten list-shaped OpenRouter deltas; without it a "
            "list-shaped delta is silently dropped and the router raises "
            "the 'no deltas' OpenRouterError.",
        )

    def test_stream_chat_extracts_text_field_from_each_part(self) -> None:
        src = self._stream_chat_source()
        # The flatten walks each dict in the parts list and pulls the
        # ``text`` field. Pin that the structural shape is preserved
        # so a future refactor cannot accidentally stringify the
        # whole dict (which would dump JSON into the bubble).
        self.assertIn('item.get("text")', src)
        # And the per-part contribution must land in ``parts`` so the
        # final ``"".join(parts)`` returns the assembled reply.
        self.assertIn("parts.append(text_value)", src)
        self.assertIn('"".join(parts)', src)

    def test_stream_chat_still_yields_only_non_empty_deltas(self) -> None:
        src = self._stream_chat_source()
        # The original ``isinstance(content, str) and content`` truthiness
        # guard must remain after the flatten so a list whose parts
        # all stringify to "" does not produce a spurious yield that
        # would then be filtered out by ``st.write_stream``'s own
        # empty-string check (and waste a chunk budget).
        self.assertIn(
            'isinstance(content, str) and content',
            src,
            "stream_chat must keep the non-empty delta guard after the "
            "flatten; without it an empty flattening result would yield "
            "an empty string and the bubble would gain a literal "
            "no-op delta.",
        )


if __name__ == "__main__":
    unittest.main()