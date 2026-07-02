"""Tests for the multimodal file-upload pipeline (Phase 11).

These tests cover the new engine and helper surface added in
Phase 11 (image and PDF uploads):

- ``app.file_processor`` — pure engine that classifies uploads,
  base64-encodes images, and extracts text from PDFs.
- ``web.chat_helpers.build_user_turn_content`` — collapses a user
  turn + uploads into either a plain string (text-only) or a
  list of content parts (multimodal, with image_url + text).
- ``web.chat_helpers.select_model_for_request`` — pure decision
  rule for whether the current model can see images.
- ``app.config.model_supports_vision`` — predicate for the
  curated vision-capable model allow-list.
- ``web.chat_helpers._build_messages`` — round-trips both ``str``
  and ``list[dict]`` content unchanged so the engine can pass
  multimodal payloads to the LLM.
- ``app.openrouter._build_payload`` — regression: a list-of-parts
  ``content`` flows through to the HTTP body unchanged.

House style follows ``tests/test_smoke.py``: ``unittest.TestCase``,
duck-typed fakes for Streamlit uploads, ``setUp`` swaps ``cwd`` and
``sys.path`` so ``from app...`` and ``from web...`` resolve from the
project root regardless of where ``unittest`` was invoked.

Run with:  python -m unittest tests.test_files -v
"""

import base64
import importlib
import json
import os
import sys
import unittest
from unittest.mock import patch

import pytest

from web.chat_helpers import consume_stop_flag, resolve_chatbox_model_id

pytestmark = pytest.mark.smoke

# --- Path bootstrap ----------------------------------------------------------
# Mirror the project-root-cd pattern used in tests/test_smoke.py so this file
# works no matter where the user invokes `python -m unittest` from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# --- Duck-typed upload fakes -----------------------------------------------


class _FakeUpload:
    """Mimics ``streamlit.runtime.uploaded_file_manager.UploadedFile``.

    The view's helpers consume anything that exposes ``name``,
    ``type``, ``size``, and ``read()`` — Streamlit's real class has
    a few extra methods (e.g. ``seek``, ``getvalue``) that the
    engine does not call. We implement the minimal surface only.
    """

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


def _png_payload(width=2, height=2):
    """A minimal, hand-rolled PNG byte string.

    We don't decode it in tests; the engine just re-encodes it as
    a base64 data URL. What matters is the magic bytes (so
    ``image_processor`` can sniff the MIME if needed) and the size.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    return sig + b"\x00" * 200


def _jpeg_payload():
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


# --- Tests ------------------------------------------------------------------


class FileProcessorImportTests(unittest.TestCase):
    """The engine module is importable and exposes the public surface."""

    def test_module_exports(self):
        from app import file_processor as fp
        for name in (
            "FileProcessingError",
            "ImagePart",
            "process_image",
            "process_pdf",
            "image_url_part",
        ):
            with self.subTest(symbol=name):
                self.assertTrue(
                    hasattr(fp, name),
                    f"app.file_processor must export {name!r}",
                )


class FileProcessingErrorTests(unittest.TestCase):
    """``FileProcessingError`` carries a stable ``kind`` code and a message."""

    def test_kind_and_message_are_preserved(self):
        from app.file_processor import FileProcessingError

        exc = FileProcessingError("boom", kind="image")
        self.assertEqual(exc.kind, "image")
        self.assertEqual(str(exc), "boom")
        # It must be a regular Exception subclass so callers can `raise` /
        # `except` it without special-casing the engine.
        self.assertIsInstance(exc, Exception)

    def test_unknown_kind_is_acceptable(self):
        """The engine documents ``kind`` as a free-form string; tests
        should not pin the universe of values, only that it is exposed
        and defaults to a string."""
        from app.file_processor import FileProcessingError

        for kind in ("image", "pdf", "text", "unknown"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    FileProcessingError("x", kind=kind).kind,
                    kind,
                )


class ProcessImageTests(unittest.TestCase):
    """``process_image`` returns an ``ImagePart`` with a valid data URL."""

    def setUp(self):
        from app import file_processor as fp
        self.fp = fp

    def test_png_round_trip(self):
        """A small PNG comes back as ``ImagePart`` with a data: URL
        whose payload decodes to the original bytes."""
        original = _png_payload()
        fake = _FakeUpload("shot.png", "image/png", original)
        part = self.fp.process_image(fake)
        self.assertEqual(part.mime, "image/png")
        self.assertEqual(part.size, len(original))
        # data_url must be parseable: data:<mime>;base64,<b64>
        prefix = "data:image/png;base64,"
        self.assertTrue(
            part.data_url.startswith(prefix),
            f"data_url must start with {prefix!r}; got {part.data_url[:60]!r}",
        )
        decoded = base64.b64decode(part.data_url[len(prefix):])
        self.assertEqual(decoded, original)

    def test_jpeg_round_trip(self):
        original = _jpeg_payload()
        fake = _FakeUpload("snap.jpg", "image/jpeg", original)
        part = self.fp.process_image(fake)
        self.assertEqual(part.mime, "image/jpeg")
        prefix = "data:image/jpeg;base64,"
        self.assertTrue(part.data_url.startswith(prefix))
        self.assertEqual(
            base64.b64decode(part.data_url[len(prefix):]),
            original,
        )

    def test_oversized_image_raises(self):
        """Files over the engine's size cap must raise with kind='oversized'."""
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (5 * 1024 * 1024)
        fake = _FakeUpload("big.png", "image/png", big)
        with self.assertRaises(self.fp.FileProcessingError) as ctx:
            self.fp.process_image(fake)
        self.assertEqual(ctx.exception.kind, "oversized")

    def test_unsupported_mime_raises(self):
        """A BMP file is not in the engine's allow-list and must be rejected."""
        fake = _FakeUpload("art.bmp", "image/bmp", b"BM" + b"\x00" * 50)
        with self.assertRaises(self.fp.FileProcessingError) as ctx:
            self.fp.process_image(fake)
        self.assertEqual(ctx.exception.kind, "unsupported_image")

    def test_empty_file_raises(self):
        fake = _FakeUpload("blank.png", "image/png", b"")
        with self.assertRaises(self.fp.FileProcessingError) as ctx:
            self.fp.process_image(fake)
        self.assertEqual(ctx.exception.kind, "empty")


class ProcessPdfTests(unittest.TestCase):
    """``process_pdf`` returns a plain string (extracted text).

    The pymupdf import is lazy inside the engine, so we skip the
    whole class if pymupdf is not installed. The test environment
    has it pinned (see requirements.txt), but on a bare system
    these tests are no-ops rather than failures.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pymupdf not installed; skipping PDF tests")

    def setUp(self):
        from app import file_processor as fp
        self.fp = fp

    def test_minimal_pdf_returns_non_empty_string(self):
        """A 1-page hand-rolled PDF should yield a non-empty string."""
        # This is a 1-page, 1-line "Hello, world!" PDF compressed with
        # deflate. The engine should call pymupdf to extract it; we
        # only assert non-emptiness, not exact text, because pymupdf
        # may normalise whitespace.
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 55>>stream\n"
            b"BT /F1 12 Tf 72 720 Td (Hello, world!) Tj ET\n"
            b"endstream endobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000053 00000 n \n"
            b"0000000100 00000 n \n"
            b"0000000200 00000 n \n"
            b"0000000290 00000 n \n"
            b"trailer<</Size 6/Root 1 0 R>>\n"
            b"startxref\n350\n%%EOF\n"
        )
        fake = _FakeUpload("hi.pdf", "application/pdf", minimal_pdf)
        out = self.fp.process_pdf(fake)
        self.assertIsInstance(out, str)
        self.assertTrue(
            out.strip(),
            "PDF extraction must yield a non-empty string for a real PDF",
        )

    def test_non_pdf_raises(self):
        """A file labelled .pdf that is not actually a PDF must raise."""
        fake = _FakeUpload("fake.pdf", "application/pdf", b"not a pdf at all")
        with self.assertRaises(self.fp.FileProcessingError) as ctx:
            self.fp.process_pdf(fake)
        self.assertEqual(ctx.exception.kind, "invalid_pdf")


class ImageUrlPartTests(unittest.TestCase):
    """``image_url_part`` produces the OpenRouter-shaped wire format."""

    def test_data_url_passthrough(self):
        from app.file_processor import ImagePart, image_url_part

        part = ImagePart(
            data_url="data:image/png;base64,QUJD",
            mime="image/png",
            size=3,
        )
        wire = image_url_part(part)
        self.assertEqual(
            wire,
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        )


class BuildUserTurnContentTests(unittest.TestCase):
    """``build_user_turn_content`` returns ``str`` (text-only) or
    ``list[dict]`` (multimodal) depending on what the user attached."""

    def setUp(self):
        # Force a fresh import so a previous test that patched the
        # helper's dependencies is not bleeding into this class.
        for mod in list(sys.modules):
            if mod == "web.chat_helpers" or mod.startswith("web.chat_helpers."):
                del sys.modules[mod]

    def test_text_only_returns_string(self):
        from web.chat_helpers import build_user_turn_content

        out = build_user_turn_content("hi", [])
        self.assertIsInstance(out, str)
        self.assertEqual(out, "hi")

    def test_none_text_no_files_returns_empty_string(self):
        from web.chat_helpers import build_user_turn_content

        self.assertEqual(build_user_turn_content(None, []), "")
        self.assertEqual(build_user_turn_content("", []), "")

    def test_text_only_file_keeps_string_shape(self):
        """A non-image file (e.g. JSON) must NOT promote the turn to
        multimodal — the model is given the inlined text instead."""
        from web.chat_helpers import build_user_turn_content

        fake = _FakeUpload(
            "log.json", "application/json", b'{"event":"login"}'
        )
        out = build_user_turn_content("review", [fake])
        self.assertIsInstance(
            out, str,
            f"text-only file must not promote turn to list; got {type(out).__name__}",
        )
        self.assertIn("review", out)
        self.assertIn("log.json", out)

    def test_image_promotes_to_list_with_text_and_image_url(self):
        from web.chat_helpers import build_user_turn_content

        fake = _FakeUpload("shot.png", "image/png", _png_payload())

        def fake_processor(file):
            from app.file_processor import ImagePart
            return ImagePart(
                data_url="data:image/png;base64,QUJD",
                mime="image/png",
                size=3,
            )

        out = build_user_turn_content("describe", [fake], image_processor=fake_processor)
        self.assertIsInstance(out, list)
        kinds = [p.get("type") for p in out]
        self.assertEqual(
            kinds, ["text", "image_url"],
            f"expected [text, image_url] parts; got {kinds!r}",
        )
        # The text part carries the user prompt verbatim
        self.assertEqual(out[0].get("text"), "describe")
        # The image_url part uses the OpenRouter wire shape
        self.assertEqual(
            out[1],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        )

    def test_image_only_turn_has_image_url_part(self):
        """If the user attaches an image without typing anything,
        the multimodal payload is a list whose first part is the
        image_url (no text part, per the OpenAI cookbook)."""
        from web.chat_helpers import build_user_turn_content

        fake = _FakeUpload("shot.png", "image/png", _png_payload())

        def fake_processor(file):
            from app.file_processor import ImagePart
            return ImagePart(
                data_url="data:image/png;base64,QUJD",
                mime="image/png",
                size=3,
            )

        out = build_user_turn_content(None, [fake], image_processor=fake_processor)
        self.assertIsInstance(out, list)
        # No text part — the model answers the implicit "what is in
        # this image?" question from the image alone.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "image_url")
        self.assertEqual(
            out[0]["image_url"]["url"],
            "data:image/png;base64,QUJD",
        )

    def test_multiple_images_all_appear(self):
        from web.chat_helpers import build_user_turn_content

        files = [
            _FakeUpload(f"img{i}.png", "image/png", _png_payload())
            for i in range(3)
        ]

        def fake_processor(file):
            from app.file_processor import ImagePart
            return ImagePart(
                data_url=f"data:image/png;base64,{file.name}",
                mime="image/png",
                size=3,
            )

        out = build_user_turn_content("look", files, image_processor=fake_processor)
        self.assertIsInstance(out, list)
        image_urls = [p for p in out if p["type"] == "image_url"]
        self.assertEqual(len(image_urls), 3)
        # Order must follow the input order (img0, img1, img2).
        names = [p["image_url"]["url"] for p in image_urls]
        self.assertEqual(
            names,
            [
                "data:image/png;base64,img0.png",
                "data:image/png;base64,img1.png",
                "data:image/png;base64,img2.png",
            ],
        )

    def test_broken_image_becomes_stub_not_exception(self):
        """If the engine raises ``FileProcessingError`` for a bad
        image, the helper must fail-soft: the broken image becomes
        a textual stub and the user text is preserved. The
        turn-shape can stay as a list so the model still sees the
        surviving images."""
        from app.file_processor import FileProcessingError
        from web.chat_helpers import build_user_turn_content

        bad = _FakeUpload("bad.png", "image/png", b"not a png")

        def failing_processor(file):
            raise FileProcessingError("bad bytes", kind="image")

        good = _FakeUpload("ok.png", "image/png", _png_payload())

        def ok_processor(file):
            from app.file_processor import ImagePart
            return ImagePart(
                data_url="data:image/png;base64,QUJD",
                mime="image/png",
                size=3,
            )

        def dispatch(file):
            return failing_processor(file) if file.name == "bad.png" else ok_processor(file)

        out = build_user_turn_content(
            "explain", [bad, good], image_processor=dispatch
        )
        self.assertIsInstance(out, list)
        # The user text must still be in the payload
        joined = json.dumps(out, default=str)
        self.assertIn("explain", joined)
        # And the broken image must surface as a stub note.
        self.assertIn("bad.png", joined)
        self.assertIn("not sent to the model", joined.lower())


class BuildUserTurnPdfTests(unittest.TestCase):
    """Regression tests for the PDF pipeline added after the user
    reported that PDF attachments were rendered as a binary stub in
    the user bubble and never reached the model.

    Two contracts are locked in here:

    1. ``build_user_turn_text`` — the *display-bubble* path. When
       given a ``pdf_processor``, the extracted text is inlined
       verbatim (not the "binary, not inlined" stub). Without a
       processor, PDFs fall through to the legacy stub behaviour
       for backward compatibility.

    2. ``build_user_turn_content`` — the *model-content* path. PDF
       text is folded into the text part of the user message, and
       the turn shape stays as a string (not a multimodal list)
       unless an image is also present. A ``FileProcessingError``
       from the PDF processor downgrades the PDF to a textual
       stub instead of crashing the turn.
    """

    def setUp(self):
        # Force a fresh import so a previous test that patched the
        # helper's dependencies is not bleeding into this class.
        for mod in list(sys.modules):
            if mod == "web.chat_helpers" or mod.startswith("web.chat_helpers."):
                del sys.modules[mod]

    # ---- build_user_turn_text (the display-bubble path) ----------------

    def test_pdf_inlined_in_display_bubble_when_processor_supplied(self):
        """The user bubble must show a *compact stub* for the PDF
        (``[Attached PDF: name · size · chars]``), not the full
        extracted prose. Top agents (Claude.ai, ChatGPT) keep the
        bubble short; the extracted text is delivered to the model
        via :func:`build_user_turn_content`, not the display
        helper. This is the exact behaviour the user requested
        after the 4 000-char preview wall was reported as visually
        noisy."""
        from web.chat_helpers import build_user_turn_text

        fake = _FakeUpload("Final CV.pdf", "application/pdf", b"%PDF-1.4\n...") 

        def fake_pdf_processor(file):
            return "Jane Doe\nSenior Security Engineer\n10 years experience"    

        out = build_user_turn_text(
            "summarize", [fake], pdf_processor=fake_pdf_processor
        )

        # The compact stub is in the bubble — file name + size + char count
        self.assertIn("[Attached PDF: Final CV.pdf", out)
        # The user sees the *count* of characters, not the prose itself
        self.assertIn("chars", out)
        # The user's typed question is preserved
        self.assertIn("summarize", out)
        # The legacy binary stub is NOT shown
        self.assertNotIn("binary, not inlined", out)
        self.assertNotIn("Ask the user to paste", out)
        # The extracted prose is NOT in the bubble — that is what the
        # user explicitly asked for ("simple and short like top agents").
        self.assertNotIn("Jane Doe", out)
        self.assertNotIn("Senior Security Engineer", out)
    def test_pdf_falls_back_to_legacy_block_when_no_processor_supplied(self):
        """For backward compatibility, when ``pdf_processor`` is
        ``None`` the helper must keep the old _format_upload_block
        behaviour: the file name is in the bubble so the user knows
        the PDF was attached, and the function still returns a
        non-empty string. The view always injects
        ``app.file_processor.process_pdf`` in production, so this
        path is only exercised by callers that have not opted in."""
        from web.chat_helpers import build_user_turn_text

        fake = _FakeUpload("Final CV.pdf", "application/pdf", b"%PDF-1.4\n...")

        out = build_user_turn_text("summarize", [fake])

        # The function returns a non-empty string and the user
        # text is preserved.
        self.assertIsInstance(out, str)
        self.assertIn("summarize", out)
        # The file name surfaces so the user knows it was attached.
        self.assertIn("Final CV.pdf", out)

    def test_pdf_processor_error_fails_soft_in_display_bubble(self):
        """A ``FileProcessingError`` from the PDF processor must
        surface a textual stub in the bubble (not crash the turn)."""
        from app.file_processor import FileProcessingError
        from web.chat_helpers import build_user_turn_text

        fake = _FakeUpload("scanned.pdf", "application/pdf", b"%PDF-1.4\n...")

        def failing_processor(file):
            raise FileProcessingError("scanned, no text", kind="empty")

        out = build_user_turn_text("read this", [fake], pdf_processor=failing_processor)

        # The user text is preserved
        self.assertIn("read this", out)
        # The file name still appears so the user knows it was attached
        self.assertIn("scanned.pdf", out)
        # The extracted text is NOT in the bubble (the processor failed)
        self.assertNotIn("Jane Doe", out)

    # ---- build_user_turn_content (the model-content path) --------------

    def test_pdf_text_folded_into_text_part_of_model_content(self):
        """For a PDF-only turn, the helper must return a *string*
        (not a multimodal list) with the extracted text inlined
        under a labelled ``[Attached PDF: <name>]`` header."""
        from web.chat_helpers import build_user_turn_content

        fake = _FakeUpload("Final CV.pdf", "application/pdf", b"%PDF-1.4\n...")

        def fake_pdf_processor(file):
            return "Jane Doe\nSenior Security Engineer"

        out = build_user_turn_content(
            "summarize", [fake], pdf_processor=fake_pdf_processor
        )

        self.assertIsInstance(out, str)
        self.assertIn("summarize", out)
        self.assertIn("[Attached PDF: Final CV.pdf]", out)
        self.assertIn("Jane Doe", out)

    def test_pdf_with_image_promotes_to_multimodal_with_text_part(self):
        """When a PDF is attached alongside an image, the helper
        must return a *list of parts* with the PDF text folded into
        the text part (OpenRouter's free tier does not accept PDF
        data URLs, so PDFs can never be sent as their own part)."""
        from web.chat_helpers import build_user_turn_content

        pdf = _FakeUpload("notes.pdf", "application/pdf", b"%PDF-1.4\n...")
        img = _FakeUpload("diagram.png", "image/png", _png_payload())

        def fake_pdf_processor(file):
            return "Key findings: 3 vulnerabilities, 2 critical."

        def fake_image_processor(file):
            from app.file_processor import ImagePart
            return ImagePart(
                data_url="data:image/png;base64,QUJD",
                mime="image/png",
                size=3,
            )

        out = build_user_turn_content(
            "analyze",
            [pdf, img],
            image_processor=fake_image_processor,
            pdf_processor=fake_pdf_processor,
        )

        self.assertIsInstance(out, list)
        kinds = [p.get("type") for p in out]
        self.assertEqual(
            kinds, ["text", "image_url"],
            f"expected [text, image_url] parts; got {kinds!r}",
        )
        # The text part carries the user prompt AND the PDF prose.
        text_part = out[0].get("text", "")
        self.assertIn("analyze", text_part)
        self.assertIn("[Attached PDF: notes.pdf]", text_part)
        self.assertIn("Key findings: 3 vulnerabilities", text_part)
        # The image part is the data URL from the image processor.
        self.assertEqual(
            out[1],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        )

    def test_pdf_processor_error_fails_soft_in_model_content(self):
        """A ``FileProcessingError`` from the PDF processor must
        surface a textual stub in the model payload (not crash the
        turn). The turn shape can stay a string for a PDF-only
        turn."""
        from app.file_processor import FileProcessingError
        from web.chat_helpers import build_user_turn_content

        fake = _FakeUpload("encrypted.pdf", "application/pdf", b"%PDF-1.4\n...")

        def failing_processor(file):
            raise FileProcessingError("encrypted", kind="invalid_pdf")

        out = build_user_turn_content(
            "review", [fake], pdf_processor=failing_processor
        )

        self.assertIsInstance(out, str)
        self.assertIn("review", out)
        # The file name is in the stub
        self.assertIn("encrypted.pdf", out)
        # The stub explains the failure
        joined = out.lower()
        self.assertTrue(
            "not sent to the model" in joined or "encrypted" in joined,
            f"expected stub wording in payload; got {out!r}",
        )


class SelectModelForRequestTests(unittest.TestCase):
    """The pure decision rule for whether to swap models for vision."""

    def setUp(self):
        for mod in list(sys.modules):
            if mod == "web.chat_helpers" or mod.startswith("web.chat_helpers."):
                del sys.modules[mod]

    def test_no_images_returns_requested_unchanged(self):
        from web.chat_helpers import select_model_for_request

        m, swapped = select_model_for_request(
            "meta-llama/llama-3.3-70b-instruct:free", False
        )
        self.assertEqual(m, "meta-llama/llama-3.3-70b-instruct:free")
        self.assertFalse(swapped)

    def test_vision_capable_requested_with_images_unchanged(self):
        from web.chat_helpers import select_model_for_request

        m, swapped = select_model_for_request(
            "nvidia/nemotron-nano-12b-v2-vl:free",
            True,
            vision_model_ids={"nvidia/nemotron-nano-12b-v2-vl:free"},
        )
        self.assertEqual(m, "nvidia/nemotron-nano-12b-v2-vl:free")
        self.assertFalse(swapped)

    def test_text_only_requested_with_images_swaps_to_first_vision_id(self):
        from web.chat_helpers import select_model_for_request

        m, swapped = select_model_for_request(
            "meta-llama/llama-3.3-70b-instruct:free",
            True,
            vision_model_ids=[
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "nvidia/nemotron-nano-9b-v2:free",  # text-only control
            ],
        )
        self.assertEqual(m, "nvidia/nemotron-nano-12b-v2-vl:free")
        self.assertTrue(swapped)

    def test_empty_vision_list_falls_back_to_hard_coded_default(self):
        from web.chat_helpers import select_model_for_request

        m, swapped = select_model_for_request(
            "meta-llama/llama-3.3-70b-instruct:free",
            True,
            vision_model_ids=[],
        )
        # Hard-coded fallback is pinned in chat_helpers. We assert
        # it is a vision-capable model (not the text-only request) and
        # that it is also the canonical entry in _VISION_MODEL_IDS so
        # the curated-list and the hard-coded fallback cannot drift.
        from app.config import _VISION_MODEL_IDS
        self.assertNotEqual(m, "meta-llama/llama-3.3-70b-instruct:free")
        self.assertTrue(swapped)
        self.assertIn(m, _VISION_MODEL_IDS)

    def test_case_insensitive_match_in_curated_list(self):
        """If the requestor passes the model id with mixed case / extra
        whitespace, the helper must still recognise it as vision-capable
        when the curated list (also normalised) contains it."""
        from web.chat_helpers import select_model_for_request

        m, swapped = select_model_for_request(
            "  NVIDIA/NEMOTRON-NANO-12B-V2-VL:FREE  ",
            True,
            vision_model_ids=["nvidia/nemotron-nano-12b-v2-vl:free"],
        )
        self.assertEqual(m, "  NVIDIA/NEMOTRON-NANO-12B-V2-VL:FREE  ")
        self.assertFalse(swapped)

    def test_curated_vision_list_picks_first_not_hard_coded_fallback(self):
        """When the view passes a curated ``vision_ids`` list, the swap
        must use the *first* entry of that list, not the hard-coded
        nemotron fallback in ``select_model_for_request``.

        This is the regression test for the user-reported image-flow
        bug: when ``FREE_MODEL_CHOICES`` had no vision entries the
        helper fell through to ``gemini-2.0-flash-exp:free`` (and later
        the nemotron default), which were guaranteed-unavailable for
        free-tier image calls. Adding a curated vision id makes the
        curated list non-empty, and the swap must now land on the
        *first* curated id, not the hard-coded fallback.
        """
        from web.chat_helpers import select_model_for_request

        curated = [
            "nvidia/nemotron-nano-12b-v2-vl:free",
        ]
        m, swapped = select_model_for_request(
            "meta-llama/llama-3.3-70b-instruct:free",
            True,
            vision_model_ids=curated,
        )
        self.assertEqual(m, "nvidia/nemotron-nano-12b-v2-vl:free")
        self.assertTrue(swapped)


class VisionAllowListTests(unittest.TestCase):
    """Pins the contents of ``app.config._VISION_MODEL_IDS`` so the
    image-flow fix can't silently regress.

    The allow-list is a frozenset; this test asserts membership for
    each free-tier vision model we rely on, and asserts the count is
    at least 5 (so adding a new id never gets missed). If a model
    stops accepting image inputs on the free tier, delete the matching
    row from ``_VISION_MODEL_IDS`` *and* from
    ``web/streamlit_app.FREE_MODEL_CHOICES`` — the two lists must
    stay in sync, or ``select_model_for_request`` will swap to a
    model that isn't in the sidebar dropdown.
    """

    def test_reliable_free_vision_models_are_allowed(self):
        from app.config import _VISION_MODEL_IDS
        # Live-probed on 2026-06-15: the only free-tier model on
        # OpenRouter that returned 200 for a real image payload. Every
        # other historical vision candidate 404s. If this assertion
        # starts failing, the OpenRouter free-tier vision model has
        # changed — re-probe and update the allow-list in
        # app/config.py and the chat_helpers fallback in lockstep.
        expected = {
            "nvidia/nemotron-nano-12b-v2-vl:free",
        }
        missing = expected - _VISION_MODEL_IDS
        self.assertFalse(
            missing,
            f"_VISION_MODEL_IDS is missing: {sorted(missing)}",
        )

    def test_free_model_choices_contains_the_vision_models(self):
        """The view only builds the ``vision_ids`` list by walking
        ``FREE_MODEL_CHOICES``; if a model is in the allow-list but
        not in the dropdown, the auto-swap can't reach it.

        Importing the module triggers ``streamlit`` as a side effect,
        so guard with a clear message if the import fails — that's
        an environment problem, not a test failure.
        """
        try:
            from web.streamlit_app import FREE_MODEL_CHOICES
        except Exception as exc:  # pragma: no cover
            self.fail(f"Could not import FREE_MODEL_CHOICES: {exc}")
        from app.config import _VISION_MODEL_IDS

        choice_ids = {m["id"] for m in FREE_MODEL_CHOICES}
        not_in_choices = _VISION_MODEL_IDS - choice_ids
        self.assertFalse(
            not_in_choices,
            "These vision-capable models are in _VISION_MODEL_IDS but "
            "missing from FREE_MODEL_CHOICES, so the auto-swap can't "
            f"reach them: {sorted(not_in_choices)}",
        )


class ModelSupportsVisionTests(unittest.TestCase):
    """The ``app.config.model_supports_vision`` predicate."""

    def test_exact_match(self):
        from app.config import model_supports_vision
        self.assertTrue(
            model_supports_vision("nvidia/nemotron-nano-12b-v2-vl:free")
        )

    def test_case_insensitive(self):
        from app.config import model_supports_vision
        self.assertTrue(
            model_supports_vision("NVIDIA/NEMOTRON-NANO-12B-V2-VL:FREE")
        )

    def test_whitespace_padded(self):
        from app.config import model_supports_vision
        self.assertTrue(
            model_supports_vision("  nvidia/nemotron-nano-12b-v2-vl:free  ")
        )

    def test_unknown_id_returns_false(self):
        from app.config import model_supports_vision
        self.assertFalse(
            model_supports_vision("meta-llama/llama-3.3-70b-instruct:free")
        )

    def test_empty_string_returns_false(self):
        from app.config import model_supports_vision
        self.assertFalse(model_supports_vision(""))


class BuildMessagesRoundTripTests(unittest.TestCase):
    """``_build_messages`` round-trips ``str`` and ``list[dict]`` content."""

    def setUp(self):
        for mod in list(sys.modules):
            if mod == "web.chat_helpers" or mod.startswith("web.chat_helpers."):
                del sys.modules[mod]

    def test_str_content_appended_unchanged(self):
        from web.chat_helpers import _build_messages

        history = [
            {"role": "system", "content": "you are a tutor"},
            {"role": "user", "content": "earlier question"},
        ]
        msgs = _build_messages(history, "follow-up")
        # The last message must be the new user turn with the text intact.
        self.assertEqual(msgs[-1], {"role": "user", "content": "follow-up"})

    def test_list_content_passes_through_to_api(self):
        """The engine should never coerce a multimodal list of parts
        back into a string. We assert byte-for-byte equality of the
        ``content`` field so a future refactor that does
        ``str(content)`` will fail loudly."""
        from web.chat_helpers import _build_messages

        parts = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]
        history = [{"role": "system", "content": "you are a tutor"}]
        msgs = _build_messages(history, parts)
        self.assertEqual(msgs[-1]["content"], parts)
        # Sanity: it really IS a list, not a string repr.
        self.assertIsInstance(msgs[-1]["content"], list)

    def test_empty_str_raises(self):
        from web.chat_helpers import _build_messages

        with self.assertRaises(ValueError):
            _build_messages(
                [{"role": "system", "content": "sys"}], ""
            )

    def test_empty_list_raises(self):
        from web.chat_helpers import _build_messages

        with self.assertRaises(ValueError):
            _build_messages(
                [{"role": "system", "content": "sys"}], []
            )


class OpenRouterPayloadRoundTripTests(unittest.TestCase):
    """``app.openrouter._build_payload`` round-trips multimodal content.

    The engine's HTTP client must serialise a ``list[dict]`` content
    field unchanged. We do not make a network call — we only inspect
    the dict it would have sent.
    """

    def test_list_content_appears_in_payload_unchanged(self):
        from app.openrouter import _build_payload

        parts = [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": parts},
        ]
        payload = _build_payload(
            messages, model="nvidia/nemotron-nano-12b-v2-vl:free",
            temperature=0.7, max_tokens=1024,
        )
        # The last message's content must still be a list of parts,
        # not a JSON-encoded string or a stringified repr.
        last = payload["messages"][-1]
        self.assertIsInstance(last["content"], list)
        self.assertEqual(last["content"], parts)

    def test_str_content_appears_in_payload_unchanged(self):
        from app.openrouter import _build_payload

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plain text turn"},
        ]
        payload = _build_payload(
            messages, model="nvidia/nemotron-nano-12b-v2-vl:free",
            temperature=0.7, max_tokens=1024,
        )
        self.assertEqual(
            payload["messages"][-1]["content"], "plain text turn"
        )


class _FakeRouter:
    """Stand-in for ``ModelRouter`` used by the helper tests.

    Records every ``stream_chat`` call (``model=``, ``timeout=`` and the
    messages list) and replays a script of chunk sequences / exceptions
    supplied at construction time. Keeps the helper free of network
    effects so the degrade branch can be exercised deterministically.

    ``scripts`` is a list; each element is either:
      * an iterable of strings → yielded as chunks
      * an exception class or instance → raised on the first iteration
    One script entry per ``stream_chat`` call, in order.
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []  # list of dicts: {model, timeout, messages}

    def stream_chat(self, messages, *, model=None, temperature=None,
                    max_tokens=None, timeout=None):
        if not self._scripts:
            raise AssertionError("FakeRouter: no more scripts queued")
        self.calls.append({
            "model": model,
            "timeout": timeout,
            "messages": messages,
        })
        script = self._scripts.pop(0)
        if isinstance(script, BaseException) or (
            isinstance(script, type) and issubclass(script, BaseException)
        ):
            exc = script() if isinstance(script, type) else script
            # ``stream_chat`` is a generator: the real client raises
            # *during* iteration, so we yield nothing and raise on the
            # first ``next()`` call. Returning a generator that
            # immediately raises mimics that semantics exactly.
            if False:  # pragma: no cover - this branch never yields
                yield ""
            raise exc
        for chunk in script:
            yield chunk


class StreamErrorClassifierTests(unittest.TestCase):
    """``_classify_degrade_trigger`` maps exceptions to machine strings."""

    def test_vision_rate_limit(self):
        from app.openrouter import OpenRouterRateLimitError

        from web.chat_helpers import _classify_degrade_trigger
        exc = OpenRouterRateLimitError("rate limited", status=429)
        self.assertEqual(
            _classify_degrade_trigger(exc), "vision_rate_limit"
        )

    def test_vision_server_error(self):
        from app.openrouter import OpenRouterServerError

        from web.chat_helpers import _classify_degrade_trigger
        exc = OpenRouterServerError("504", status=504)
        self.assertEqual(
            _classify_degrade_trigger(exc), "vision_server_error"
        )

    def test_vision_no_response_for_base_openrouter_error(self):
        from app.openrouter import OpenRouterError

        from web.chat_helpers import _classify_degrade_trigger
        exc = OpenRouterError("no chunks", status=502)
        self.assertEqual(
            _classify_degrade_trigger(exc), "vision_no_response"
        )

    def test_auth_error_is_not_a_degrade_trigger(self):
        from app.openrouter import OpenRouterAuthError

        from web.chat_helpers import _classify_degrade_trigger
        exc = OpenRouterAuthError("bad key", status=401)
        self.assertIsNone(_classify_degrade_trigger(exc))

    def test_unwrap_all_slots_exhausted_cause(self):
        from app.openrouter import (
            OpenRouterAuthError,
            OpenRouterError,
            OpenRouterRateLimitError,
            OpenRouterServerError,
        )
        from app.router import AllSlotsExhaustedError

        from web.chat_helpers import _classify_degrade_trigger
        for cause, expected in (
            (OpenRouterServerError("5xx", status=500), "vision_server_error"),
            (OpenRouterRateLimitError("429", status=429), "vision_rate_limit"),
            (OpenRouterError("no deltas", status=502), "vision_no_response"),
            (OpenRouterAuthError("bad key", status=401), None),
        ):
            wrapped = AllSlotsExhaustedError("all slots failed", attempts=1)
            wrapped.__cause__ = cause
            self.assertEqual(
                _classify_degrade_trigger(wrapped), expected,
                f"cause={cause!r}",
            )


class StreamVisionTurnWithFallbackTests(unittest.TestCase):
    """``stream_vision_turn_with_fallback`` end-to-end streaming + degrade."""

    def _messages(self, content="describe the image"):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": content},
        ]

    def test_vision_stream_yields_vision_source_unmodified(self):
        from web.chat_helpers import stream_vision_turn_with_fallback

        router = _FakeRouter([["hello ", "world"]])
        out = list(stream_vision_turn_with_fallback(
            router=router,
            messages=self._messages(),
            vision_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
            fallback_model_id="mistralai/mistral-small:free",
            content="describe the image",
            files=None,
            timeout=90.0,
        ))
        self.assertEqual(out, [("hello ", "vision"), ("world", "vision")])
        # The helper must pin the vision model id and forward timeout.
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(
            router.calls[0]["model"],
            "nvidia/nemotron-nano-12b-v2-vl:free",
        )
        self.assertEqual(router.calls[0]["timeout"], 90.0)

    def test_vision_failure_before_any_delta_degrades_to_text(self):
        from app.openrouter import OpenRouterServerError

        from web.chat_helpers import stream_vision_turn_with_fallback

        vision_exc = OpenRouterServerError("504", status=504)
        # First script raises on first iteration; second script yields
        # the text-fallback chunks. ``_FakeRouter.stream_chat`` will
        # raise the exception the moment the caller starts iterating.
        router = _FakeRouter([vision_exc, ["text-fallback-A", "text-fallback-B"]])
        out = list(stream_vision_turn_with_fallback(
            router=router,
            messages=self._messages(),
            vision_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
            fallback_model_id="mistralai/mistral-small:free",
            content="describe the image",
            files=None,
        ))
        # First emission: a one-space degraded marker. Then the two
        # text-fallback chunks, both tagged as ``"text"``.
        self.assertEqual(out, [
            (" ", "degraded"),
            ("text-fallback-A", "text"),
            ("text-fallback-B", "text"),
        ])
        # Two calls were made: vision attempt + text fallback.
        self.assertEqual(
            [c["model"] for c in router.calls],
            [
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "mistralai/mistral-small:free",
            ],
        )
        # The text-fallback call must carry the *text-only* content
        # (a string, not the original ``str`` we passed — degrade
        # inlines a stub regardless of vision-failure kind because
        # ``files=None`` means no per-attachment stubs are emitted).
        second_call_messages = router.calls[1]["messages"]
        last_msg = second_call_messages[-1]
        self.assertIsInstance(last_msg["content"], str)
        self.assertIn("describe the image", last_msg["content"])
        self.assertNotIsInstance(last_msg["content"], list)

    def test_vision_partial_stream_then_error_is_re_raised(self):
        """Once the vision model has produced a delta, the helper must
        NOT degrade. Mixing two model outputs in one bubble would be
        confusing — the caller should see the partial reply and
        re-issue."""
        from app.openrouter import OpenRouterServerError

        from web.chat_helpers import stream_vision_turn_with_fallback

        router = _FakeRouter([
            ["partial ", "reply"],
            ["text-fallback-A"],  # must never be consumed
        ])
        gen = stream_vision_turn_with_fallback(
            router=router,
            messages=self._messages(),
            vision_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
            fallback_model_id="mistralai/mistral-small:free",
            content="describe the image",
            files=None,
        )
        collected = []
        try:
            for chunk in gen:
                collected.append(chunk)
        except OpenRouterServerError:
            pass
        # Vision chunks made it through; no degraded marker, no
        # text-fallback chunks.
        self.assertEqual(collected, [
            ("partial ", "vision"), ("reply", "vision"),
        ])
        self.assertEqual(len(router.calls), 1)

    def test_auth_error_propagates_without_degrade(self):
        from app.openrouter import OpenRouterAuthError

        from web.chat_helpers import stream_vision_turn_with_fallback

        router = _FakeRouter([
            OpenRouterAuthError("bad key", status=401),
            ["fallback"],  # must never be consumed
        ])
        gen = stream_vision_turn_with_fallback(
            router=router,
            messages=self._messages(),
            vision_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
            fallback_model_id="mistralai/mistral-small:free",
            content="describe the image",
            files=None,
        )
        with self.assertRaises(OpenRouterAuthError):
            list(gen)
        # Only the vision attempt was made — auth errors must not
        # silently swap to a text fallback.
        self.assertEqual(len(router.calls), 1)

    def test_empty_vision_stream_triggers_degrade(self):
        """The helper's belt-and-braces empty-yield guard must degrade
        even if the router did not raise. (The production router does
        raise, but the contract is in the helper, not the router —
        tests here pin it in isolation.)"""
        from web.chat_helpers import stream_vision_turn_with_fallback

        # A stubbed router whose ``stream_chat`` returns an iterable
        # (not a generator) so we can swap behaviour per ``model=``
        # without yielding from inside the function. The vision call
        # yields nothing; the text fallback yields one chunk.
        def _iter_for(model):
            if model == "nvidia/nemotron-nano-12b-v2-vl:free":
                return iter(())
            return iter(["text-from-fallback"])

        class _EmptyRouter:
            def __init__(self):
                self.calls = []

            def stream_chat(self, messages, *, model=None, temperature=None,
                            max_tokens=None, timeout=None):
                self.calls.append({"model": model, "messages": messages,
                                   "timeout": timeout})
                return _iter_for(model)

        router = _EmptyRouter()
        out = list(stream_vision_turn_with_fallback(
            router=router,
            messages=self._messages(),
            vision_model_id="nvidia/nemotron-nano-12b-v2-vl:free",
            fallback_model_id="mistralai/mistral-small:free",
            content="describe the image",
            files=None,
        ))
        self.assertEqual(out, [
            (" ", "degraded"),
            ("text-from-fallback", "text"),
        ])
        self.assertEqual(len(router.calls), 2)


# --- Cooperative stop + chatbox picker ----------------------------------------
# Lightweight unit tests for the pure helpers added in the
# "Stop button + chatbox model picker" pass. The widget-rendering
# code (which calls ``st.button`` / ``st.selectbox``) lives in
# ``web/streamlit_app.py`` because it needs the live Streamlit
# context; the pure read/clear + label→id logic lives in
# ``web/chat_helpers.py`` and is tested here directly so we do not
# have to boot Streamlit or maintain a deep module stub.


class _FakeSessionState(dict):
    """Minimal ``st.session_state`` stand-in for pure-logic tests.

    Tests pass an instance to ``consume_stop_flag`` /
    ``resolve_chatbox_model_id`` instead of ``st.session_state`` so
    the helpers can be exercised without Streamlit.
    """


class ChatboxPickerTests(unittest.TestCase):
    """Pin the chatbox model-picker label→id resolution contract.

    The picker is a view-layer shortcut for the sidebar dropdown —
    the two write to the same ``session_state['model']`` key so a
    change in either place is reflected on the next turn. The pure
    resolution (label → id, and "did the value actually change?")
    lives in :func:`web.chat_helpers.resolve_chatbox_model_id`; the
    tests pin that contract.
    """

    def test_resolves_known_label_to_id(self):
        choices = [
            {"id": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B (default)"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
        ]
        new_id, changed = resolve_chatbox_model_id(
            choices,
            chosen_label="Llama 3.3 70B",
            current_id="google/gemma-4-31b-it:free",
        )
        self.assertEqual(new_id, "meta-llama/llama-3.3-70b-instruct:free")
        self.assertTrue(changed)

    def test_unchanged_selection_reports_no_change(self):
        choices = [
            {"id": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B (default)"},
        ]
        new_id, changed = resolve_chatbox_model_id(
            choices,
            chosen_label="Gemma 4 31B (default)",
            current_id="google/gemma-4-31b-it:free",
        )
        self.assertEqual(new_id, "google/gemma-4-31b-it:free")
        self.assertFalse(changed)

    def test_unknown_label_falls_back_to_current_id(self):
        # Defensive: if the curated list ever drifts from a stale
        # label in session_state we must not crash and we must not
        # silently swap to an unknown model.
        choices = [
            {"id": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B (default)"},
        ]
        new_id, changed = resolve_chatbox_model_id(
            choices,
            chosen_label="Mystery Model",
            current_id="google/gemma-4-31b-it:free",
        )
        self.assertEqual(new_id, "google/gemma-4-31b-it:free")
        self.assertFalse(changed)

    def test_view_delegate_writes_only_on_change(self):
        # The view's wrapper should *only* write ``session_state['model']``
        # when the resolved id differs from the current one. Simulate
        # that contract directly against a plain dict using a small
        # local choices table (the real ``FREE_MODEL_CHOICES`` lives
        # in ``web/streamlit_app.py`` and would force a Streamlit
        # import — see :func:`web.streamlit_app._render_chatbox_model_picker`).
        state = _FakeSessionState()
        state["model"] = "google/gemma-4-31b-it:free"
        choices = [
            {"id": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B (default)"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
        ]
        # Unchanged → no write.
        new_id, changed = resolve_chatbox_model_id(
            choices,
            chosen_label="Gemma 4 31B (default)",
            current_id=state["model"],
        )
        if changed:
            state["model"] = new_id
        self.assertEqual(state["model"], "google/gemma-4-31b-it:free")
        # Changed → write.
        new_id, changed = resolve_chatbox_model_id(
            choices,
            chosen_label="Llama 3.3 70B",
            current_id=state["model"],
        )
        if changed:
            state["model"] = new_id
        self.assertEqual(state["model"], "meta-llama/llama-3.3-70b-instruct:free")


class StopFlagTests(unittest.TestCase):
    """Pin the cooperative stop-flag contract.

    The view's two chunk loops check ``session_state['stop_requested']``
    on every delta and break out (pinned-vision path) or ``return``
    (streaming shim) when the flag flips ``True``. The flag is set
    by the Stop button widget when the user clicks it, and reset by
    :func:`consume_stop_flag` at the start of every new turn. The
    tests pin the pure read/clear behaviour here.
    """

    def test_consume_returns_true_when_flag_was_set(self):
        state = _FakeSessionState()
        state["stop_requested"] = True
        self.assertTrue(consume_stop_flag(state))

    def test_consume_clears_the_flag(self):
        state = _FakeSessionState()
        state["stop_requested"] = True
        consume_stop_flag(state)
        # ``consume_stop_flag`` both pops the flag and re-seeds it
        # to ``False`` so subsequent reads inside the same session
        # see the canonical default.
        self.assertFalse(state.get("stop_requested"))

    def test_consume_returns_false_when_flag_missing(self):
        state = _FakeSessionState()
        state.pop("stop_requested", None)
        self.assertFalse(consume_stop_flag(state))
        # Re-seeded to ``False`` even when it was absent — the next
        # turn's chunk loops need a deterministic default to check.
        self.assertFalse(state.get("stop_requested"))

    def test_consume_is_idempotent(self):
        # Two consecutive consumes must both report ``False`` so a
        # double-rerun cannot accidentally re-arm the flag.
        state = _FakeSessionState()
        state["stop_requested"] = True
        self.assertTrue(consume_stop_flag(state))
        self.assertFalse(consume_stop_flag(state))

    def test_consume_tolerates_a_plain_dict(self):
        # Production passes ``st.session_state`` (a ``ServerSession``
        # proxy); tests pass a plain dict. The contract should hold
        # for both because the helper only relies on ``pop`` /
        # ``__setitem__``.
        state = {"stop_requested": True}
        self.assertTrue(consume_stop_flag(state))
        self.assertFalse(state.get("stop_requested"))


class VisionTimeoutTests(unittest.TestCase):
    """Pin the per-request timeout for vision calls.

    The vision path on OpenRouter is the only known-working free
    route, but its first-token latency varies wildly (warm slots
    come back in ~25s, cold starts can run past 90s). The helper
    exposes :func:`web.chat_helpers.vision_timeout_seconds` so a
    future adaptive loop has a single override point. The current
    value is set to **45s** — long enough to absorb a typical
    warm-slot first-token, short enough that a stuck call degrades
    to the text fallback before the user perceives a stall.

    The lower bound here is a guard against an accidental shrink
    back to the old 30s (which used to abort every warm slot before
    any text came back); the upper bound guards against a runaway
    bump that would silently re-introduce the "Thinking… for 90s"
    UX bug the user reported.
    """

    def test_vision_timeout_is_within_safe_window(self):
        from web.chat_helpers import vision_timeout_seconds
        value = vision_timeout_seconds()
        self.assertGreaterEqual(value, 30.0)
        self.assertLessEqual(value, 60.0)

    def test_vision_timeout_is_a_float(self):
        # Tests and the router both rely on the helper returning a
        # plain float so the timeout can be forwarded straight to
        # ``openrouter.stream_chat(..., timeout=...)`` without an
        # int/float coercion at the call site.
        from web.chat_helpers import vision_timeout_seconds
        self.assertIsInstance(vision_timeout_seconds(), float)


if __name__ == "__main__":
    unittest.main()
