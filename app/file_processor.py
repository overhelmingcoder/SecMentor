"""File processing for chat attachments (Phase 11).

The chat-completions endpoint that ``app.openrouter`` talks to accepts
text and image inputs (as data-URL parts in the user message). This
module is the single place that knows how to turn a raw uploaded file
into the wire shape the engine expects.

Layering
--------

* ``app/openrouter.py``  — pure HTTP. Accepts ``messages`` as a list of
  dicts and serialises the ``content`` field verbatim into JSON. It has
  no knowledge of MIME types or base64.
* ``web/chat_helpers.py`` — turns the user's upload list into either a
  plain string (text-only / PDF-as-text) or a list of OpenRouter parts
  (text + ``image_url``). The view consumes this.
* ``app/file_processor.py`` (this file) — the *one* place that knows
  what an "image" or a "PDF" actually is, and the *one* place that
  imports a third-party library to do real work (``pymupdf``). Keeping
  it isolated here means the rest of the engine has no ``pymupdf``
  import to fail on, and tests can monkey-patch ``process_pdf`` /
  ``process_image`` to avoid the real library.

Why the laziness matters
------------------------

``pymupdf`` is a native-extension wheel that ships with a small C
library. On systems where the wheel is not available (e.g. an exotic
Linux distro) the import would fail and break the entire app even for
text-only chats. We import ``pymupdf`` lazily inside ``process_pdf``
so the rest of the app keeps starting and the error only surfaces when
the user actually attaches a PDF.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from typing import Any, Protocol


# --- Constants ----------------------------------------------------------------
# Hard upper bound on the *decoded* size of a single image, after the
# data-URL wrap. OpenRouter's free tier has a per-request payload cap
# (8 MB by default across the providers we route to). 4 MB leaves
# headroom for the rest of the request (system prompt + history).
_MAX_IMAGE_BYTES: int = 4 * 1024 * 1024

# Hard upper bound on the *extracted* text from a single PDF. A 1000-
# page PDF full of dense prose would otherwise blow up the context
# window of even the 262K Gemma 4 model. 200 KB is roughly 50K words,
# which is more than enough for any realistic security-tutor question.
_MAX_PDF_TEXT_CHARS: int = 200_000

# Image MIME types we know how to base64-wrap. Other image/* types
# (e.g. image/avif) are treated as unsupported by ``process_image``
# and raise ``FileProcessingError(kind="unsupported_image")`` so the
# helper layer can fall back to a textual stub.
_SUPPORTED_IMAGE_MIMES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",  # rare but valid; OpenRouter normalises to image/jpeg
    "image/gif",
    "image/webp",
})


# --- Errors -------------------------------------------------------------------
class FileProcessingError(Exception):
    """Raised when an upload cannot be turned into a wire-shape part.

    ``kind`` is a short machine-readable tag the helper layer can
    branch on without parsing the human-readable message:

    * ``"empty"``                — the file had zero bytes
    * ``"oversized"``            — the file exceeds the size cap
    * ``"unsupported_image"``    — image/* MIME not in the allow-list
    * ``"invalid_image_b64"``    — base64 round-trip failed (corrupt)
    * ``"invalid_pdf"``          — pymupdf refused the bytes
    * ``"encrypted_pdf"``        — PDF requires a password
    * ``"pdf_text_extraction"``  — pymupdf raised something else
    * ``"no_vision_model"``      — caller asked for an image but no
                                    vision-capable model is configured
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


# --- Public dataclass ---------------------------------------------------------
@dataclass(frozen=True)
class ImagePart:
    """A single image ready to be sent as an OpenRouter part.

    ``data_url`` is the ``data:image/<mime>;base64,<...>`` string the
    chat-completions endpoint expects in an ``{"type": "image_url",
    "image_url": {"url": <data_url>}}`` part.
    """

    data_url: str
    mime: str
    size: int  # original raw byte length, for the stub header


# --- Public protocol ---------------------------------------------------------
class _UploadedFileLike(Protocol):
    """Tiny subset of the Streamlit ``UploadedFile`` surface we use.

    Kept here (and again in ``web/chat_helpers``) so the processor has
    no Streamlit import. The Protocol is structural — anything with the
    four attributes is accepted.
    """

    name: str
    type: str | None
    size: int

    def read(self) -> bytes: ...


# --- Image handling ----------------------------------------------------------
def process_image(file: _UploadedFileLike) -> ImagePart:
    """Turn an uploaded image file into an OpenRouter ``image_url`` part.

    Reads the full payload (image attachments are typically under 1 MB
    in the browser picker; the 4 MB cap above is a safety net, not a
    target), validates the MIME type, base64-encodes it, and returns
    the data-URL string the engine wants.

    Raises :class:`FileProcessingError` for empty / oversized /
    unsupported / corrupt inputs. The caller is expected to catch and
    downgrade to a textual stub so the user still gets a useful prompt
    when an image is broken.
    """
    # 1) Read & basic validation. The Streamlit UploadedFile's ``read()``
    #    may have been advanced by an earlier sniff in the helper; we
    #    rewind defensively so a calling order change can't silently
    #    produce empty payloads.
    if hasattr(file, "seek"):
        try:
            file.seek(0)  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass
    raw = file.read() if hasattr(file, "read") else b""
    if not raw:
        raise FileProcessingError(
            "Uploaded image is empty.", kind="empty",
        )
    if len(raw) > _MAX_IMAGE_BYTES:
        raise FileProcessingError(
            f"Image is {len(raw)} bytes; the cap is {_MAX_IMAGE_BYTES}.",
            kind="oversized",
        )

    # 2) MIME gate. The browser sometimes gives us a generic
    #    ``application/octet-stream`` even for a real PNG; we accept
    #    the union of what the browser said and what the extension
    #    implies so a mislabelled image still goes through.
    mime = (getattr(file, "type", None) or "").lower()
    if mime not in _SUPPORTED_IMAGE_MIMES:
        name = (getattr(file, "name", "") or "").lower()
        if name.endswith(".png"):
            mime = "image/png"
        elif name.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif name.endswith(".gif"):
            mime = "image/gif"
        elif name.endswith(".webp"):
            mime = "image/webp"
        else:
            raise FileProcessingError(
                f"Image MIME {mime!r} is not in the supported set "
                f"({sorted(_SUPPORTED_IMAGE_MIMES)}).",
                kind="unsupported_image",
            )

    # 3) Base64 wrap. We use ``base64.b64encode`` directly (not
    #    ``base64.encodebytes`` which adds newlines). The data-URL
    #    form is the one the OpenRouter examples use, and it is
    #    identical to what the OpenAI chat-completions endpoint
    #    expects for vision inputs.
    try:
        encoded = base64.b64encode(raw, altchars=None)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise FileProcessingError(
            f"Failed to base64-encode the image: {exc}",
            kind="invalid_image_b64",
        ) from exc

    return ImagePart(
        data_url=f"data:{mime};base64,{encoded.decode('ascii')}",
        mime=mime,
        size=len(raw),
    )


# --- PDF handling -------------------------------------------------------------
def process_pdf(file: _UploadedFileLike) -> str:
    """Extract plain text from an uploaded PDF.

    The extracted text is what gets sent to the model — we deliberately
    do NOT base64-attach the PDF. Reasons:

    * OpenRouter's free-tier providers do not all accept PDF inputs
      as data URLs (most accept images only).
    * The model's value-add on a PDF is the *prose* in it, not the
      byte stream. Extracting the text keeps the prompt small and
      keeps the conversation reproducible across models.
    * A text-only result slots into the existing string ``content``
      shape, so a single PDF attachment costs zero engine changes.

    Truncation kicks in at :data:`_MAX_PDF_TEXT_CHARS` characters and
    a ``[truncated]`` marker is appended so the model knows the input
    was clipped. This is the same contract as
    ``web/chat_helpers._MAX_INLINE_BYTES`` for the textual snippet
    path.
    """
    raw = file.read() if hasattr(file, "read") else b""
    if not raw:
        raise FileProcessingError(
            "Uploaded PDF is empty.", kind="empty",
        )

    # Lazy import so a missing or broken pymupdf wheel only breaks PDF
    # uploads — not the whole app. We surface a typed error so the
    # helper layer can fall back to a stub.
    try:
        import pymupdf  # type: ignore[import-untyped]
    except (ImportError, OSError) as exc:
        raise FileProcessingError(
            f"PDF support requires pymupdf but it failed to import: {exc}",
            kind="pdf_text_extraction",
        ) from exc

    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except pymupdf.FileDataError as exc:  # type: ignore[attr-defined]
        raise FileProcessingError(
            f"PDF is not a valid PDF document: {exc}",
            kind="invalid_pdf",
        ) from exc
    except pymupdf.PasswordError as exc:  # type: ignore[attr-defined]
        raise FileProcessingError(
            "PDF is password-protected; the chatbot cannot read "
            "encrypted PDFs. Remove the password and re-upload.",
            kind="encrypted_pdf",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise FileProcessingError(
            f"pymupdf refused the PDF: {exc}",
            kind="pdf_text_extraction",
        ) from exc

    # pymupdf's get_text() is robust on mixed-content PDFs. We do not
    # sort by position here (a more advanced layout-preserving mode);
    # the chat-completions model is happy with per-page blocks joined
    # by blank lines.
    pieces: list[str] = []
    total_chars = 0
    truncated = False
    try:
        for page in doc:
            text = page.get_text()  # type: ignore[attr-defined]
            if not text:
                continue
            if total_chars + len(text) > _MAX_PDF_TEXT_CHARS:
                remaining = max(0, _MAX_PDF_TEXT_CHARS - total_chars)
                pieces.append(text[:remaining])
                truncated = True
                break
            pieces.append(text)
            total_chars += len(text)
    finally:
        doc.close()

    if not pieces:
        # A PDF that has zero extractable text is either a scanned
        # image or a real PDF with no text layer. Either way, the
        # text-only pipeline cannot help — raise so the helper can
        # fall back to a stub that tells the user what happened.
        raise FileProcessingError(
            "PDF contains no extractable text (likely a scan).",
            kind="pdf_text_extraction",
        )

    joined = "\n\n".join(pieces)
    if truncated:
        joined += (
            f"\n\n... [truncated, PDF text exceeded "
            f"{_MAX_PDF_TEXT_CHARS} characters]"
        )
    return joined


# --- Internal: a serialisable part list --------------------------------------
# Used by the tests in tests/test_files.py so we can assert the exact
# JSON shape the engine will see. The shape mirrors the OpenRouter
# example for vision inputs.
def image_url_part(image: ImagePart) -> dict[str, Any]:
    """Wrap an :class:`ImagePart` in the OpenRouter ``image_url`` shape.

    The returned dict is what the helper layer splices into the user
    message's ``content`` list when an image is present:

        {"type": "image_url", "image_url": {"url": <data_url>}}
    """
    return {
        "type": "image_url",
        "image_url": {"url": image.data_url},
    }


# --- Internal: streamlit UploadedFile .size quirk ----------------------------
# The Streamlit ``UploadedFile.size`` is a property that is set when
# the file is first read. When we get a duck-typed test double without
# a real size, we still want ``process_image`` to work. This helper
# normalises that. It is intentionally a one-liner so the test code
# can either set ``size`` correctly or rely on us reading the bytes.
def _safe_size(file: _UploadedFileLike, fallback: int) -> int:
    raw = getattr(file, "size", None)
    if isinstance(raw, int) and raw >= 0:
        return raw
    return fallback


# Exposed so the test file can assert the size normalisation
# behaviour without us forcing a particular Streamlit version.
__all__ = [
    "FileProcessingError",
    "ImagePart",
    "_MAX_IMAGE_BYTES",
    "_MAX_PDF_TEXT_CHARS",
    "_SUPPORTED_IMAGE_MIMES",
    "image_url_part",
    "process_image",
    "process_pdf",
    "_safe_size",
]
