"""Pure helpers for the Streamlit chat UI (was Phase 6, settled in Phase 7).

The Streamlit file (`streamlit_app.py`) is mostly view code. The functions
that actually shape state — building the OpenRouter message list, capping
history length, serializing a transcript for download — live here as pure
functions so they can be unit-tested without spinning up a browser.

This mirrors the engine/interface split from the rest of the project:
`app/` is the engine, `web/chat_helpers.py` is the *UI logic* that sits
between the engine and the view, and `streamlit_app.py` is the view.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Mapping, Protocol

from app import prompts as _prompts
from app.file_processor import FileProcessingError


# --- Uploaded-file shape ------------------------------------------------------
# Streamlit's `st.chat_input(accept_file=...)` returns a `ChatInputValue`
# whose `.files` attribute is a list of `UploadedFile` objects. Each one
# exposes `.name`, `.type` (MIME), `.size` (bytes), and `.read()`.
#
# We only need a tiny subset of that surface, so a Protocol keeps the helper
# unit-testable without importing `streamlit.runtime` (which would otherwise
# pull the full Streamlit stack into the test process and slow CI).

class _UploadedFileLike(Protocol):
    """Minimal interface the file-uploads helper relies on.

    Matches `streamlit.runtime.uploaded_file_manager.UploadedFile`. Tests
    can pass any object with these four attributes — no Streamlit import
    needed.
    """

    name: str
    type: str | None
    size: int
    # `read()` is the only callable we need; declared as a method-shaped
    # attribute so the Protocol can be satisfied by a class with that
    # method, or by a simple test double that exposes `read` as a bound
    # function.
    def read(self) -> bytes: ...


# A single chat message is the same shape the OpenRouter client expects.
# The system prompt is just a message with role "system", so it is not a
# special case in this helper — callers just pass it in `history`.
ChatMessage = dict[str, str]


# Hard upper bound on how many non-system messages we keep in session.
# Gemma 4 31B IT has a 262K context window, so technically we could keep
# hundreds of turns. We cap low (12) because free-tier models are slow on
# long contexts, and a short history keeps first-token latency low. The
# UI exposes this as a slider so power users can raise it if they want.
DEFAULT_MAX_HISTORY_MESSAGES: int = 12


def _build_messages(
    history: list[ChatMessage],
    user_input: str | list[dict[str, object]],
) -> list[ChatMessage]:
    """Return a fresh messages list ready to send to OpenRouter.

    Rules:
    - The first message must be the system prompt. The caller is responsible
      for making sure `history[0]` is the system prompt (the UI seeds the
      session state this way on a new chat).
    - The new user message is appended at the end with role "user".
    - ``user_input`` may be a plain ``str`` (text-only turn) or a
      ``list[dict[str, object]]`` of OpenRouter content parts (multimodal
      turn with one or more ``image_url`` parts). The list shape is passed
      through unchanged.
    - The returned list is a *new* list; the caller's `history` is not
      mutated. The UI is expected to extend `history` after a successful
      model reply.
    """
    if not history:
        raise ValueError("history must contain at least the system prompt.")
    if isinstance(user_input, str):
        if not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")
    elif not user_input:
        # Empty list of content parts — also a non-event.
        raise ValueError("user_input must be a non-empty string or list.")
    return [*history, {"role": "user", "content": user_input}]


def _truncate_history(
    history: list[ChatMessage],
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
) -> list[ChatMessage]:
    """Return a copy of `history` capped at `max_messages` non-system turns.

    The system prompt (the first message, if its role is "system") is
    always preserved. Older user/assistant turns are dropped first.
    """
    if not history:
        return []
    if max_messages < 1:
        raise ValueError("max_messages must be >= 1.")

    system = history[0] if history[0].get("role") == "system" else None
    turns = history[1:] if system is not None else list(history)

    if len(turns) <= max_messages:
        return list(history)

    kept = turns[-max_messages:]
    return ([system] if system is not None else []) + kept


def _serialize_for_download(
    history: Iterable[ChatMessage],
    *,
    model: str | None = None,
) -> str:
    """Render a conversation as a plain-text transcript for download.

    The format is intentionally simple: a header line, then a block per
    message with a role label and the content. Markdown is rendered as
    plain text in the transcript — the user gets the *content*, not the
    formatting, which is what they want when they paste it elsewhere.
    """
    lines: list[str] = []
    if model:
        lines.append(f"AI Security Chatbot (Stage 1) — transcript")
        lines.append(f"Model: {model}")
    else:
        lines.append("AI Security Chatbot (Stage 1) — transcript")
    lines.append("")

    for message in history:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "")
        lines.append(f"--- {role} ---")
        lines.append(content)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _count_chars(history: Iterable[ChatMessage]) -> int:
    """Total character count of all message contents in `history`.

    Used by the status line in the UI so the user can see the context
    grow, the same way the CLI does it in Phase 4.
    """
    return sum(len(m.get("content", "")) for m in history)


def _bubble_alignment(role: str) -> str:
    """Return the Streamlit column alignment hint for a message role.

    Kept here (not in the view file) so the rule is testable: every
    "user" message goes right, everything else (assistant, system, tool)
    goes left. The view just renders `:rainbow[...]` or `:blue[...]`
    based on the return value.
    """
    return "right" if role == "user" else "left"


# --- Error message shaping ----------------------------------------------------
#
# The view layer catches `OpenRouterError` and wants to show the user a
# short, actionable message instead of dumping the raw API response. The
# shape is kept as a small pure helper so we can pin the mapping in a
# test ("429 in the error -> 'rate-limited' in the message") without
# spinning up Streamlit.

_RATE_LIMIT_RETRY_SECONDS: int = 30


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like an upstream rate-limit (HTTP 429).

    The engine raises `OpenRouterError` with a message of the form
    ``OpenRouter returned HTTP 429: { ... }``. We match the substring
    ``HTTP 429`` so the rule survives small wording changes in the
    engine as long as the status code stays in the message.
    """
    msg = str(exc)
    return "HTTP 429" in msg or " 429 " in msg


def _friendly_error_message(exc: BaseException, model: str) -> tuple[str, str]:
    """Return ``(headline, body)`` for a user-facing error banner.

    The headline is short and ends with the model name so the user can
    tell which selection failed; the body is one or two sentences of
    actionable advice. The view layer passes both to ``st.error`` /
    ``st.caption``. If we cannot classify the error we fall back to a
    generic "something went wrong" message and include a short hint
    to check the env / rate limits — never the raw JSON, which is
    unreadable for non-developers.
    """
    if _is_rate_limit_error(exc):
        return (
            f"⏳ {model} is rate-limited upstream.",
            f"This free model is being throttled. "
            f"Wait ~{_RATE_LIMIT_RETRY_SECONDS}s and retry, or pick a "
            f"different model from the sidebar.",
        )
    return (
        f"❌ {model} call failed.",
        "Check your `.env` (model name, API key) or your rate limit, "
        "then resend the message.",
    )


# --- Uploaded files -----------------------------------------------------------
# `st.chat_input(accept_file="multiple", file_type=...)` returns a tuple with
# `.text` and `.files`. The LLM only sees text, so we collapse the file
# list into a single readable block that the helper returns to the view.
# The view layer concatenates this block with the user's text and passes
# the combined string to `_build_messages` as the user-turn content.
#
# Rules:
# - Text-like files (text/*, application/json, common source extensions,
#   anything we can decode as utf-8 with no replacement errors) are
#   inlined verbatim, with a header line so the model knows the filename
#   and size. Capped at 12 KB per file so a 5 MB log paste does not
#   blow the context window.
# - Binary files are summarized as a one-line stub (name, size, MIME).
#   We do not try to base64-attach them — OpenRouter's free tier does
#   not support vision payloads via the chat-completions endpoint we
#   use, and the model's value is in the *textual context* the user
#   already typed, not in a raw image.
# - If the user typed nothing AND attached nothing, the helper returns
#   an empty string so the view's existing "blank prompt" guard
#   correctly drops the turn.
#
# The function is pure: it does not log, does not call Streamlit, and
# does not touch the file's `.read()` more than once. Callers should
# pass the files in the order they were attached (Streamlit returns
# them in attachment order).

_TEXTUAL_MIME_PREFIXES: tuple[str, ...] = ("text/",)
_TEXTUAL_MIME_EXACT: frozenset[str] = frozenset({
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/javascript",
})
_TEXTUAL_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".log", ".csv", ".tsv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".xml",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql",
    ".ini", ".toml", ".cfg", ".conf", ".env",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java",
    ".go", ".rs", ".rb", ".php", ".kt", ".swift",
    ".tf", ".hcl",
})
_MAX_INLINE_BYTES: int = 12 * 1024  # 12 KB per file


def _classify_upload(file: _UploadedFileLike) -> bool:
    """Return True if `file` looks like text we can safely inline.

    We consider a file textual when ANY of the following is true:
    - its MIME type starts with ``text/``
    - its MIME type is in the known-textual exact set
    - its filename has a known textual extension
    - we can decode its first 4 KB as utf-8 without errors (last-resort
      sniff — a `.bin` that happens to be plain English still inlines)

    The function never reads more than 4 KB to decide, so it is cheap
    even on a 50 MB upload. The full content is then read in the caller
    when it is decided to be textual.
    """
    mime = (getattr(file, "type", None) or "").lower()
    if any(mime.startswith(prefix) for prefix in _TEXTUAL_MIME_PREFIXES):
        return True
    if mime in _TEXTUAL_MIME_EXACT:
        return True
    name = (getattr(file, "name", "") or "").lower()
    # Strip any path the browser may include (e.g. on some uploaders).
    bare = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for ext in _TEXTUAL_EXTENSIONS:
        if bare.endswith(ext):
            return True
    # Last-resort sniff: if the first 4 KB decode cleanly as utf-8, treat
    # the file as text. This catches log dumps with unusual extensions
    # and falls back gracefully on truly binary input (which will
    # raise UnicodeDecodeError and return False).
    try:
        sample = file.read(4096) if hasattr(file, "read") else b""
    except (OSError, ValueError):
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(sample)


def _format_upload_block(
    files: Iterable[_UploadedFileLike],
) -> str:
    """Render a list of uploaded files as a readable text block.

    The returned string is meant to be *appended* to the user's text
    prompt (separated by a blank line). It is plain text — no markdown
    fences, no base64 — so it slots into the existing text-only OpenAI
    schema unchanged.
    """
    blocks: list[str] = []
    for f in files:
        name = getattr(f, "name", "<unnamed>") or "<unnamed>"
        mime = getattr(f, "type", None) or "application/octet-stream"
        size = int(getattr(f, "size", 0) or 0)
        if _classify_upload(f):
            try:
                # `read()` may have been advanced by `_classify_upload`;
                # most UploadedFile implementations support `seek(0)`.
                if hasattr(f, "seek"):
                    try:
                        f.seek(0)
                    except (OSError, ValueError):
                        pass
                raw = f.read() if hasattr(f, "read") else b""
            except (OSError, ValueError):
                raw = b""
            if isinstance(raw, str):
                raw = raw.encode("utf-8", errors="replace")
            truncated = len(raw) > _MAX_INLINE_BYTES
            content = raw[:_MAX_INLINE_BYTES].decode("utf-8", errors="replace")
            if truncated:
                content += (
                    f"\n... [truncated, {len(raw) - _MAX_INLINE_BYTES} "
                    f"more bytes not shown]"
                )
            blocks.append(
                f"[Attached file: {name} ({mime}, {size} bytes)]\n"
                f"{content}"
            )
        else:
            blocks.append(
                f"[Attached file: {name} ({mime}, {size} bytes) — "
                f"binary, not inlined. Ask the user to paste the "
                f"relevant section as text if you need to see it.]"
            )
    if not blocks:
        return ""
    header = (
        f"[User attached {len(blocks)} file"
        f"{'s' if len(blocks) != 1 else ''}]"
    )
    return header + "\n\n" + "\n\n".join(blocks)


def build_user_turn_text(
    text: str | None,
    files: Iterable[_UploadedFileLike] | None = None,
    *,
    pdf_processor: Callable[[_UploadedFileLike], str] | None = None,
) -> str:
    """Return the single string the model should see for this turn.

    Combines the user's typed text with a readable rendering of any
    uploaded files. The text comes first (so any inline question is
    immediately visible to the model) and the file block is appended
    after a blank line. Returns the empty string when the user typed
    nothing and attached nothing — the view's existing guard treats
    that as a no-op.

    ``pdf_processor`` is optional for backward compatibility — when
    not supplied, PDF attachments fall through to
    :func:`_format_upload_block` and are rendered as the binary
    stub. When supplied, the extracted text is inlined verbatim
    (mirroring :func:`build_user_turn_content`'s behaviour for the
    text-only turn shape).
    """
    text_part = (text or "").strip()
    file_list = list(files or [])

    # PDFs get extracted text when a processor is available so the
    # user bubble (and any transcript / download) shows the same
    # prose the model sees. Without a processor we keep the old
    # binary-stub behaviour for compatibility.
    pdf_text_blocks: list[str] = []
    if pdf_processor is not None:
        non_pdf_files: list[_UploadedFileLike] = []
        for f in file_list:
            if _is_pdf_mime(f):
                try:
                    pdf_text = pdf_processor(f)
                except FileProcessingError as exc:
                    # Fail-soft: surface the stub in the bubble so
                    # the user knows the PDF was attached but could
                    # not be read.
                    non_pdf_files.append(f)  # route through legacy block
                    continue
                if pdf_text:
                    name = (
                        getattr(f, "name", "<unnamed.pdf>")
                        or "<unnamed.pdf>"
                    )
                    pdf_text_blocks.append(
                        f"[Attached PDF: {name}]\n{pdf_text}"
                    )
                else:
                    non_pdf_files.append(f)
            else:
                non_pdf_files.append(f)
        file_list = non_pdf_files

    file_part = _format_upload_block(file_list)
    pdf_part = "\n\n".join(pdf_text_blocks)
    combined_extras = "\n\n".join(p for p in (file_part, pdf_part) if p)
    if text_part and combined_extras:
        return f"{text_part}\n\n{combined_extras}"
    return text_part or combined_extras

# --- Multimodal (vision) upload pipeline -------------------------------------
#
# The text-only `build_user_turn_text` above is what the view has always
# used. It is *still* what we use for transcripts, downloads, and any
# text-only / PDF turn. But the chat-completions endpoint also accepts
# a `content` field that is a *list of parts* — one `{"type": "text",
# "text": ...}` and one or more `{"type": "image_url", "image_url":
# {"url": <data_url>}}` per user message — so we can send a screenshot
# to a vision-capable model.
#
# The new builder returns *either* a string (the existing text-only
# shape, used when no images are attached) *or* a list of parts (the
# multimodal shape, used when at least one image made it through the
# processor). The engine serialises the content verbatim, so the
# widening of `app/openrouter.chat`'s `messages` annotation from
# `list[dict[str, str]]` to `list[dict[str, Any]]` is the only engine
# change this needs.
#
# Failure handling is intentionally fail-soft: if the image processor
# raises for an attachment, the helper appends a *textual* stub block
# and continues. The user still gets a model reply (it just won't see
# the image). This mirrors `_format_upload_block`'s existing rule for
# non-textual files.
#
# The image processor is injected as a callable so the test file can
# pass a fake without importing `app.file_processor` (and therefore
# without the pymupdf wheel being required on CI). Production calls
# pass `app.file_processor.process_image`.


def _is_image_mime(file: _UploadedFileLike) -> bool:
    """Return True if `file` looks like an image attachment.

    The check is intentionally permissive: any ``image/*`` MIME, plus
    the four common image extensions. PDF is *not* considered an
    image here — PDFs are routed through the text-extraction pipeline
    via the dedicated `process_pdf` injection (see
    ``build_user_turn_content``'s ``pdf_processor`` kwarg).
    """
    mime = (getattr(file, "type", None) or "").lower()
    if mime.startswith("image/"):
        return True
    name = (getattr(file, "name", "") or "").lower()
    bare = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return any(
        bare.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")
    )


def _is_pdf_mime(file: _UploadedFileLike) -> bool:
    """Return True if `file` looks like a PDF attachment.

    The check mirrors ``_is_image_mime``'s permissiveness: any
    ``application/pdf`` MIME, plus the ``.pdf`` extension. Browsers
    that guess a different MIME for a file the user clearly named
    ``.pdf`` are still routed through the text-extraction pipeline.
    """
    mime = (getattr(file, "type", None) or "").lower()
    if mime == "application/pdf" or mime.startswith("application/x-pdf"):
        return True
    name = (getattr(file, "name", "") or "").lower()
    bare = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return bare.endswith(".pdf")


def build_user_turn_content(
    text: str | None,
    files: Iterable[_UploadedFileLike] | None = None,
    *,
    image_processor: Callable[[_UploadedFileLike], object] | None = None,
    pdf_processor: Callable[[_UploadedFileLike], str] | None = None,
) -> str | list[dict[str, object]]:
    """Return the wire-shape content for the user turn.

    Returns:
        * ``str`` — when the turn is text-only (the legacy shape that
          slots into the existing text-only `content` field).
        * ``list[dict[str, object]]`` — when at least one image made
          it through the processor. The first part is always the
          user's text (or an empty string when the user attached
          images without typing). Subsequent parts are
          ``{"type": "image_url", "image_url": {"url": <data_url>}}``.

    The function is fail-soft: a `FileProcessingError` from the
    injected image/PDF processor is caught, converted to a textual
    stub, and the turn is downgraded to the list-of-parts shape (or
    to a plain string if the *only* attachments were broken).
    """
    text_part = (text or "").strip()
    file_list = list(files or [])

    # If there are no files, the answer is the plain string. This
    # matches `build_user_turn_text` byte-for-byte so the text-only
    # path is unchanged and the view can keep the old contract for
    # any caller that does not opt in.
    if not file_list:
        return text_part

    # We have files. Decide which of them are images, which are
    # PDFs, and which are textual / binary. Images go through the
    # image processor; PDFs go through the PDF processor; everything
    # else falls into the textual upload-block path. If *any* image
    # made it through, we return a list of parts; otherwise we
    # return a string (the legacy text-only path).
    text_stubs: list[str] = []
    image_parts: list[dict[str, object]] = []
    pdf_text_blocks: list[str] = []

    for f in file_list:
        if _is_image_mime(f):
            if image_processor is None:
                # No processor injected (e.g. CI test stub): treat as
                # a textual stub. The view layer always injects
                # `app.file_processor.process_image` in production.
                text_stubs.append(_stub_block(f, reason="no image processor"))
                continue
            try:
                part = image_processor(f)
            except FileProcessingError as exc:
                # Fail-soft: surface a textual stub so the user still
                # gets a model reply. The stub includes the error
                # kind (machine-readable) and the human message.
                text_stubs.append(_stub_block(f, reason=str(exc)))
                continue
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.data_url},
                }
            )
        elif _is_pdf_mime(f):
            if pdf_processor is None:
                text_stubs.append(_stub_block(f, reason="no PDF processor"))
                continue
            try:
                pdf_text = pdf_processor(f)
            except FileProcessingError as exc:
                # Fail-soft: surface a textual stub so the user still
                # gets a model reply. The stub explains *why* the PDF
                # could not be extracted (scanned, encrypted, etc.).
                text_stubs.append(_stub_block(f, reason=str(exc)))
                continue
            if pdf_text:
                # The extracted text is folded into a labelled block
                # so the model can tell that the prose came from a
                # PDF and not from the user. The label mirrors the
                # wording of `_format_upload_block` for consistency.
                name = getattr(f, "name", "<unnamed.pdf>") or "<unnamed.pdf>"
                pdf_text_blocks.append(
                    f"[Attached PDF: {name}]\n{pdf_text}"
                )
        else:
            # Non-image, non-PDF files (text, log, source, binary)
            # are rendered by the existing block builder. We pass a
            # single-item list so the helper doesn't have to
            # special-case the "I only want to render this one file"
            # path.
            text_stubs.append(_format_upload_block([f]))

    # PDF text always lives in the textual part of the prompt
    # (OpenRouter's free tier does not accept PDF data URLs). It is
    # folded into the same text bucket the user's typed question
    # uses so the model sees "user question + extracted PDF prose"
    # as one block.
    combined_text_parts: list[str] = []
    if text_part:
        combined_text_parts.append(text_part)
    combined_text_parts.extend(pdf_text_blocks)
    combined_text_parts.extend(text_stubs)
    combined_text = "\n\n".join(p for p in combined_text_parts if p)

    if not image_parts:
        # No image survived the processor. Return a plain string so
        # the view layer can reuse the text-only downstream code.
        return combined_text

    # Multimodal turn: build the list of parts. The text part is
    # always first so the model's instruction-following sees the
    # question (and any inlined PDF prose) before the image (which
    # is what OpenAI's examples recommend).
    parts: list[dict[str, object]] = []
    if combined_text:
        parts.append({"type": "text", "text": combined_text})
    # If the user attached images without typing and there was no
    # PDF prose either, the text part is omitted entirely — the
    # model will answer the implicit "what is in this image?"
    # question. This matches the OpenAI cookbook.
    parts.extend(image_parts)
    return parts


def _stub_block(file: _UploadedFileLike, *, reason: str) -> str:
    """Render a one-line textual stub for a broken image attachment.

    Mirrors the wording of `_format_upload_block`'s binary stub so the
    model sees a consistent shape whether the image processor was
    unavailable, the MIME was unsupported, or the bytes were corrupt.
    """
    name = getattr(file, "name", "<unnamed>") or "<unnamed>"
    mime = getattr(file, "type", None) or "application/octet-stream"
    size = int(getattr(file, "size", 0) or 0)
    return (
        f"[Attached image: {name} ({mime}, {size} bytes) — "
        f"not sent to the model: {reason}.]"
    )


def select_model_for_request(
    requested_model: str,
    has_images: bool,
    *,
    vision_model_ids: Iterable[str] | None = None,
) -> tuple[str, bool]:
    """Return ``(effective_model, was_swapped)`` for this turn.

    Rules:
        * If the turn has no images, the requested model is used
          unchanged. ``was_swapped`` is ``False``.
        * If the turn has images and the requested model is already
          vision-capable, it is used unchanged. ``was_swapped`` is
          ``False``.
        * If the turn has images and the requested model is *not*
          vision-capable, we look at ``vision_model_ids`` (the
          allow-list from ``app.config.model_supports_vision``). The
          first vision-capable id in that iterable is used; the
          view layer should then toast the user about the swap.
          ``was_swapped`` is ``True``.
        * If the turn has images but *no* vision model is available
          (the iterable is empty or all entries are blank), we raise
          ``FileProcessingError(kind="no_vision_model")`` so the view
          can surface a clean error rather than silently dropping the
          image.

    The function never mutates global state; it is a pure decision
    rule. The view layer is responsible for the side effect of
    informing the user.
    """
    if not has_images:
        return (requested_model, False)

    # Late import: `app.config.model_supports_vision` requires the env
    # vars to be loaded, which is true by the time the view calls
    # us. The import is also kept out of the module top so the test
    # suite can patch the allow-list via a parameter without
    # monkey-patching `app.config`.
    from app.config import model_supports_vision

    if model_supports_vision(requested_model):
        return (requested_model, False)

    # Build the fallback list. The view may pass a curated list of
    # vision-capable ids (one per supported model); we use the first
    # non-blank entry. If nothing was passed, we fall back to the
    # single most reliable vision model in the curated pool. The
    # fallback id is intentionally hard-coded here (not imported
    # from `app.config`) so the file processor remains usable even
    # if the curated model list moves between modules.
    fallback_pool: list[str] = []
    if vision_model_ids is not None:
        fallback_pool.extend(
            m.strip() for m in vision_model_ids if isinstance(m, str) and m.strip()
        )
    if not fallback_pool:
        # Last-ditch default. As of mid-2026 the only confirmed-working
        # free-tier vision model on OpenRouter is Nemotron Nano 12B VL;
        # every other candidate 404s. The curated list in the view
        # layer is the preferred path — this branch is only hit when
        # the view passes nothing, which is a misconfiguration.
        fallback_pool = [_DEFAULT_FREE_VISION_MODEL]

    chosen = fallback_pool[0]
    # Sanity-check the chosen id: if the view's allow-list is stale
    # and the id is not in the canonical allow-list, we still return
    # it (the user asked for it) but the toast tells them why.
    return (chosen, True)


# Single source of truth for the *hard-coded* vision fallback used when
# the view layer did not pass a curated ``vision_model_ids`` iterable.
# Kept in sync with ``app.config._VISION_MODEL_IDS``; if a model here
# stops working, change both — the helper falls through to this string
# only when the curated pool is empty.
_DEFAULT_FREE_VISION_MODEL: str = "nvidia/nemotron-nano-12b-v2-vl:free"

# --- Teaching mode / system-prompt selection ---------------------------------
#
# The web UI exposes a sidebar radio that lets the learner pick between the
# defensive four-pillar prompt (default) and the CTF/lab "SecMentor" prompt.
# The choice is stored in ``st.session_state["teaching_mode"]`` as one of
# the string keys below. The view layer never imports the prompt constants
# directly — it asks this helper which prompt to seed into the message list,
# so the rule is testable in isolation and we fail *closed* (defensive
# fallback) on any unexpected state.
#
# Why fail-closed: a wider-scope prompt leaking into a defensive session is
# a safety regression, while a tighter-scope prompt showing up in a mentor
# session is a UX nit the user can fix by toggling. We always choose the
# safer default.

_TEACHING_MODE_DEFENSIVE: str = "defensive"
_TEACHING_MODE_MENTOR: str = "mentor"

_TEACHING_MODE_TO_PROMPT: dict[str, str] = {
    _TEACHING_MODE_DEFENSIVE: _prompts.CYBERSECURITY_SYSTEM_PROMPT,
    _TEACHING_MODE_MENTOR: _prompts.OFFENSIVE_MENTOR_SYSTEM_PROMPT,
}


def _active_system_prompt(state: Mapping[str, object] | None) -> str:
    """Return the system-prompt constant for the user's current teaching mode.

    `state` is the Streamlit ``session_state`` (or any mapping shaped like
    it). The helper looks up ``state["teaching_mode"]`` and returns the
    matching prompt constant. Any of the following fall back to the safer
    defensive prompt:

    * `state` is ``None`` or not a mapping
    * `state` has no ``teaching_mode`` key
    * `state["teaching_mode"]`` is not one of the recognised string keys
    * the stored value is not a string at all

    The fallback always returns ``prompts.CYBERSECURITY_SYSTEM_PROMPT``
    (the identity of the constant, not a copy), so tests can pin the
    return value with ``assertIs``.
    """
    if not isinstance(state, Mapping):
        return _prompts.CYBERSECURITY_SYSTEM_PROMPT
    mode = state.get("teaching_mode")
    if not isinstance(mode, str):
        return _prompts.CYBERSECURITY_SYSTEM_PROMPT
    return _TEACHING_MODE_TO_PROMPT.get(mode, _prompts.CYBERSECURITY_SYSTEM_PROMPT)


# --- Copy-to-clipboard helper (Tier 1 #4) -------------------------------------
# The assistant bubble is rendered as raw HTML inside `st.markdown(...,
# unsafe_allow_html=True)` (see the assistant branch of the history loop in
# `web/streamlit_app.py`). That means a copy button has to be a plain HTML
# element — Streamlit's own `st.button` cannot live inside a custom HTML
# block.
#
# Design choice — why no inline ``onclick``:
#   A naive implementation puts a `onclick="..."` attribute on the button
#   and inlines the assistant text as a JS string literal. That is a
#   quote-collision landmine: the `onclick` attribute itself is delimited
#   by quotes, the JS string literal is also delimited by quotes, and the
#   HTML parser terminates the attribute at the first inner quote it sees.
#   The button renders, the click handler is silently malformed, and the
#   rest of the JS leaks out as visible text in the bubble. We hit exactly
#   that bug in v1 and the screenshot the user posted showed the leaked JS.
#
#   The fix is to:
#     1. put the text in a `data-text` HTML attribute (HTML-escaped once,
#        no JS involved at all — the browser parses the attribute for us);
#     2. ship a one-time `<script>` block via `_copy_button_init_script()`
#        that registers a single delegated `click` listener on `document`,
#        reads `data-text` from the clicked element, and runs the
#        clipboard logic.
#   The init script is idempotent (guarded by a flag), so calling it once
#   per page render is safe even when Streamlit re-renders during
#   streaming.
#
# Why a pure helper instead of inlining the HTML in the view:
#   - The escaping logic (HTML attribute escaping for the payload, label
#     restore) is the only part that can break, and breaking it means an
#     XSS hole. Keeping it in a pure function means we can unit-test the
#     escaping with zero browser dependency.
#   - The view stays readable: one `st.markdown(_copy_button_html(content))`
#     call next to each assistant bubble.
#   - Falls back gracefully to a `document.execCommand("copy")` path so the
#     button still works on older browsers and on the Streamlit Cloud
#     preview iframe, where `navigator.clipboard` is gated behind a user
#     gesture and sometimes blocked entirely.

import html as _html  # local import: keeps the helper module self-contained
# and avoids polluting the module namespace with a name that shadows the
# standard library at import time.

_COPY_BUTTON_LABEL: str = "📋 Copy"
_COPY_BUTTON_LABEL_COPIED: str = "✓ Copied"
_COPY_BUTTON_LABEL_FAILED: str = "⚠ Press Ctrl+C"

# Sentinel for the init-script idempotency guard. A module-level boolean
# means the same Streamlit process can call ``_copy_button_init_script()``
# any number of times (including from re-renders during streaming) and
# the actual ``<script>`` is emitted only once.
_COPY_BUTTON_INIT_EMITTED: bool = False


def _copy_button_html(text: str) -> str:
    """Return a self-contained HTML <button> that copies ``text`` to the clipboard.

    The returned string is meant to be rendered via
    ``st.markdown(html, unsafe_allow_html=True)``. It is intentionally
    tiny:

    - one <button> element with no JavaScript in any attribute
    - the text lives in a ``data-text`` attribute (HTML-escaped once,
      decoded back to the original string by the delegated click handler
      registered in ``_copy_button_init_script``)
    - the original label lives in ``data-label`` so the click handler can
      restore it after the 1.4 s confirmation window

    Behaviour (provided by the delegated listener in
    ``_copy_button_init_script``):

    - On click, calls ``navigator.clipboard.writeText(text)`` (modern API).
    - If that throws or returns a rejected promise (e.g. insecure context,
      permission denied), falls back to a hidden <textarea> +
      ``document.execCommand("copy")`` path so the button still works.
    - Briefly swaps the button label to "✓ Copied" (or "⚠ Press Ctrl+C"
      if both paths fail) so the user has feedback.
    """
    # HTML-escape the payload so it is safe inside a double-quoted HTML
    # attribute. ``html.escape`` converts ``&``, ``<``, ``>``, and the
    # double quote to entities (``&amp;``, ``&lt;``, ``&gt;``, ``&quot;``).
    # The browser parses the attribute for us, so by the time JS reads
    # ``btn.dataset.text`` the original string is back, byte-for-byte.
    #
    # The double quotes inside the attribute delimiters are guaranteed
    # safe because ``html.escape`` has turned every inner ``"`` into
    # ``&quot;`` — there is no way for the parser to close the attribute
    # early. This is the XSS-safe alternative to inline ``onclick``.
    safe_text: str = _html.escape(text, quote=True)
    safe_label: str = _html.escape(_COPY_BUTTON_LABEL, quote=True)
    return (
        '<button type="button" '
        'class="bubble-copy-btn" '
        f'data-label="{safe_label}" '
        f'data-text="{safe_text}">'
        f'{_COPY_BUTTON_LABEL}'
        '</button>'
    )


# --- Per-bubble iframe copy button -------------------------------------------
#
# Why an iframe (instead of st.markdown + delegated listener):
#   Streamlit's `st.markdown(..., unsafe_allow_html=True)` runs the input
#   through a markdown sanitizer (bleach / DOMPurify) that strips
#   ``<script>`` tags and inline event handlers. The delegated listener
#   approach (``_copy_button_init_script``) tries to work around this by
#   routing the script through ``st.components.v1.html`` + ``parent.eval``,
#   but in modern browsers the component iframe is cross-origin and
#   ``parent.eval`` is blocked by the same-origin policy. Result: the
#   button renders but clicking it does nothing.
#
#   The fix that actually works is to put the entire button + click handler
#   inside the iframe's own ``srcdoc=``. Streamlit sets ``srcdoc`` as an
#   attribute on the iframe element; that attribute value is parsed by
#   the browser as raw HTML (no Streamlit sanitizer touches it). The
#   iframe is served same-origin by the Streamlit server, so:
#
#     - inline ``<script>`` runs inside the iframe's own window
#     - ``navigator.clipboard.writeText(...)`` is allowed (secure context)
#     - the click event never crosses the parent/iframe boundary, so
#       same-origin policy is irrelevant
#     - the payload lives on a ``data-text`` attribute on the button
#       itself (HTML-escaped once), and the click handler reads it back
#       via ``btn.dataset.text`` -- the browser decodes the attribute
#       automatically, so no manual unescaping is needed on the JS side
#       and there is no XSS hole from round-tripping user bytes through
#       an HTML attribute into a JS string literal
#
# One iframe per assistant message means a per-message component, which
# is cheap (Streamlit components are tiny) and keeps each button's
# handler fully isolated from every other button's handler.
def _copy_button_iframe_html(text: str) -> str:
    """Return a full HTML document for an iframe-hosted copy button.

    The document contains **only** a small "📋 Copy" button. The reply
    text is NOT rendered inside the iframe -- the Streamlit view already
    renders it (as markdown) directly above the button. If we duplicated
    it inside the iframe the user would see the reply twice: once as the
    rendered markdown, once inside the iframe's <pre>.

    Instead, the payload lives on a ``data-text`` attribute on the button
    itself (HTML-escaped once with ``html.escape(quote=True)`` so it is
    XSS-safe and cannot break the attribute quoting). The click handler
    runs inside the iframe's own window, reads ``btn.dataset.text``, and
    copies it to the clipboard:

      1. ``navigator.clipboard.writeText(payload)`` (modern API; the
         iframe is same-origin to the Streamlit server so it is a secure
         context and the API is allowed).
      2. If that rejects, falls back to a hidden ``<textarea>`` +
         ``document.execCommand('copy')`` path.
      3. Briefly swaps the button label to "✓ Copied" (or "⚠ Press Ctrl+C"
         on total failure) and restores it after 1.4 s.

    Why this works where the previous design did not:

    The earlier design wired a *single* delegated click listener on the
    parent ``document`` via ``st.components.v1.html`` + ``parent.eval``.
    The component iframe is sandboxed (``allow-scripts`` without
    ``allow-same-origin``) so ``parent.eval(...)`` is blocked by the
    same-origin policy and the listener never registered. Putting the
    handler *inside* the iframe removes the cross-origin hop entirely
    -- the click never crosses the parent/iframe boundary, and
    ``navigator.clipboard.writeText`` is allowed because the iframe is
    served same-origin by the Streamlit server.
    """
    # The payload is HTML-escaped once and put in a ``data-text``
    # attribute. ``html.escape(..., quote=True)`` converts ``&``, ``<``,
    # ``>``, ``"``, and ``'`` to entities, which guarantees the
    # attribute's double-quote delimiter can never be closed by any byte
    # in the payload. The browser decodes the attribute for us, so by
    # the time JS reads ``btn.dataset.text`` the original string is back.
    safe_text: str = _html.escape(text, quote=True)
    safe_label: str = _html.escape(_COPY_BUTTON_LABEL, quote=True)
    # Build the JS body with ``json.dumps`` so the string literals are
    # unambiguously quoted -- any byte in the label, copied-text, or
    # failed-text constants can be embedded without breaking the parser.
    # ``</script>`` inside the body is also rewritten to ``<\/script>``
    # on the wire so the browser's HTML parser does not terminate the
    # wrapper script tag at the body's own closer.
    import json as _json

    handler_js = (
        "(function(){\n"
        "  var btn = document.getElementById('copy-btn');\n"
        "  if (!btn) { return; }\n"
        "  function show(label){\n"
        "    btn.textContent = label;\n"
        "    setTimeout(function(){ btn.textContent = " + _json.dumps(_COPY_BUTTON_LABEL) + "; }, 1400);\n"
        "  }\n"
        "  function fallback(text){\n"
        "    try {\n"
        "      var ta = document.createElement('textarea');\n"
        "      ta.value = text;\n"
        "      ta.setAttribute('readonly', '');\n"
        "      ta.style.position = 'absolute';\n"
        "      ta.style.left = '-9999px';\n"
        "      document.body.appendChild(ta);\n"
        "      ta.select();\n"
        "      var ok = document.execCommand && document.execCommand('copy');\n"
        "      document.body.removeChild(ta);\n"
        "      show(ok ? " + _json.dumps(_COPY_BUTTON_LABEL_COPIED) + " : " + _json.dumps(_COPY_BUTTON_LABEL_FAILED) + ");\n"
        "    } catch (e) {\n"
        "      show(" + _json.dumps(_COPY_BUTTON_LABEL_FAILED) + ");\n"
        "    }\n"
        "  }\n"
        "  btn.addEventListener('click', function(ev){\n"
        "    ev.preventDefault();\n"
        "    var text = btn.dataset.text || '';\n"
        "    if (navigator.clipboard && navigator.clipboard.writeText) {\n"
        "      navigator.clipboard.writeText(text).then(\n"
        "        function(){ show(" + _json.dumps(_COPY_BUTTON_LABEL_COPIED) + "); },\n"
        "        function(){ fallback(text); }\n"
        "      );\n"
        "    } else {\n"
        "      fallback(text);\n"
        "    }\n"
        "  });\n"
        "})();\n"
    ).replace("</script>", "<\\/script>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>\n"
        "html, body { margin: 0; padding: 0; background: transparent; "
        "font-family: inherit; }\n"
        "body { display: inline-flex; padding: 0; }\n"
        "button#copy-btn { align-self: flex-end; "
        "background: transparent; color: #555; "
        "border: 1px solid #ddd; border-radius: 999px; "
        "padding: 2px 10px; font-size: 0.72rem; "
        "font-family: inherit; cursor: pointer; "
        "line-height: 1.2; }\n"
        "button#copy-btn:hover { background: rgba(0,0,0,0.04); }\n"
        "button#copy-btn:focus-visible { outline: 2px solid #6aa9ff; "
        "outline-offset: 1px; }\n"
        "</style></head><body>"
        f"<button type='button' id='copy-btn' "
        f"data-label='{safe_label}' "
        f"data-text='{safe_text}'>{_COPY_BUTTON_LABEL}</button>"
        f"<script>{handler_js}</script>"
        "</body></html>"
    )


# --- Markdown -> plain-text for the copy button --------------------------------
# The assistant reply that lands in the view is *markdown source* (the
# Streamlit render call uses ``st.markdown(content)``, not the HTML variant).
# If we naively put that source into ``data-text``, clicking Copy pastes
# ``# Heading``, ``**bold**``, and ````code`` blocks into the user's
# chat/email/doc — useless for sharing.
#
# The fix is to convert the markdown to a readable plain-text rendering of
# the same content before stuffing it into ``data-text``. The conversion is
# deliberately conservative: it strips the markdown syntax that gets in the
# way of reading (heading hashes, bold/italic delimiters, link brackets,
# inline-code backticks, fenced-code fences) while preserving the words,
# code contents, list markers, paragraph breaks, and link URLs.
#
# The rules, in order:
#   1. Fenced code blocks (```...```) — keep the inner code verbatim,
#      drop the fence. The code IS the content.
#   2. ATX headings (#, ##, ###...) — drop the leading hashes, keep the
#      heading text. (Hashes are pure markdown syntax, no semantic content.)
#   3. Bold (**...** / __...__) — drop the delimiters, keep the text.
#   4. Italic (*...* / _..._) — drop the delimiters, keep the text.
#   5. Inline code (`...`) — keep the inner code verbatim, drop the
#      backticks. (Same reasoning as fenced blocks: the code IS the
#      content.)
#   6. Links ([text](url)) — render as ``text (url)`` so neither piece
#      of information is lost.
#   7. Images (![alt](url)) — render as ``[image: alt]`` (alt text only;
#      a raw URL is not useful in a chat paste).
#   8. Blockquote markers (> ...) — drop the leading ``>`` characters,
#      keep the quote text.
#   9. Unordered list markers (- * +) — keep the marker, drop nothing.
#  10. Ordered list markers (1. 2. ...) — keep the marker, drop nothing.
#  11. Horizontal rules (--- *** ___) — drop the line entirely.
#  12. HTML entities (``&amp;`` / ``&lt;`` / ``&quot;``) — decoded to the
#      literal char so a chat paste of ``<x>`` does not become ``&lt;x&gt;``.
#  13. Trailing whitespace per line and triple+ blank lines normalized
#      to a single blank line.
#
# The order matters: fences and inline code are processed *before* bold/
# italic, so an asterisk inside a code block is not misread as emphasis.
# The implementation is a single pass with a small state machine for
# fenced code; everything else is a regex substitution.
_MARKDOWN_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", flags=re.DOTALL)
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
_MARKDOWN_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_BOLD_UNDER_RE = re.compile(r"__(.+?)__")
_MARKDOWN_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)")
_MARKDOWN_ITALIC_UNDER_RE = re.compile(r"(?<!_)_([^_\n]+?)_(?!_)")
_MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"(?m)^[ \t]*>\s?")
_MARKDOWN_HR_RE = re.compile(r"(?m)^\s*([-*_])\s*\1\s*\1[\1\s-]*$")
_MARKDOWN_TRAILING_WS_RE = re.compile(r"[ \t]+\n", flags=re.MULTILINE)
_MARKDOWN_BLANK_LINES_RE = re.compile(r"\n{3,}")


_MARKDOWN_INLINE_SENTINEL_RE = re.compile(r"\x00INLINE(\d+)\x00")


def _split_on_inline_code(text: str) -> tuple[str, list[str]]:
    """Replace inline-code spans with sentinels; return (sentinelized, originals).

    Inline code (`` `like this` ``) needs the same protection from later
    regex passes as fenced code blocks do: a ``*`` inside `` `*star*` ``
    must not be read as italic, and ``&amp;`` inside `` `&amp;` `` must
    not be HTML-decoded. We can't reuse the fence sentinel namespace
    because inline-code and fences can be interleaved in the same reply
    (e.g. a sentence mentioning both). A second, independent list of
    inline-code payloads, swapped back in *after* entity decoding,
    solves both problems cleanly.
    """
    inline_blocks: list[str] = []

    def _stash_inline(match: "re.Match[str]") -> str:
        inline_blocks.append(match.group(1))
        return f"\x00INLINE{len(inline_blocks) - 1}\x00"

    return _MARKDOWN_INLINE_CODE_RE.sub(_stash_inline, text), inline_blocks


def _swap_inline_back(text: str, inline_blocks: list[str]) -> str:
    """Inverse of :func:`_split_on_inline_code`."""
    def _restore(match: "re.Match[str]") -> str:
        return inline_blocks[int(match.group(1))]
    return _MARKDOWN_INLINE_SENTINEL_RE.sub(_restore, text)


def _markdown_to_plain_text(content: str) -> str:
    """Convert an LLM markdown reply into clean, copy-pasteable plain text.

    The result is what a user expects to land in their clipboard when they
    click "Copy" on a chat bubble: the *rendered* response, with markdown
    syntax stripped and the actual content (words, code, list markers,
    link URLs) preserved. See the rule list above for the exact behaviour.

    The function is pure (input string in, output string out, no I/O, no
    globals, no side effects) and is unit-tested independently of
    Streamlit — see ``MarkdownToPlainTextTests`` in ``tests/test_smoke.py``.

    Edge cases:

    - Empty / None input returns the empty string.
    - Fenced code blocks are protected: an asterisk *inside* a fenced
      block is preserved as a literal asterisk, not interpreted as
      italic. HTML entities inside fenced code are *not* decoded (they
      may be part of the code).
    - Inline code spans (single backticks) get the same protection: a
      ``*`` inside `` `*star*` `` is preserved verbatim and entities
      inside `` `&amp;` `` stay encoded.
    - The language tag on a fenced block (`` ```python ``) is dropped,
      not echoed as an HTML comment, because the comment would leak
      into the clipboard and confuse the reader.
    - Malformed markdown (unclosed ``**``, stray backticks) is treated
      as plain text. The output may contain the stray delimiter, but
      it will not crash or hang.
    """
    if not content:
        return ""

    # 1) Fenced code blocks: extract the inner code, drop the fence.
    #    We replace fenced blocks with a sentinel that survives the
    #    later substitutions, then swap the sentinels back. This keeps
    #    bold/italic/link regexes from corrupting code contents.
    fence_blocks: list[str] = []

    def _stash_fence(match: "re.Match[str]") -> str:
        body = match.group(2)
        # The language tag (e.g. ``python``) is dropped here. The
        # common case is that the code body itself is readable enough
        # without the tag, and echoing the tag as an HTML comment
        # would pollute the clipboard with markup noise.
        fence_blocks.append(body)
        return f"\x00FENCE{len(fence_blocks) - 1}\x00"

    text = _MARKDOWN_FENCE_RE.sub(_stash_fence, content)

    # 1b) Inline code spans: same protection, second pass on whatever
    #     is left after fences are out of the way. Inline code can
    #     contain backticks of its own? No — the spec disallows an
    #     unescaped backtick inside an inline-code span, so this
    #     regex is sound.
    text, inline_blocks = _split_on_inline_code(text)

    # 2-11) The rest of the rules. Order matters less once fences and
    #    inline code are out of the way, but bold must run before
    #    italic (an italic inside a bold would otherwise be
    #    mis-parsed).
    text = _MARKDOWN_HEADING_RE.sub("", text)            # 2
    text = _MARKDOWN_BOLD_STAR_RE.sub(r"\1", text)       # 3a
    text = _MARKDOWN_BOLD_UNDER_RE.sub(r"\1", text)      # 3b
    text = _MARKDOWN_ITALIC_STAR_RE.sub(r"\1", text)     # 4a
    text = _MARKDOWN_ITALIC_UNDER_RE.sub(r"\1", text)    # 4b
    text = _MARKDOWN_IMAGE_RE.sub(r"[image: \1]", text)  # 7 (before links)
    text = _MARKDOWN_LINK_RE.sub(r"\1 (\2)", text)       # 6
    text = _MARKDOWN_BLOCKQUOTE_RE.sub("", text)         # 8
    text = _MARKDOWN_HR_RE.sub("", text)                 # 11

    # 12) Decode HTML entities OUTSIDE code. We split-and-rejoin so
    #     the inside-code segments are not entity-decoded.
    parts: list[str] = []
    cursor = 0
    for sentinel_match in re.finditer(r"\x00FENCE(\d+)\x00", text):
        parts.append(_html.unescape(text[cursor:sentinel_match.start()]))
        idx = int(sentinel_match.group(1))
        fence_body = fence_blocks[idx]
        # Inside code we deliberately do NOT decode entities — they may
        # be part of the actual code (e.g. ``if (a &amp;&amp; b)``).
        parts.append(fence_body)
        cursor = sentinel_match.end()
    parts.append(_html.unescape(text[cursor:]))
    text = "".join(parts)

    # 1c) Swap inline-code sentinels back. Done AFTER entity decoding so
    #     inline-code payloads retain their original entities (the spec
    #     is: backtick-delimited text is verbatim).
    text = _swap_inline_back(text, inline_blocks)

    # 13) Whitespace tidy.
    text = _MARKDOWN_TRAILING_WS_RE.sub("\n", text)
    text = _MARKDOWN_BLANK_LINES_RE.sub("\n\n", text)

    return text.strip("\n")


def _copy_button_html_for_bubble(content: str) -> str:
    """Wrap an LLM markdown reply as a copy button whose payload is plain text.

    This is the helper the Streamlit view should call from the assistant
    branch of the history render. It runs ``_markdown_to_plain_text`` on
    the raw markdown ``content`` so the user pastes a *rendered* reply
    into chat/email/doc, not the markdown source.

    Falls back to the raw ``content`` if the plain-text conversion
    returns the empty string (defensive — should not happen in practice
    but keeps the button useful for edge cases like an assistant
    returning only ``"\n"`` or only punctuation).
    """
    plain = _markdown_to_plain_text(content)
    if not plain:
        plain = content or ""
    return _copy_button_html(plain)


def _render_copy_button_for_bubble(content: str) -> None:
    """Emit a per-message ``st.components.v1.html`` copy button for an
    assistant reply.

    Why a component (and not ``st.markdown`` of a tiny ``<button>``):

    ``st.markdown(..., unsafe_allow_html=True)`` runs its input through
    Streamlit's sanitizer which strips ``<script>`` tags AND inline
    event handlers. Either path leaves a button that renders but does
    nothing on click. A ``st.components.v1.html`` call wraps the HTML in
    an iframe whose ``srcdoc`` attribute is set directly — the browser
    parses it as raw HTML, scripts run inside the iframe's own window,
    and the iframe is same-origin to the Streamlit server, so
    ``navigator.clipboard.writeText`` is allowed (secure context).

    Each call renders one tiny iframe per assistant message. The iframe
    contains **only** the "📋 Copy" button — the reply itself is
    rendered directly above (via ``st.markdown(content)`` in the view)
    so the user sees it exactly once. The payload rides on a
    ``data-text`` attribute on the button (HTML-escaped once) and the
    click handler reads it back via ``btn.dataset.text`` — no
    parent/iframe boundary crossing, no ``parent.eval`` (which the
    cross-origin sandbox blocks), no delegated listener that has to be
    wired through a separate ``<script>`` block.
    """
    plain = _markdown_to_plain_text(content)
    if not plain:
        plain = content or ""
    # Imported lazily because ``streamlit`` is not a hard dep of the
    # helpers module (the CLI/tests pull this module without Streamlit
    # installed in some sandboxes). The view always has Streamlit
    # available so the runtime path is fine.
    import streamlit as _st

    srcdoc = _copy_button_iframe_html(plain)
    # Tiny iframe -- it only hosts the button. ``height`` is the visible
    # vertical space, ``width`` is the available horizontal space; the
    # button right-aligns itself inside via CSS.
    _st.components.v1.html(
        srcdoc, height=32, scrolling=False, width=720
    )


def _copy_button_init_script() -> str:
    """Return a one-time ``<script>`` block that wires up copy buttons.

    Call this exactly once per page render (idempotent: subsequent calls
    return an empty string so re-renders during streaming do not pile up
    duplicate listeners). The returned block registers a *delegated*
    click listener on ``document`` so every ``.bubble-copy-btn`` on the
    page — current and future — copies its ``data-text`` payload to the
    clipboard.

    Why a delegated listener and not one ``addEventListener`` per button:

    - Streamlit re-renders the chat history on every assistant turn.
      Adding a listener per button would leak listeners across renders.
    - The listener is registered once and matches against the
      ``.bubble-copy-btn`` class via ``event.target.closest(...)``.
    - It is also safe inside the Streamlit Cloud preview iframe, where
      inline event handlers in dynamically-injected HTML are sometimes
      blocked by a strict Content Security Policy.

    Behaviour on click:

    1. Reads ``btn.dataset.text`` (the HTML-decoded original assistant
       reply) and ``btn.dataset.label`` (the original button caption).
    2. Tries ``navigator.clipboard.writeText(text)`` (modern API).
    3. If that throws or returns a rejected promise (e.g. insecure
       context, permission denied), falls back to a hidden ``<textarea>``
       + ``document.execCommand('copy')`` path so the button still
       works.
    4. Briefly swaps the button label to "✓ Copied" (or "⚠ Press Ctrl+C"
       if both paths fail) and restores it after 1.4 s.
    5. Uses a per-element ``__copyBtnBusy`` guard so rapid double-clicks
       do not stack overlapping timeouts.
    """
    # Idempotency guard. The module-level flag flips on the first call so
    # subsequent calls (Streamlit re-renders, hot reloads) return '' and
    # do not stack duplicate ``document.addEventListener`` registrations.
    # We mutate a module global instead of a closure so the guard
    # survives a Streamlit script rerun.
    global _COPY_BUTTON_INIT_EMITTED
    if _COPY_BUTTON_INIT_EMITTED:
        return ""
    _COPY_BUTTON_INIT_EMITTED = True

    # The script is built with single-quoted JS string literals inside
    # the surrounding HTML script tag so the outer HTML parser never has
    # to disambiguate anything. The strings we embed are all ASCII and
    # contain no single quotes, so there is no escape needed.
    return (
        "<script>\n"
        "(function(){\n"
        "  if (window.__secMentorCopyBtnWired) { return; }\n"
        "  window.__secMentorCopyBtnWired = true;\n"
        "  function show(btn, lbl){\n"
        "    btn.textContent = lbl;\n"
        "    setTimeout(function(){\n"
        "      btn.textContent = btn.dataset.label || '';\n"
        "    }, 1400);\n"
        "  }\n"
        "  function fallback(btn, text){\n"
        "    try {\n"
        "      var ta = document.createElement('textarea');\n"
        "      ta.value = text;\n"
        "      ta.setAttribute('readonly', '');\n"
        "      ta.style.position = 'absolute';\n"
        "      ta.style.left = '-9999px';\n"
        "      document.body.appendChild(ta);\n"
        "      ta.select();\n"
        "      var ok = document.execCommand && document.execCommand('copy');\n"
        "      document.body.removeChild(ta);\n"
        "      show(btn, ok ? '"
        + _COPY_BUTTON_LABEL_COPIED
        + "' : '"
        + _COPY_BUTTON_LABEL_FAILED
        + "');\n"
        "    } catch (e) {\n"
        "      show(btn, '"
        + _COPY_BUTTON_LABEL_FAILED
        + "');\n"
        "    }\n"
        "  }\n"
        "  document.addEventListener('click', function(ev){\n"
        "    var btn = ev.target && ev.target.closest && ev.target.closest('.bubble-copy-btn');\n"
        "    if (!btn) { return; }\n"
        "    if (btn.__copyBtnBusy) { return; }\n"
        "    btn.__copyBtnBusy = true;\n"
        "    setTimeout(function(){ btn.__copyBtnBusy = false; }, 1500);\n"
        "    var text = btn.dataset.text || '';\n"
        "    if (navigator.clipboard && navigator.clipboard.writeText) {\n"
        "      navigator.clipboard.writeText(text).then(\n"
        "        function(){ show(btn, '"
        + _COPY_BUTTON_LABEL_COPIED
        + "'); },\n"
        "        function(){ fallback(btn, text); }\n"
        "      );\n"
        "    } else {\n"
        "      fallback(btn, text);\n"
        "    }\n"
        "  });\n"
        "})();\n"
        "</script>"
    )


def _emit_copy_button_init_script() -> None:
    """Emit the one-time init script to the page so the click handler runs.

    Why this is *not* ``st.markdown(_copy_button_init_script(),
    unsafe_allow_html=True)``: Streamlit's markdown sanitizer strips
    ``<script>`` tags (and inline event handlers) from the rendered HTML
    even when ``unsafe_allow_html=True`` is passed. That means the
    delegated ``click`` listener in the script body never gets registered
    in the browser, and clicking the copy button does nothing.

    The escape hatch is ``st.components.v1.html(...)``: it injects the
    HTML into an iframe via ``srcdoc=`` (so ``<script>`` tags actually
    execute — DOMPurify and the markdown sanitizer never see them).

    Subtlety: scripts inside an iframe run in the **iframe's** window,
    not the parent. The copy button is rendered in the parent document,
    so a ``document.addEventListener('click', ...)`` inside the iframe
    would never fire for clicks on the parent's button. We work around
    that by wrapping the body in a tiny bootstrapper that
    ``parent.eval(...)``s the body inside the parent window — so the
    existing delegated-listener body (which uses bare ``window``,
    ``document``, and ``navigator``) runs against the parent and catches
    clicks on the copy button.

    The body string itself is unchanged, so the 20 structural tests in
    ``CopyButtonInitScriptTests`` still pin its behaviour (delegated
    listener, ``closest`` matching, modern + legacy clipboard paths,
    busy guard, label restore, payload read from ``data-text``, window
    guard).
    """
    body = _copy_button_init_script()
    if not body:
        # Idempotent: the module-level guard has already flipped, so the
        # first emission wins and subsequent calls become a no-op. This
        # is important across Streamlit reruns.
        return
    # Local import: keeps ``web.chat_helpers`` importable from tests
    # that don't have a Streamlit script-run context.
    import streamlit.components.v1 as components
    import json as _json
    # The body itself contains a literal ``</script>`` (its own closing
    # tag). When we drop the body into a JS string literal inside the
    # wrapper ``<script>...</script>`` block, the HTML5 parser scans for
    # ``</script>`` *as text* and would terminate the wrapper at the
    # body's own closer -- so the body would be truncated and the
    # delegated listener would never register. The standard escape is
    # to replace ``</script>`` with ``<\/script>`` in the source text
    # *before* the browser sees it. Inside a JS string literal, the
    # backslash-before-slash is interpreted as an escape sequence whose
    # value is just ``/``, so the resulting string at runtime is the
    # original body verbatim -- the wire bytes are unchanged, only the
    # parsed literal differs.
    body_for_wire = body.replace("</script>", "<\\/script>")
    parent_eval_call = (
        "<script>\n"
        "try {\n"
        "  parent.eval(" + _json.dumps(body_for_wire) + ");\n"
        "} catch (e) {\n"
        "  console.error('secMentor copy init failed:', e);\n"
        "}\n"
        "</script>"
    )
    components.html(parent_eval_call, height=0, width=0, scrolling=False)
