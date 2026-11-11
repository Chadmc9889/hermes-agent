"""Byte-integrity checks for Kanban attachments written by the agent tools.

A model that relays a file as base64 can corrupt it silently: a flipped or
dropped character still decodes, and the store path used to keep whatever
arrived. ``verify_attachment`` fails closed before anything is written: an
``expected_sha256`` / ``expected_size`` taken from the command that produced
the bytes must match, and anything named or typed as a zip/gzip archive must
pass a full CRC/stream self-test.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import re
import zipfile
import zlib
from typing import Any, Optional

# Ceiling on bytes inflated by a self-test: a 25 MB gzip bomb must not pin the agent
# process. An archive past it is rejected as unverifiable rather than stored unchecked.
ARCHIVE_VERIFY_MAX_EXPANDED = 2 * 1024 ** 3

_ZIP_TYPES = frozenset({"application/zip", "application/x-zip", "application/x-zip-compressed"})
_GZIP_TYPES = frozenset({"application/gzip", "application/x-gzip"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESEND_HINT = ("Nothing was stored. If the bytes were hand-copied as base64, re-send the "
                "file with path= instead.")


class AttachmentIntegrityError(ValueError):
    """Bytes failed a requested hash/size check or an archive self-test (nothing stored)."""


def archive_kind(filename: str, content_type: Optional[str]) -> Optional[str]:
    """``"zip"`` / ``"gzip"`` when the name or MIME type declares an archive, else None."""
    ctype = (content_type or "").split(";")[0].strip().lower()
    name = (filename or "").strip().lower()
    if ctype in _ZIP_TYPES or name.endswith(".zip"):
        return "zip"
    if ctype in _GZIP_TYPES or name.endswith((".gz", ".tgz")):
        return "gzip"
    return None


def _zip_problem(data: bytes) -> Optional[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            declared = sum(info.file_size for info in zf.infolist())
            if declared > ARCHIVE_VERIFY_MAX_EXPANDED:
                return f"declares {declared} expanded bytes, above the self-test ceiling"
            bad_member = zf.testzip()
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError,
            NotImplementedError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if bad_member is None else f"member {bad_member!r} fails its CRC check"


def _gzip_problem(data: bytes) -> Optional[str]:
    if not data:
        return "empty file is not a gzip stream"
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as fh:
            while chunk := fh.read(1 << 20):
                total += len(chunk)
                if total > ARCHIVE_VERIFY_MAX_EXPANDED:
                    return "expands past the self-test ceiling"
    except (OSError, EOFError, zlib.error) as exc:  # BadGzipFile is an OSError
        return f"{type(exc).__name__}: {exc}"
    return None


_SELF_TESTS = {"zip": _zip_problem, "gzip": _gzip_problem}


def _expected_size(value: Any) -> int:
    if isinstance(value, bool):
        raise AttachmentIntegrityError("expected_size must be a non-negative integer")
    try:
        size = int(str(value).strip())
    except ValueError:
        raise AttachmentIntegrityError("expected_size must be a non-negative integer") from None
    if size < 0:
        raise AttachmentIntegrityError("expected_size must be a non-negative integer")
    return size


def verify_attachment(
    data: bytes, filename: str, content_type: Optional[str] = None, *,
    expected_sha256: Any = None, expected_size: Any = None,
) -> str:
    """Return the sha256 hex digest of ``data`` or raise :class:`AttachmentIntegrityError`.

    Blank ``expected_*`` values mean "not supplied". The archive self-test is keyed on
    the declared name/type, so a ``.zip`` that is not a valid zip is rejected too.
    """
    digest = hashlib.sha256(data).hexdigest()
    if expected_size is not None and str(expected_size).strip():
        want_size = _expected_size(expected_size)
        if want_size != len(data):
            raise AttachmentIntegrityError(
                f"size mismatch for {filename!r}: expected {want_size} bytes, received "
                f"{len(data)}. {_RESEND_HINT}")
    if expected_sha256 is not None and str(expected_sha256).strip():
        want = str(expected_sha256).strip().lower().removeprefix("sha256:")
        if not _SHA256.fullmatch(want):
            raise AttachmentIntegrityError("expected_sha256 must be 64 hexadecimal characters")
        if want != digest:
            raise AttachmentIntegrityError(
                f"sha256 mismatch for {filename!r}: expected {want}, received {digest} "
                f"({len(data)} bytes). {_RESEND_HINT}")
    kind = archive_kind(filename, content_type)
    problem = _SELF_TESTS[kind](data) if kind else None
    if problem:
        raise AttachmentIntegrityError(
            f"{filename!r} is declared {kind} but fails its archive self-test ({problem}). "
            f"{_RESEND_HINT}")
    return digest
