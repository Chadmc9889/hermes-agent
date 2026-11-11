"""kanban_attach byte integrity: path uploads, expected hash/size, archive self-test.

Regression for silently corrupted board archives: a worker relayed base64 through
the model, one character drifted, and ``kanban_attach`` stored the damaged bytes
(a zip failing its member CRC, a truncated gzip). Every check here runs through
the real registry dispatch against a temp ``HERMES_HOME`` board.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="attach-integrity", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _call(args: dict) -> dict:
    import tools.kanban_tools  # noqa: F401  (registers the handlers)
    from tools.registry import registry
    return json.loads(registry.dispatch("kanban_attach", args))


def _stored(tid: str) -> list:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.list_attachments(conn, tid)
    finally:
        conn.close()


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("REPORT.txt", "checkpoint report\n" * 200)
        zf.writestr("OBSERVATION.json", json.dumps({"rows": list(range(400))}))
    return buf.getvalue()


def _flip_inside_member(data: bytes, member: str) -> bytes:
    """One flipped bit inside a member's compressed payload: still a parseable zip
    whose member fails CRC (the shape of the corrupted checkpoint archive)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        info = zf.getinfo(member)
    fname_len = int.from_bytes(data[info.header_offset + 26:info.header_offset + 28], "little")
    extra_len = int.from_bytes(data[info.header_offset + 28:info.header_offset + 30], "little")
    start = info.header_offset + 30 + fname_len + extra_len
    pos = start + info.compress_size // 2
    out = bytearray(data)
    out[pos] ^= 0x40
    return bytes(out)


CORRUPTIONS = {
    "zip_member_crc": lambda: ("evidence.zip", _flip_inside_member(_zip_bytes(), "OBSERVATION.json")),
    "zip_not_a_zip": lambda: ("evidence.zip", _zip_bytes()[1:]),
    "gzip_truncated": lambda: ("probes.py.gz", gzip.compress(b"print('x')\n" * 500, mtime=0)[:-9]),
    "gzip_bad_stream": lambda: ("probes.py.gz", gzip.compress(bytes(range(256)) * 40, mtime=0)[:40]
                                + bytes(64) + gzip.compress(bytes(range(256)) * 40, mtime=0)[104:]),
}


@pytest.mark.parametrize("case", sorted(CORRUPTIONS))
def test_corrupt_archive_base64_is_rejected_and_nothing_stored(worker, case):
    filename, data = CORRUPTIONS[case]()
    out = _call({"filename": filename, "content_base64": base64.b64encode(data).decode()})
    assert "error" in out and "self-test" in out["error"], out
    assert _stored(worker) == []
    # The attachments dir must not keep an orphan blob either.
    from hermes_cli import kanban_db as kb
    assert not any(kb.task_attachments_dir(worker).glob("*"))


def test_one_char_base64_drift_is_caught_by_expected_sha256(worker):
    """A non-archive payload has no self-test; the producer's hash is what catches drift."""
    data = b"report body " * 300
    b64 = base64.b64encode(data).decode()
    drifted = b64[:500] + ("A" if b64[500] != "A" else "B") + b64[501:]
    out = _call({"filename": "REPORT.txt", "content_base64": drifted,
                 "expected_sha256": hashlib.sha256(data).hexdigest()})
    assert "sha256 mismatch" in out.get("error", ""), out
    assert _stored(worker) == []


def test_expected_size_mismatch_rejected(worker):
    data = b"x" * 1000
    out = _call({"filename": "a.bin", "content_base64": base64.b64encode(data).decode(),
                 "expected_size": 999})
    assert "size mismatch" in out.get("error", ""), out
    assert _stored(worker) == []


def test_path_attach_is_byte_exact_and_reports_sha(worker, tmp_path):
    data = _zip_bytes()
    src = tmp_path / "ops-checkpoint.zip"
    src.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    out = _call({"path": str(src), "expected_sha256": want.upper(), "expected_size": len(data)})
    assert out.get("ok") is True, out
    assert out["sha256"] == want and out["size"] == len(data)
    [att] = _stored(worker)
    assert att.filename == "ops-checkpoint.zip"
    assert att.content_type == "application/zip"
    assert hashlib.sha256(Path(att.stored_path).read_bytes()).hexdigest() == want


def test_path_and_base64_are_mutually_exclusive(worker, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("hi")
    both = _call({"path": str(src), "content_base64": "aGk=", "filename": "a.txt"})
    neither = _call({"filename": "a.txt"})
    assert "exactly one" in both.get("error", "")
    assert "exactly one" in neither.get("error", "")
    assert _stored(worker) == []


def test_path_must_be_absolute_and_not_a_credential_store(worker, tmp_path):
    rel = _call({"path": "relative/report.txt"})
    assert "absolute" in rel.get("error", ""), rel
    home = Path(tmp_path / ".hermes")
    (home / ".env").write_text("SECRET=1\n")
    denied = _call({"path": str(home / ".env")})
    assert "Access denied" in denied.get("error", ""), denied
    assert _stored(worker) == []


def test_valid_base64_archive_and_plain_file_still_attach(worker):
    """Existing base64 callers keep working; verification only adds a reported digest."""
    gz = gzip.compress(b"payload\n" * 100, mtime=0)
    ok_gz = _call({"filename": "a.py.gz", "content_base64": base64.b64encode(gz).decode()})
    ok_txt = _call({"filename": "notes.txt", "content_base64": base64.b64encode(b"hi").decode()})
    assert ok_gz.get("ok") is True and ok_gz["sha256"] == hashlib.sha256(gz).hexdigest()
    assert ok_txt.get("ok") is True
    assert sorted(a.filename for a in _stored(worker)) == ["a.py.gz", "notes.txt"]


def test_attach_url_honours_expected_sha256(worker, monkeypatch):
    """The same fail-closed check guards server-side URL fetches."""
    from tools import kanban_tools as kt
    body = b"fetched"
    monkeypatch.setattr(kt, "_download_url_with_cap", lambda url, cap: (body, "text/plain"))
    import tools.kanban_tools  # noqa: F401
    from tools.registry import registry
    bad = json.loads(registry.dispatch("kanban_attach_url", {
        "url": "https://files.example.com/a.txt", "expected_sha256": "0" * 64}))
    good = json.loads(registry.dispatch("kanban_attach_url", {
        "url": "https://files.example.com/a.txt", "expected_sha256": hashlib.sha256(body).hexdigest()}))
    assert "sha256 mismatch" in bad.get("error", ""), bad
    assert good.get("ok") is True, good
    assert [a.filename for a in _stored(worker)] == ["a.txt"]
