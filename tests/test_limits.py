"""tests/test_limits.py — every limit is a setting, and none cuts an input quietly.

atrium-project#53 (factor III), for this repo: the limits are declared in tool_limits.py, the
[TEXT_INGEST] caps can be set from the environment as well as from the config file, an input
over a size limit is refused with 413 ``limit_exceeded``, and every limit that shapes a result
without refusing it is returned in ``limits_applied``. tests/test_limits_contract.py
(canonical) checks the declaration against .env.example and service/README.md.
"""

from __future__ import annotations

import zipfile
from unittest.mock import MagicMock, patch

import pytest

import text_formats as tf
import tool_limits
from atrium_limits import LimitNotes
from tests.text_format_fixtures import make_zip, odf_bytes


def _clear_ingest_settings():
    """Reset the cached [TEXT_INGEST] settings of BOTH loaded copies of text_inference.

    service/text_api.py imports the bare `text_inference` (its sys.path bootstrap), while
    these tests import `service.text_inference`: two module objects, each with its own
    lru_cache. Clearing only the second left the cache /process reads untouched, so a test
    that posted through the app after any earlier /process call had filled it (a sniffed
    upload is enough) ran with that call's limits. Nothing ran first until the conformance
    tests of atrium-project#32 round 2, which sort before this file.
    """
    import sys

    import service.text_inference  # noqa: F401  (loaded, so the loop below always finds it)

    for name in ("text_inference", "service.text_inference"):
        module = sys.modules.get(name)
        if module is not None:
            module.ingest_settings.cache_clear()


@pytest.fixture
def ingest_config(tmp_path, monkeypatch):
    """Point LANGID_CONFIG at a scratch config and reset the cached settings around the test."""

    def write(text):
        path = tmp_path / "config.txt"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("LANGID_CONFIG", str(path))
        _clear_ingest_settings()

    _clear_ingest_settings()
    yield write
    _clear_ingest_settings()


# ── [TEXT_INGEST]: environment → config file → default ──────────────────────────────────


def test_the_environment_wins_over_the_config_file(ingest_config, monkeypatch):
    import service.text_inference as ti

    ingest_config("[TEXT_INGEST]\nMAX_PAGES = 7\nMAX_LINE_CHARS = 80\n")
    limits, options, _c, _k = ti.ingest_settings()
    assert (limits.max_pages, options.max_line_chars) == (7, 80)
    assert tool_limits.LIMITS.values()["max_pages"] == 7
    assert tool_limits.LIMITS.meta()["max_pages"]["source"] == "config"

    monkeypatch.setenv("ATRIUM_TEXT_INGEST_MAX_PAGES", "3")
    ti.ingest_settings.cache_clear()
    limits, _o, _c, _k = ti.ingest_settings()
    assert limits.max_pages == 3
    assert tool_limits.LIMITS.values()["max_pages"] == 3
    assert tool_limits.LIMITS.meta()["max_pages"]["source"] == "env"


def test_the_batch_stages_honour_the_environment_too(monkeypatch):
    """text_split.py, extract_TEXT_2_TXT.py and text_stats_create.py call load_settings directly."""
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read_string("[TEXT_INGEST]\nMAX_PAGES = 7\n")
    assert tf.load_settings(cfg)[0].max_pages == 7
    monkeypatch.setenv("ATRIUM_TEXT_INGEST_MAX_PAGES", "3")
    monkeypatch.setenv("ATRIUM_TEXT_INGEST_MAX_LINE_CHARS", "90")
    limits, options = tf.load_settings(cfg)
    assert (limits.max_pages, options.max_line_chars) == (3, 90)
    assert tf.load_settings(None)[0].max_pages == 3


def test_a_malformed_ingest_variable_fails_like_a_malformed_config_key(ingest_config, monkeypatch):
    import service.text_inference as ti

    ingest_config("[TEXT_INGEST]\n")
    monkeypatch.setenv("ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBERS", "many")
    with pytest.raises(ValueError, match="ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBERS"):
        ti.ingest_settings()


# ── reader notes ─────────────────────────────────────────────────────────────────────────


def test_a_zip_member_over_max_file_mb_is_noted_apart_from_other_skips(tmp_path):
    # Deflated, so the file itself is under the 0.001 MB cap and only the member, unpacked, is over it.
    data = make_zip([("1.txt", "Strana jedna"), ("2.txt", "x" * 5000)], compression=zipfile.ZIP_DEFLATED)
    path = tmp_path / "b.zip"
    path.write_bytes(data)
    doc = tf.read_document(str(path), limits=tf.Limits(max_file_mb=0.001))
    assert "zip_members_over_max_file_mb=1" in doc.notes


def test_the_odf_repeat_cap_is_a_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("ATRIUM_TEXT_INGEST_ODF_REPEAT_CAP", "3")
    body = (
        '<table:table table:name="T"><table:table-row table:number-rows-repeated="50">'
        '<table:table-cell office:value-type="string"><text:p>opak</text:p></table:table-cell>'
        "</table:table-row></table:table>"
    )
    path = tmp_path / "s.ods"
    path.write_bytes(odf_bytes("spreadsheet", body))
    doc = tf.read_document(str(path))
    assert len(doc.pages[0].lines) == 3
    assert "sheet_repeat_capped" in doc.notes


def test_reader_notes_become_limit_notes():
    from service.text_inference import _note_reader_limits

    notes = LimitNotes()
    _note_reader_limits(
        ["bundle_members=3", "zip_members_over_max_file_mb=2", "page_overflow_split", "pdf_objects_capped", "x=1"],
        notes,
    )
    got = {(n["limit"], n["effect"], n["count"]) for n in notes.as_list()}
    assert got == {("max_file_mb", "skipped", 2), ("max_lines_per_page", "split", 1), ("pdf_object_cap", "sampled", 1)}


# ── the models ───────────────────────────────────────────────────────────────────────────


def _manager():
    from service.text_inference import TextModelManager

    m = TextModelManager()
    ft = MagicMock()
    ft.predict.return_value = ([["__label__ces"]], [[0.9]])  # as tests/test_text_inference.py's _mock_ft
    m.ft_model = ft
    m.ppl_model = MagicMock()
    m.ppl_model.config.max_position_embeddings = 8
    m.ppl_tokenizer = MagicMock(side_effect=lambda text: {"input_ids": text.split()})
    m.device = "cpu"
    return m


def test_long_lines_are_wrapped_and_noted(tmp_path, monkeypatch, ingest_config):
    ingest_config("[TEXT_INGEST]\nMAX_LINE_CHARS = 20\n")
    monkeypatch.setattr("service.text_inference.calculate_perplexity_batch", lambda t, *a: [50.0] * len(t))
    (tmp_path / "doc.txt").write_text("krátký\n" + "slovo " * 10 + "\n", encoding="utf-8")
    notes = LimitNotes()
    result = _manager().process_text_file(str(tmp_path / "doc.txt"), notes=notes)
    assert len(result["cleaned_lines"]) > 2
    [entry] = [n for n in notes.as_list() if n["limit"] == "max_line_chars"]
    assert (entry["effect"], entry["count"], entry["value"]) == ("split", 1, 20)


def test_perplexity_is_batched_on_every_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_BATCH_LINES", "4")
    calls = []

    def fake(texts, *_a):
        calls.append(len(texts))
        return [50.0] * len(texts)

    monkeypatch.setattr("service.text_inference.calculate_perplexity_batch", fake)
    lines = [f"řádek {i}" for i in range(10)]
    assert len(_manager()._classify_lines(lines)) == 10
    assert calls == [4, 4, 2]


def test_a_failed_perplexity_batch_is_noted(monkeypatch):
    monkeypatch.setattr("service.text_inference.calculate_perplexity_batch", lambda t, *a: [99999.0] * len(t))
    notes = LimitNotes()
    _manager()._classify_lines(["jedna", "dva"], notes)
    [entry] = notes.as_list()
    assert (entry["limit"], entry["effect"], entry["count"]) == ("ppl_batch_lines", "skipped", 2)


def test_a_line_over_the_perplexity_window_is_noted(monkeypatch):
    monkeypatch.setattr("service.text_inference.calculate_perplexity_batch", lambda t, *a: [50.0] * len(t))
    notes = LimitNotes()
    # The fake tokenizer makes one token per word; the window is 8 tokens.
    _manager()._classify_lines(["krátký řádek", "a b c d e f g h i j k l"], notes)
    [entry] = notes.as_list()
    assert (entry["limit"], entry["effect"], entry["count"], entry["value"]) == ("ppl_max_tokens", "trimmed", 1, 8)


# ── the service ──────────────────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from service.text_api import app

    return TestClient(app)


def test_info_reports_every_limit(client):
    data = client.get("/info").json()
    assert data["limits"] == tool_limits.LIMITS.values()
    assert data["limits_meta"]["max_pages"]["env"] == "ATRIUM_TEXT_INGEST_MAX_PAGES"
    assert data["limits_meta"]["ppl_max_tokens"]["source"] == "derived"


def test_a_document_over_max_pages_is_413_limit_exceeded(client, ingest_config, monkeypatch):
    ingest_config("[TEXT_INGEST]\n")
    monkeypatch.setenv("ATRIUM_TEXT_INGEST_MAX_PAGES", "1")
    files = {"file": ("doc.md", "strana\fdruhá strana\n".encode(), "text/markdown")}
    response = client.post("/process", files=files, data={"task_type": "document"})
    assert response.status_code == 413
    body = response.json()
    assert body["reason"] == "limit_exceeded" and body["detail"].startswith("too_large: 2 pages > MAX_PAGES=1")
    assert body["limit"]["env"] == "ATRIUM_TEXT_INGEST_MAX_PAGES" and body["limit"]["value"] == 1


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_the_response_carries_limits_applied(mock_process, client):
    def process(path, notes=None, components=None):
        notes.note(tool_limits.MAX_LINE_CHARS, "split", 3)
        return {"type": "plain_text", "cleaned_lines": []}

    mock_process.side_effect = process
    response = client.post("/process", files={"file": ("d.txt", b"text", "text/plain")}, data={"task_type": "text"})
    assert response.status_code == 200
    assert response.json()["limits_applied"][0]["count"] == 3


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_a_file_name_with_a_path_or_over_the_os_limit_is_not_a_500(mock_process, client):
    mock_process.return_value = {"type": "plain_text", "cleaned_lines": []}
    for name in ("../../etc/passwd.txt", "dlouhý-název-" * 40 + ".txt"):
        response = client.post("/process", files={"file": (name, b"text", "text/plain")}, data={"task_type": "text"})
        assert response.status_code == 200, name
        assert response.json()["filename"] == name


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_an_oversized_document_record_is_413(mock_process, client, monkeypatch):
    mock_process.return_value = {"type": "plain_text", "cleaned_lines": []}
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    response = client.post(
        "/process",
        files={
            "file": ("CTX01.txt", b"text", "text/plain"),
            "document_record": ("r.json", b"{" + b" " * 4096 + b"}", "application/json"),
        },
        data={"task_type": "text"},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "document_record too large: over 0.001 MB (MAX_UPLOAD_MB)."


def test_paradata_reads_this_repos_para_config(tmp_path, monkeypatch):
    """config_dir points at setup/, so the API record carries the real tool version."""
    from pathlib import Path

    import atrium_paradata
    from service.text_api import PARA_CONFIG_PATH

    logger = atrium_paradata.ParadataLogger(
        config=PARA_CONFIG_PATH,
        program="alto-postprocess",
        paradata_dir=str(tmp_path),
        config_dir=str(Path(PARA_CONFIG_PATH).parent),
    )
    assert logger.version not in ("", "unknown")
