"""tests/test_api_contract.py — ATRIUM API meta-contract conformance (strategy §4, issue #32).

Hermetic contract test: asserts the ``/info`` envelope, ``/health``, ``/ready`` (issue #55), the advertised endpoint
set, and OpenAPI validity against the in-process app. ``importorskip``-guarded and tolerant of
missing service dependencies, so it is a clean no-op in the fast lane and a real check in CI.
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# --- per-service contract parameters -----------------------------------------------------------
SERVICE = "atrium-alto-postprocess"
APP_IMPORT = "service.text_api"
PRIMARY_ENDPOINTS = ["/process"]
# -----------------------------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient

    app = __import__(APP_IMPORT, fromlist=["app"]).app
    client = TestClient(app)
# Only a missing dependency skips (atrium-project#53). This used to be `except Exception`,
# which turned ANY import-time failure into a green skip — including a malformed limit
# (atrium_limits.LimitConfigError), which must fail loudly.
except ImportError as exc:
    app = None
    client = None
    # Pytest will collect the tests but skip running them
    pytestmark = pytest.mark.skip(reason=f"cannot load app: {exc}")


def test_info_envelope_required_fields():
    """§4.1: /info always carries service, version, endpoints, limits.max_upload_mb."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == SERVICE
    assert data["version"] and data["version"] == app.version
    assert isinstance(data["endpoints"], list) and data["endpoints"]
    assert isinstance(data["limits"], dict)
    assert "max_upload_mb" in data["limits"]


def test_info_reports_every_declared_limit():
    """atrium-project#53: /info `limits` is tool_limits.LIMITS, value for value, and
    `limits_meta` names the variable that sets each one. tests/test_limits_contract.py checks
    the declaration against .env.example and the README."""
    from tool_limits import LIMITS

    data = client.get("/info").json()
    assert data["limits"] == LIMITS.values()
    assert data["limits_meta"] == LIMITS.meta()


def test_errors_have_the_harmonised_body():
    """§4.4 (atrium-project#32 item 2): every error is {status, reason, detail}."""
    body = client.get("/no-such-route").json()
    assert body == {"status": 404, "reason": None, "detail": "Not Found"}


def test_info_endpoints_match_real_routes():
    """Advertised endpoints are real routes, and every primary endpoint is advertised."""
    advertised = set(client.get("/info").json()["endpoints"])
    real = {r.path for r in app.routes if getattr(r, "methods", None)}
    assert advertised <= real
    for path in PRIMARY_ENDPOINTS:
        assert path in advertised, f"{path} missing from /info endpoints"


def test_health_shallow_ok():
    """§4.1: shallow /health is a cheap 200 liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_primary_endpoints_documented_in_openapi():
    paths = app.openapi()["paths"]
    for path in PRIMARY_ENDPOINTS:
        assert path in paths, f"{path} missing from OpenAPI paths"


def test_openapi_document_is_spec_valid():
    """The runtime /openapi.json validates against the OpenAPI 3.x spec (§2.2)."""
    spec_validator = pytest.importorskip("openapi_spec_validator")
    spec_validator.validate(app.openapi())


# --- §4.6 readiness + shutdown contract (issue #55) --------------------------------------------
# The state-machine itself is unit-tested once, in the hub
# (atrium-project/docs/templates/shared/test_atrium_service.py). What these assert is that THIS
# repo actually wired it up: the route exists, it is advertised, and — the one that matters —
# liveness does not start failing just because the service is draining.

try:
    _state = getattr(__import__(APP_IMPORT, fromlist=["app"]), "_state", None)
except Exception:  # noqa: BLE001 - same missing-heavy-deps case this file already guards
    # Repos guard the app import two different ways (module-level pytest.skip vs a
    # `deps_present` flag + pytestmark.skipif). Under the second style this module keeps
    # loading after a failed import, so this must not raise at import time; the skip
    # marker already stops the tests below from running.
    _state = None


def test_ready_route_is_registered_and_advertised():
    """§4.6: /ready exists, and /info advertises it like any other route."""
    assert _state is not None, (
        f"{APP_IMPORT} has no module-level `_state` — the service has not adopted "
        "ServiceState/attach_health(state=...) (issue #55)"
    )
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    assert response.json()["status"] in {"ready", "starting", "draining"}
    assert "/ready" in client.get("/info").json()["endpoints"]


def test_ready_reports_starting_before_warmup_and_ready_after():
    """503 until the service's own lifespan marks it warm, 200 once it has.

    `client` above is a bare TestClient, so the ASGI lifespan has NOT run and the service is
    genuinely un-warm here — which is exactly the pre-warmup state a Kubernetes startupProbe
    sees on a cold pod.
    """
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.draining = False
        _state.warm = False
        assert client.get("/ready").status_code == 503
        assert client.get("/ready").json()["status"] == "starting"

        _state.warm = True
        assert client.get("/ready").status_code == 200
        assert client.get("/ready").json()["status"] == "ready"
    finally:
        _state.warm, _state.draining = was_warm, was_draining


def test_liveness_stays_200_while_draining_but_readiness_does_not():
    """The load-bearing distinction of issue #55.

    If shallow /health went 503 on SIGTERM, an orchestrator's livenessProbe would SIGKILL the
    container before its drain finished — the very failure the drain exists to prevent. Routing
    traffic away from a draining pod is /ready's job.
    """
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.warm = True
        _state.draining = True

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "draining"
    finally:
        _state.warm, _state.draining = was_warm, was_draining


def test_deep_health_reports_draining_with_operator_fields():
    """`?deep=true` had no coverage in any repo before issue #55."""
    assert _state is not None
    was_warm, was_draining = _state.warm, _state.draining
    try:
        _state.warm = True
        _state.draining = True
        response = client.get("/health?deep=true")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["detail"] == "shutting down"
        assert body["draining"] is True
        assert "in_flight" in body
    finally:
        _state.warm, _state.draining = was_warm, was_draining


# --- the typed contract (atrium-project#32 round 2) --------------------------------------------
# tests/test_openapi_contract.py (canonical, vendored) checks the committed spec itself. What
# these add is the part only this repo can do: drive /process (with the models replaced by
# canned results shaped exactly like text_inference's, as tests/test_service_api.py does) and
# hold every response — 200s and refusals alike — to the schema the PUBLISHED spec declares
# for it.

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import atrium_openapi  # noqa: E402

_SPEC = atrium_openapi.load(Path(__file__).resolve().parent.parent / "service" / "openapi.json")


def _line(text, category, line_num, **extra):
    """One entry exactly as service/text_inference._classify_line builds it."""
    return {
        "text": text,
        "lang": "ces",
        "lang_score": 0.97,
        "original_lang": "ces",
        "orig_lang_score": 0.97,
        "perplexity": 41.5,
        "garbage_density": 0.02,
        "sym_count": 0,
        "upper_count": 0,
        "repeated_count": 0,
        "ldl_fuses": 0,
        "gibberish": 0,
        "word_weird": 0.1,
        "quality_score": 0.9 if category == "Clear" else 0.2,
        "category": category,
        "line_num": line_num,
        **extra,
    }


_ALTO_RESULT = {"type": "alto_xml", "cleaned_lines": [_line("První řádek", "Clear", 1), _line("|||| ~~", "Trash", 2)]}
_DOC_RESULT = {
    "type": "document",
    "format": "pdf",
    "media_type": "application/pdf",
    "origin": "ocr:pdf-text-layer",
    "pages": [
        {"page": "1", "page_label": "i", "lines": 1, "text_layer": "ocr", "needs_ocr_reason": None},
        {"page": "2", "page_label": "ii", "lines": 1, "text_layer": "ocr", "needs_ocr_reason": None},
    ],
    "cleaned_lines": [
        _line("Strana jedna", "Clear", 1, page="1", page_label="i"),
        _line("Str4na dv4", "Trash", 1, page="2", page_label="ii"),
    ],
}
_ONE_PAGE_ALTO = (
    b'<?xml version="1.0" encoding="UTF-8"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v3#"><Layout>'
    b'<Page ID="P1" PHYSICAL_IMG_NR="7" WIDTH="1000" HEIGHT="2000"><PrintSpace/></Page></Layout></alto>'
)
_SEED_ID = "C-202000543A-DT-27"
#: A #67 R1 seed: the AMČR file id and the source, nothing else.
_SEED = {"doc_id": _SEED_ID, "source": {"sha256": "a" * 64, "filename": "scan.alto.xml", "origin": "ABBYY-ALTO"}}


def _conforms(status, response, path="/process", method="post"):
    pytest.importorskip("jsonschema")
    assert response.status_code == status, response.text
    atrium_openapi.validate_response(_SPEC, path, method, status, response.json())
    return response.json()


@patch("service.text_api.text_manager.process_alto", create=True)
def test_an_alto_response_conforms_to_the_published_schema(mock_process):
    mock_process.return_value = json.loads(json.dumps(_ALTO_RESULT))
    body = _conforms(200, client.post("/process", files={"file": ("p.alto.xml", _ONE_PAGE_ALTO, "application/xml")}))
    assert body["type"] == "alto_xml" and body["limits_applied"] == []
    assert "document_json" not in body and "document_json_out" not in body


@patch("service.text_api.text_manager.process_document", create=True)
def test_a_document_response_conforms_including_its_pages(mock_process):
    mock_process.return_value = json.loads(json.dumps(_DOC_RESULT))
    response = client.post(
        "/process", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, data={"task_type": "document"}
    )
    body = _conforms(200, response)
    assert [page["page_label"] for page in body["pages"]] == ["i", "ii"]


@patch("service.text_api.text_manager.process_document", create=True)
def test_a_pdf_run_logs_pdfium_in_its_paradata(mock_process, tmp_path, monkeypatch):
    """The API path logs the reader components the CLI logs (atrium-project#6, #72): a PDF read
    by PDFium puts `pypdfium2` in the run's licence detail, as text_split.py always did."""
    from atrium_paradata import ParadataLogger

    monkeypatch.chdir(tmp_path)  # ParadataLogger writes ./paradata
    logged = []
    monkeypatch.setattr(ParadataLogger, "log_component", lambda self, name, license=None: logged.append(name))

    def read(path, notes=None, components=None):
        components.add("pypdfium2")
        return json.loads(json.dumps(_DOC_RESULT))

    mock_process.side_effect = read
    response = client.post(
        "/process", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, data={"task_type": "document"}
    )
    assert response.status_code == 200, response.text
    assert logged == ["pypdfium2"]


@pytest.mark.parametrize("part, key", [("document_json", "document_json"), ("document_record", "document_json_out")])
@patch("service.text_api.text_manager.process_alto", create=True)
def test_a_seed_comes_back_accreted_under_the_name_it_was_sent_with(mock_process, part, key, tmp_path, monkeypatch):
    """`document_json` in → `document_json` out, as every ATRIUM service; the deprecated
    `document_record` → `document_json_out`, as before. The returned record is held to the
    vendored record schema, through the spec's AtriumDocument component — the type AMČR's
    generated client deserialises it into."""
    monkeypatch.chdir(tmp_path)  # ParadataLogger writes ./paradata
    mock_process.return_value = json.loads(json.dumps(_ALTO_RESULT))
    files = {
        "file": ("scan_0001.alto.xml", _ONE_PAGE_ALTO, "application/xml"),
        part: ("seed.document.json", json.dumps(_SEED).encode("utf-8"), "application/json"),
    }
    body = _conforms(200, client.post("/process", files=files))
    other = "document_json_out" if key == "document_json" else "document_json"
    assert other not in body
    record = body[key]
    assert record["doc_id"] == _SEED_ID
    assert [line["text"] for line in record["lines"]] == ["První řádek", "|||| ~~"]


@patch("service.text_api.text_manager.process_alto", create=True)
def test_a_seed_with_a_byte_order_mark_is_accepted(mock_process, tmp_path, monkeypatch):
    """parse_record_part accepts a UTF-8 BOM, and so does atrium_document.load_document
    (utf-8-sig): the record the gate lets through must not fail later as a 500."""
    monkeypatch.chdir(tmp_path)
    mock_process.return_value = json.loads(json.dumps(_ALTO_RESULT))
    files = {
        "file": ("scan_0001.alto.xml", _ONE_PAGE_ALTO, "application/xml"),
        "document_json": (
            "seed.document.json",
            b"\xef\xbb\xbf" + json.dumps(_SEED).encode("utf-8"),
            "application/json",
        ),
    }
    assert _conforms(200, client.post("/process", files=files))["document_json"]["doc_id"] == _SEED_ID


@pytest.mark.parametrize("record", [b"[1, 2]", b"{not json", b'{"schema_version": "9.0", "doc_id": "x"}'])
@patch("service.text_api.text_manager.process_alto", create=True)
def test_a_record_that_cannot_be_opened_is_422_invalid_record_before_any_model_runs(mock_process, record):
    files = {
        "file": ("p.alto.xml", _ONE_PAGE_ALTO, "application/xml"),
        "document_record": ("r.document.json", record, "application/json"),
    }
    body = _conforms(422, client.post("/process", files=files))
    assert body["reason"] == "invalid_record" and "document_record" in body["detail"]
    assert mock_process.call_count == 0


@patch("service.text_api.text_manager.process_alto", create=True)
def test_an_empty_record_part_counts_as_none(mock_process):
    mock_process.return_value = json.loads(json.dumps(_ALTO_RESULT))
    files = {
        "file": ("p.alto.xml", _ONE_PAGE_ALTO, "application/xml"),
        "document_json": ("r.document.json", b"", "application/json"),
    }
    assert "document_json" not in _conforms(200, client.post("/process", files=files))


@patch("service.text_api.text_manager.process_alto", create=True)
def test_a_record_sent_under_both_names_is_refused(mock_process):
    seed = json.dumps(_SEED).encode("utf-8")
    files = {
        "file": ("p.alto.xml", _ONE_PAGE_ALTO, "application/xml"),
        "document_json": ("a.document.json", seed, "application/json"),
        "document_record": ("b.document.json", seed, "application/json"),
    }
    body = _conforms(422, client.post("/process", files=files))
    assert body["reason"] is None and "not both" in body["detail"] and mock_process.call_count == 0


def test_an_unsupported_upload_is_415_with_cause_and_accepted():
    files = {"file": ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png")}
    body = _conforms(415, client.post("/process", files=files))
    assert (body["reason"], body["cause"]) == ("unsupported_media_type", "image_needs_ocr")
    assert ".pdf" in body["accepted"] and body["detail"].startswith("Cannot auto-detect a supported file type")


@patch("service.text_api.text_manager.process_document", create=True)
def test_a_missing_reader_dependency_is_501_not_a_client_error(mock_process):
    from text_formats import IngestError

    mock_process.side_effect = IngestError("dependency_missing", "pypdfium2 is required for PDF input")
    response = client.post(
        "/process", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}, data={"task_type": "document"}
    )
    body = _conforms(501, response)
    assert (body["reason"], body["cause"]) == (None, "dependency_missing")


@pytest.mark.parametrize(
    "content, cause", [(b'{"pages": [1, 2', "malformed"), (b'{"text": "\xff\xfe"}', "decode_failed")]
)
def test_a_json_upload_that_does_not_parse_is_422_not_500(content, cause):
    """text_manager.process_json opens the file before any model runs, so the real one is used."""
    body = _conforms(422, client.post("/process", files={"file": ("p.json", content, "application/json")}))
    assert (body["reason"], body["cause"]) == (None, cause) and body["detail"].startswith(cause)


def test_an_unknown_task_type_is_422():
    """It used to be read as `text`; the spec lists the five values."""
    response = client.post("/process", files={"file": ("p.txt", b"x", "text/plain")}, data={"task_type": "txt"})
    body = _conforms(422, response)
    assert body["errors"] and body["reason"] is None
    schema = _SPEC["components"]["schemas"]["Body_process_document"]["properties"]["task_type"]
    assert schema["enum"] == ["auto", "alto", "text", "json", "document"]


def test_a_draining_replica_refuses_new_work_with_the_error_body():
    assert _state is not None
    was_draining = _state.draining
    try:
        _state.draining = True
        body = _conforms(503, client.post("/process", files={"file": ("p.txt", b"x", "text/plain")}))
        assert body["reason"] is None
    finally:
        _state.draining = was_draining


def test_info_conforms_to_the_published_schema():
    _conforms(200, client.get("/info"), path="/info", method="get")
