"""tool_limits.py — every limit atrium-alto-postprocess has (atrium-project#53, factor III).

One declaration, read by the service (``service/text_api.py``, ``service/text_inference.py``)
and the readers (``text_formats.py``), and reported by ``GET /info`` (``limits`` and
``limits_meta``). ``.env.example`` and ``service/README.md``'s ``## Limits`` table list the
same set; ``tests/test_limits_contract.py`` checks that they agree.

**The ``[TEXT_INGEST]`` caps** are also keys of ``setup/config.txt`` (the file ``LANGID_CONFIG``
names), which the batch pipeline reads. Each is now an environment setting too, named in
alto's existing override pattern ``ATRIUM_<SECTION>_<KEY>`` (``text_util.py``): the environment
wins, then the config file, then the default below. Their ``/info`` keys are the config keys
in lower case. A malformed value — in the environment or the file — stops the service at
startup, naming it.

What happens over each limit — refused (with the HTTP status), or processed in full with a
``limits_applied`` note (with the effect) — is said beside it.

Standard library only (``atrium_limits`` is the hub's canonical module at the repo root).
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Dict, Optional

from atrium_limits import LimitSet, limit, upload_limit

_REPO_ROOT = Path(__file__).resolve().parent

#: §4.5 upload limit, per uploaded part (the file and the document record). Over it → 413.
MAX_UPLOAD = upload_limit(25)

# ── [TEXT_INGEST]: caps applied before and while a document is read ─────────────────────
#: Largest file (or decompressed stream) a reader takes. Over it → 413 ``too_large``; a ZIP
#: bundle member over it is skipped instead → ``skipped`` note.
MAX_FILE_MB = limit("ATRIUM_TEXT_INGEST_MAX_FILE_MB", 256, unit="MB", kind=float, key="max_file_mb", minimum=0.001)
#: Most members a ZIP container may have. Over it → 413 ``zip_limits_exceeded``.
ZIP_MAX_MEMBERS = limit("ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBERS", 10000, unit="members", key="zip_max_members", minimum=1)
#: Most a ZIP container may unpack to in total. Over it → 413 ``zip_limits_exceeded``.
ZIP_MAX_TOTAL_MB = limit(
    "ATRIUM_TEXT_INGEST_ZIP_MAX_TOTAL_MB", 1024, unit="MB", kind=float, key="zip_max_total_mb", minimum=0.001
)
#: Largest single ZIP member, unpacked. Over it → 413 ``zip_limits_exceeded``.
ZIP_MAX_MEMBER_MB = limit(
    "ATRIUM_TEXT_INGEST_ZIP_MAX_MEMBER_MB", 256, unit="MB", kind=float, key="zip_max_member_mb", minimum=0.001
)
#: Highest compression ratio a ZIP member or compressed stream may have. Over it → 413.
ZIP_MAX_RATIO = limit("ATRIUM_TEXT_INGEST_ZIP_MAX_RATIO", 200, unit="ratio", kind=float, key="zip_max_ratio", minimum=1)
#: Most pages a document may have. Over it → 413 ``too_large``.
MAX_PAGES = limit("ATRIUM_TEXT_INGEST_MAX_PAGES", 20000, unit="pages", key="max_pages", minimum=1)
#: Most lines one page may have. A paged format (PDF, ALTO, PAGE, hOCR, …) over it → 413
#: ``too_large``; any other format is split onto continuation pages and read in full →
#: ``split`` note.
MAX_LINES_PER_PAGE = limit(
    "ATRIUM_TEXT_INGEST_MAX_LINES_PER_PAGE", 100000, unit="lines", key="max_lines_per_page", minimum=1
)
#: Seconds the isolated PDF reader may take. Over it → 422 ``timeout`` (a per-input budget,
#: not the request's size).
READER_TIMEOUT_S = limit(
    "ATRIUM_TEXT_INGEST_READER_TIMEOUT_S", 300, unit="s", kind=float, key="reader_timeout_s", minimum=1, status=422
)
#: Longest line, in characters; a longer one is wrapped at a word boundary (hard-split when
#: it has none) and classified in full, as several lines → ``split`` note.
MAX_LINE_CHARS = limit("ATRIUM_TEXT_INGEST_MAX_LINE_CHARS", 1000, unit="chars", key="max_line_chars")

# ── reader caps that were hardcoded in text_formats.py ──────────────────────────────────
#: Most times an ODS/ODT cell or row is repeated (``number-*-repeated``); the rest is not
#: read → ``trimmed`` note.
ODF_REPEAT_CAP = limit("ATRIUM_TEXT_INGEST_ODF_REPEAT_CAP", 100, unit="repeats", key="odf_repeat_cap", minimum=1)
#: Most objects of one PDF page scanned to judge its text layer (the text itself is always
#: read in full); a page with more is judged on the first N → ``sampled`` note.
PDF_OBJECT_CAP = limit("ATRIUM_TEXT_INGEST_PDF_OBJECT_CAP", 20000, unit="objects", key="pdf_object_cap", minimum=1)

# ── the models ──────────────────────────────────────────────────────────────────────────
#: Lines per LayoutReader window when predicting an ALTO page's reading order. A longer page
#: is ordered window by window and the windows joined in document order → ``split`` note.
LR_CHUNK_SIZE = limit("LR_CHUNK_SIZE", 350, unit="lines", minimum=1)
#: Smallest window the out-of-memory fallback halves down to; below it the page keeps its
#: document order unordered by the model → ``skipped`` note.
LR_MIN_CHUNK_SIZE = limit("LR_MIN_CHUNK_SIZE", 50, unit="lines", minimum=1)
#: Lines per perplexity forward pass (one batch is padded to its longest line). A failed
#: batch scores its lines 99999 (worst quality) → ``skipped`` note.
PPL_BATCH_LINES = limit("PPL_BATCH_LINES", 128, unit="lines", minimum=1)

#: The [TEXT_INGEST] keys, by the environment variable that overrides each.
_INGEST_KEYS: Dict[str, str] = {
    spec.env: spec.key.upper()
    for spec in (
        MAX_FILE_MB,
        ZIP_MAX_MEMBERS,
        ZIP_MAX_TOTAL_MB,
        ZIP_MAX_MEMBER_MB,
        ZIP_MAX_RATIO,
        MAX_PAGES,
        MAX_LINES_PER_PAGE,
        READER_TIMEOUT_S,
        MAX_LINE_CHARS,
    )
}


def config_path() -> Path:
    """The config file the service and the batch stages read (``LANGID_CONFIG``)."""
    return Path(os.getenv("LANGID_CONFIG", str(_REPO_ROOT / "setup" / "config.txt")))


def ingest_config_values() -> Dict[str, str]:
    """``{environment variable: raw [TEXT_INGEST] value}`` for the keys the config file sets."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(config_path(), encoding="utf-8")
    if not cfg.has_section("TEXT_INGEST"):
        return {}
    out = {}
    for env, key in _INGEST_KEYS.items():
        raw = cfg.get("TEXT_INGEST", key, fallback="").strip()
        if raw:
            out[env] = raw
    return out


#: The perplexity model's token window, set by the service once the model is loaded.
_PPL_WINDOW: Dict[str, Optional[int]] = {"tokens": None}


def set_ppl_max_tokens(tokens: Optional[int]) -> None:
    _PPL_WINDOW["tokens"] = tokens


def ppl_max_tokens() -> Optional[int]:
    """Tokens of one line the perplexity model reads (its context window, from
    ``GPT2_MODEL_NAME``); a longer line is scored on its first N → ``trimmed`` note.
    ``None`` until the model is loaded."""
    return _PPL_WINDOW["tokens"]


LIMITS = LimitSet(
    MAX_UPLOAD,
    MAX_FILE_MB,
    ZIP_MAX_MEMBERS,
    ZIP_MAX_TOTAL_MB,
    ZIP_MAX_MEMBER_MB,
    ZIP_MAX_RATIO,
    MAX_PAGES,
    MAX_LINES_PER_PAGE,
    READER_TIMEOUT_S,
    MAX_LINE_CHARS,
    ODF_REPEAT_CAP,
    PDF_OBJECT_CAP,
    LR_CHUNK_SIZE,
    LR_MIN_CHUNK_SIZE,
    PPL_BATCH_LINES,
    config=ingest_config_values,
)
LIMITS.derived("ppl_max_tokens", ppl_max_tokens, unit="tokens", derived_from=["GPT2_MODEL_NAME"])
