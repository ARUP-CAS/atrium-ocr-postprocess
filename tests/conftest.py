"""
tests/conftest.py
=================
Shared pytest fixtures for the atrium-alto-postprocess test suite.

Nothing here requires ML models, GPU, or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── Ensure repo root is importable ────────────────────────────────────────
# pytest.ini already sets pythonpath = . for pytest ≥ 7.
# This guard keeps things working with older pytest versions too.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── The archive's allow list, held out ────────────────────────────────────
# (#30 Q5a/Q5b) Since 2026-10-01 `setup/word_lists.txt` `[allowed]` ships
# @david-spacil's reviewed entries, and a listed word is never evidence of damage
# to the shape witness. Tests of OTHER mechanisms -- the lexicon, the language
# split, the witness report -- use exactly those words (`ssuti`, `Dauerleihe`,
# `Kaukasus`, `Triticum monococcum`) as their "convicted by shape alone" examples,
# and would otherwise be measuring the list instead of the mechanism they name.
#
# This fixture is the shipped file with ONLY the `[allowed]` entries removed:
# every other section, and every comment, stays exactly as shipped. The list's
# own behaviour is pinned in tests/test_word_lists.py, at the shipped config.
@pytest.fixture
def without_allowed_words(tmp_path):
    import text_util as tu

    shipped = (_ROOT / "setup" / "word_lists.txt").read_text(encoding="utf-8")
    kept: list[str] = []
    in_allowed = False
    for line in shipped.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_allowed = stripped.lower() == "[allowed]"
        elif in_allowed and stripped and not stripped.startswith("#"):
            continue
        kept.append(line)
    path = tmp_path / "word_lists_without_allowed.txt"
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with tu.override_constants({"WORD_LISTS_PATH": str(path)}):
        assert tu.word_list("allowed", frozenset()) == frozenset()
        yield path
