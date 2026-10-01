"""
tests/test_domain_notation_categ.py
===================================
(#30 D43) Covers the web/e-mail route and the config key that decides its answer.

@david-spacil settled what the five categories mean on 2026-09-22: *"`Trash` =
illegible. Anything legible is `Clear`, easily decipherable is `Noisy` --
regardless of how useful the line is to us."* `http://www.arub.cz` is scanned
perfectly correctly on 5,309 lines, so by that definition it is `Clear`. D33 had
already moved it `Trash` -> `Noisy`, the right direction and one step short.

`DOMAIN_NOTATION_CATEG` is a CATEGORY NAME, not an on/off switch, because this
one line has had three different answers inside this issue and a fourth is
defensible. Settling it should be a config edit and a re-score, not a code change
per answer. It shipped EMPTY (off) until 2026-10-01, when @david-spacil and
@DanaKriv answered Q4: `Clear`.

What is asserted here is the WIRING and the BOUNDARIES: that the shipped value
is the answer given, that empty is off, that every valid value is honoured, that an invalid value fails loudly
rather than silently doing nothing, that the route reaches LONG addresses and not
only short ones, and that it does not touch lines carrying no address.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from classify_TEXT import score_line
from text_util import CONFIG_GATED_RULES, override_constants, rule_fire_capture

_KNOWN_BASES = frozenset(["ces", "deu", "eng", "fra", "pol", "ita", "slk"])
_EXPECTED = ["ces", "deu", "eng"]

#: Addresses of both shapes, and both lengths. The long one is the point of the
#: rule being in the cascade rather than in the short-line gate: a citation and a
#: bare domain are the same kind of thing to a reader.
ADDRESSES = [
    "http://www.arub.cz",
    "www.archaiabrno.cz",
    "e-mail: mhauer@zip-ops.cz",
    "roku 1820 (http://www.hrady.cz/index.php?OID=1291).",
    "1 https://www.mza.cz/indikacniskici/skica/detail/1669",
]

#: No address anywhere in them. The route must not reach these in any setting.
NOT_ADDRESSES = ["oueussd", "vrstva 3", "OUUITN", "Dauerleihe", "sonda 9"]


def _categ(text: str, *, perplexity: float = 900.0) -> str:
    return score_line(
        text_content=text,
        original_text=text,
        original_lang="ces_Latn",
        original_lang_score=0.4,
        perplexity=perplexity,
        known_lang_bases=_KNOWN_BASES,
        expected_langs=_EXPECTED,
    )["categ"]


def test_the_shipped_value_is_the_q4_answer():
    """Q4, answered 2026-10-01 by @david-spacil and @DanaKriv: `Clear`."""
    import text_util as tu

    assert tu.DOMAIN_NOTATION_CATEG == "Clear"


@pytest.mark.parametrize("text", ADDRESSES)
def test_every_address_is_clear_at_the_shipped_config(text):
    assert _categ(text) == "Clear"


@pytest.mark.parametrize("text", ADDRESSES + NOT_ADDRESSES)
def test_empty_is_off(text):
    """Empty still means the cascade decides as it did before the route existed."""
    with override_constants({"DOMAIN_NOTATION_CATEG": ""}):
        with rule_fire_capture() as counts:
            _categ(text)
    assert counts.get("rule_domain_notation_categ", 0) == 0


@pytest.mark.parametrize("categ", ["Clear", "Noisy", "Trash", "Non-text"])
@pytest.mark.parametrize("text", ADDRESSES)
def test_every_configured_category_is_honoured(text, categ):
    with override_constants({"DOMAIN_NOTATION_CATEG": categ}):
        assert _categ(text) == categ


@pytest.mark.parametrize("text", NOT_ADDRESSES)
def test_a_line_with_no_address_is_never_touched(text):
    """The route's reach is the pattern and nothing else."""
    before = _categ(text)
    for categ in ("Clear", "Noisy", "Trash", "Non-text"):
        with override_constants({"DOMAIN_NOTATION_CATEG": categ}):
            assert _categ(text) == before, f"{text!r} moved under DOMAIN_NOTATION_CATEG={categ}"


def test_it_reaches_a_line_the_hard_sweep_would_otherwise_take():
    """Why the rule runs FIRST, stated as a measurement rather than a preference.

    `roku 1820 (http://www.hrady.cz/...)` is Trash today at this perplexity --
    `rule_hard_sweep` takes it, and hard sweep is deliberately NOT exempted for
    notation. Perplexity on a domain name is noise, so if the archive has a rule
    for addresses, that rule has to be reached before the sweep or it is only a
    rule for the addresses the sweep happened to leave alone.
    """
    citation = "roku 1820 (http://www.hrady.cz/index.php?OID=1291)."
    with override_constants({"DOMAIN_NOTATION_CATEG": ""}):
        assert _categ(citation, perplexity=5000.0) == "Trash"
    with override_constants({"DOMAIN_NOTATION_CATEG": "Clear"}):
        assert _categ(citation, perplexity=5000.0) == "Clear"


@pytest.mark.parametrize("bad", ["Readable", "clear", "TRASH", "nontext"])
def test_an_unrecognised_category_fails_loudly(bad):
    """A typo must not read as "off".

    The whole point of a category-valued key is that someone will edit it, and a
    value of `clear` or `Readable` silently doing nothing is the failure mode
    this issue has logged twice already under other names. Case matters too: the
    five labels are spelled one way and compared by identity elsewhere.

    Run in a SUBPROCESS, not by reloading the module. The constant is read at
    import, so the in-process version of this test would have to
    `importlib.reload(text_util)` -- which replaces the module object and breaks
    every other test that compares a function's identity across modules
    (tests/test_scoring_single_source.py does exactly that). Reloading passed here
    and failed four tests in three other files.
    """
    env = {**os.environ, "ATRIUM_TEXT_UTILS_DOMAIN_NOTATION_CATEG": bad}
    proc = subprocess.run(
        [sys.executable, "-c", "import text_util"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, f"{bad!r} was accepted silently"
    assert "DOMAIN_NOTATION_CATEG" in proc.stderr


def test_a_valid_category_imports_cleanly_from_the_environment():
    """The counterpart, so the test above cannot pass because import is broken."""
    env = {**os.environ, "ATRIUM_TEXT_UTILS_DOMAIN_NOTATION_CATEG": "Noisy"}
    proc = subprocess.run(
        [sys.executable, "-c", "import text_util; print(text_util.DOMAIN_NOTATION_CATEG)"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "Noisy"


def test_the_rule_name_fires_only_when_a_category_is_configured():
    """Coverage and ablation must see the site, not just its absence."""
    with override_constants({"DOMAIN_NOTATION_CATEG": ""}):
        with rule_fire_capture() as counts:
            _categ("http://www.arub.cz")
    assert counts.get("rule_domain_notation_categ", 0) == 0

    with override_constants({"DOMAIN_NOTATION_CATEG": "Clear"}):
        with rule_fire_capture() as counts:
            _categ("http://www.arub.cz")
    assert counts.get("rule_domain_notation_categ", 0) == 1


def test_the_rule_is_declared_config_gated():
    """Without this, rule_coverage_report calls the site DEAD and nominates it
    for deletion -- which is exactly what happened to the shape witness (D35).
    """
    assert CONFIG_GATED_RULES["rule_domain_notation_categ"] == "DOMAIN_NOTATION_CATEG"


def test_a_mis_scanned_address_is_not_separable_and_gets_the_same_category():
    """The route's KNOWN LIMIT, pinned rather than described.

    `e-mail: officeauappmost.cz` is the same label with the `@` lost to the
    scanner. Under the settled definition it is `Noisy` -- there IS a mistake in
    it -- but nothing in this pipeline separates it from a correctly scanned
    address, so a configured category applies to both.

    This test exists because an earlier draft of this route asserted the
    opposite, that a damaged address would stay `Noisy`, and running it showed
    `detect_fused_words` fires on the address SHAPE, correct and damaged alike.
    Recording the limit is honest; asserting a separation the code cannot make
    is not. It is also the reason `Noisy` is a defensible setting for this key.
    """
    with override_constants({"DOMAIN_NOTATION_CATEG": "Clear"}):
        assert _categ("e-mail: officeauappmost.cz") == "Clear"
        assert _categ("e-mail: mhauer@zip-ops.cz") == "Clear"
