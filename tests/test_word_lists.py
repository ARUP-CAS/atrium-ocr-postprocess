"""
tests/test_word_lists.py
========================
(#30) Covers `setup/word_lists.txt` — the hand-maintained word lists — and the
loader that reads it.

WHY THE FILE EXISTS, because the reason is the thing a future reader will need.
Every word list this repository carried was CLOSED-class: Czech function words,
units, academic titles, catalogue markers. The OPEN-class vocabulary this issue
keeps mis-convicting — `Dauerleihe`, `Kaukasus`, `Schuhleistenkeilbruchstueck`,
`ssuti`, `ppole`, the Latin taxonomy — lived only in fixtures in this directory,
where no archivist will ever find it. And the one mechanism meant to cover
open-class words, `SHORT_GARBAGE_LEXICON_PATH`, ships EMPTY with no frequency
table anywhere in the repository. So the shipped configuration had no open-class
protection at all. The file's `[allowed]` section is where that layer goes, not a
duplicate of an existing one -- but it is not the layer yet: every entry ships
commented out (which to switch on is Q5a), and an entry reaches the quality score
and -- since Q5b was answered yes on 2026-10-01 -- the witness's shape tests.

WHAT IS PINNED HERE. Mostly that the migration is a NO-OP: each list read from
the file must equal the literal that was compiled into `text_util.py` before it.
That is the assertion that fails if someone edits the file and drops a member, and
it is the only reason the migration commit can be trusted without a re-score.
"""

import pytest

import text_util as tu

MIGRATED = [
    ("neutral", "_NEUTRAL_LEXICON"),
    ("notation_labels", "_NOTATION_LABELS"),
    ("header_labels", "_DOCUMENT_HEADER_LABELS"),
]


@pytest.mark.parametrize("section,const", MIGRATED)
def test_the_shipped_file_reproduces_the_compiled_in_list_exactly(section, const):
    """The migration no-op proof, mechanical rather than argued.

    `<const>_DEFAULT` is the literal that shipped before the file existed. If the
    file and the default ever disagree, either someone edited the file — which is
    allowed, and this test is where they find out it has consequences — or the
    migration dropped a member, which is not.
    """
    live = getattr(tu, const)
    default = getattr(tu, f"{const}_DEFAULT")
    assert live == default, (
        f"setup/word_lists.txt [{section}] no longer matches {const}_DEFAULT.\n"
        f"  only in the file:    {sorted(live - default)}\n"
        f"  only in the default: {sorted(default - live)}\n"
        "If the edit was deliberate, update the default in the same commit so the two "
        "cannot drift; if it was not, the file lost a member production depends on."
    )


@pytest.mark.parametrize("section,const", MIGRATED)
def test_every_migrated_list_is_non_empty_and_lowercase(section, const):
    live = getattr(tu, const)
    assert live, f"[{section}] is empty — a missing section must fall back, not blank out"
    assert all(t == t.lower() for t in live), f"[{section}] carries an uncased token"


def test_the_allowed_section_ships_empty():
    """It is wired and not armed, the same split used for the shape witness.

    The candidates are in the file as a COMMENTED block with their provenance.
    Uncommenting changes stored categories, so it is a deliberate act and not a
    default — and it keeps the migration commit a true no-op, so a number that
    moves has exactly one possible cause.
    """
    assert tu.word_list("allowed", frozenset()) == frozenset()


def test_an_empty_path_falls_back_to_the_compiled_in_values():
    """No file, no change: the module must still import and behave as before."""
    with tu.override_constants({"WORD_LISTS_PATH": ""}):
        for _, const in MIGRATED:
            assert tu.word_list("x", frozenset()) == frozenset()
            assert getattr(tu, f"{const}_DEFAULT")


def test_a_missing_section_falls_back_rather_than_blanking_out(tmp_path):
    """Deleting a section must not silently empty a list production depends on.

    This is the failure mode that makes a hand-edited file dangerous: a person
    removes a heading they do not recognise, and a veto quietly stops existing.
    """
    path = tmp_path / "partial.txt"
    path.write_text("[allowed]\nkaukasus\n", encoding="utf-8")
    with tu.override_constants({"WORD_LISTS_PATH": str(path)}):
        assert tu.word_list("allowed", frozenset()) == {"kaukasus"}
        assert tu.word_list("neutral", tu._NEUTRAL_LEXICON_DEFAULT) == tu._NEUTRAL_LEXICON_DEFAULT


def test_an_unreadable_file_degrades_to_the_fallback(tmp_path):
    """Configuration, not input: a missing file must not stop the pipeline."""
    with tu.override_constants({"WORD_LISTS_PATH": str(tmp_path / "does-not-exist.txt")}):
        assert tu.word_list("neutral", tu._NEUTRAL_LEXICON_DEFAULT) == tu._NEUTRAL_LEXICON_DEFAULT


def test_the_format_is_the_plainest_thing_that_holds_several_lists(tmp_path):
    """Comments anywhere, blanks skipped, case folded, tokens before any section ignored."""
    path = tmp_path / "w.txt"
    path.write_text(
        "stray token before any section\n"
        "# a comment\n"
        "\n"
        "[allowed]\n"
        "Kaukasus   # the Caucasus\n"
        "  HALLSTATTHAUS\n"
        "# commented-out candidate\n"
        "# dauerleihe\n"
        "\n"
        "[neutral]\n"
        "mm\n",
        encoding="utf-8",
    )
    with tu.override_constants({"WORD_LISTS_PATH": str(path)}):
        assert tu.word_list("allowed", frozenset()) == {"kaukasus", "hallstatthaus"}
        assert tu.word_list("neutral", frozenset()) == {"mm"}


def test_an_edit_is_seen_without_a_restart(tmp_path):
    """The cache is keyed on mtime, so an operator editing the file is not told to
    restart a long-running service. Pinned because the obvious implementation --
    a zero-argument cache -- would freeze the first read, and that is the exact
    bug class `_CACHES_FROM_FLAG` exists to close (#30 D28)."""
    path = tmp_path / "w.txt"
    path.write_text("[allowed]\nkaukasus\n", encoding="utf-8")
    with tu.override_constants({"WORD_LISTS_PATH": str(path)}):
        assert tu.word_list("allowed", frozenset()) == {"kaukasus"}
        import os

        path.write_text("[allowed]\nkaukasus\nhallstatthaus\n", encoding="utf-8")
        os.utime(path, (0, 0))  # force a different mtime rather than racing the clock
        assert tu.word_list("allowed", frozenset()) == {"kaukasus", "hallstatthaus"}


def test_the_shipped_file_documents_every_section_it_defines():
    """A section an archivist cannot tell apart from its neighbours is a trap.

    The sections reach different distances -- `[allowed]` touches only the quality
    score, `[notation_labels]` decides whether a line reads as a reference -- so
    the file must say so per section, not once at the top.
    """
    from pathlib import Path

    lines = Path(tu.WORD_LISTS_PATH).read_text(encoding="utf-8").splitlines()
    for section in ("allowed", "neutral", "notation_labels", "header_labels"):
        # The HEADING, not a mention in prose -- the header names `[allowed]` when
        # it tells a reader which section to default to, and matching that instead
        # would pass while the heading itself sat undocumented.
        where = [i for i, ln in enumerate(lines) if ln.strip() == f"[{section}]"]
        assert len(where) == 1, f"expected exactly one [{section}] heading, found {len(where)}"
        head = "\n".join(lines[: where[0]])
        assert "WHAT IT DOES" in head.split(f"── [{section}]")[-1], (
            f"[{section}] is not preceded by its own 'WHAT IT DOES' note"
        )


# ---------------------------------------------------------------------------
# [allowed] — the quality-score effect (#30)
# ---------------------------------------------------------------------------
#
# A listed token contributes nothing to the quality score (below) and is never
# evidence of damage to the shape witness (the Q5b section after it).


def _armed(tmp_path, *tokens):
    path = tmp_path / "allow.txt"
    path.write_text("[allowed]\n" + "\n".join(tokens) + "\n", encoding="utf-8")
    return tu.override_constants({"WORD_LISTS_PATH": str(path)})


def test_nothing_is_allowed_in_the_shipped_configuration():
    """Step 2's no-op proof at the predicate. [allowed] ships empty, so every call
    site below is inert until an archive writes a list."""
    for token in ("kaukasus", "ssuti", "Dauerleihe", "oueussd"):
        assert tu._is_allowed_token(token) is False


def test_a_listed_token_stops_counting_in_all_four_score_components(tmp_path):
    subject = "Schuhleistenkeilbruchstueck"  # 27 chars: fused by length alone
    assert tu.detect_fused_words(subject) == 1, "premise: it is penalised before listing"
    with _armed(tmp_path, subject.lower()):
        assert tu.detect_fused_words(subject) == 0
        assert tu.score_word(subject) == 0.0
        assert tu.detect_gibberish_words(subject) == 0
        assert tu.detect_wx_words(subject) == 0


def test_a_listed_token_is_non_evaluable_not_valid(tmp_path):
    """ "Not debuffed" is the ask. Counting it VALID would raise the ratio, which is
    more, so it is skipped the way `_is_neutral_token` skips a unit."""
    # `Roe<toeovy` is a real string from the archive (a mis-scanned
    # `Počítačový`). The `<` is what makes the shape heuristic reject it —
    # `wwwxxx` would NOT do, because ">=3 chars, >=70% alphabetic, nothing
    # strange" calls that valid, which is the heuristic's known weakness and the
    # reason the vocabulary branch exists at all.
    line = "Roe<toeovy kostra"
    assert tu.compute_valid_ratio(line) == pytest.approx(0.5), "premise: one of two is invalid"
    with _armed(tmp_path, "roe<toeovy"):
        # 1 of 1 evaluable, not 2 of 2 — the listed token left the denominator
        # rather than joining the numerator.
        assert tu.compute_valid_ratio(line) == pytest.approx(1.0)
        assert tu.compute_valid_ratio("Roe<toeovy") == pytest.approx(1.0)  # evaluable == 0 -> 1.0


def test_listing_a_token_that_was_never_penalised_does_nothing(tmp_path):
    """The safe direction to be wrong in, and worth pinning: an archivist adding a
    word "just in case" must not move anything."""
    before = tu.score_word("Kaukasus"), tu.detect_fused_words("Kaukasus"), tu.compute_valid_ratio("Kaukasus")
    with _armed(tmp_path, "kaukasus"):
        after = tu.score_word("Kaukasus"), tu.detect_fused_words("Kaukasus"), tu.compute_valid_ratio("Kaukasus")
    assert before == after


def test_an_unlisted_token_is_untouched(tmp_path):
    with _armed(tmp_path, "dauerleihe"):
        assert tu.score_word("oueussd") > 0.0
        assert tu.detect_fused_words("oueussd") == 1


def test_matching_is_case_folded_but_does_not_fold_diacritics(tmp_path):
    """Listing `jáma` must not also excuse `jama`.

    `_NOTATION_LABELS_FOLDED` folds diacritics on purpose, because OCR drops
    accents and a label is still a label. An allow list is the opposite case: the
    accent-stripped form is the DAMAGED reading, and a list of words the archive
    says are real should not quietly also cover the ways they come out wrong. An
    archive that wants both, lists both.
    """
    with _armed(tmp_path, "jáma"):
        assert tu._is_allowed_token("JÁMA") is True
        assert tu._is_allowed_token("Jáma") is True
        assert tu._is_allowed_token("jama") is False


# ---------------------------------------------------------------------------
# [allowed] — the shape witness (#30 Q5b)
# ---------------------------------------------------------------------------
#
# @david-spacil, 2026-10-01, for himself and @DanaKriv: "yes, listed means never
# evidence of damage". Before the answer, `ssuti vfetennl` went to `Trash` on
# `ssuti`'s doubled first letter with `ssuti` listed (AT1, probed 2026-09-22).


@pytest.mark.parametrize(
    "token,clause",
    [("ssuti", "initial_geminate"), ("Dauerleihe", "vowel_run"), ("Kaukasus", "low_variety")],
)
def test_a_listed_word_is_never_evidence_of_damage(tmp_path, token, clause):
    assert clause in tu.shape_garbage_clauses(token), "premise: convicted before listing"
    with _armed(tmp_path, token.lower()):
        assert tu.shape_garbage_clauses(token) == []


def test_a_listed_word_does_not_shield_the_rest_of_the_line(tmp_path):
    """Veto only: the listed word stops being evidence, the others are still read."""
    with _armed(tmp_path, "ssuti"):
        assert tu.shape_garbage_clauses("ssuti") == []
        assert tu.shape_garbage_clauses("ssuti oueussd") == ["vowel_run"]


def test_the_listed_word_is_exempt_from_the_unattested_clause_too(tmp_path):
    """`no_vocabulary` is the one clause that can add a conviction; a listed word
    must not reach it either, lexicon or not."""
    table = tmp_path / "df.tsv"
    table.write_text("# documents\t10\nvrstva\t500\n", encoding="utf-8")
    overrides = {"SHORT_GARBAGE_LEXICON_PATH": str(table), "SHORT_GARBAGE_LEXICON_CONVICT": True}
    with tu.override_constants(overrides):
        assert "no_vocabulary" in tu.shape_garbage_clauses("Hallstatthaus"), "premise"
        with _armed(tmp_path, "hallstatthaus"):
            assert tu.shape_garbage_clauses("Hallstatthaus") == []


def test_the_short_line_penalty_makes_the_effect_a_cliff_not_a_weight_share(tmp_path):
    """MEASURED, because the plan said to measure it rather than reason about it.

    The four per-token components are 0.60 of the score's weight, so the naive
    expectation is that un-debuffing one token moves a one-word line by some part
    of that. It does not. `short_penalty` (text_util.py) subtracts a FLAT 0.20
    when the line is <= 12 characters and any weirdness remains, so listing the
    only token releases the penalty as well:

        `ssuti`       (5 chars)  0.6160 -> 0.8790   +0.2630
        `Dauerleihe` (10 chars)  0.5870 -> 0.8800   +0.2930

    That crosses a category boundary -- `CATEG_TRASH_SCORE_MAX` is 0.55 and the
    `Clear` band opens at 0.70 -- which is precisely why `[allowed]` ships empty
    and its candidates ship commented out. Arming it is a categorisation change,
    not a tuning nudge, and it is measured before it is switched on rather than
    after.
    """

    def score(text: str) -> float:
        weird = tu.compute_word_weird_ratio(tu.score_words_in_line(text))
        wc = max(len(text.split()), 1)
        return tu.compute_quality_score(
            tu.compute_valid_ratio(text),
            900.0,
            len(text),
            weird,
            vowel_ratio=tu.compute_vowel_ratio(text),
            garbage_density=tu.compute_garbage_density(text),
            lang_score=0.4,
            gibberish_ratio=(tu.detect_gibberish_words(text) + tu.detect_wx_words(text)) / wc,
            fused_ratio=tu.detect_fused_words(text) / wc,
        )

    before = score("ssuti")
    with _armed(tmp_path, "ssuti"):
        after = score("ssuti")

    assert after - before > 0.20, "the flat short-line penalty should be released as well"
    assert before < 0.70 <= after, "and that is enough to cross a category band"


# ---------------------------------------------------------------------------
# Precedence: the config key still wins (#30)
# ---------------------------------------------------------------------------


MIGRATED_WITH_KEY = [
    ("short_valid", "SHORT_VALID_WORDS"),
    ("short_exception", "SHORT_EXCEPTION_TOKENS"),
    ("ldl_units", "LDL_UNITS"),
    ("rot_whitelist", "ROT_WHITELIST"),
    ("ghost_collisions", "_GHOST_REAL_WORD_COLLISIONS"),
]


@pytest.mark.parametrize("section,const", MIGRATED_WITH_KEY)
def test_the_file_reproduces_these_lists_too(section, const):
    assert getattr(tu, const) == getattr(tu, f"{const}_DEFAULT")


def test_an_explicit_config_value_still_beats_the_file(tmp_path):
    """Moving a list into the file must not TAKE AWAY the override.

    `tests/test_config_constants.py::test_tier1_key_roundtrip_from_alternate_config`
    caught exactly that: it points LANGID_CONFIG at an alternate config and
    expects ROT_WHITELIST to follow it. After the move it did not, because the
    file had quietly become the only source.

    So the keys stay in setup/config.txt, EMPTY. Empty means "not overridden, use
    the file", which keeps the member list in one place while leaving the
    override path -- config file, alternate LANGID_CONFIG, or
    ATRIUM_TEXT_UTILS_<KEY> -- open.
    """
    path = tmp_path / "w.txt"
    path.write_text("[rot_whitelist]\nfromfile\n", encoding="utf-8")
    with tu.override_constants({"WORD_LISTS_PATH": str(path)}):
        assert tu.word_list("rot_whitelist", frozenset()) == {"fromfile"}
        # A non-empty override wins outright; it does not merge.
        assert tu.word_list("rot_whitelist", frozenset(), "po,do") == {"po", "do"}
    # And with no file at all the override still wins over the in-code default.
    with tu.override_constants({"WORD_LISTS_PATH": ""}):
        assert tu.word_list("rot_whitelist", tu.ROT_WHITELIST_DEFAULT, "po,do") == {"po", "do"}


def test_the_config_keys_ship_empty_so_the_file_is_the_single_source():
    """If a key ever ships non-empty again, the member list lives in two places
    and they will drift. This is the test that says so."""
    import configparser
    from pathlib import Path

    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(Path(tu.__file__).resolve().parent / "setup" / "config.txt")
    for key in ("SHORT_VALID_WORDS", "SHORT_EXCEPTION_TOKENS", "LDL_UNITS", "ROT_WHITELIST", "GHOST_WORD_COLLISIONS"):
        assert parser.get("TEXT_UTILS", key).strip() == "", (
            f"{key} ships non-empty; the members belong in setup/word_lists.txt and the key is "
            "an override that should be blank unless someone is deliberately overriding it"
        )
