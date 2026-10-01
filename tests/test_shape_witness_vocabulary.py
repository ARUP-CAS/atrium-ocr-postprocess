"""
tests/test_shape_witness_vocabulary.py
======================================
The two things issue #30 left open about ``_has_shape_garbage_evidence()``:
the loan false-positive class that gates the flag, and the lexical signal for
the residue (D14).

Why this file exists
--------------------
The design block above the predicate named its own weak point and then did not
close it:

    "KNOWN false positives of the vowel-run clause: Latin/French/German loans
     with a 3+ vowel run (`Poaceae`, `Naiade`, ...). Most carry weird_ratio 0.0
     and so never reach the route at all -- a real but THIN margin, since it
     depends on a signal outside this predicate. Measuring that class against
     annotated lines is a precondition for enabling the flag, not a follow-up."

``agent_dev_logs/plans/30.plan.md`` carried it for a month as "still a stated
precondition for flipping the flag, still unmeasured". Measured, it fails, and
not marginally: ``-aceae`` is the botanical FAMILY suffix, so the vowel-run
clause did not make an occasional mistake on this class -- it convicted all of
it, in a corpus whose archaeobotany reports are where that class lives.

The margin the block relies on is a signal this predicate does not own. These
tests pin the predicate's own behaviour instead, which is the thing a future
change can break without noticing.

Every test here is pure text. No model, no corpus, no GPU -- which is the point:
the class that gated this flag was always measurable, and nothing about it
needed the delivered batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402
from text_util import _has_shape_garbage_evidence, shape_garbage_clauses  # noqa: E402


@pytest.fixture(autouse=True)
def _judged_without_the_archive_allow_list(without_allowed_words):
    """This file is about the shape clauses and the lexicon veto, not about `[allowed]`.

    Since @david-spacil's Q5a list landed (2026-10-01) the shipped file spares
    several of the words used below as examples, and Q5b makes a listed word never
    evidence of damage. Held out here so these tests keep measuring those;
    the list itself is pinned in tests/test_word_lists.py.
    """


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

#: Botanical family names. Every one ends in the family suffix `-aceae`, which is
#: why this class is systematic rather than anecdotal: the suffix carries the
#: `eae` vowel run, so before the exemption the clause convicted the naming
#: convention itself. Measured 2026-09-17 on the shipped predicate: 10 of 10.
BOTANICAL_FAMILIES = [
    "Poaceae",
    "Rosaceae",
    "Fabaceae",
    "Brassicaceae",
    "Cyperaceae",
    "Chenopodiaceae",
    "Asteraceae",
    "Betulaceae",
    "Fagaceae",
    "Polygonaceae",
]

#: Binomials, genera and osteological terms from the #30 thread and from the
#: report classes Dana reviewed. These already passed before the exemption; they
#: are here so a future widening of any clause has to break a named line.
TAXONOMIC_VOCABULARY = [
    "Equus caballus",
    "Canis familiaris",
    "Pinus silvestris",
    "Unio crassus",
    "Corylus avellana",
    "Triticum aestivum",
    "Capreolus capreolus",
    "Cervus elaphus",
    "Occipitale",
    "Phalanx proximalis",
    "Maxilla+dentes",
    "Ossa tarsi",
    "radius prox.sin.",
]

#: The loans the SUFFIX rule deliberately does not reach: name-shaped rather than
#: suffixed. They are the vocabulary half of the D14 case -- the reason a lexical
#: signal is needed at all, rather than one more orthographic rule.
UNSUFFIXED_LOANS = ["Naiade", "Beuern", "Oueste"]

#: Garbage the witness must keep convicting. An exemption that buys a false
#: positive back by giving up a true one is not a narrowing, it is a retreat.
MUST_STILL_CONVICT = [
    "oueussd",
    "sektlll",
    "cuxoaid",
    "rragment",
    "vansasaasasa",
    "NINNNIC",
    "Tthts I",
    "IDIDIDIDIDIDUOID",
    "lllll",
]

#: Phonotactically legal garbage. Out of reach of every SHAPE clause by
#: construction -- that is the residue D14 is about.
RESIDUE = ["edelite", "vfetennl k.", "zcv7"]


def _table(tmp_path: Path, rows: dict[str, int]) -> str:
    """Write a token/document-frequency table in build_token_lexicon.py's format."""
    path = tmp_path / "token_df.tsv"
    body = "# synthetic table for tests\n" + "".join(f"{t}\t{n}\n" for t, n in rows.items())
    path.write_text(body, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# The precondition that gated the flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BOTANICAL_FAMILIES)
def test_botanical_family_names_are_not_witnessed(name):
    """The `-aceae` class, which the predicate convicted 10 of 10 before the exemption.

    This is the "loan false-positive class" the plan recorded as a precondition
    for enabling `SHORT_GARBAGE_WITNESS_ENABLE`. With the flag on and without the
    exemption, every botanical family name in an archaeobotanical report would
    route to Trash on a single clause.
    """
    assert _has_shape_garbage_evidence(name) is False, (
        f"{name!r} is witnessed as garbage: {shape_garbage_clauses(name)}. "
        "This is the Latin family suffix, not a rare spelling."
    )


@pytest.mark.parametrize("text", TAXONOMIC_VOCABULARY)
def test_taxonomic_and_osteological_vocabulary_is_not_witnessed(text):
    assert _has_shape_garbage_evidence(text) is False, f"{text!r}: {shape_garbage_clauses(text)}"


def test_the_exemption_is_a_suffix_rule_not_a_latin_amnesty():
    """`-aceae` is exempt; a token that merely looks Latinate is not.

    The exemption has to stay narrow enough that it cannot be reached by
    accident. `oueussd` and `cuxoaid` are not Latin and must not become exempt
    because some future edit widens the pattern.
    """
    assert tu._RE_TAXONOMIC_SUFFIX.search("poaceae")
    assert tu._RE_TAXONOMIC_SUFFIX.search("rosoideae")
    assert not tu._RE_TAXONOMIC_SUFFIX.search("oueussd")
    assert not tu._RE_TAXONOMIC_SUFFIX.search("cuxoaid")
    # Length floor: a short token ending in the pattern is a coincidence.
    assert tu._TAXONOMIC_SUFFIX_MIN_ALPHA >= 5


@pytest.mark.parametrize("text", MUST_STILL_CONVICT)
def test_the_exemptions_cost_no_conviction(text):
    """Every named garbage line still fires at least one clause.

    Measured together with the exemption rather than after it: 0 convictions lost
    across this list. An exemption that traded one of these away would be moving
    the error, not removing it.
    """
    assert _has_shape_garbage_evidence(text) is True, f"{text!r} stopped being witnessed"


#: One excavation's grid series, the whole of it in the 822-document corpus. The
#: sub-letter is fused to the roman segment (`S-VIIIb`, not `S-VIII-b`), which is
#: all that separates it from a shape `is_domain_notation()` already accepts.
FUSED_GRID_REFS = ["AA-VIIIb", "E-VIIIb", "F-VIIIb", "J-VIIIb", "K-VIIIc", "L-VIIIb", "S-VIIIb"]


@pytest.mark.parametrize("text", FUSED_GRID_REFS)
def test_fused_grid_references_are_not_witnessed(text):
    """`S-VIIIb` is annotated `Clear` in the gold sidecar.

    PINNED BECAUSE IT WAS LOST ONCE. The guard landed on 2026-09-18 and was
    deleted the same evening by an unrelated lexicon commit (`cc4990e`); with no
    test holding it, the suite stayed green and every later run measured the
    witness without it, until @david-spacil's 508-line re-check on 2026-09-23
    found `S-VIIIb` among its two breaks. The roman-numeral exemption does not
    reach this shape: the fused lowercase letter keeps `VIIIb` from matching it.
    """
    assert shape_garbage_clauses(text) == [], f"{text!r} is witnessed again: {shape_garbage_clauses(text)}"


def test_the_grid_guard_is_the_hyphenless_twin_and_nothing_wider():
    """The hyphenated form is already notation; the guard must not become an amnesty.

    Anchored at both ends: a find identifier with a trailing number, a stamp and a
    stutter all stay outside it, so nothing that the witness convicts today loses
    its conviction to this pattern except the grid shape itself.
    """
    assert tu.is_domain_notation("S-VIII-b") is True
    assert tu._RE_FUSED_GRID_REF.match("S-VIIIb")
    for text in ["Aa/III 116", "oueussd", "sektlll", "OUUITN", "Lokolieace: •VIII,", "VIIIb"]:
        assert not tu._RE_FUSED_GRID_REF.match(text), f"{text!r} would be exempted by the grid guard"


# ---------------------------------------------------------------------------
# The vowel-run knob
# ---------------------------------------------------------------------------


def test_vowel_run_minimum_is_decoupled_from_fused_vowel_run_min():
    """The witness must be tunable without moving the quality score.

    `_RE_FUSED_VOWEL_RUN` feeds `detect_fused_words()`, which feeds `fused_ratio`
    in `compute_quality_score` and the `fused_words` CSV column. While the witness
    shared that constant, the only clause that discriminates on the #30 population
    could not be tuned without changing scores on every line in the corpus.
    """
    assert tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN == tu.FUSED_VOWEL_RUN_MIN, (
        "the shipped default must reproduce the previous behaviour exactly"
    )
    # And it must actually steer the clause, not merely exist.
    assert tu._compile_vowel_run(3).search("oai")
    assert not tu._compile_vowel_run(4).search("oai")
    assert tu._compile_vowel_run(4).search("oueu")


def test_raising_the_vowel_run_is_a_measured_trade_in_both_directions():
    """Pins the trade at 4 so it is a decision, not a surprise.

    Documented in the clause and in setup/config.txt: `oueussd` survives, the
    unsuffixed loans stop firing, and `cuxoaid` escapes with no other clause
    catching it. It was then raised against gold (stage 10d) and rejected --
    errors 503 = 503, fixes 2 / breaks 2, p = 1, `Trash`-recall 34 -> 32/180 --
    so this stays pinned as the record of why, and the D44 language split is
    the answer rather than a global 4.
    """
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN": 4}):
        assert _has_shape_garbage_evidence("oueussd") is True
        assert _has_shape_garbage_evidence("cuxoaid") is False, "the cost side of the trade moved"
        for loan in UNSUFFIXED_LOANS:
            assert _has_shape_garbage_evidence(loan) is False


# ---------------------------------------------------------------------------
# D14: the vocabulary signal
# ---------------------------------------------------------------------------


def test_lexicon_is_inert_when_no_table_is_configured():
    """The shipped state. With no table the predicate is the shape-only predicate."""
    assert tu.SHORT_GARBAGE_LEXICON_PATH == ""
    assert not tu.token_lexicon()
    assert tu.SHORT_GARBAGE_LEXICON_CONVICT is False
    for text in MUST_STILL_CONVICT:
        assert _has_shape_garbage_evidence(text) is True
    for text in RESIDUE:
        assert _has_shape_garbage_evidence(text) is False, (
            "with no lexicon the residue must remain out of reach — that is the "
            "documented limit this table is what changes"
        )


#: (#30 stage 8, D32.) Zooarchaeological and archaeobotanical binomials measured
#: against the shipped predicate WITHOUT a lexicon: every one still fires,
#: `Lepus europaeus` on vowel_run, the rest on low_variety in the epithet
#: (`monococcum`, `terrestris`, `usitatissimum`) or the genus itself
#: (`Mammalia`). That is not the taxonomy exemption failing -- there never was
#: one for binomials, only for the `-aceae` family suffix above. It is the
#: reason this list exists: measured WITH the full-collection lexicon armed
#: (values below from the stage-8 delivery's `08g_distinct_evidence.csv`),
#: every one of these is already `attested` and exempt through
#: `_has_vocabulary_support()`, the same D14 mechanism that exempts `ppole`.
#: No predicate change closes this class; arming the lexicon already does.
STAGE8_BINOMIALS_WITHOUT_LEXICON = [
    "Mammalia indet.",
    "Mammalia",
    "Triticum monococcum",
    "Lepus europaeus",
    "Arvicola terrestris",
    "Linum usitatissimum",
]

#: The df values `08g_distinct_evidence.csv` measured them at, full collection
#: (113,100 documents). Comfortably above SHORT_GARBAGE_LEXICON_MIN_DF (3).
STAGE8_BINOMIAL_LEXICON = {
    "mammalia": 342 + 819,
    "indet": 819,
    "triticum": 188,
    "monococcum": 188,
    "lepus": 951,
    "europaeus": 951,
    "arvicola": 83,
    "terrestris": 83,
    "linum": 85,
    "usitatissimum": 85,
}


def test_stage8_binomials_convict_without_a_lexicon():
    """The measurement this class needed, made explicit: no lexicon, no exemption.

    Pinned separately from `test_lexicon_is_inert_when_no_table_is_configured`
    because this is the specific class the stage-8 delivery's own witness
    queue was read against a lexicon-off run (T1) -- the false-positive class
    it appeared to show was itself measured in the wrong configuration.
    """
    assert not tu.token_lexicon()
    for text in STAGE8_BINOMIALS_WITHOUT_LEXICON:
        assert _has_shape_garbage_evidence(text) is True, (
            f"{text!r} is not witnessed with no lexicon configured: {shape_garbage_clauses(text)}"
        )


def test_stage8_binomials_are_exempt_once_the_lexicon_is_armed(tmp_path):
    """The corrected finding: arming the lexicon (D14) already reaches this class.

    Measured against the real full-collection document frequencies, not
    synthetic round numbers, so this pins the actual delivered table's shape
    rather than a convenient fixture.
    """
    path = _table(tmp_path, STAGE8_BINOMIAL_LEXICON)
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        for text in STAGE8_BINOMIALS_WITHOUT_LEXICON:
            assert _has_shape_garbage_evidence(text) is False, (
                f"{text!r} is still witnessed with the lexicon armed: {shape_garbage_clauses(text)}"
            )


#: (#30 stage 8, D34.) The residual `low_variety` false positives that survive
#: even a lexicon armed with everything above: real words, too rare across
#: documents to be attested (`Kaukasus` occurs 5 times in the stage-8 delivery,
#: all in few documents), that are not name-shaped loans either so D14 cannot
#: reach them. `issue30_annotation_guide.md` already names both as the
#: canonical "recoverability zero, still a real word" example.
#:
#: NOT fixed here, deliberately, and measured before that decision rather than
#: assumed: a length cap (mirroring `SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA`)
#: cannot separate them from real `low_variety` garbage at the same length --
#: `PSSPPOP` (garbage) and `vodovod` (real, but attested once the lexicon is
#: armed, so not this class) are both 7 letters at ratio 0.43; `Kaukasus` (real,
#: unattested) and `RARRRPRIR` (garbage) are both effectively 8-9 letters at
#: ratio ~0.44-0.50. Case shape does not separate them either: `VODOVOD`
#: (real) and `PSSPPOP` (garbage) are both ALLCAPS at the same ratio. Retuning
#: `SHORT_GARBAGE_WITNESS_VARIETY_MAX`/`_MIN_ALPHA` from two named examples
#: would be the same mistake D25/D30 already named in this issue -- fitting a
#: production threshold to a population too small to fit one to. Accepted,
#: named debt -- the same shape as the doubled-initial class, where three
#: signals were tried and none separated language from damage either.
STAGE8_RARE_WORDS_UNRESOLVED = [
    "Kaukasus",
    "Hallstatthaus",
    "Schuhleistenkeilbruchstueck",
]


def test_stage8_rare_words_remain_unresolved_even_with_a_lexicon(tmp_path):
    """D34, pinned rather than silently accepted.

    A lexicon armed with unrelated vocabulary must not accidentally rescue
    these -- they are absent from this table on purpose, matching them being
    absent from the real one at full scale.
    """
    path = _table(tmp_path, STAGE8_BINOMIAL_LEXICON)
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        for text in STAGE8_RARE_WORDS_UNRESOLVED:
            assert _has_shape_garbage_evidence(text) is True, (
                f"{text!r} became exempt from an unrelated lexicon table: {shape_garbage_clauses(text)}"
            )


def test_attested_vocabulary_vetoes_the_unsuffixed_loans(tmp_path):
    """The half the suffix rule deliberately leaves alone.

    `Naiade`, `Beuern` and `Oueste` are name-shaped, so no suffix rule reaches
    them and inventing a legal-vowel-sequence list to do it is the guessing that
    cost twelve lines the last time this predicate was widened by reading rather
    than measuring. Attestation reaches them without anyone deciding which vowel
    sequences a European language may contain.
    """
    path = _table(tmp_path, {"naiade": 14, "beuern": 9, "oueste": 5, "oueussd": 1, "sektlll": 2})
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        for loan in UNSUFFIXED_LOANS:
            assert _has_shape_garbage_evidence(loan) is False, f"{loan!r} was attested and still convicted"
        # Below the threshold is not attested: one document is what OCR noise looks like.
        assert _has_shape_garbage_evidence("oueussd") is True
        assert _has_shape_garbage_evidence("sektlll") is True


def test_the_veto_is_a_veto_and_never_adds_a_conviction(tmp_path):
    """Presence in the table can only remove a clause, never introduce one."""
    path = _table(tmp_path, {"kaaden": 40, "pinii": 12})
    before = {t: shape_garbage_clauses(t) for t in MUST_STILL_CONVICT + BOTANICAL_FAMILIES + RESIDUE}
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        after = {t: shape_garbage_clauses(t) for t in MUST_STILL_CONVICT + BOTANICAL_FAMILIES + RESIDUE}
    for text, clauses in after.items():
        assert set(clauses) <= set(before[text]), f"{text!r} gained a clause from a veto-only table"


def test_unattested_conviction_reaches_the_residue_and_ships_off(tmp_path):
    """D14, the part no character-level rule can do.

    `edelite` and `vfetennl k.` are spelled the way words are spelled. The only
    property that separates them from `malakofauna` is that nothing else in the
    collection contains them. This is the one clause of the witness that can ADD
    a conviction, so it carries its own key and that key ships false.
    """
    path = _table(
        tmp_path,
        {
            "malakofauna": 31,
            "diapozitiv": 22,
            "occipitale": 17,
            "equus": 40,
            "caballus": 38,
            "kaaden": 8,
            "vrstva": 900,
        },
    )
    overrides = {"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}

    # Table loaded, conviction OFF — the residue is still out of reach.
    with tu.override_constants(overrides):
        assert tu.SHORT_GARBAGE_LEXICON_CONVICT is False
        for text in RESIDUE:
            assert _has_shape_garbage_evidence(text) is False

    # Conviction ON — the residue is reached, and vocabulary survives.
    with tu.override_constants({**overrides, "SHORT_GARBAGE_LEXICON_CONVICT": True}):
        assert "no_vocabulary" in shape_garbage_clauses("edelite")
        assert "no_vocabulary" in shape_garbage_clauses("vfetennl k.")
        for text in ["malakofauna", "diapozitiv", "Equus caballus", "Occipitale", "Kaaden", "vrstva"]:
            assert _has_shape_garbage_evidence(text) is False, (
                f"{text!r} is attested and must survive the unattested-conviction clause"
            )


def test_unattested_conviction_cannot_fire_without_a_table(tmp_path):
    """Belt and braces: the flag alone must not turn every token into garbage.

    Without a table `token_lexicon()` is empty, so "unattested" would be true of
    everything. The clause checks for a loaded table first; this pins that it
    does, because the failure mode is the entire corpus routing to Trash.
    """
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_CONVICT": True}):
        assert not tu.token_lexicon()
        for text in ["malakofauna", "Equus caballus", "vrstva", "Poaceae"]:
            assert _has_shape_garbage_evidence(text) is False


def test_a_corrupt_table_degrades_to_no_signal_rather_than_raising(tmp_path):
    """This predicate runs per sub-token inside the categoriser.

    An optional, operator-supplied file that raises would take a multi-hour
    corpus run down at whatever line it reached. It must degrade to "no
    vocabulary signal" instead.
    """
    bad = tmp_path / "broken.tsv"
    bad.write_bytes(b"\xff\xfe not a table at all\n\x00\x00")
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": str(bad)}):
        assert _has_shape_garbage_evidence("oueussd") is True
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": str(tmp_path / "does_not_exist.tsv")}):
        assert not tu.token_lexicon()
        assert _has_shape_garbage_evidence("oueussd") is True


def test_min_df_is_honoured_rather_than_assumed(tmp_path):
    path = _table(tmp_path, {"naiade": 2})
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        assert _has_shape_garbage_evidence("Naiade") is True, "df 2 is below the threshold of 3"
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "SHORT_GARBAGE_LEXICON_MIN_DF": 2}):
        assert _has_shape_garbage_evidence("Naiade") is False


# ---------------------------------------------------------------------------
# The builder must tokenise the way the predicate looks up
# ---------------------------------------------------------------------------


def test_builder_tokenisation_matches_the_predicate_lookup():
    """The failure this prevents is silent and one-directional.

    If the table is keyed differently from the way the predicate looks tokens up,
    every lookup misses, which reads as "no vocabulary support" — i.e. as a
    witness that convicts MORE. That is the direction that costs real lines, and
    nothing in the output would say so. Four harness divergences in this
    repository came from a tool carrying its own copy of text_util logic; the
    builder imports `_witness_subtokens` and `_STRIP_CHARS` for that reason.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_btl", _ROOT / "tools" / "build_token_lexicon.py")
    btl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btl)

    for line in [
        "radius prox.sin.",
        "Reg.Bez.Aussig.",
        "Equus caballus",
        "Sonda VIII/3",
        "0,2-0,4 m",
        "gut erhalten,Siedelungsfund,gefunden",
    ]:
        built = list(btl.iter_tokens(line))
        # The predicate's own path, reproduced from the same helpers.
        expected = [
            core.lower()
            for word in line.split()
            for sub in tu._witness_subtokens(word)
            if (core := sub.strip(tu._STRIP_CHARS))
        ]
        assert built == expected, f"{line!r}: builder {built} != predicate {expected}"


# ---------------------------------------------------------------------------
# Table format: a token that looks like a comment
# ---------------------------------------------------------------------------


def test_a_token_beginning_with_hash_is_data_not_a_comment(tmp_path):
    """Found in the real 822-document table: 27 tokens start with `#`.

    `#` is not in `_STRIP_CHARS`, so the builder emits tokens like `#rdisico` and
    `#žkami` verbatim — and the reader skipped every line starting with `#` as a
    provenance comment, dropping them at load. Nothing said so.

    Impact on the delivered table is nil: all 27 are df 1, below any threshold,
    and all garbage. It is fixed because it is the same silent-drop shape as the
    rest of this issue — the file says one thing, the loader reads another, and
    the only symptom is a number that is quietly slightly wrong.

    The format needs no escaping to tell them apart: a header line has no TAB
    (`# columns: token<TAB>document_frequency` is literal text), a data line
    always does.
    """
    path = tmp_path / "token_df.tsv"
    path.write_text(
        "# token document-frequency table — provenance header\n"
        "# columns: token<TAB>document_frequency\n"
        "#rdisico\t40\n"
        "ordinary\t40\n",
        encoding="utf-8",
    )
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": str(path), "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        lex = tu.token_lexicon()

    assert "#rdisico" in lex, "a token that merely looks like a comment was dropped"
    assert "ordinary" in lex
    assert not any(t.startswith("# ") for t in lex), "header lines leaked in as tokens"
    assert len(lex) == 2


def test_header_lines_are_still_skipped(tmp_path):
    """The other half of the same contract — the fix must not admit the header."""
    path = tmp_path / "token_df.tsv"
    path.write_text(
        "# built: 2026-09-17T11:36:51+00:00\n# documents: 822  lines: 12716706\nordinary\t5\n",
        encoding="utf-8",
    )
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": str(path), "SHORT_GARBAGE_LEXICON_MIN_DF": 3}):
        assert dict(tu.token_lexicon()) == {"ordinary": 5}


# ---------------------------------------------------------------------------
# The coupling between the two flags
# ---------------------------------------------------------------------------


def test_the_witness_without_a_lexicon_is_reported_as_a_known_bad_configuration():
    """The two settings are one decision, and the reason is corpus exposure.

    Over both full collections the witness with no table would newly convict
    8,529 strings / 37,555 lines the pipeline currently keeps (stage 08f); with
    the 113,100-document table it is 5,563 / 6,714 (stage 9b, which also carries
    D33). What the table spares is led by real words -- `ppole` (an
    abbreviation), `ARCHAIA`, `Lepus europaeus`, `Triticum monococcum`. Gold
    cannot show the difference: it labels 23 of the ~20k lines the witness
    reaches, and there shape-only and armed score alike (round-2 5a 500 errors,
    5a-bis 502; the no-table 508 re-check goes 326 -> 336).

    History: this used to quote 15,217 convictions against 5,107 ("1:3
    against"), scored against `categ`, and 90.5% of that gap was `ppole`. The
    coupling survived; that arithmetic did not.

    Advisory rather than a gate: every gold A/B from stage 5a on measures the
    shape-only configuration on purpose, and refusing it would make those
    measurements impossible. But nobody should reach it by accident.
    """
    assert tu.uncoupled_witness_warning(witness_enabled=True, lexicon_path="") is not None
    assert tu.uncoupled_witness_warning(witness_enabled=True, lexicon_path="   ") is not None
    # Both other combinations are fine, including the shipped one.
    assert tu.uncoupled_witness_warning(witness_enabled=False, lexicon_path="") is None
    assert tu.uncoupled_witness_warning(witness_enabled=True, lexicon_path="tools/gold/token_df.tsv") is None
    assert tu.uncoupled_witness_warning(witness_enabled=False, lexicon_path="tools/gold/token_df.tsv") is None


def test_the_shipped_configuration_raises_no_advisory():
    assert tu.uncoupled_witness_warning() is None


# ---------------------------------------------------------------------------
# Attested doubled-initial tokens (#30 D40, guard removed 2026-09-22)
# ---------------------------------------------------------------------------
#
# `_has_vocabulary_support` rests on "OCR noise is idiosyncratic to the scan that
# produced it". A pre-printed form scanned across the collection breaks that: the
# same misread recurs once per document and accrues document frequency like a
# word. A DE-GEMINATION GUARD used to carve that shape out -- it let the witness
# convict a doubled-initial token although the table attested it, on a ratio
# against the de-geminated form plus an absolute document-count cap.
#
# It is gone, and these tests now pin the opposite contract. Three reasons, in
# the order they arrived:
#
#   1. `ppole` -- the flagship case, and 57% of the witnessed population on the
#      thin table -- is not an artefact at all. It is *popelnicová pole*,
#      urnfield culture, `pp` doubled for a plural as in `pp.` for pages
#      (@david-spacil, 2026-09-19).
#   2. `ssuti` / `ssutí` / `ssutě` are not artefacts either. They are an old
#      spelling of `suť` (@david-spacil, 2026-09-22). So of the eight tokens the
#      guard was fitted to, FOUR are real language and four are damage -- and the
#      empty frequency gap the ratio sat in had vocabulary on BOTH sides of it.
#   3. At full scale (113,100 documents) the table attests all eight and the
#      guard fired on zero of 42,853 witness-queue rows.
#
# The frequencies below are still the real ones from the 822-document table
# (issue30_out/03_lookup.log and the 347,097-row token_df.tsv), kept because they
# are what the removed guard was calibrated on and what these tests replay.

#: Doubled-initial tokens whose own source word is also attested and commoner.
#: THREE of these are scanning damage -- `oobjekt`, `jjámy`, `vvkop` (the fourth
#: damaged token, `llocm`, is not in this 822-document fixture) -- and the three
#: `ssut*` forms are an old spelling of `suť`, i.e. real language.
#: The removed guard could not tell them apart, and nothing else can either:
#: ratio, absolute document frequency and per-collection concentration were all
#: measured and all failed. They are one class to this code now.
ATTESTED_DOUBLED_INITIALS = {
    "oobjekt": (3, "objekt", 378),
    "jjámy": (5, "jámy", 356),
    "ssuti": (8, "suti", 42),
    "ssutí": (5, "sutí", 42),
    "vvkop": (5, "vkop", 31),
    "ssutě": (4, "sutě", 23),
}

#: Roman numerals and character runs — the natural false positive of a
#: doubled-initial test, and the regression that matters. `xxiii` is not a
#: doubling of `xiii`; they are different numerals that happen to share a suffix.
GEMINATE_LOOKALIKES = {
    "xxiii": (54, "xiii", 87),
    "xxviii": (41, "xviii", 76),
    "xxvii": (42, "xvii", 59),
    "xxxiv": (34, "xxiv", 56),
    "xxxxx": (8, "xxxx", 18),
    "iiiii": (6, "iiii", 9),
}


#: ABBREVIATIONS. A doubled initial that is a real convention, not a scan error.
#: `pp` for a plural is standard Czech -- `pp.` for pages, `ss.` for sections --
#: and `ppole` is *popelnicová pole*, urnfield culture (@david-spacil,
#: 2026-09-19). This file previously called it the flagship OCR artefact and
#: pinned a guard that convicted it on 11,562 lines.
GEMINATE_ABBREVIATIONS = {
    "ppole": (35, "pole", 163),
}


def _doubled_initial_table(tmp_path: Path) -> str:
    rows: dict[str, int] = {}
    merged = {**ATTESTED_DOUBLED_INITIALS, **GEMINATE_LOOKALIKES, **GEMINATE_ABBREVIATIONS}
    for tok, (df_tok, source, df_source) in merged.items():
        rows[tok] = df_tok
        rows[source] = df_source
    # Real vocabulary that must keep its exemption, at its measured frequency.
    rows.update({"triticum": 39, "monococcum": 20, "lepus": 38, "europaeus": 31, "poaceae": 23, "kaaden": 6})
    return _table(tmp_path, rows)


@pytest.mark.parametrize(
    "token",
    sorted(ATTESTED_DOUBLED_INITIALS) + sorted(GEMINATE_LOOKALIKES) + sorted(GEMINATE_ABBREVIATIONS),
)
def test_every_attested_token_keeps_its_veto(tmp_path, token):
    """Attestation is the whole test again: if the table has it, it is exempt.

    This is the contract the de-gemination guard's removal creates (#30 D40), and
    it is the assertion that would fail if anyone re-introduced a carve-out. It
    replaces `test_templated_geminate_artefacts_lose_their_veto`, which asserted
    exactly the opposite and was right about at most three of the six tokens it
    covered -- the three `ssut*` forms are an old spelling of `suť`, not damage.

    The three fixture dicts are listed separately because they record what the
    removed guard tried to distinguish -- damage, numerals, abbreviations. To
    this predicate they are now one class, which is the point.
    """
    with tu.override_constants(
        {"SHORT_GARBAGE_LEXICON_PATH": _doubled_initial_table(tmp_path), "SHORT_GARBAGE_LEXICON_MIN_DF": 3}
    ):
        assert token in tu.token_lexicon(), "fixture error: the token should be attested"
        assert tu._has_vocabulary_support(token) is True, (
            f"{token!r} is attested, so nothing here may withdraw its exemption"
        )


@pytest.mark.parametrize("token", ["triticum", "monococcum", "lepus", "europaeus", "poaceae", "kaaden"])
def test_real_vocabulary_keeps_its_veto(tmp_path, token):
    """The veto's actual work, unchanged by the removal.

    `Triticum monococcum` (121 lines) and `Lepus europaeus` (86) are real `Clear`
    taxonomy that the shape clauses convict and attestation rescues.
    """
    with tu.override_constants(
        {"SHORT_GARBAGE_LEXICON_PATH": _doubled_initial_table(tmp_path), "SHORT_GARBAGE_LEXICON_MIN_DF": 3}
    ):
        assert tu._has_vocabulary_support(token) is True


def test_no_table_means_no_vocabulary_support(tmp_path):
    """No table, no exemption: the shipped configuration is byte-identical.

    Both flags ship false and the path ships empty, so this is the state
    production runs in.
    """
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": ""}):
        assert not tu.token_lexicon()
        for token in list(ATTESTED_DOUBLED_INITIALS) + list(GEMINATE_LOOKALIKES):
            assert tu._has_vocabulary_support(token) is False


def test_attestation_is_what_withdraws_the_clause_not_a_guard(tmp_path):
    """`ssuti` is the worked example of D40, pinned so the correction cannot be lost.

    `initial_geminate` matches it on shape (`^([bcdfghjklmnpqrstvwxz])\\1`), and
    with no table that is the verdict. With the table armed, attestation
    withdraws the conviction and nothing puts it back -- which is correct, because
    `ssuti` is an old spelling of `suť` and not damage at all (@david-spacil,
    2026-09-22). Before the guard was removed the armed arm returned
    `["initial_geminate"]` here.

    `suti` is the control: a clean word, attested, never witnessed either way.
    """
    table = _doubled_initial_table(tmp_path)
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": "", "SHORT_GARBAGE_WITNESS_ENABLE": True}):
        assert shape_garbage_clauses("ssuti") == ["initial_geminate"], (
            "premise: the shape clause reaches ssuti when nothing attests it"
        )
    with tu.override_constants(
        {
            "SHORT_GARBAGE_LEXICON_PATH": table,
            "SHORT_GARBAGE_LEXICON_MIN_DF": 3,
            "SHORT_GARBAGE_WITNESS_ENABLE": True,
        }
    ):
        assert shape_garbage_clauses("ssuti") == []
        assert shape_garbage_clauses("suti") == []
        assert _has_shape_garbage_evidence("suti") is False


# ---------------------------------------------------------------------------
# The abbreviation class (#30, 2026-09-19) — what killed the guard first
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", sorted(GEMINATE_ABBREVIATIONS))
def test_an_abbreviation_keeps_its_veto(tmp_path, token):
    """`ppole` is *popelnicová pole*, not a scanning error.

    @david-spacil, 2026-09-19: *"`ppole` is not an OCR artefact (at least not
    every time), it's an abbreviation for urnfield culture, which also explains
    its frequency (and also the same position on pages with forms)."*

    `pp` doubled for a plural is a standard Czech convention — `pp.` for pages,
    `ss.` for sections — so `initial_geminate`'s premise that no European
    orthography opens a word that way is simply false for abbreviations. The
    de-gemination guard convicted it on 11,562 lines, the largest single
    population in the witnessed queue, and this is the test that recorded why.

    Kept after the guard's removal (#30 D40) deliberately: the guard is gone, the
    reason it had to go is not, and a bare `_has_vocabulary_support` assertion
    with no provenance would invite someone to rebuild it.
    """
    with tu.override_constants(
        {"SHORT_GARBAGE_LEXICON_PATH": _doubled_initial_table(tmp_path), "SHORT_GARBAGE_LEXICON_MIN_DF": 3}
    ):
        assert token in tu.token_lexicon(), "fixture error: the token should be attested"
        assert tu._has_vocabulary_support(token) is True, f"{token!r} lost the veto that protects it"


# ---------------------------------------------------------------------------
# D26 / D27 — two signals that exist and ship off
# ---------------------------------------------------------------------------


def test_valid_ratio_falls_back_to_shape_and_that_is_the_problem():
    """The measurement behind D26, pinned so it cannot be quietly disputed.

    `compute_valid_ratio` has always taken a `word_set`; production never passes
    one, so the fallback is shape — >=3 characters, >=70% alphabetic, nothing
    strange. Under it, three garbage tokens score a perfect 1.00, and that value
    feeds `compute_quality_score` and every threshold below it.
    """
    assert tu.compute_valid_ratio("oueussd edelite sektlll") == 1.0
    assert tu.compute_valid_ratio("oueussd edelite sektlll", {"vrstva"}) == 0.0


def test_the_vocabulary_signal_ships_off(tmp_path):
    """Off, `quality_word_set()` is None and the call is byte-identical to before."""
    assert tu.QUALITY_VOCABULARY_ENABLE is False
    assert tu.quality_word_set() is None

    table = _table(tmp_path, {"vrstva": 429, "nalez": 200})
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": table, "QUALITY_VOCABULARY_ENABLE": True}):
        tu.quality_word_set.cache_clear()
        got = tu.quality_word_set()
        assert got is not None and "vrstva" in got
    tu.quality_word_set.cache_clear()
    assert tu.quality_word_set() is None, "the flag must not leak past its context"


def test_symbol_glyph_stripping_ships_off():
    """Off, tokenisation is unchanged — every table already built stays valid."""
    assert tu.STRIP_SYMBOL_GLYPHS is False
    for glyph in "♦✓■•":
        assert glyph not in tu._STRIP_CHARS, "a symbol glyph reached the shipped strip set"
    # The constant exists and is non-empty, so arming it is a config change only.
    assert set("♦✓■•") <= set(tu._SYMBOL_GLYPHS)


def test_symbol_glyph_stripping_actually_arms_under_override(tmp_path):
    """`_STRIP_CHARS` is DERIVED from the flag, built once at import time, and
    `override_constants()` only setattrs the names it is given -- so overriding
    `STRIP_SYMBOL_GLYPHS` alone used to be a silent no-op: the flag moved, the
    strip set it is supposed to feed never did, and `.strip(_STRIP_CHARS)` at
    every call site never saw the difference.

    This is exactly how `tools/ab_constant_eval.py` measured `STRIP_SYMBOL_GLYPHS`
    true vs false as bit-identical on all 2,064 gold rows (issue #30 stage 07c) --
    not because the flag is inert on this population, but because the harness
    could not arm it at all. `_DERIVED_FROM_FLAG` closes that gap.
    """
    marker = "\u2666zkoumaná"  # ♦zkoumaná — the corpus's own case: a marker fused to a real word
    assert marker.strip(tu._STRIP_CHARS) == marker, "premise: unarmed, the glyph is not stripped"

    with tu.override_constants({"STRIP_SYMBOL_GLYPHS": True}):
        assert tu.STRIP_SYMBOL_GLYPHS is True
        assert "\u2666" in tu._STRIP_CHARS, "_STRIP_CHARS must pick up the glyph set once the flag is armed"
        assert marker.strip(tu._STRIP_CHARS) == "zkoumaná"

    # Restores cleanly, same as every other constant override_constants() touches.
    assert tu.STRIP_SYMBOL_GLYPHS is False
    assert "\u2666" not in tu._STRIP_CHARS
    assert marker.strip(tu._STRIP_CHARS) == marker

    # Round-trip the other direction: forcing False from a False baseline (a
    # no-op arm, as ab_constant_eval.py's reference value always is) must not
    # accidentally strip anything extra or corrupt the saved/restored value.
    with tu.override_constants({"STRIP_SYMBOL_GLYPHS": False}):
        assert "\u2666" not in tu._STRIP_CHARS
    assert "\u2666" not in tu._STRIP_CHARS


# ---------------------------------------------------------------------------
# The same bug as D27, one step further in (#30, 2026-09-20)
#
# `_DERIVED_FROM_FLAG` fixed one way a flag's value gets frozen: a module-level
# constant built from it at import. There is a second way, and stage 07b walked
# into it with the identical symptom -- a zero-argument `functools.lru_cache`
# that reads the flag on its first call and answers from the cache forever
# after. `ab_constant_eval.py` runs both arms in ONE process, reference first,
# so the True arm was handed the False arm's cached answer.
# ---------------------------------------------------------------------------


def test_quality_vocabulary_flag_actually_arms_under_override(tmp_path):
    """The D26 counterpart of `test_symbol_glyph_stripping_actually_arms_under_override`.

    `quality_word_set()` is `lru_cache(maxsize=1)` over a zero-argument function
    that reads `QUALITY_VOCABULARY_ENABLE`. Without a cache clear the two arms of
    an in-process A/B cannot differ, which is exactly what stage 07b reported:
    bit-identical on all 2,064 gold rows, 0 discordant. That was written up as a
    coverage result. It was an unarmed flag.
    """
    path = _table(tmp_path, {"vrstva": 429, "kontext": 216, "malakofauna": 63})

    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path}):
        # Arm 1 is always the reference value, and it populates the cache.
        with tu.override_constants({"QUALITY_VOCABULARY_ENABLE": False}):
            off = tu.quality_word_set()
        # Arm 2 must not inherit it.
        with tu.override_constants({"QUALITY_VOCABULARY_ENABLE": True}):
            on = tu.quality_word_set()

        assert off is None, "flag off must yield no word set — that is the shipped behaviour"
        assert on is not None, "flag on must yield a word set; None here is the 07b bug"
        assert "malakofauna" in on

    # And the signal it is wired to actually moves, which is the whole point of
    # D26: shape alone calls pure garbage a perfect line of valid words.
    garbage = "oueussd edelite sektlll"
    assert tu.compute_valid_ratio(garbage, None) == 1.00
    assert tu.compute_valid_ratio(garbage, frozenset({"vrstva", "kontext", "malakofauna"})) == 0.00


def test_the_cache_is_cleared_on_the_way_out_too(tmp_path):
    """Leaving an arm's answer cached is the same defect one step later."""
    path = _table(tmp_path, {"vrstva": 429})
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": path, "QUALITY_VOCABULARY_ENABLE": True}):
        assert tu.quality_word_set() is not None
    # Back on the shipped configuration, and not reading the override's table.
    assert tu.quality_word_set() is None


def test_lexicon_path_and_min_df_also_clear_the_quality_cache(tmp_path):
    """`quality_word_set()` resolves through `token_lexicon()`.

    So the keys that decide WHICH table it gets have to clear it as well —
    otherwise an A/B over the path or the threshold reads the first arm's table
    under the second arm's name.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    thin = _table(tmp_path / "a", {"vrstva": 429})
    thick = _table(tmp_path / "b", {"vrstva": 429, "malakofauna": 63})

    with tu.override_constants({"QUALITY_VOCABULARY_ENABLE": True}):
        with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": thin}):
            first = tu.quality_word_set()
        with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": thick}):
            second = tu.quality_word_set()

    assert first is not None and second is not None
    assert "malakofauna" not in first
    assert "malakofauna" in second, "the second arm read the first arm's table"


def test_no_unregistered_zero_arg_cache_reads_a_flag():
    """A source-level guard, in the same spirit as the five-caller gold test.

    Every `functools.lru_cache` in this module is either keyed on its arguments
    -- in which case a changed constant is a changed cache key and it cannot go
    stale -- or takes none, in which case it freezes whatever module state it
    read first and `override_constants()` has to be told about it. A new
    zero-argument cache that nobody registers is a new 07b, and it would arrive
    looking like a clean null result.
    """
    import ast
    import inspect

    source = inspect.getsource(tu)
    tree = ast.parse(source)
    registered = {name for names in tu._CACHES_FROM_FLAG.values() for name in names}

    zero_arg_caches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        cached = any(
            "lru_cache" in ast.unparse(dec) or "cache" == ast.unparse(dec).split(".")[-1] for dec in node.decorator_list
        )
        if not cached:
            continue
        takes_no_arguments = not (node.args.args or node.args.posonlyargs or node.args.kwonlyargs)
        if takes_no_arguments:
            zero_arg_caches.append(node.name)

    unregistered = sorted(set(zero_arg_caches) - registered)
    assert not unregistered, (
        "zero-argument lru_cache(s) not registered in _CACHES_FROM_FLAG: "
        f"{unregistered}. Either key the cache on the constants it reads, or add it "
        "to _CACHES_FROM_FLAG so override_constants() can clear it between A/B arms. "
        "See the 07b write-up in agent_dev_logs/digests/30.digest.md."
    )


# ---------------------------------------------------------------------------
# The token table's provenance header (#30)
# ---------------------------------------------------------------------------
#
# Three tests stood here until 2026-09-22, all about the de-gemination cap: that
# it separated the eight doubled-initial tokens on the 822-document table it was
# fitted to, that it separated nothing at all on the 113,100-document one, and
# that `geminate_cap_scale_warning()` said so on stderr. The cap and its advisory
# are gone with the guard (#30 D40), so the tests went with them; the measurement
# they pinned is preserved in docs/issue30/issue30_gold_ab_findings.md, which is
# where a historical table belongs.
#
# `lexicon_document_count()` survives them, because it is not part of the guard:
# tools/short_garbage_witness_report.py reads it for the banner that names the
# configuration a run was produced in.


def test_lexicon_document_count_reads_the_provenance_header(tmp_path):
    path = tmp_path / "lex.tsv"
    path.write_text(
        "# token document-frequency table — tools/build_token_lexicon.py v1.0\n"
        "# documents: 113,100  lines: 72,306,182\n"
        "# columns: token<TAB>document_frequency\n"
        "ppole\t229\n",
        encoding="utf-8",
    )
    assert tu.lexicon_document_count(str(path)) == 113100
    assert tu.lexicon_document_count("") is None
    assert tu.lexicon_document_count(str(tmp_path / "missing.tsv")) is None
