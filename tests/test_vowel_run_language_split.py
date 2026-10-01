"""
tests/test_vowel_run_language_split.py
======================================
(#30 D44) The vowel-run clause is gated on the detected language.

Three vowels in a row is evidence of scanning damage because CZECH has no
triphthongs. German and French have them natively, so the same run is worth
nothing there -- which is why the clause reached `Dauerleihe` (*aue*, permanent
loan) and `FEUILLETON` (*eui*), both scanned perfectly correctly, on hundreds of
lines.

@david-spacil, 2026-09-22: *"For Czech, 3+ vowels in a row is a good rule --
Czech has no triphthongs. For German and French it does damage. So either split
by language, or try 4+."*

The rule implemented is the split, and four as well:

    3 vowels, any language except the exempt ones  -> damaged
    4 vowels, ANY language including the exempt ones -> damaged

Blunting to 4 everywhere was measured and is the worse trade: it releases
`J. Vysoean` (*Vysočan*), `POSTKRANIAINY SKELET`, `lenaye` and `noienm k.` --
all damage -- to spare two German words, giving up 3.5 lines that currently agree
with `Trash` per at-risk line spared (stage 10b). @david-spacil corrected two of
those examples himself, having found them listed among the correctly-read.
"""

import pytest

import text_util as tu
from text_util import shape_garbage_clauses


@pytest.fixture(autouse=True)
def _judged_without_the_archive_allow_list(without_allowed_words):
    """This file is about the language split, not about `[allowed]`.

    Since @david-spacil's Q5a list landed (2026-10-01) the shipped file spares
    several of the words used below as examples, and Q5b makes a listed word never
    evidence of damage. Held out here so these tests keep measuring the split;
    the list itself is pinned in tests/test_word_lists.py.
    """


#: Correctly scanned German and French, with a three-vowel run. These are the
#: lines the split exists for.
NATIVE_TRIPHTHONGS = {
    "Dauerleihe": "deu_Latn",
    "Bauerleihe": "deu_Latn",
    "Neuotting": "deu_Latn",
    "FEUILLETON.": "fra_Latn",
}

#: Scanning damage with a three-vowel run, in Czech. @david-spacil confirmed the
#: first two as damage on 2026-09-22, correcting this repository, which had
#: listed them among the correctly-read text a higher threshold would spare.
CZECH_DAMAGE = {
    "J. Vysoean": "ces_Latn",
    "B/ POSTKRANIAINY SKELET:": "ces_Latn",
    "lenaye": "ces_Latn",
    "Poeitaecxy soubor": "ces_Latn",
}

#: Four or more in a row. Not a word in any of these languages, so the exemption
#: must not reach them.
FOUR_RUNS = ["oueussd", "eaual to:"]


@pytest.mark.parametrize("text,lang", sorted(NATIVE_TRIPHTHONGS.items()))
def test_three_vowels_in_an_exempt_language_are_not_damage(text, lang):
    assert "vowel_run" not in shape_garbage_clauses(text, lang)


@pytest.mark.parametrize("text,lang", sorted(NATIVE_TRIPHTHONGS.items()))
def test_the_same_strings_still_convict_when_the_language_is_czech(text, lang):
    """The premise: it is the language that spares them, not the string.

    Without this the test above would also pass on a predicate that had simply
    stopped firing.
    """
    assert "vowel_run" in shape_garbage_clauses(text, "ces_Latn")


@pytest.mark.parametrize("text,lang", sorted(CZECH_DAMAGE.items()))
def test_three_vowels_in_czech_are_still_damage(text, lang):
    """This is what a global threshold of 4 would have released."""
    assert "vowel_run" in shape_garbage_clauses(text, lang)


@pytest.mark.parametrize("text", FOUR_RUNS)
@pytest.mark.parametrize("lang", ["ces_Latn", "deu_Latn", "fra_Latn", None])
def test_four_vowels_convict_in_every_language(text, lang):
    assert "vowel_run" in shape_garbage_clauses(text, lang)


@pytest.mark.parametrize("text", sorted(NATIVE_TRIPHTHONGS) + sorted(CZECH_DAMAGE))
def test_an_unknown_language_gets_the_strict_threshold(text):
    """Not knowing the language must not silently exempt a line.

    `None` is what every caller that has no language passes -- the offline
    re-scorer, the service, and every test written before D44 -- so the default
    has to be the behaviour the clause had before the split, not the exemption.
    """
    assert shape_garbage_clauses(text, None) == shape_garbage_clauses(text, "ces_Latn")
    assert shape_garbage_clauses(text) == shape_garbage_clauses(text, None)


def test_an_unrecognised_label_is_not_an_exemption():
    """A label outside the exempt set gets the strict threshold, whatever it is."""
    for lang in ("zzz_Latn", "", "und", "lat_Latn"):
        assert "vowel_run" in shape_garbage_clauses("Dauerleihe", lang)


def test_a_bare_base_works_as_well_as_a_full_label():
    """FastText emits `deu_Latn`; a caller holding a base should not be surprised."""
    assert "vowel_run" not in shape_garbage_clauses("Dauerleihe", "deu")
    assert "vowel_run" not in shape_garbage_clauses("Dauerleihe", "deu_Latn")


def test_an_empty_exempt_list_restores_one_global_threshold():
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS": frozenset()}):
        assert "vowel_run" in shape_garbage_clauses("Dauerleihe", "deu_Latn")


def test_the_exempt_threshold_is_configurable_and_separate():
    """Raising the exempt threshold spares more; lowering it to the general value
    collapses the split without needing the language list emptied."""
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT": 3}):
        assert "vowel_run" in shape_garbage_clauses("Dauerleihe", "deu_Latn")
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT": 5}):
        assert "vowel_run" not in shape_garbage_clauses("oueussd", "deu_Latn")
        assert "vowel_run" in shape_garbage_clauses("oueussd", "ces_Latn")


def test_the_shipped_exempt_set_is_the_two_languages_the_answer_named():
    assert tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS == frozenset({"deu", "fra"})
    assert tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT == 4
    assert tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN == 3


def test_the_raw_language_reaches_the_clause_through_the_pipeline():
    """End to end, and the reason `original_lang` is the column that is passed.

    The stored `lang` column has been through `remap_lang()`, which rewrites any
    base outside EXPECTED_LANGS + TRUSTED_FOREIGN_LANGS to Czech. Passing that
    would answer a phonotactic question with a policy default, in the direction
    that costs the most -- unrecognised foreign text relabelled Czech and then
    held to the stricter threshold.
    """
    from classify_TEXT import score_line

    def categ(text: str, lang: str) -> str:
        return score_line(
            text_content=text,
            original_text=text,
            original_lang=lang,
            original_lang_score=0.4,
            perplexity=900.0,
            known_lang_bases=frozenset(["ces", "deu", "eng", "fra", "pol", "ita", "slk"]),
            expected_langs=["ces", "deu", "eng"],
        )["categ"]

    with tu.override_constants({"SHORT_GARBAGE_WITNESS_ENABLE": True}):
        assert categ("Dauerleihe", "ces_Latn") == "Trash"
        assert categ("Dauerleihe", "deu_Latn") == "Clear"
        # And the four-run is unaffected by the exemption, end to end.
        assert categ("oueussd", "deu_Latn") == "Trash"
