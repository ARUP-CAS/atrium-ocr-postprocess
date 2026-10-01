"""
tests/test_witness_fused_tokens.py
==================================
(#30 D46) The shape witness splits words joined by punctuation without a space.

`_split_subtokens` splits on `. - –` only, so the witness used to read a
comma-list as one long token. `low_variety` then fired on alphabet saturation --
a 30-letter string simply runs out of new letters -- and a roman numeral fused by
a comma stopped looking like a roman numeral.

Found in @DanaKriv's 357 decisions (returned 2026-10-01,
docs/issue30/answers/2026-10-01/). The only two at-risk lines she labelled
`Clear` are both this shape, and so are 41 of the rows she left blank -- readable
German find descriptions. One gold-`Trash` row is lost and is pinned below as the
accepted cost, so a later change has to break a named line to move it.

The split is WITNESS-LOCAL: `_split_subtokens` also feeds the quality metrics,
and the flag still ships off, so nothing here changes a stored category.
"""

import pytest

import text_util as tu
from text_util import shape_garbage_clauses

#: Dana labelled these `Clear`. Before D46 the witness convicted both, on a
#: fused roman numeral (`XXX,1937,str`, `III,konec`).
DANA_CLEAR = ["XXX,1937,str. 21", "okraj sekt.III,konec"]

#: Left blank by Dana (she judged Czech only) and readable German. Each fired
#: only because commas or a slash joined several words into one token.
FUSED_READABLE = [
    "gut erhalten,Siedelungsfund,gefunden",
    "Kulturschichte,teilweise inkrustiert,",
    "Bronzezeit,Siedelungsfund,gefunden",
    "erhalten,Skelettgrabfund,Henkel osen-",
    "Steinhammerfragment,Schlagfläehe beim",
    "metacarpus/metatarsus",
    "Vicia hirsuta/craca/tetrasperma",
]

#: Labelled `Trash` by Dana and still convicted after the split: the damaged
#: piece is long enough to be read on its own.
DANA_TRASH_STILL_CAUGHT = [
    "IIIe/str. 45.",
    "Ô/IOAP",
    "zapsal ..N..EAMEAA(k",
    "OUUITN",
    "Kat. obec: CtUAAUCA",
]

#: The accepted cost. `IEU` drops below SHORT_GARBAGE_WITNESS_MIN_ALPHA once it
#: is separated from `MZMRISCH`, and nothing else on the line fires.
DANA_TRASH_RELEASED = "MZMRISCH,IEU Stxdter"


@pytest.mark.parametrize("text", DANA_CLEAR)
def test_dana_clear_lines_are_not_convicted(text):
    assert shape_garbage_clauses(text) == []


@pytest.mark.parametrize("text", FUSED_READABLE)
def test_comma_and_slash_lists_are_read_word_by_word(text):
    assert shape_garbage_clauses(text) == []


@pytest.mark.parametrize("text", DANA_TRASH_STILL_CAUGHT)
def test_dana_trash_lines_stay_convicted(text):
    assert shape_garbage_clauses(text)


def test_the_one_released_trash_line_is_named():
    """If this starts failing because the line is convicted again, good -- update D46's note."""
    assert shape_garbage_clauses(DANA_TRASH_RELEASED) == []


@pytest.mark.parametrize(
    "word,pieces",
    [
        ("erhalten,Siedelungsfund,gefunden", ["erhalten", "Siedelungsfund", "gefunden"]),
        ("III,konec", ["III", "konec"]),
        ("hirsuta/craca/tetrasperma", ["hirsuta", "craca", "tetrasperma"]),
        ("prox.sin.", ["prox", "sin"]),
        ("I-VIII-c", ["I", "VIII", "c"]),
        ("ab", ["ab"]),
        (",,", []),
    ],
)
def test_witness_subtokens(word, pieces):
    assert tu._witness_subtokens(word) == pieces


def test_split_subtokens_itself_is_unchanged():
    """The quality metrics read `_split_subtokens`; D46 must not move them."""
    assert tu._split_subtokens("erhalten,Siedelungsfund,gefunden") == ["erhalten,Siedelungsfund,gefunden"]
