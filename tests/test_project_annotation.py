"""
tests/test_project_annotation.py
================================
``tools/project_annotation.py`` -- reading a RETURNED annotation ask (issue #30).

What must not fail silently:

1. a label outside the five categories stops the run; a blank is skipped;
2. a row settles every spelling it lists, and a spelling family claimed by two
   different answers is not guessed;
3. the join writes the ``(file, page_num, line_num)`` keys ``--gold-sidecar``
   reads, and does not depend on whether the witness fires today;
4. the tail is projected by LINES through the frame, not by rows;
5. the figures quoted on the issue for the 2026-10-01 return are what the tool
   prints from the files in the tree.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.project_annotation import (  # noqa: E402
    Decision,
    Decisions,
    main,
    normalize_label,
    parse_mix,
    project,
    read_ask,
    read_decisions,
)

ANSWERS = _ROOT / "docs" / "issue30"
HEADER = [
    "text",
    "variants",
    "lines_settled",
    "tranche",
    "stratum",
    "sampling_weight",
    "clauses",
    "categ_current",
    "gold_categ",
]


def _write(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def _d(text, label, mix, stratum="at_risk/none", variants=()):
    return Decision(text, label, "t", "sample_tail", stratum, sum(mix.values()), mix, list(variants))


@pytest.mark.parametrize(
    "raw,canon", [("trash", "Trash"), ("Clear", "Clear"), (" NOISY ", "Noisy"), ("non_text", "Non-text"), ("", "")]
)
def test_labels_are_normalised(raw, canon):
    assert normalize_label(raw, "x") == canon


@pytest.mark.parametrize("raw", ["rubbish", "Readable", "?"])
def test_an_unknown_label_stops_the_run(raw):
    with pytest.raises(ValueError, match="gold_categ"):
        normalize_label(raw, "census.csv:7")


def test_parse_mix():
    assert parse_mix("Clear:277|Trash:19|Noisy:9") == {"Clear": 277, "Trash": 19, "Noisy": 9}
    assert parse_mix("") == {}


def test_a_row_settles_every_spelling_it_lists():
    d = Decisions([_d("Keao '4", "Trash", {"Noisy": 3}, variants=["Keao '4", "Keao 1", "Keao 93"])])
    assert d.lookup("Keao 1").label == "Trash"
    # Not listed, same spelling family: settled by the same decision.
    assert d.lookup("KEAO 12").label == "Trash"
    assert d.lookup("Keaox") is None


def test_blank_rows_settle_nothing():
    d = Decisions([_d("Dauerleihe", "", {"Clear": 5})])
    assert d.lookup("Dauerleihe") is None


def test_a_family_with_two_answers_is_not_guessed():
    d = Decisions([_d("ssoe", "Trash", {"Clear": 1}), _d("SSOE -", "Clear", {"Clear": 1})])
    assert d.conflicts == ["ssoe"]
    assert d.lookup("ssoe").label == "Trash"  # exact spelling still matches
    assert d.lookup("Ssoe!") is None  # the family no longer does


def test_projection_weights_by_lines_not_rows():
    """Two strata of very different size: the large one must dominate."""
    frame = [
        {"stratum": "at_risk/full", "lines_in_stratum": 100},
        {"stratum": "at_risk/none", "lines_in_stratum": 900},
    ]
    sample = [
        _d("vrstva", "Clear", {"Clear": 1}, stratum="at_risk/full"),
        _d("OUUITN", "Trash", {"Clear": 1}, stratum="at_risk/none"),
    ]
    out = project([], sample, frame)
    assert out["tail_projection"]["labels"] == {"Clear": 100.0, "Trash": 900.0}
    # Witness on: OUUITN fires (vowel run) and is caught; vrstva does not fire.
    on = out["tail_projection"]["arms"]["on"]
    assert on["trash_caught"] == 900.0 and on["errors"] == 0.0


def test_join_writes_a_sidecar_independent_of_the_witness(tmp_path):
    census = _write(
        tmp_path / "census.csv",
        [
            [
                "XXX,1937,str. 21",
                "XXX,1937,str. 21",
                "1",
                "census_contested",
                "at_risk/none",
                "1.0",
                "triple",
                "Noisy:1",
                "clear",
            ]
        ],
    )
    sample = _write(
        tmp_path / "sample.csv",
        [
            ["OUUITN", "OUUITN", "1", "sample_tail", "at_risk/none", "27.2", "vowel_run", "Clear:1", "trash"],
            ["Dauerleihe", "Dauerleihe", "1", "sample_tail", "at_risk/none", "27.2", "vowel_run", "Clear:1", ""],
        ],
    )
    corpus = tmp_path / "ARUP"
    corpus.mkdir()
    with (corpus / "CTX1.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "page_num", "line_num", "categ", "text"])
        w.writerow(["CTX1", "1", "1", "Noisy", "XXX,1937,str. 21"])  # the witness no longer fires (D46)
        w.writerow(["CTX1", "1", "2", "Clear", "OUUITN"])
        w.writerow(["CTX1", "1", "3", "Clear", "Dauerleihe"])  # blank: not joined
        w.writerow(["CTX1", "1", "4", "Clear", "vrstva"])
    out = tmp_path / "sidecar.csv"
    docs = tmp_path / "docs.txt"
    assert (
        main(
            [
                "join",
                "--census",
                str(census),
                "--sample",
                str(sample),
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--docs-out",
                str(docs),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [(r["file"], r["page_num"], r["line_num"], r["gold_categ"]) for r in rows] == [
        ("CTX1", "1", "1", "Clear"),
        ("CTX1", "1", "2", "Trash"),
    ]
    assert docs.read_text(encoding="utf-8").strip() == str(corpus / "CTX1.csv")


def test_join_refuses_an_empty_corpus(tmp_path):
    census = _write(tmp_path / "census.csv", [])
    sample = _write(tmp_path / "sample.csv", [])
    (tmp_path / "empty").mkdir()
    assert (
        main(
            [
                "join",
                "--census",
                str(census),
                "--sample",
                str(sample),
                "--corpus",
                str(tmp_path / "empty"),
                "--out",
                str(tmp_path / "o.csv"),
            ]
        )
        == 2
    )


# ---------------------------------------------------------------------------
# The 2026-10-01 return, as it sits in the tree
# ---------------------------------------------------------------------------


def test_the_returned_files_are_the_ask_with_only_gold_filled():
    for name in ("census.csv", "sample.csv"):
        ask = list(csv.DictReader((_ROOT / "docs" / "issue30" / name).open(encoding="utf-8")))
        ret = list(csv.DictReader((ANSWERS / name).open(encoding="utf-8")))
        assert len(ask) == len(ret)
        for a, r in zip(ask, ret, strict=True):
            assert {k: v for k, v in a.items() if k != "gold_categ"} == {
                k: v for k, v in r.items() if k != "gold_categ"
            }
    assert json.loads((ANSWERS / "frame.json").read_text()) == json.loads(
        (_ROOT / "docs" / "issue30" / "frame.json").read_text()
    )


def test_the_2026_10_01_figures():
    """The numbers quoted on the issue. If a predicate change moves them, re-quote."""
    census = read_ask(ANSWERS / "census.csv", "census")
    sample = read_ask(ANSWERS / "sample.csv", "sample")
    frame = json.loads((ANSWERS / "frame.json").read_text(encoding="utf-8"))
    out = project(census, sample, frame)

    assert out["coverage"]["census"]["labelled"] == 119
    assert out["coverage"]["sample"]["labelled"] == 130

    head = out["census_at_risk"]["arms"]
    assert (head["off"]["errors"], head["on"]["errors"]) == (339, 358)
    assert head["on"]["clear_loss"] == 0

    tail = out["tail_projection"]["arms"]
    assert round(tail["off"]["errors"]) == 4844
    assert round(tail["on"]["errors"]) == 1180
    assert tail["on"]["clear_loss"] == 0

    assert out["witness_convicts_gold_clear"] == ["Svatoslavova"]


def test_every_blank_in_the_return_is_counted():
    d = read_decisions([ANSWERS / "census.csv", ANSWERS / "sample.csv"])
    assert len(d.labelled) == 249
    assert not d.conflicts
