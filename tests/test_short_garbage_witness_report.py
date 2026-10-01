"""
Fast, model-free tests for `tools/short_garbage_witness_report.py` (issue #30).

The report exists because `_has_shape_garbage_evidence()` is wired but disabled
(`SHORT_GARBAGE_WITNESS_ENABLE` defaults to false), so re-scoring a collection
cannot show what it WOULD reach. Asking the predicate directly is the only way to
measure its exposure before the flag is flipped. Two properties are worth pinning:

  * the report's per-clause diagnosis must not drift from the predicate it
    describes — the module asserts this per line, and
    `test_clause_diagnosis_matches_the_predicate` exercises that assertion
    across the whole #30 fixture population rather than one line at a time;
  * the report must not become a second scoring path. `test_no_signal_reconstruction`
    is the guard: the same class of drift that
    `tests/test_scoring_single_source.py` pins for the three real scorers;
  * and it must hand the witness the same INPUTS production does. Since #30 D44
    that includes the row's raw language, and a report that dropped it judged
    every German row by the Czech threshold -- see the D44 section below.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402

_TOOL_PATH = _ROOT / "tools" / "short_garbage_witness_report.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("short_garbage_witness_report", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R = _load_tool()


# The #30 population, both sides. Kept local rather than imported from
# calibration_fixtures because these are inputs to a *predicate*, not scored
# fixtures, and they carry no ppl/lang columns.
_WITNESSED = ["oueussd", "sektlll", "cuxoaid", "Tthts I", "vansasaasasa", "NINNNIC", "rragment", "IDIDIDIDIDIDUOID"]
_NOT_WITNESSED = [
    "malakofauna",
    "diapozitiv",
    "Equus caballus",
    "Occipitale",
    "Kaaden",
    "Pinii",
    "kontext",
    "Uniocrassus",
    "Skelettmaterial",
    "Ossa tarsi",
    "radius prox.sin.",
    "Reg.Bez.Aussig.",
    "1 ks",
    "II/C",
    "I-VIII-c",
    "vrstva 3",
    "ctvrtek",
    "Hannah",
    "Schifffahrt",
    "mm",
    "edelite",
    # The roman-numeral class. Added because its absence is precisely why the
    # guard below passed while the tool was crashing on 0.74% of real lines:
    # the predicate gained a roman-numeral exemption, the report's own copy of
    # the clauses did not, and no case here could see the difference. These are
    # the seven lines the 2026-09-10 witness measurement broke.
    "Sonda VIII/3",
    "12.VIII.1977,",
    "CCV. CCVI.",
    "205; CCLXII).",
    "166. Hr.XLIII.1.",
    "Lokalisace: I-VIII-eneol.II",
    "w XVIII.",
    # Drawn from the delivered collection rather than invented, so the corpus
    # keeps a foothold in the shapes that actually occur.
    "XVIII. '",
    "23. VIII.1947.",
    "580. Hr. VIII.4.",
    "Cod.Mor. II,40.-D.O.VII,721.-VIII,99,100.-",
]


@pytest.mark.parametrize("text", _WITNESSED + _NOT_WITNESSED)
def test_clause_diagnosis_matches_the_predicate(text):
    """`classify_line` asserts internally; this proves it holds on both sides.

    A clause list that is non-empty exactly when the predicate is True is what
    makes the report's "which clause fired" column trustworthy. If the predicate
    grows a clause the report does not know about, this goes red rather than
    silently under-reporting.
    """
    verdict = R.classify_line(text)
    assert bool(verdict["clauses"]) == verdict["witness"]
    assert verdict["witness"] == tu._has_shape_garbage_evidence(text)


@pytest.mark.parametrize("text", _WITNESSED)
def test_witnessed_lines_name_a_clause(text):
    verdict = R.classify_line(text)
    assert verdict["witness"], f"{text!r} should be witnessed"
    assert verdict["clauses"], f"{text!r} is witnessed but names no clause"
    for clause in verdict["clauses"].split(","):
        assert clause in R._CLAUSE_ORDER


@pytest.mark.parametrize("text", _NOT_WITNESSED)
def test_vocabulary_and_notation_are_not_witnessed(text):
    verdict = R.classify_line(text)
    assert not verdict["witness"], f"{text!r} must not be witnessed (clauses={verdict['clauses']})"


def test_route_eligibility_is_the_text_only_half_of_the_entry_condition():
    """Eligibility must mirror gate 6's text-only terms, and nothing more."""
    assert R.classify_line("oueussd")["route_eligible"]
    # a diacritic escapes the route entirely
    assert not R.classify_line("oueussdá")["route_eligible"]
    # notation is exempt
    assert not R.classify_line("II/C")["route_eligible"]
    # too many tokens for the route
    long_line = " ".join(["oueussd"] * (tu.ISOLATED_CHAR_MIN_TOKENS + 1))
    assert not R.classify_line(long_line)["route_eligible"]
    # ...but the witness itself is length-agnostic, and the report must not
    # conflate the two: eligibility is about the ROUTE, the witness is about the
    # spelling.
    assert R.classify_line(long_line)["witness"]


def test_no_signal_reconstruction():
    """The report must never grow a second scoring path.

    `lang_score` in a DOC_LINE_CATEG CSV is the `remap_lang` cap, not the
    two-tier trust score the rules read; deriving routing decisions from the
    stored columns is the harness bug already fixed in
    tests/test_rotation_regression.py and tests/test_calibration.py::_categ.
    Re-scoring belongs to tools/recategorize_from_csv.py, which reuses
    classify_TEXT.score_line. Source-inspected for the same reason
    tests/test_scoring_single_source.py inspects its subjects.
    """
    source = _TOOL_PATH.read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    for forbidden in (
        "compute_quality_score(",
        "categorize_line(",
        "determine_category(",
        "score_line(",
        "TRUST_TIER_UNKNOWN",
        "trust_lang_score",
    ):
        assert forbidden not in body, (
            f"{_TOOL_PATH.name} references {forbidden!r}: it is re-scoring, or reconstructing a signal. "
            "It must stay on text-only predicates."
        )


def test_reads_a_doc_line_categ_csv_and_writes_candidates(tmp_path):
    src = tmp_path / "CTX000000000.csv"
    with src.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["categ", "text", "word_count"])
        writer.writerow(["Trash", "oueussd", "1"])
        writer.writerow(["Clear", "malakofauna", "1"])
        writer.writerow(["Clear", "v klášteře Strahovském.", "3"])
        writer.writerow(["Trash", "", "0"])

    out = tmp_path / "candidates.csv"
    assert R.main([str(src), "--out", str(out)]) == 0

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["text"] for r in rows] == ["oueussd"], "only witnessed lines belong in the candidate file"
    assert rows[0]["categ_current"] == "Trash"
    assert rows[0]["gold_categ"] == "", "gold column must ship blank so it can be filled blind"


def test_from_distinct_reads_a_returned_census_and_repeats(tmp_path):
    """(#30, 2026-10-01) The returned ask is family-shaped and lower-case.

    `census.csv` rows carry a representative `text` plus every spelling in
    `variants`, and @DanaKriv's labels came back as `trash` / `noisy`. The old
    reader matched `text` exactly and passed the label through verbatim, so a
    variant spelling stayed unlabelled and `trash` was not `Trash`.
    """
    src = tmp_path / "CTX000000000.csv"
    with src.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "page_num", "line_num", "categ", "text", "word_count"])
        writer.writerow(["CTX000000000", "1", "1", "Clear", "oueussd", "1"])
        writer.writerow(["CTX000000000", "1", "2", "Clear", "Oueussd .", "2"])
        writer.writerow(["CTX000000000", "1", "3", "Noisy", "OUUITN", "1"])

    header = ["text", "variants", "lines_settled", "tranche", "stratum", "categ_current", "gold_categ"]
    census = tmp_path / "census.csv"
    with census.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["oueussd", "oueussd | Oueussd .", "2", "census_head", "at_risk/none", "Clear:2", "trash"])
    sample = tmp_path / "sample.csv"
    with sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["OUUITN", "OUUITN", "1", "sample_tail", "at_risk/none", "Noisy:1", "Trash"])

    out = tmp_path / "candidates.csv"
    assert R.main([str(src), "--out", str(out), "--from-distinct", str(census), "--from-distinct", str(sample)]) == 0
    labels = {r["text"]: r["gold_categ"] for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert labels == {"oueussd": "Trash", "Oueussd .": "Trash", "OUUITN": "Trash"}


def test_from_distinct_refuses_an_unknown_label(tmp_path):
    src = tmp_path / "CTX000000000.csv"
    with src.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["categ", "text", "word_count"])
        writer.writerow(["Clear", "oueussd", "1"])
    filled = tmp_path / "filled.csv"
    filled.write_text("text,gold_categ\noueussd,rubbish\n", encoding="utf-8")
    assert R.main([str(src), "--out", str(tmp_path / "o.csv"), "--from-distinct", str(filled)]) == 2


def test_plain_lines_mode(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("oueussd\nmalakofauna\n\n", encoding="utf-8")
    assert R.main(["--lines", str(probe)]) == 0


def test_flag_state_does_not_change_the_report():
    """The report reads the predicate directly, so the ship-inert flag is moot.

    This is the property that makes the tool usable *before* the flag flips —
    and the reason its header says the flag does not affect it.
    """
    before = R.classify_line("oueussd")
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_ENABLE": True}):
        after = R.classify_line("oueussd")
    assert before == after


def test_the_report_has_no_clause_implementation_of_its_own():
    """The structural guarantee, asserted rather than trusted.

    `test_clause_diagnosis_matches_the_predicate` can only catch drift on the
    lines it happens to list -- and it did not catch the roman-numeral case,
    because none was listed. The durable fix is that there is nothing to drift:
    the module re-exports `text_util`'s vocabulary and delegates to its
    implementation. This test fails if anyone reintroduces a local copy.
    """
    assert R._CLAUSE_ORDER is tu.SHAPE_GARBAGE_CLAUSES, "the clause vocabulary must be text_util's, not a copy"
    assert not hasattr(R, "_clauses_for_token"), (
        "a per-token clause implementation is back in the report tool; "
        "delegate to text_util.shape_garbage_clauses() instead"
    )
    assert R.clauses_for_line("oueussd") == tu.shape_garbage_clauses("oueussd")


@pytest.mark.parametrize(
    "text, clause",
    [
        ("oueussd", "vowel_run"),
        ("sektlll", "triple"),
        ("rragment", "initial_geminate"),
        ("vansasaasasa", "low_variety"),
    ],
)
def test_every_clause_name_is_still_reachable(text, clause):
    """Each of the four clauses still has a line that reaches it.

    The roman-numeral exemption sits above all four, so it could in principle
    have made one unreachable. This is the check that it did not.
    """
    assert clause in R.clauses_for_line(text)


# ---------------------------------------------------------------------------
# --recursive: the two-archive layout (#30 stage 8)
#
# "All of the collections" is spelled ARUP/ and ARUB/ under one parent on the
# cluster. A bare `*.csv` glob over that parent matches nothing, which is the
# same gap `recategorize_from_csv.py --recursive` exists to close. The refusal
# below is what made it survivable — it is a refusal, not a plausible zero —
# but the corpus-scale figures still could not be produced in one invocation.
# ---------------------------------------------------------------------------


def _archive(root: Path, name: str, doc: str, rows: list[tuple[str, str, str]]) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{doc}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "page_num", "line_num", "categ", "text", "word_count"])
        for i, (categ, text, wc) in enumerate(rows, start=1):
            writer.writerow([doc, "1", str(i), categ, text, wc])
    return d


def test_a_two_archive_parent_needs_recursive(tmp_path):
    _archive(tmp_path, "ARUP", "CTX000000001", [("Trash", "oueussd", "1")])
    _archive(tmp_path, "ARUB", "MTX000000002", [("Clear", "sektlll", "1")])

    # Without --recursive the parent matches nothing, and the tool REFUSES
    # rather than reporting an empty corpus as a result.
    with pytest.raises(SystemExit):
        R.main(["--input-dir", str(tmp_path)])

    out = tmp_path / "candidates.csv"
    assert R.main(["--input-dir", str(tmp_path), "--recursive", "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert {r["text"] for r in rows} == {"oueussd", "sektlll"}, "both archives must be read"


def test_the_refusal_names_recursive_as_the_fix(tmp_path, capsys):
    (tmp_path / "ARUP").mkdir()
    with pytest.raises(SystemExit):
        R.main(["--input-dir", str(tmp_path)])
    assert "--recursive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --by-group: the modal dedup's blast radius (#30 stage 8, H6)
# ---------------------------------------------------------------------------


def test_the_tie_break_is_alphabetical_and_never_lands_on_trash():
    """`apply_document_postprocessing()` resolves with `x.mode()[0]`.

    `pandas.Series.mode()` returns its tied values SORTED, so `[0]` is the
    alphabetically first — and 'Clear' < 'Noisy' < 'Trash'. A tie therefore
    never demotes to Trash. That is an accident of the alphabet rather than a
    design, it is load-bearing for H6 (it makes a whole class of the feared
    case impossible), and nothing in production says it out loud.
    """
    from collections import Counter

    assert R._vote(Counter({"Trash": 3, "Clear": 3})) == ("tie", "Clear")
    assert R._vote(Counter({"Trash": 3, "Noisy": 3})) == ("tie", "Noisy")
    assert R._vote(Counter({"Trash": 2, "Noisy": 2, "Clear": 2})) == ("tie", "Clear")
    # And the ordering the claim rests on, asserted rather than assumed.
    assert sorted(["Trash", "Clear", "Noisy"]) == ["Clear", "Noisy", "Trash"]


def test_vote_shapes_are_distinguished():
    from collections import Counter

    assert R._vote(Counter({"Clear": 7})) == ("unanimous", "Clear")
    assert R._vote(Counter({"Trash": 39, "Clear": 11})) == ("strict majority", "Trash")
    # 4 of 9 is a plurality, not a majority — the case H6 option 2 targets.
    assert R._vote(Counter({"Trash": 4, "Clear": 3, "Noisy": 2})) == ("bare plurality", "Trash")


def test_by_group_counts_destroyed_and_rescued_lines(tmp_path, capsys):
    """The two numbers H6 needs, and they point in opposite directions."""
    doc_a = [("Trash", "cuxoaid", "1")] * 3 + [("Clear", "cuxoaid", "1")] * 1  # Trash wins, 1 Clear destroyed
    doc_b = [("Clear", "sektlll", "1")] * 5 + [("Trash", "sektlll", "1")] * 2  # Clear wins, 2 Trash rescued
    _archive(tmp_path, "ARUP", "CTX000000001", doc_a)
    _archive(tmp_path, "ARUB", "MTX000000002", doc_b)

    dump = tmp_path / "groups.csv"
    assert R.main(["--input-dir", str(tmp_path), "--recursive", "--by-group", str(dump)]) == 0
    out = capsys.readouterr().out

    assert "(document, string) groups" in out
    assert "carrying Clear down :      1 group(s),      1 Clear line(s)" in out
    assert "lifting Trash out  :      1 group(s),      2 Trash line(s)" in out
    assert "NET                                      :     +1" in out

    rows = list(csv.DictReader(dump.open(encoding="utf-8")))
    assert len(rows) == 2, "both contested groups belong in the dump"
    by_text = {r["text"]: r for r in rows}
    assert by_text["cuxoaid"]["dedup_winner"] == "Trash"
    assert by_text["cuxoaid"]["clear_at_risk"] == "1"
    assert by_text["sektlll"]["dedup_winner"] == "Clear"
    assert by_text["sektlll"]["clear_at_risk"] == "0", "a rescued group puts no Clear line at risk"


def test_by_group_summary_needs_no_path(tmp_path, capsys):
    _archive(tmp_path, "ARUP", "CTX000000001", [("Clear", "oueussd", "1")] * 2)
    assert R.main(["--input-dir", str(tmp_path), "--recursive", "--by-group"]) == 0
    assert "(document, string) groups" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.csv")), "no dump requested, none written"


def test_by_group_refuses_plain_lines_input(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("oueussd\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        R.main(["--lines", str(probe), "--by-group"])


def test_a_group_is_all_or_nothing_for_the_witness(tmp_path):
    """The property the whole group analysis rests on.

    The witness is a function of the line's text and of the raw language label
    FastText derives from that same text, so within a (document, string) group
    it convicts every member or none. It therefore
    cannot CREATE a split — only move a group that was already split, or move a
    unanimous one wholesale. If that ever stops being true, the blast-radius
    figures stop meaning what they say.
    """
    for text in ("oueussd", "sektlll", "malakofauna", "vrstva"):
        verdicts = {R.classify_line(text, 1)["witness"] for _ in range(3)}
        assert len(verdicts) == 1, f"{text!r} must classify identically every time"


# ---------------------------------------------------------------------------
# (#30 D44) The report must tell the witness what production tells it.
#
# Production passes each row's raw `original_lang` to the witness, so the
# vowel-run clause needs 4 vowels in `deu`/`fra` and 3 everywhere else. Until
# 2026-09-23 the report passed no language, so every row got the strict
# threshold: it convicted `Dauerleihe` on German rows the gate spares, and
# overstated exactly the exposure the split removes. The fourth harness
# divergence in this repository was a copied implementation; this was the same
# shape one argument over.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("without_allowed_words")  # `Dauerleihe` is in the shipped [allowed] list (Q5a)
def test_classify_line_passes_the_language_to_the_witness():
    assert R.classify_line("Dauerleihe", 1, "ces_Latn")["witness"]
    assert not R.classify_line("Dauerleihe", 1, "deu_Latn")["witness"]
    # Unknown is strict, as it is in the gate -- never an exemption.
    assert R.classify_line("Dauerleihe", 1)["witness"]
    # Still delegation, not a copy, on the language path too.
    assert R.clauses_for_line("Dauerleihe", "deu_Latn") == tu.shape_garbage_clauses("Dauerleihe", "deu_Latn")


@pytest.mark.usefixtures("without_allowed_words")  # `Dauerleihe` is in the shipped [allowed] list (Q5a)
def test_a_csv_row_is_judged_in_its_own_detected_language(tmp_path):
    src = tmp_path / "CTX000000000.csv"
    with src.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "page_num", "line_num", "categ", "text", "word_count", "original_lang"])
        writer.writerow(["CTX000000000", "1", "1", "Clear", "Dauerleihe", "1", "ces_Latn"])
        writer.writerow(["CTX000000000", "1", "2", "Clear", "Dauerleihe", "1", "deu_Latn"])
        writer.writerow(["CTX000000000", "1", "3", "Clear", "Dauerleihe", "1", ""])

    out = tmp_path / "candidates.csv"
    assert R.main([str(src), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [(r["line_num"], r["clauses"]) for r in rows] == [("1", "vowel_run"), ("3", "vowel_run")], (
        "the Czech row and the row with no language are witnessed; the German row is not"
    )

    # The split is the config's, not the report's: emptying the exempt list
    # gives back one global threshold here exactly as it does in the gate.
    with tu.override_constants({"SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS": frozenset()}):
        assert R.main([str(src), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["line_num"] for r in rows] == ["1", "2", "3"]


def test_input_dir_repeats_to_read_both_archives(tmp_path):
    """The only safe way to say "all of the documents" (#30 stage 8).

    The two archives have no common parent holding nothing else, and a staging
    directory of symlinks does NOT work: `pathlib.Path.glob("**/*.csv")` does
    not follow directory symlinks, so it returns an empty list WITHOUT failing.
    That is the shape of error this whole issue is about, so the supported way
    has to be explicit.
    """
    arup = _archive(tmp_path, "ARUP", "CTX000000001", [("Trash", "oueussd", "1")])
    arub = _archive(tmp_path, "ARUB", "MTX000000002", [("Clear", "sektlll", "1")])

    out = tmp_path / "candidates.csv"
    assert R.main(["--input-dir", str(arup), "--input-dir", str(arub), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert {r["text"] for r in rows} == {"oueussd", "sektlll"}
    assert {r["file"] for r in rows} == {"CTX000000001", "MTX000000002"}


def test_a_symlinked_staging_directory_would_read_nothing(tmp_path):
    """Pinned as a fact about the platform, because it is the trap this avoids.

    If a future Python (3.13+ has `recurse_symlinks`) changes this, the staging
    approach becomes viable and this test says so by failing — which is the
    right way to find out.
    """
    real = _archive(tmp_path / "real", "ARUP", "CTX000000001", [("Trash", "oueussd", "1")])
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "ARUP").symlink_to(real, target_is_directory=True)

    assert list(stage.glob("**/*.csv")) == [], (
        "pathlib's ** stopped ignoring directory symlinks — a symlink staging "
        "directory is now viable, and the job's per-archive --input-dir can be simplified"
    )
    with pytest.raises(SystemExit):
        R.main(["--input-dir", str(stage), "--recursive"])


def test_the_same_corpus_reached_twice_is_refused(tmp_path):
    """Double-counting every line would inflate every exposure figure silently."""
    arup = _archive(tmp_path, "ARUP", "CTX000000001", [("Trash", "oueussd", "1")])
    with pytest.raises(SystemExit):
        R.main(["--input-dir", str(arup), "--input-dir", str(arup)])


def test_per_archive_document_counts_are_printed(tmp_path, capsys):
    """So "did it see all the documents" is answered in the log, not assumed."""
    arup = _archive(tmp_path, "ARUP", "CTX000000001", [("Trash", "oueussd", "1")])
    arub = _archive(tmp_path, "ARUB", "MTX000000002", [("Clear", "sektlll", "1")])
    R.main(["--input-dir", str(arup), "--input-dir", str(arub)])
    err = capsys.readouterr().err
    assert "ARUP" in err and "ARUB" in err
    assert err.count("document CSV(s)") == 2


# ---------------------------------------------------------------------------
# (#30 D42) The banner has to carry the configuration that produced the numbers.
#
# 08f/08g ran with no lexicon configured and the delivered log did not say so —
# it had to be inferred from which warning was absent, which is how a wrong
# population survived a delivery and a 592-decision annotation ask (digest T1).
# The same shape was waiting on SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN: separately
# tunable, responsible for two thirds of the witness's exposure, and named in no
# artefact the tool wrote.
#
# These are source-level guards rather than string assertions on one line,
# because the failure mode is a NEW constant arriving unprinted, not this one
# regressing.
# ---------------------------------------------------------------------------
def _banner(capsys, argv):
    assert R.main(argv) == 0
    return capsys.readouterr().out


def test_the_banner_names_every_witness_constant_the_predicate_reads(capsys, tmp_path):
    """Every `SHORT_GARBAGE_WITNESS_*` the clause body reads must reach the log.

    Directly, or through `_vowel_run_min_for()`: that helper is where the D44
    split's two constants are read, and the clause body only calls it, so a scan
    of the body alone would let them steer the numbers without being printed.

    `SHORT_GARBAGE_WITNESS_ENABLE` is excluded: it has its own line, and the
    report states there that it does not affect the result.
    """
    import inspect
    import re as _re

    source = inspect.getsource(tu.shape_garbage_clauses) + inspect.getsource(tu._vowel_run_min_for)
    read_by_predicate = {
        name
        for name in _re.findall(r"\bSHORT_GARBAGE_WITNESS_[A-Z_]+\b", source)
        if name != "SHORT_GARBAGE_WITNESS_ENABLE"
    }
    assert read_by_predicate, "the clause body reads no witness constants — this guard has gone stale"

    probe = tmp_path / "probe.txt"
    probe.write_text("oueussd\n", encoding="utf-8")
    out = _banner(capsys, ["--lines", str(probe)])

    for name in sorted(read_by_predicate):
        short = name[len("SHORT_GARBAGE_WITNESS_") :]
        assert f"{short}=" in out, (
            f"{name} steers the predicate but never reaches the report's banner. "
            "A run's own log has to say which value produced its numbers; inferring it "
            "from an absent warning is what cost stage 8 its exposure figures (digest T1)."
        )


def test_the_banner_reports_the_lexicon_whether_or_not_one_is_configured(capsys, tmp_path):
    """The absence of a table is the case that has to be printed, not the presence."""
    probe = tmp_path / "probe.txt"
    probe.write_text("oueussd\n", encoding="utf-8")

    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": ""}):
        assert "NONE CONFIGURED" in _banner(capsys, ["--lines", str(probe)])

    table = tmp_path / "token_df.tsv"
    table.write_text("# documents: 113,100  lines: 56,599,631\nvrstva\t429\n", encoding="utf-8")
    with tu.override_constants({"SHORT_GARBAGE_LEXICON_PATH": str(table)}):
        out = _banner(capsys, ["--lines", str(probe)])
    assert str(table) in out
    assert "113,100 documents" in out, "the provenance line is the whole point of naming the path"


def test_the_banner_says_when_the_input_has_no_language(capsys, tmp_path):
    """`--lines` input carries no language, so every line meets the strict threshold.

    That changes what the vowel-run numbers mean, so the log says it rather than
    leaving a reader to infer it (D42's lesson, one constant over).
    """
    probe = tmp_path / "probe.txt"
    probe.write_text("Dauerleihe\n", encoding="utf-8")
    assert "--lines input has no language" in _banner(capsys, ["--lines", str(probe)])
