#!/usr/bin/env python3
"""
tools/short_garbage_witness_report.py — measure `_has_shape_garbage_evidence()`
against real delivered `DOC_LINE_CATEG` CSVs (issue #30).

WHY THIS EXISTS
---------------
`_has_shape_garbage_evidence()` ships behind `SHORT_GARBAGE_WITNESS_ENABLE`
(default false). Since PR #48 merged it IS wired -- read by the short-line
garbage gate as a second disjunct -- but the flag is still off, so it cannot
change any outcome until someone turns it on.

It can be exercised end to end, and has been. `tools/ab_constant_eval.py
--const SHORT_GARBAGE_WITNESS_ENABLE --values false,true` scores the flag
against the gold sidecar (#30 stages 08b and 10e), and
`tools/recategorize_from_csv.py` re-scores with the flag set through the config
file or `ATRIUM_TEXT_UTILS_SHORT_GARBAGE_WITNESS_ENABLE` -- not through
`--override`, which still rejects every `SHORT_GARBAGE_WITNESS_*` key (see
`_DELIBERATELY_NOT_TUNABLE` in `tests/test_recategorize_parity.py`). That is
how @david-spacil re-scored his 508 lines. But gold reaches only 23 of the ~20k
lines the witness fires on in the 822-document corpus, and the in-tree sample
corpus cannot help either: of its 15 lines exactly 2 reach the route's
text-only entry condition, both already `Trash`, so flipping the flag there
changes 0 categories.

This tool answers the other half, over any collection you already have on
disk: **which lines would the witness reach, and what does the pipeline
currently call them?** That is the exposure the annotation ask is built from,
and it needs no GPU, no FastText and no re-scoring.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does **not** re-score, and it does **not** reconstruct any signal. Every
column it computes is a function of the line's text -- the three vetoes
(`has_cz_diacs`, `is_structured_line`, `is_domain_notation`), the witness, and
which of the witness's clauses fired -- plus, for the vowel-run clause alone,
the row's stored raw language label. That label is read verbatim from
`original_lang`, the same value production hands the witness since #30 D44; it
is an input, not a reconstruction. `--lines` input has no language, so there
every line gets the general threshold. The signal-dependent terms of the
route's condition — `lang_score`, `rot_ratio`, `gibberish_present`,
`weird_ratio` — are **not** re-derived here, because the stored CSV columns are
not the values the rules see (`lang_score` in the CSV is the `remap_lang` cap;
the rules read the two-tier trust score) and reconstructing them is exactly the
harness bug already fixed twice in this repo, in
`tests/test_rotation_regression.py` and `tests/test_calibration.py::_categ`.
Re-scoring is `tools/recategorize_from_csv.py`'s job and it reuses
`classify_TEXT.score_line`; this tool stays on the side of the line where text
is all it needs.

Because of that, the population it reports is an **upper bound** on the lines
the route acts on: every line it counts satisfies the text-only half of the
entry condition (`word_count <= ISOLATED_CHAR_MIN_TOKENS`, no Czech diacritics,
not structured, not notation), but some of them fail the signal half and never
reach the rule. Read the counts as exposure, not as an effect size.

The `categ` column is the **current pipeline's** answer, not ground truth. A
witnessed line sitting at `Clear` is a false-positive *candidate*; whether it is
actually wrong is an annotation question, which is what `--out` is for.

USAGE
-----
    # exposure over a delivered collection
    python3 tools/short_garbage_witness_report.py --input-dir /path/to/DOC_LINE_CATEG

    # a single document, with examples of every clause
    python3 tools/short_garbage_witness_report.py DOC_LINE_CATEG/CTX200205348.csv --examples 8

    # write the candidate lines out for blind annotation
    python3 tools/short_garbage_witness_report.py --input-dir DOC_LINE_CATEG \
        --out /tmp/witness_candidates.csv

    # ad-hoc: one line per row of a plain text file (no CSV schema needed)
    python3 tools/short_garbage_witness_report.py --lines /tmp/probe.txt

    # BOTH COLLECTIONS in one invocation (#30 stage 8). --input-dir repeats.
    # There is no common parent holding only these two, and a staging directory
    # of symlinks does NOT work: pathlib's `**` glob does not follow directory
    # symlinks, so it would read zero rows and not fail.
    python3 tools/short_garbage_witness_report.py \
        --input-dir ../ARUP/DOC_LINE_CATEG_307 \
        --input-dir ../ARUB/DOC_LINE_CATEG_307 \
        --distinct queue.csv --by-group groups.csv

Stdlib only (`csv`, not pandas), so it runs anywhere `text_util` imports.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_util as tu  # noqa: E402

# The clause vocabulary, imported rather than restated. This module USED TO
# carry its own copy of the four tests, mirroring `_has_shape_garbage_evidence`,
# with an assertion in `classify_line` that the two agreed.
#
# The assertion earned its keep: the roman-numeral exemption (#30, 2026-09-10)
# landed in the predicate and not in the copy, and the guard fired on 0.74% of
# real lines -- every line carrying a roman numeral, which in archaeological
# field documentation means `Sonda VIII/3`, `23. VIII.1947.`, `580. Hr. VIII.4.`
# and thousands more. The tool aborted on the first one, which made it useless
# for exactly the measurement the witness flag is gated on.
#
# So the copy is gone. `text_util.shape_garbage_clauses()` is the single
# implementation and `_has_shape_garbage_evidence()` is `bool()` of it; this
# module just re-exports the vocabulary. There is no longer a second thing to
# drift, which is a better guarantee than an assertion that it has not.
_CLAUSE_ORDER = tu.SHAPE_GARBAGE_CLAUSES


def clauses_for_line(text: str, lang: str | None = None) -> list[str]:
    """Clauses fired across a line's sub-tokens, in fixed order.

    A thin alias for `text_util.shape_garbage_clauses`, kept because this
    module's CLI, tests and `--examples` output all name it.
    """
    return tu.shape_garbage_clauses(text, lang)


def classify_line(text: str, word_count: int | None = None, lang: str | None = None) -> dict:
    """Text-only verdicts for one line. No scoring, no signal reconstruction.

    ``lang`` is the row's RAW FastText label (`original_lang`), passed to the
    witness exactly as `classify_TEXT.score_line` passes it (#30 D44): 3 vowels
    convict outside `SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS`, 4 inside it.
    ``None`` means unknown and gets the general threshold -- production's own
    default, so a row with no language is judged as strictly as the gate would.
    Without this the report convicted `Dauerleihe` on German rows the gate
    spares, overstating the exposure the split removes.
    """
    wc = len(text.split()) if word_count is None else word_count
    diacs = tu.has_cz_diacs(text)
    structured = tu.is_structured_line(text)
    notation = tu.is_domain_notation(text)
    witness = tu._has_shape_garbage_evidence(text, lang)
    clauses = clauses_for_line(text, lang)

    # Structurally guaranteed now that both come from `shape_garbage_clauses`,
    # and kept as the tripwire if anyone reintroduces a second implementation.
    assert bool(clauses) == witness, (
        f"clause diagnosis disagrees with _has_shape_garbage_evidence() on {text!r}: "
        f"clauses={clauses} witness={witness}. Both must come from "
        f"text_util.shape_garbage_clauses(); do not reintroduce a local copy."
    )

    return {
        "text": text,
        "word_count": wc,
        "has_cz_diacs": diacs,
        "structured": structured,
        "notation": notation,
        # The text-only half of rule_short_garbage's entry condition. An upper
        # bound on the population: the signal half is not evaluated here.
        "route_eligible": bool(wc) and wc <= tu.ISOLATED_CHAR_MIN_TOKENS and not (diacs or structured or notation),
        "witness": witness,
        "clauses": ",".join(clauses),
    }


def _iter_csv_rows(paths: list[Path]):
    """Yield (locator, text, word_count, stored_categ, lang) per scoreable line.

    ``locator`` is ``(file, page_num, line_num)`` -- the gold sidecar key from
    ``recategorize_from_csv.GOLD_SIDECAR_KEYS``, carried so that an annotated
    ``--out`` file can be joined straight back onto the batch. It used to be
    ``path.name`` alone, which made the candidate file a dead end: see the
    comment on the writer.
    """
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                raw_wc = (row.get("word_count") or "").strip()
                try:
                    wc = int(raw_wc) if raw_wc else None
                except ValueError:
                    wc = None
                locator = (
                    (row.get("file") or path.stem).strip(),
                    (row.get("page_num") or "").strip(),
                    (row.get("line_num") or "").strip(),
                )
                # (#30 stage 11b) `original_lang`, NOT `lang`. The stored `lang`
                # column has been through `remap_lang()`, which rewrites any base
                # outside EXPECTED_LANGS + TRUSTED_FOREIGN_LANGS to Czech -- so it
                # is a policy output, not a detection, and reading it here would
                # answer "how much of this queue is non-Czech" with a number the
                # remap partly decided. This module's own docstring already warns
                # that `lang_score` in the CSV is the remap cap; this is the same
                # trap one column over.
                lang = (row.get("original_lang") or "").strip() or "?"
                yield locator, text, wc, (row.get("categ") or "").strip() or "?", lang


def _iter_plain_lines(path: Path):
    """--lines mode has no locators or language, so those columns come back blank."""
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            yield (path.stem, "", ""), text, None, "?", "?"


def _collect_csvs(path: Path, recursive: bool = False) -> list[Path]:
    """Every document CSV under ``path``. ``recursive`` walks sub-directories.

    (#30 stage 8.) The non-recursive default is the same gap
    ``recategorize_from_csv.py --recursive`` was added to close, and it bites the
    same way: a two-archive layout (``ARUP/`` and ``ARUB/`` under one parent) is
    how "all of the collections" is spelled on the cluster, and a bare
    ``*.csv`` glob over that parent matches nothing at all.

    It does NOT report success on nothing -- ``main()`` refuses an empty match --
    so the failure mode here is a refusal rather than a plausible zero. That is
    the only reason this was survivable; it still meant the corpus-scale exposure
    figures could not be produced in one invocation.
    """
    if path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        return sorted(p for p in path.glob(pattern) if p.is_file())
    return [path]


def _refuses_gold_dir(path: Path, flag: str) -> bool:
    """Refuse the one directory these files break, printing why. True == refused.

    `tools/gold/` is one CSV per document; these queues span many and carry no
    `categ`, so dropping one there poisons every consumer of that directory --
    which is exactly what happened, because this tool used to close by calling the
    annotated result "a gold set gold_gate() can consume". Sidecars go in
    tools/gold/sidecars/.
    """
    gold_dir = (Path(__file__).resolve().parent / "gold").resolve()
    if path.resolve().parent != gold_dir:
        return False
    print(
        f"error: refusing to write {path.name} into {gold_dir} ({flag}).\n"
        f"       That directory is one CSV per document and this file spans many;\n"
        f"       a multi-document annotation set is a SIDECAR.\n"
        f"       Use: {flag} {gold_dir / 'sidecars' / path.name}",
        file=sys.stderr,
    )
    return True


def _write_distinct(path: Path, witnessed_rows: list) -> None:
    """One row per distinct STRING, most frequent first.

    (#30, 2026-09-17.) The line-level queue is 20,324 rows but only 5,243
    strings, 94% of which occur exactly once, and `ppole` alone is 57% of the
    lines. Annotating per line spends almost all of an archivist's attention
    re-deciding `ppole` 11,562 times.

    `occurrences` is carried so the reader can work in frequency order and know
    what each decision buys, and `categ_current` records the MIX rather than one
    value, because the pipeline does not always agree with itself on a repeated
    string -- `Linum usitatissimum` is 22 Clear / 40 Trash. That disagreement is
    worth seeing while deciding, not averaging away.
    """
    by_text: dict[str, dict] = {}
    for (file_id, page_num, line_num), verdict, categ in witnessed_rows:
        entry = by_text.setdefault(
            verdict["text"],
            {
                "n": 0,
                "wc": verdict["word_count"],
                "clauses": verdict["clauses"],
                "categs": Counter(),
                "example": (file_id, page_num, line_num),
            },
        )
        entry["n"] += 1
        entry["categs"][categ] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "text",
                "occurrences",
                "word_count",
                "clauses",
                "categ_current",
                "example_file",
                "example_page_num",
                "example_line_num",
                "gold_categ",
            ]
        )
        for text, e in sorted(by_text.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
            mix = "|".join(f"{c}:{n}" for c, n in e["categs"].most_common())
            writer.writerow([text, e["n"], e["wc"], e["clauses"], mix, *e["example"], ""])

    n_lines = sum(e["n"] for e in by_text.values())
    print(f"\nwrote {len(by_text)} distinct string(s) to {path}  (covering {n_lines} lines)")
    running = 0
    for cut in (1, 20, 100, 500):
        running = sum(e["n"] for _, e in sorted(by_text.items(), key=lambda kv: -kv[1]["n"])[:cut])
        if cut <= len(by_text):
            print(f"  top {cut:5d} string(s) settle {running:8d} lines  ({running / n_lines:5.1%})")
    print("  Fill `gold_categ` in frequency order, then project it back onto the line-level")
    print(f"  queue with:  --from-distinct {path} --out <sidecar>.csv")


#: Categories in the order `pandas.Series.mode()` sorts them. This is not a
#: stylistic choice and it decides real lines: `apply_document_postprocessing()`
#: resolves a document's repeated text with `x.mode()[0]`, and `mode()` returns
#: its tied values SORTED, so `[0]` is the alphabetically first of them.
#: `Clear` < `Empty` < `Noisy` < `Non-text` < `Trash`, so a TIE CAN NEVER LAND
#: ON `Trash` -- an accident of the alphabet that happens to be the safe
#: direction, and which nothing in the code says out loud.
_MODE_TIE_ORDER = ("Clear", "Empty", "Noisy", "Non-text", "Trash")


def _vote(counts: Counter) -> tuple[str, str]:
    """(shape, winner) for one (document, string) group under the modal dedup.

    ``shape`` is ``unanimous`` / ``strict majority`` / ``bare plurality`` /
    ``tie``; ``winner`` is the category ``mode()[0]`` would actually pick.
    """
    if len(counts) == 1:
        only = next(iter(counts))
        return "unanimous", only
    total = sum(counts.values())
    top = max(counts.values())
    tied = sorted((c for c, n in counts.items() if n == top), key=lambda c: (c not in _MODE_TIE_ORDER, c))
    winner = tied[0]
    if len(tied) > 1:
        return "tie", winner
    return ("strict majority" if top > total / 2 else "bare plurality"), winner


def _write_by_group(path: Path | None, witnessed_rows: list) -> None:
    """The modal dedup's blast radius, in the unit the dedup actually votes in.

    (#30 stage 8, H6.) Issue #30 argued for weeks that **per-line precision
    does not bound Clear-loss**, because `apply_document_postprocessing()`
    rewrites every occurrence of a repeated string in a document to that
    string's modal category -- so convicting a few occurrences can flip the vote
    and carry a correct one down with it. The mechanism is real. Its SIZE was
    argued from rather than measured until this flag, and the decision it
    informed (stop a bare plurality demoting `Clear` -> `Trash`?) would have been
    a production-wide change.

    This reports it. The unit is the **(document, string) group**, because that
    is what `groupby("text")` inside one document's frame votes on, and the
    witness is a function of the line's text and of the raw language label
    FastText derives from that same text -- so within a group it convicts ALL
    or NONE. It cannot create a split; it can only move a group that was
    already split, or move a unanimous group wholesale.

    Two numbers matter and neither is the group count:

      * **groups whose vote lands on `Trash` while some member is `Clear`** --
        the lines the cascade destroys, and
      * **groups whose vote lands off `Trash` while some member is `Trash`** --
        the lines it rescues.

    On the 822-document queue those were 4 groups / 17 Clear lines against 17
    groups / 31 Trash lines: the cascade is NET PROTECTIVE on this population,
    which is the opposite of how the mechanism had been read. It survives at
    collection scale, measured with this flag over both archives: 24 Clear
    lines destroyed against 305 Trash rescued with no table (stage 08f, net
    +281), 10 against 51 with the 113,100-document table (9b/9d, net +41). A
    bare-plurality rule would change 0 groups either way, and H6 closed on
    option 1: keep the vote as it is.

    ONE LIMIT, STATED HERE BECAUSE IT IS EASY TO FORGET. `categ` in a delivered
    `DOC_LINE_CATEG` CSV is POST-cascade: the vote has already run. A group the
    dedup unified reads as unanimous here, so the contested count is a LOWER
    BOUND on how much the vote actually decided. For the pre-cascade picture,
    re-score with `recategorize_from_csv.py --no-postprocessing --out DIR` and
    point this flag at `DIR`.
    """
    groups: dict[tuple[str, str], Counter] = {}
    for (file_id, _page, _line), verdict, categ in witnessed_rows:
        groups.setdefault((file_id, verdict["text"]), Counter())[categ] += 1

    shapes: Counter = Counter()
    shape_lines: Counter = Counter()
    sizes: Counter = Counter()
    contested: list[tuple] = []
    for (file_id, text), counts in groups.items():
        n = sum(counts.values())
        shape, winner = _vote(counts)
        shapes[shape] += 1
        shape_lines[shape] += n
        sizes[_band(n)] += n
        if shape != "unanimous":
            mix = "|".join(f"{c}:{k}" for c, k in counts.most_common())
            contested.append((file_id, text, n, shape, mix, winner, counts))

    n_groups = len(groups)
    n_lines = sum(sum(c.values()) for c in groups.values())
    print("\n=== (document, string) groups — what the modal dedup votes on ===")
    print(f"  lines                        {n_lines:8d}")
    print(f"  (document, string) groups    {n_groups:8d}")
    print(f"  {'shape':18} {'groups':>8} {'lines':>9}")
    for shape in ("unanimous", "strict majority", "bare plurality", "tie"):
        if shapes[shape]:
            print(f"  {shape:18} {shapes[shape]:8d} {shape_lines[shape]:9d}")

    print("\n=== group size — where the blast radius actually is ===")
    print(f"  {'lines per group':18} {'lines':>9} {'share':>7}")
    for band in _SIZE_BANDS:
        if sizes[band]:
            print(f"  {band:18} {sizes[band]:9d} {sizes[band] / n_lines if n_lines else 0:6.1%}")

    destroys = [c for c in contested if c[5] == "Trash" and c[6].get("Clear", 0)]
    rescues = [c for c in contested if c[5] != "Trash" and c[6].get("Trash", 0)]
    clear_lost = sum(c[6].get("Clear", 0) for c in destroys)
    trash_saved = sum(c[6].get("Trash", 0) for c in rescues)
    bare = [c for c in destroys if c[3] == "bare plurality"]

    print("\n=== what the vote does to contested groups ===")
    print(f"  vote lands on Trash, carrying Clear down : {len(destroys):6d} group(s), {clear_lost:6d} Clear line(s)")
    print(f"  vote lands off Trash, lifting Trash out  : {len(rescues):6d} group(s), {trash_saved:6d} Trash line(s)")
    print(
        f"  NET                                      : {trash_saved - clear_lost:+6d} line(s) in the cascade's favour"
    )

    print("\n=== H6 option 2: stop a BARE PLURALITY demoting Clear -> Trash ===")
    print(f"  groups that change under that rule       : {len(bare):6d}")
    print(f"  Clear lines it would save                : {sum(c[6].get('Clear', 0) for c in bare):6d}")
    print("  A TIE is already safe and cannot be part of this: apply_document_postprocessing()")
    print("  resolves with `x.mode()[0]`, mode() returns its tied values SORTED, and")
    print("  'Clear' < 'Noisy' < 'Trash' — so a tie never lands on Trash. That is an")
    print("  accident of the alphabet, not a design, and it is load-bearing.")
    print("  Read this against the NET line above before changing production: on the")
    print("  822-document queue the cascade rescued 31 Trash lines to destroy 17 Clear ones.")

    if path is None:
        return
    if _refuses_gold_dir(path, "--by-group"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "text", "lines", "vote_shape", "categ_mix", "dedup_winner", "clear_at_risk"])
        for file_id, text, n, shape, mix, winner, counts in sorted(contested, key=lambda r: (-r[2], r[0])):
            writer.writerow([file_id, text, n, shape, mix, winner, counts.get("Clear", 0) if winner == "Trash" else 0])
    print(f"\nwrote {len(contested)} contested group(s) to {path}")
    print("  Carries line text — keep it on the cluster, like the annotation queues.")


#: Size bands for the blast-radius table. Open-ended at the top because one
#: string (`ppole`) held 11,562 lines of the 822-document queue on its own.
_SIZE_BANDS = ("1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+")


def _band(n: int) -> str:
    for edge, label in ((1, "1"), (2, "2"), (4, "3-4"), (9, "5-9"), (19, "10-19"), (49, "20-49"), (99, "50-99")):
        if n <= edge:
            return label
    return "100+"


def _read_filled_distinct(paths: list[Path]):
    """Read one or more FILLED queues into a lookup, skipping blanks.

    Delegates to ``tools/project_annotation.py`` so there is one reader of a
    returned ask, not two. That reader also takes the census/sample shape
    (``variants`` joined by ``" | "``, family rows) and validates the labels --
    `trash` and `Trash` are the same answer, a typo stops the run. (#30, the
    2026-10-01 return used lower-case labels throughout.)
    """
    from tools.project_annotation import read_decisions

    for path in paths:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        missing = [c for c in ("text", "gold_categ") if c not in fields]
        if missing:
            raise ValueError(
                f"{path} has no {', '.join(missing)} column, so it is not a filled --distinct "
                "file. Generate one with --distinct, fill gold_categ, then pass it here."
            )
    return read_decisions(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report which lines _has_shape_garbage_evidence() would reach, against the category "
            "the pipeline currently assigns. Does not re-score and does not reconstruct signals."
        )
    )
    parser.add_argument("path", nargs="?", help="DOC_LINE_CATEG CSV file or directory of them")
    parser.add_argument(
        "--input-dir",
        action="append",
        metavar="DIR",
        help=(
            "A directory of DOC_LINE_CATEG CSVs. REPEATABLE: give it once per archive to read "
            "both collections in one invocation. That is the only safe way to say 'all of the "
            "documents' -- the two archives have no common parent that holds nothing else, and "
            "a staging directory of symlinks does NOT work, because pathlib's `**` glob does not "
            "follow directory symlinks and would return zero rows without failing."
        ),
    )
    parser.add_argument("--lines", help="Plain text file, one candidate line per row (no CSV schema).")
    parser.add_argument("--out", help="Write the witnessed candidate lines to this CSV, for annotation.")
    parser.add_argument(
        "--distinct",
        metavar="PATH",
        help=(
            "Write ONE ROW PER DISTINCT STRING instead of per line, with an occurrence count. "
            "The queue is heavily repeated -- on the 822-document corpus 20,324 lines are only "
            "5,243 strings, and the single string `ppole` is 57%% of them -- so annotating by "
            "string is a tenth of the work for the same coverage."
        ),
    )
    parser.add_argument(
        "--from-distinct",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Read a FILLED --distinct file and project its gold_categ onto every line carrying "
            "that string, writing the result to --out as a joinable sidecar. This is the step "
            "that turns string-level decisions back into (file, page_num, line_num) rows. "
            "Repeatable, and it also reads a returned census.csv / sample.csv (variants and "
            "spelling families). It only labels lines the witness fires on NOW; for a sidecar "
            "that survives a predicate change use `tools/project_annotation.py join`."
        ),
    )
    parser.add_argument(
        "--by-group",
        metavar="PATH",
        nargs="?",
        const="",
        help=(
            "Report the modal dedup's blast radius in (document, string) groups -- the unit "
            "apply_document_postprocessing() actually votes in -- and write the contested groups "
            "to PATH. Give the flag with no PATH for the summary only. This is the corpus-wide "
            "denominator issue #30's dedup decision (H6) was taken on: over both archives the "
            "vote rescues more Trash lines than it destroys Clear ones (stages 08f and 9d)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Recurse into sub-directories. The default `*.csv` glob matches nothing at all over a "
            "two-archive parent (ARUP/ and ARUB/), which is how 'all of the collections' is spelled "
            "on the cluster -- the same gap recategorize_from_csv.py --recursive exists to close."
        ),
    )
    parser.add_argument("--examples", type=int, default=0, metavar="N", help="Print up to N examples per clause.")
    parser.add_argument(
        "--all-lengths",
        action="store_true",
        help="Report every line, not only those meeting the route's text-only entry condition.",
    )
    args = parser.parse_args(argv)

    if args.by_group is not None and args.lines:
        parser.error("--by-group needs document locators; it cannot run on --lines input")

    if args.lines:
        source = _iter_plain_lines(Path(args.lines))
    else:
        targets = list(args.input_dir or ([args.path] if args.path else []))
        if not targets:
            parser.error("give a CSV path, --input-dir, or --lines")
        csvs = []
        for target in targets:
            found = _collect_csvs(Path(target), recursive=args.recursive)
            # Per-root, so a typo in ONE of two archives is a named failure and
            # not a halved corpus reported as a result. Every corpus-scale figure
            # this tool produces is only as complete as this list.
            print(f"  corpus: {target} -> {len(found)} document CSV(s)", file=sys.stderr)
            if not found:
                hint = "" if args.recursive else " (a nested layout needs --recursive)"
                parser.error(f"no CSV files found under {target}{hint}")
            csvs.extend(found)
        if len({p.resolve() for p in csvs}) != len(csvs):
            parser.error("the same document CSV was reached through more than one --input-dir")
        source = _iter_csv_rows(csvs)

    total = 0
    eligible = 0
    witnessed_rows: list[tuple[tuple[str, str, str], dict, str]] = []
    by_categ: Counter = Counter()
    witnessed_by_categ: Counter = Counter()
    clause_counts: Counter = Counter()
    examples: dict[str, list[str]] = {c: [] for c in _CLAUSE_ORDER}
    # (#30 stage 11b) The language distribution of the witnessed queue. It decides
    # whether the vowel-run language split (D44) is worth anything: 3% non-Czech
    # and it buys nothing, 40% and it is the whole answer. Two counters, because
    # the at-risk half is the only half a label can change.
    lang_witnessed: Counter = Counter()
    lang_at_risk: Counter = Counter()

    for doc, text, wc, categ, lang in source:
        total += 1
        # (#30 D44) The row's raw language, as production passes it; "?" (no
        # `original_lang`, or --lines input) means unknown and gets the general
        # vowel-run threshold, which is what the gate does with an unknown label.
        verdict = classify_line(text, wc, None if lang == "?" else lang)
        in_scope = args.all_lengths or verdict["route_eligible"]
        if not in_scope:
            continue
        eligible += 1
        by_categ[categ] += 1
        if verdict["witness"]:
            witnessed_by_categ[categ] += 1
            witnessed_rows.append((doc, verdict, categ))
            lang_witnessed[lang] += 1
            if categ in ("Clear", "Noisy"):
                lang_at_risk[lang] += 1
            for clause in verdict["clauses"].split(","):
                clause_counts[clause] += 1
                if args.examples and len(examples[clause]) < args.examples:
                    examples[clause].append(text)

    print(f"\nSHORT_GARBAGE_WITNESS_ENABLE = {tu.SHORT_GARBAGE_WITNESS_ENABLE}  (does not affect this report)")
    print(
        f"witness constants: MIN_ALPHA={tu.SHORT_GARBAGE_WITNESS_MIN_ALPHA} "
        f"VARIETY_MIN_ALPHA={tu.SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA} "
        f"VARIETY_MAX={tu.SHORT_GARBAGE_WITNESS_VARIETY_MAX} "
        f"TRIPLE_MAX_ALPHA={tu.SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA} "
        f"VOWEL_RUN_MIN={tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN}"
    )
    # (#30 D44) The language split's two constants reach the log for the reason
    # the D42 note below gives for VOWEL_RUN_MIN: they steer the clause that
    # carries most of the exposure, so a run's own log has to say which values
    # produced it. --lines input has no language at all, and that changes what
    # the numbers mean, so it is said on the same line.
    exempt_langs = ",".join(sorted(tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS))
    print(
        f"vowel-run language split: VOWEL_RUN_EXEMPT_LANGS={exempt_langs or '(empty: one global threshold)'} "
        f"VOWEL_RUN_MIN_EXEMPT={tu.SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT}  "
        + (
            "(--lines input has no language: every line gets VOWEL_RUN_MIN)"
            if args.lines
            else "(language from each row's raw `original_lang`; a row without one gets VOWEL_RUN_MIN)"
        )
    )
    # (#30 D42) The lexicon line prints WHETHER OR NOT a table is configured, and
    # VOWEL_RUN_MIN joins the constants line above. Both are here for one reason.
    #
    # 08f and 08g were run with no lexicon configured, and that could not be read
    # off the log they produced -- it had to be INFERRED from which warning did
    # and did not appear, which is how the defect survived a delivery and a
    # 592-decision annotation ask built on top of it (digest T1). VOWEL_RUN_MIN
    # is the same shape waiting to happen: it is the clause that accounts for
    # two thirds of the witness's exposure, it is separately tunable, and until
    # now no artefact this tool wrote said which value produced it.
    #
    # "no table configured" is the line whose ABSENCE was the problem, so it is
    # printed as loudly as its presence.
    lexicon_path = tu.SHORT_GARBAGE_LEXICON_PATH
    if lexicon_path:
        documents = tu.lexicon_document_count(lexicon_path)
        built_over = f"{documents:,} documents" if documents else "document count not recorded in the table"
        print(f"vocabulary lexicon: {lexicon_path}  ({built_over})")
    else:
        print(
            "vocabulary lexicon: NONE CONFIGURED -- _has_vocabulary_support() is inert, so this "
            "report measures a population the witness never ships against"
        )
    scope = (
        "all lines"
        if args.all_lengths
        else f"word_count <= {tu.ISOLATED_CHAR_MIN_TOKENS}, no diacritics/structure/notation"
    )
    print(f"\n{total} lines read; {eligible} in scope ({scope})")

    print("\n=== stored category x witnessed ===")
    print(f"  {'category':10} {'in scope':>9} {'witnessed':>10} {'share':>7}")
    for categ in sorted(by_categ):
        n = by_categ[categ]
        w = witnessed_by_categ[categ]
        print(f"  {categ:10} {n:9d} {w:10d} {(w / n if n else 0):6.1%}")
    n_tot = sum(by_categ.values())
    w_tot = sum(witnessed_by_categ.values())
    print(f"  {'TOTAL':10} {n_tot:9d} {w_tot:10d} {(w_tot / n_tot if n_tot else 0):6.1%}")

    if clause_counts:
        print("\n=== which clause fired (a line may fire several) ===")
        for clause in _CLAUSE_ORDER:
            if clause_counts[clause]:
                print(f"  {clause:18} {clause_counts[clause]:8d}")

    if lang_witnessed:
        # (#30 stage 11b) The measurement that sizes D44, the vowel-run language
        # split. @david-spacil: three vowels in a row is a fact about Czech
        # phonotactics, so the clause is a category error in German and French
        # rather than a badly chosen threshold. This says how much that is worth.
        #
        # THE COLUMN READ IS `original_lang`, NOT `lang`. The stored `lang` has
        # been through `remap_lang()`, which rewrites any base outside
        # EXPECTED_LANGS + TRUSTED_FOREIGN_LANGS to Czech -- so reading it would
        # answer "how much of this queue is non-Czech" with a number the remap
        # partly decided, and in the direction that understates the answer.
        print("\n=== detected language of the witnessed queue (raw `original_lang`) ===")
        print("  Sizes the vowel-run language split (#30 D44). The at-risk column is the")
        print("  half a label can still change; the other half is already Trash.")
        print(f"  {'lang':12} {'witnessed':>10} {'share':>7} {'at risk':>9} {'share':>7}")
        w_all = sum(lang_witnessed.values())
        r_all = sum(lang_at_risk.values())
        for lang, n in lang_witnessed.most_common():
            at_risk = lang_at_risk.get(lang, 0)
            print(
                f"  {lang:12} {n:10d} {n / w_all if w_all else 0:7.1%} "
                f"{at_risk:9d} {at_risk / r_all if r_all else 0:7.1%}"
            )
        non_czech = sum(v for k, v in lang_at_risk.items() if not k.startswith("ces"))
        print(
            f"  at-risk lines NOT detected as Czech: {non_czech} of {r_all} ({non_czech / r_all if r_all else 0:.1%})"
        )
        print("  Read against the decision rule: a few per cent and the split buys almost")
        print("  nothing; tens of per cent and it is the whole answer.")

    # The two numbers the flag decision rests on. Neither is an error rate:
    # `categ` is the pipeline's own answer, so these are exposure counts that
    # tell you how many lines an annotator has to look at, and where.
    retained = witnessed_by_categ.get("Trash", 0)
    exposure = witnessed_by_categ.get("Clear", 0) + witnessed_by_categ.get("Noisy", 0)
    print("\n=== exposure (NOT an error rate — `categ` is the pipeline's answer, not gold) ===")
    print(f"  witnessed and currently Trash        : {retained:8d}   would stay Trash with the witness armed")
    print(f"  witnessed and currently Clear/Noisy  : {exposure:8d}   false-positive candidates — annotate these first")

    # The line counts above are the wrong unit for a decision and always have
    # been, so the distinct count is printed here rather than only under
    # --distinct. On the 822-document corpus the two differ by ~4x overall and by
    # 20x on the worst single string: 20,324 witnessed lines are 5,243 strings,
    # and `ppole` alone is 11,562 lines but one decision. Every ratio anyone has
    # quoted from this block -- the "1:3 against" that argued the vocabulary veto
    # was mandatory included -- was a line ratio, and `ppole` was 76.8% of the
    # false-positive candidates in it. Archival tables repeat one string thousands
    # of times; the decision surface is the strings.
    distinct_texts = {v["text"] for _, v, _ in witnessed_rows}
    distinct_fp = {v["text"] for _, v, c in witnessed_rows if c in ("Clear", "Noisy")}
    print(
        f"  distinct strings                     : {len(distinct_texts):8d}   "
        f"({len(distinct_fp)} of them currently Clear/Noisy)"
    )
    if witnessed_rows and distinct_texts:
        ratio = len(witnessed_rows) / len(distinct_texts)
        print(f"  lines per distinct string            : {ratio:8.1f}   read the string counts, not the line counts")
        top_text, top_n = Counter(v["text"] for _, v, _ in witnessed_rows).most_common(1)[0]
        print(f"  most repeated string                 : {top_n:8d}   {top_text!r} — one decision, not {top_n}")
    print("  Use --distinct to get the annotation queue in that unit.")

    if args.examples:
        print("\n=== examples ===")
        for clause in _CLAUSE_ORDER:
            if examples[clause]:
                print(f"  [{clause}]")
                for text in examples[clause]:
                    print(f"    {text!r}")

    if args.by_group is not None:
        _write_by_group(Path(args.by_group) if args.by_group else None, witnessed_rows)

    if args.distinct:
        distinct_path = Path(args.distinct)
        if _refuses_gold_dir(distinct_path, "--distinct"):
            return 2
        _write_distinct(distinct_path, witnessed_rows)

    if args.out:
        out_path = Path(args.out)
        if _refuses_gold_dir(out_path, "--out"):
            return 2
        projected = None
        if args.from_distinct:
            try:
                projected = _read_filled_distinct([Path(p) for p in args.from_distinct])
            except (OSError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            covered = sum(1 for _, v, _ in witnessed_rows if projected.lookup(v["text"]) is not None)
            print(
                f"\nprojecting {len(projected.labelled)} annotated row(s) from {', '.join(args.from_distinct)} "
                f"onto {covered} of {len(witnessed_rows)} lines "
                f"({covered / len(witnessed_rows) if witnessed_rows else 0:.1%} covered)"
            )

        def _label(text: str) -> str:
            hit = projected.lookup(text) if projected is not None else None
            return hit.label if hit is not None else ""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            # (file, page_num, line_num) FIRST, and named exactly as
            # GOLD_SIDECAR_KEYS names them, so an annotated file is a usable gold
            # sidecar with no reshaping.
            #
            # This used to emit a single `document` column holding `path.name`
            # -- the filename, extension included -- and no locators at all, while
            # the closing message promised the annotated result was "a gold set
            # gold_gate() can consume". It was not: `--gold-sidecar` joins on
            # (file, page_num, line_num) and refuses a frame without them, and
            # dropping the file into tools/gold/ breaks the per-document
            # invariant that directory is checked for. The annotation queue was a
            # dead end in both directions, which is the worst possible defect in
            # a file whose entire purpose is to be filled in by hand.
            writer.writerow(
                ["file", "page_num", "line_num", "text", "word_count", "categ_current", "clauses", "gold_categ"]
            )
            for (file_id, page_num, line_num), verdict, categ in witnessed_rows:
                writer.writerow(
                    [
                        file_id,
                        page_num,
                        line_num,
                        verdict["text"],
                        verdict["word_count"],
                        categ,
                        verdict["clauses"],
                        _label(verdict["text"]),
                    ]
                )
        print(f"\nwrote {len(witnessed_rows)} candidate lines to {out_path}")
        print("  `gold_categ` is left blank on purpose: fill it blind, then join it back with")
        print(f"    --gold-sidecar {out_path} --gold-column gold_categ")
        print("  Keep it OUT of tools/gold/ -- that directory is one CSV per document and this")
        print("  file spans many. Sidecars live in tools/gold/sidecars/.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
