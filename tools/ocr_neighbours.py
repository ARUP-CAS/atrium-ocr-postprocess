#!/usr/bin/env python3
"""
tools/ocr_neighbours.py
=======================
Is this token garbage, or a damaged rendering of a real word?

Why this exists
---------------
Issue #30 is one question wearing several hats: a 1-3 token line with no
diacritics is either archaeological shorthand, a Latin binomial, an excavation
code, or OCR noise, and the pipeline has to choose. Every mechanism tried so far
answers a *different* question and is then asked to stand in for this one.

* **Shape rules** (`_has_shape_garbage_evidence`) ask "does this look like noise?"
  They convict `sektlll` correctly and `Poaceae`, `Triticum` and the abbreviation
  `ppole` wrongly,
  and no amount of narrowing fixes the second group -- deciding it by shape means
  deciding which vowel sequences a language is allowed to contain.
* **Attestation** (`_has_vocabulary_support`) asks "has this exact string been seen
  elsewhere?" It rescues the binomials. But a *damaged* word is by definition
  unattested in its damaged form, so attestation is silent on exactly the
  population that matters, and worse, it is fooled by a misread that repeats --
  a misread that repeats on a pre-printed form accrues df exactly like a word.

Neither can reach the question this module asks. `1 fraament okraie` is **530 lines,
all currently `Trash`**, and it is "1 fragment okraje" -- an ordinary count of rim
sherds under heavy OCR damage. Judged by shape it is noise. Judged by attestation it
is unattested. Judged by *distance to something attested* it is obviously real text,
and that is a question with an answer.

The two mechanisms, and why both
--------------------------------
**Generic edit distance 1**, via a symmetric-delete index. Catches single
substitutions, insertions and deletions: `fraament -> fragment`, `oobjekt -> objekt`.
Cheap and unbiased -- it knows nothing about what OCR does, which is its virtue
and its limit.

**Directed OCR-confusion expansion.** Applies known scan confusions and keeps the
results that are attested. This earns its place on **diacritic loss**, which edit
distance cannot reach: a Czech word stripped of three diacritics sits at edit
distance 3 from its real form, invisible to any ED1 index, and it is the single
most common transformation in this corpus. `zakladni` -> `základní` is three edits
and one obvious rule.

What this module must NOT be read as saying
-------------------------------------------
**Recoverability does not decide `Trash` vs `Noisy`.** `oobjekt` is maximally
recoverable -- it *is* `objekt` -- and the shape clauses convict it. Whether a
corrupted-but-readable rendering belongs in `Trash` (it is not usable text) or
`Noisy` (it is text, damaged) is a **policy question for the archive**, not a fact
this or any other tool can settle. This module reports the evidence and stops.

**A suggestion is not a correction.** The same index proposes `Linum -> ilium` and
`Lepus -> lupus`, where both forms are real Latin and neither is an OCR error. Any
consumer -- human or code -- must treat the output as evidence to weigh. That is
why `nearest_attested` returns the *mechanism* alongside the candidate: "one
substitution" and "diacritics restored" are different strengths of claim.

**THE SIGNAL IS ASYMMETRIC, and this is the caveat most likely to cause a wrong
call.** A high score is strong evidence: producing it takes a specific attested
target, and coincidences at that specificity are rare. A ZERO score is weak
evidence of anything, because "not attested and not near anything attested" is
precisely what a rare true word looks like. Measured on the full-collection table
(113,100 documents), the zero-recoverability rows include `sektlll` and `IOIAL`,
which are noise -- and `Kaukasus`, `Feuersteinspan` and
`Schuhleistenkeilbruchstueck`, which are correct German archaeological terms too
rare for any corpus-derived lexicon to hold. No threshold separates those two
groups, because the thing that separates them is knowledge this corpus does not
contain.

So: *recoverable* means "probably real text". *Not recoverable* means "this tool
has nothing to say", and a consumer that reads it as "garbage" has invented a
verdict the data does not support.

**Nothing here is wired into the categoriser.** This issue has three
instrument-level errors on record -- a self-referential sweep objective, an
exposure ratio scored against `categ`, and an A/B verdict ranked by a metric that
preferred the worst arm -- and each was a measurement trusted before it was
checked. Recoverability becomes a signal in the commit that has gold for it, not
before. `--report` reads `categ` only to *display* a distribution; nothing here
fits anything to it.

Usage
-----
    # What does the machinery say about these tokens?
    python tools/ocr_neighbours.py --lexicon tools/gold/token_df.tsv \\
        --lookup oobjekt fraament okraie oueussd Triticum

    # Add evidence columns to an annotation queue
    python tools/ocr_neighbours.py --lexicon tools/gold/token_df.tsv \\
        --annotate issue30_out/04_witness_distinct.csv --out with_evidence.csv

    # Recoverability distribution per stored category (diagnostic only)
    python tools/ocr_neighbours.py --lexicon tools/gold/token_df.tsv \\
        --report data_samples/DOC_LINE_CATEG

Exit codes
----------
  0  Completed.
  2  Bad arguments, missing path, or an unreadable lexicon.
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TOOL_VERSION = "1.0"

#: A token needs this many documents before it can be a correction TARGET. Higher
#: than SHORT_GARBAGE_LEXICON_MIN_DF on purpose: a df-3 token is attested enough to
#: escape conviction itself, but far too thin to be evidence that some *other*
#: string is a damaged copy of it. Suggesting `edelite -> edelita (3)` would
#: manufacture exactly the false confidence this module exists to avoid.
DEFAULT_STRONG_DF = 10

#: Below this many alphabetic characters a token is not queried at all. Short
#: strings have too many neighbours for a neighbour to mean anything: at three
#: letters most of the lexicon is within one edit.
DEFAULT_MIN_ALPHA = 4

#: Single-character OCR confusions, as `wrong -> right`. Sourced from the failure
#: modes visible in this corpus rather than from a generic list.
#:
#: The diacritic block is the load-bearing one: Czech scanned as ASCII loses every
#: accent at once, which no edit-distance index can follow.
_DIACRITIC_RESTORE: dict[str, tuple[str, ...]] = {
    "a": ("á", "ä"),
    "c": ("č",),
    "d": ("ď",),
    "e": ("é", "ě"),
    "i": ("í",),
    "n": ("ň",),
    "o": ("ó", "ö"),
    "r": ("ř",),
    "s": ("š",),
    "t": ("ť",),
    "u": ("ú", "ů", "ü"),
    "y": ("ý",),
    "z": ("ž",),
}

#: Glyph confusions: pairs a scanner conflates because they look alike in print or
#: typescript. Applied in both directions.
_GLYPH_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("i", "j"),  # okraje -> okraie, the measured case
    ("i", "l"),
    ("i", "1"),
    ("l", "1"),
    ("o", "0"),
    ("s", "5"),
    ("b", "6"),
    ("g", "9"),
    ("b", "8"),
    ("z", "2"),
    ("u", "v"),
    ("c", "e"),
    ("f", "t"),
    ("h", "b"),
    ("m", "n"),
)

#: Multi-character confusions: a ligature or letter pair read as one glyph, or the
#: reverse. These are why a plain character-substitution model is not enough.
_SHAPE_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("rn", "m"),
    ("cl", "d"),
    ("vv", "w"),
    ("ii", "u"),
    ("li", "h"),
    ("nn", "m"),
    ("ri", "n"),
    ("lt", "h"),
    ("ss", "ß"),
)


def strip_diacritics(text: str) -> str:
    """Fold to ASCII-ish by dropping combining marks. `základní` -> `zakladni`."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


class OCRNeighbours:
    """A lexicon plus the two lookup mechanisms over it.

    Build once, query many times. The symmetric-delete index costs roughly one
    list entry per (strong token x its length), which is a few million small
    entries on a full-collection table -- seconds to build, and the queries are
    then dictionary hits rather than scans.
    """

    def __init__(self, lexicon: "dict[str, int] | None" = None, strong_df: int = DEFAULT_STRONG_DF):
        self.lexicon: dict[str, int] = dict(lexicon or {})
        self.strong_df = strong_df
        self.strong: dict[str, int] = {t: d for t, d in self.lexicon.items() if d >= strong_df}
        # Fold -> the accented forms that fold to it. This is what makes diacritic
        # restoration a dictionary hit instead of a search.
        self._by_fold: dict[str, list[str]] = defaultdict(list)
        for token in self.strong:
            folded = strip_diacritics(token)
            if folded != token:
                self._by_fold[folded].append(token)
        # Symmetric-delete index for edit distance 1.
        self._deletes: dict[str, list[str]] = defaultdict(list)
        for token in self.strong:
            self._deletes[token].append(token)
            for i in range(len(token)):
                self._deletes[token[:i] + token[i + 1 :]].append(token)

    # -- mechanism (a): generic edit distance 1 --------------------------------

    def _ed1_candidates(self, token: str) -> set[str]:
        out = set(self._deletes.get(token, ()))
        for i in range(len(token)):
            out.update(self._deletes.get(token[:i] + token[i + 1 :], ()))
        out.discard(token)
        return out

    # -- mechanism (b): directed OCR-confusion expansion ------------------------

    def _confusion_candidates(self, token: str) -> set[str]:
        """Attested forms reachable by applying known scan confusions.

        Diacritic restoration is handled wholesale rather than one accent at a
        time: a scan that loses accents loses all of them, so the useful question
        is "which attested tokens fold to this string?", not "which single accent
        would fix it?".
        """
        out: set[str] = set()
        # All-at-once diacritic restoration, any distance.
        out.update(self._by_fold.get(token, ()))
        # Single glyph / shape confusions, applied one at a time.
        for wrong, right in _GLYPH_CONFUSIONS + _SHAPE_CONFUSIONS:
            for a, b in ((wrong, right), (right, wrong)):
                start = 0
                while (idx := token.find(a, start)) != -1:
                    cand = token[:idx] + b + token[idx + len(a) :]
                    if cand in self.strong:
                        out.add(cand)
                    start = idx + 1
        # A doubled initial; de-geminating is one rule. NOTE this also reaches
        # abbreviations -- see `stronger_twin` for why that is a real limitation.
        if len(token) >= 2 and token[0] == token[1] and token[1:] in self.strong:
            out.add(token[1:])
        out.discard(token)
        return out

    # -- public API ------------------------------------------------------------

    def nearest_attested(self, token: str, limit: int = 3) -> list[tuple[str, int, str]]:
        """``[(candidate, document_frequency, mechanism)]``, strongest first.

        ``mechanism`` is part of the answer, not decoration. "diacritics" is a
        near-certain correction; "edit1" may be coincidence -- `Linum` and `ilium`
        are one edit apart and both real Latin. A caller that collapses the two
        has thrown away the only thing that separates evidence from noise.
        """
        token = token.lower()
        if not self.strong or sum(c.isalpha() for c in token) < DEFAULT_MIN_ALPHA:
            return []
        if token in self.lexicon and self.lexicon[token] >= self.strong_df:
            return []  # attested in its own right; nothing to recover

        found: dict[str, str] = {}
        for cand in self._confusion_candidates(token):
            found[cand] = "diacritics" if cand in self._by_fold.get(token, ()) else "confusion"
        for cand in self._ed1_candidates(token):
            found.setdefault(cand, "edit1")

        ranked = sorted(found.items(), key=lambda kv: (-self.strong[kv[0]], kv[0]))
        return [(c, self.strong[c], m) for c, m in ranked[:limit]]

    def stronger_twin(self, token: str, ratio: float = 4.0) -> "tuple[str, int, str] | None":
        """An ATTESTED token's own much-commoner source form, if it has one.

        The complement of `nearest_attested`, which stays silent on attested
        tokens because they need no recovery. That silence hides a real class:
        `oobjekt` is attested at df 3 — not because it is a word, but because the
        same misread recurred — while `objekt` sits at 378. Attestation cannot
        see that; the ratio can.

        **A twin is NOT proof of an artefact, and this is a measured limitation,
        not a hypothetical one.** `ppole` has a twin (`pole`, 163 against 35) and
        is *not* an artefact: it is the standard Czech abbreviation for
        *popelnicová pole*, urnfield culture — `pp` doubled for the plural, the
        same convention as `pp.` for pages (@david-spacil, 2026-09-19). An
        abbreviation is DERIVED from its base word, so its base is always the
        commoner of the two and the ratio test always fires. Nothing here
        distinguishes that from a scanning error.

        Absolute frequency looked like it separated them on the 822-document
        table: `ppole` at df 35 against `oobjekt` 3, `ssutě` 4, `jjámy` 5,
        `ssutí` 5, `vvkop` 5, `ssuti` 8. It does not hold. At 113,100 documents
        the margin collapses to 1.17x, and three of those six -- `ssuti`,
        `ssutí`, `ssutě` -- turned out to be an old spelling of `suť` rather than
        damage at all (@david-spacil, 2026-09-22), so the gap has real language
        on both sides of it. `text_util` used to apply such a cap; it does not
        any more (#30 D40). **No production predicate turns this evidence into a
        verdict**, and on the measurements so far none should.

        Returns the strongest twin at or above ``ratio`` times this token's own
        frequency, or ``None``. Evidence for a reader, not a verdict.
        """
        token = token.lower()
        own = self.lexicon.get(token, 0)
        if own <= 0:
            return None
        best: tuple[str, int, str] | None = None
        for cand in self._confusion_candidates(token):
            df = self.strong.get(cand, 0)
            if df >= ratio * own and (best is None or df > best[1]):
                mech = "diacritics" if cand in self._by_fold.get(token, ()) else "stronger-twin"
                best = (cand, df, mech)
        return best

    def token_status(self, token: str) -> str:
        """``attested`` / ``recoverable`` / ``unattested`` for one token."""
        token = token.lower()
        if self.lexicon.get(token, 0) >= self.strong_df:
            return "attested"
        return "recoverable" if self.nearest_attested(token, limit=1) else "unattested"

    def recoverability(self, text: str) -> tuple[float, list[tuple[str, str]]]:
        """``(score, [(token, status)])`` for one line.

        The score is the share of queryable tokens that are either attested or one
        known transformation away from something attested. It separates *damaged
        real text* from *no word at all* — `1 fraament okraie` scores 1.0 and
        `oueussd` scores 0.0 — which is the distinction neither shape nor
        attestation can draw.

        It does **not** say which category the line belongs in. See the module
        docstring: a recoverable line may still be unusable, and that call is the
        archive's.

        **A zero is not a verdict.** It arises two different ways, and neither is
        "this is garbage": the line had nothing queryable (pure notation, digits,
        strings under the alpha minimum), or its tokens simply have no attested
        target -- which is equally true of `Schuhleistenkeilbruchstueck`. The
        empty-detail case is distinguishable here (the list is empty) and
        `recoverability_note` reports it downstream; the second is not
        distinguishable at all, by design, because the data does not contain the
        distinction.
        """
        detail: list[tuple[str, str]] = []
        for word in text.split():
            for sub in tu._witness_subtokens(word):
                core = sub.strip(tu._STRIP_CHARS).lower()
                if core and sum(c.isalpha() for c in core) >= DEFAULT_MIN_ALPHA:
                    detail.append((core, self.token_status(core)))
        if not detail:
            return 0.0, []
        good = sum(1 for _, s in detail if s in ("attested", "recoverable"))
        return good / len(detail), detail


def load_lexicon(path: Path) -> tuple[dict[str, int], int]:
    """Read a table into ``({token: total_df}, documents)``.

    Field 1 after the token is the total, exactly as
    ``text_util._read_token_lexicon`` reads it, so a multi-collection table with
    per-collection columns loads here identically.

    ``documents`` comes from the builder's provenance header and is 0 when absent.
    It exists so `--strong-df` can be judged as a SHARE of the corpus rather than
    an absolute: 10 documents is 1.2% of an 822-document build and 0.009% of a
    113,100-document one, and those are very different claims about a token.
    """
    out: dict[str, int] = {}
    documents = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if stripped.startswith("# documents:"):
                head = stripped.split(":", 1)[1].split()
                if head and head[0].isdigit():
                    documents = int(head[0])
                continue
            token, sep, rest = stripped.partition("\t")
            if not sep or (not token and stripped.startswith("#")) or not token:
                continue
            try:
                out[token] = int(rest.partition("\t")[0])
            except ValueError:
                continue
    return out, documents


def strong_df_advice(strong_df: int, documents: int, strong_tokens: int) -> "str | None":
    """Warn when `--strong-df` has quietly stopped meaning much. None when it is fine.

    A warning and not an automatic adjustment: silently moving a threshold between
    runs makes two reports incomparable, and this issue has three instrument
    failures on record that all began with a number meaning something different
    from what its reader assumed.
    """
    if documents <= 0 or strong_df <= 0:
        return None
    share = strong_df / documents
    if share >= 0.001:
        return None
    suggested = max(strong_df, round(documents * 0.001))
    return (
        f"note: --strong-df {strong_df} is {share:.4%} of {documents:,} documents, which admits "
        f"{strong_tokens:,} correction targets.\n"
        f"      On a corpus this size a thin target is weak evidence that some OTHER string is a\n"
        f"      damaged copy of it — the `Linum`/`ilium` failure, scaled. Consider --strong-df "
        f"{suggested:,}\n"
        f"      (0.1% of documents). Not changed automatically: two runs at different thresholds\n"
        f"      are not comparable, and that has to be a decision rather than a default."
    )


def _format_suggestions(items: list[tuple[str, int, str]]) -> str:
    return "; ".join(f"{c} ({d}, {m})" for c, d, m in items)


def cmd_lookup(nb: OCRNeighbours, tokens: list[str]) -> int:
    print(f"\n  {'token':<24} {'status':<12} nearest attested")
    print(f"  {'-' * 24} {'-' * 12} {'-' * 44}")
    for raw in tokens:
        for word in raw.split():
            for sub in tu._witness_subtokens(word):
                core = sub.strip(tu._STRIP_CHARS).lower()
                if not core:
                    continue
                status = nb.token_status(core)
                sugg = _format_suggestions(nb.nearest_attested(core))
                twin = nb.stronger_twin(core)
                if twin and not sugg:
                    own = nb.lexicon.get(core, 0)
                    sugg = f"attested {own}x, but {twin[0]} is {twin[1]}x ({twin[1] / own:.1f}x commoner, {twin[2]})"
                print(f"  {core:<24} {status:<12} {sugg}")
    print(
        "\n  `diacritics` is a near-certain correction; `edit1` may be coincidence —\n"
        "  `Linum`/`ilium` and `Lepus`/`lupus` are one edit apart and both real Latin.\n"
        "  An `attested Nx, but ...` line can mean the token is in the lexicon only because\n"
        "  a misread repeated across documents. But an ABBREVIATION or an old spelling also\n"
        "  has a commoner base form — `ppole` (popelnicová pole) against `pole` — so a twin\n"
        "  is evidence, not a verdict, and nothing in the frequencies separates them.\n"
        "\n"
        "  THE SIGNAL IS ASYMMETRIC. `recoverable` is strong evidence of real text.\n"
        "  `unattested` is NOT evidence of garbage — it is also what a rare true word\n"
        "  looks like: `Kaukasus` and `Schuhleistenkeilbruchstueck` score zero and are\n"
        "  correct German archaeological terms. Nothing here separates those two cases.\n"
        "\n"
        "  Recoverable does NOT mean Noisy rather than Trash; that call is the archive's."
    )
    return 0


def cmd_annotate(nb: OCRNeighbours, in_path: Path, out_path: Path, text_column: str) -> int:
    with open(in_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            print(
                f"error: {in_path} has no {text_column!r} column (found: {reader.fieldnames})",
                file=sys.stderr,
            )
            return 2
        rows = list(reader)
        fields = list(reader.fieldnames)

    added = ["nearest_attested", "recoverability", "recoverability_note", "token_status"]
    for name in added:
        if name not in fields:
            fields.append(name)

    for row in rows:
        text = str(row.get(text_column) or "")
        score, detail = nb.recoverability(text)
        sugg: list[str] = []
        for core, status in detail:
            if status == "recoverable":
                best = nb.nearest_attested(core, limit=1)
                if best:
                    sugg.append(f"{core} -> {best[0][0]} ({best[0][1]}, {best[0][2]})")
            elif status == "attested":
                # Attested, but with a much commoner base form. Could be a repeated
                # misread OR an abbreviation; the reader decides, not this column.
                twin = nb.stronger_twin(core)
                if twin:
                    own = nb.lexicon.get(core, 0)
                    sugg.append(f"{core} attested {own}x but {twin[0]} is {twin[1]}x ({twin[2]})")
        row["nearest_attested"] = "; ".join(sugg)
        row["recoverability"] = f"{score:.2f}" if detail else ""
        # Why the number is what it is. A bare 0.00 reads as a verdict, and it is
        # not one: `Schuhleistenkeilbruchstueck` scores 0.00 and is a real word.
        if not detail:
            row["recoverability_note"] = "nothing queryable — no opinion"
        elif score == 0.0:
            row["recoverability_note"] = "no attested target — could be noise OR rare real vocabulary"
        elif score >= 0.99:
            row["recoverability_note"] = "every token attested or one step from attested"
        else:
            row["recoverability_note"] = "mixed — read the per-token status"
        row["token_status"] = " ".join(f"{t}:{s}" for t, s in detail)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    scored = sum(1 for r in rows if r["recoverability"])
    helped = sum(1 for r in rows if r["nearest_attested"])
    print(f"\nwrote {len(rows):,} rows to {out_path}")
    print(f"  {scored:,} carried a queryable token; {helped:,} got at least one suggestion")
    print("  `recoverability` is blank where no token was long enough to query — that is")
    print("  'no opinion', not evidence of garbage.")
    return 0


def cmd_report(nb: OCRNeighbours, corpus: Path, text_column: str, categ_column: str) -> int:
    """Recoverability distribution per stored category. DIAGNOSTIC ONLY.

    This reads `categ` — the pipeline's own answer — and it would be a circular
    objective if anything were fitted to it. Nothing is: the numbers are printed
    for a human to look at, and the module wires into no rule.
    """
    files = sorted(corpus.glob("*.csv")) if corpus.is_dir() else [corpus]
    buckets: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for path in files:
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or text_column not in reader.fieldnames:
                    continue
                for row in reader:
                    categ = str(row.get(categ_column) or "?")
                    score, detail = nb.recoverability(str(row.get(text_column) or ""))
                    if not detail:
                        buckets[categ]["no opinion"] += 1
                    elif score >= 0.99:
                        buckets[categ]["all recoverable"] += 1
                    elif score >= 0.5:
                        buckets[categ]["mostly"] += 1
                    elif score > 0:
                        buckets[categ]["some"] += 1
                    else:
                        buckets[categ]["none"] += 1
                    totals[categ] += 1
        except OSError:
            continue

    order = ["all recoverable", "mostly", "some", "none", "no opinion"]
    print(f"\n=== recoverability by stored `{categ_column}` (DIAGNOSTIC — not an objective) ===")
    print(f"  {'category':<12} {'lines':>9}  " + "".join(f"{k:>17}" for k in order))
    for categ in sorted(totals):
        n = totals[categ]
        cells = "".join(f"{buckets[categ][k] / n:>16.1%} " for k in order)
        print(f"  {categ:<12} {n:>9,}  {cells}")
    print(
        "\n  Read this as a description of the corpus, never as a rule. A line can be\n"
        "  fully recoverable and still not be usable text — see the module docstring."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ocr_neighbours.py",
        description="Decide whether a token is garbage or a damaged rendering of an attested word.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lexicon", required=True, metavar="TSV", help="Table from build_token_lexicon.py.")
    ap.add_argument(
        "--strong-df",
        type=int,
        default=DEFAULT_STRONG_DF,
        metavar="N",
        help=(
            f"A token needs N documents to be a correction TARGET (default {DEFAULT_STRONG_DF}). "
            "Higher than the veto's min_df on purpose: a thin token is attested enough to escape "
            "conviction but too thin to prove something else is a damaged copy of it."
        ),
    )
    ap.add_argument("--text-column", default="text", metavar="COL", help="Column to read (default: text).")
    ap.add_argument("--categ-column", default="categ", metavar="COL", help="Category column for --report.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lookup", nargs="+", metavar="TOKEN", help="Report on these tokens.")
    mode.add_argument("--annotate", metavar="CSV", help="Add evidence columns to this CSV (needs --out).")
    mode.add_argument("--report", metavar="CORPUS", help="Recoverability distribution per stored category.")
    ap.add_argument("--out", metavar="CSV", help="Destination for --annotate.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    lex_path = Path(args.lexicon)
    if not lex_path.exists():
        print(f"error: lexicon not found: {lex_path}", file=sys.stderr)
        return 2
    lexicon, documents = load_lexicon(lex_path)
    if not lexicon:
        print(f"error: {lex_path} yielded no rows — wrong file, or every row malformed.", file=sys.stderr)
        return 2

    nb = OCRNeighbours(lexicon, strong_df=args.strong_df)
    print(
        f"lexicon {len(lexicon):,} tokens; {len(nb.strong):,} strong enough to be a correction "
        f"target (df >= {args.strong_df})",
        file=sys.stderr,
    )
    advice = strong_df_advice(args.strong_df, documents, len(nb.strong))
    if advice:
        print(advice, file=sys.stderr)
    if not nb.strong:
        print(
            f"error: no token reaches --strong-df {args.strong_df}; lower it or build a bigger table.",
            file=sys.stderr,
        )
        return 2

    if args.lookup:
        return cmd_lookup(nb, args.lookup)

    if args.annotate:
        if not args.out:
            print("error: --annotate needs --out", file=sys.stderr)
            return 2
        in_path = Path(args.annotate)
        if not in_path.exists():
            print(f"error: path not found: {in_path}", file=sys.stderr)
            return 2
        return cmd_annotate(nb, in_path, Path(args.out), args.text_column)

    corpus = Path(args.report)
    if not corpus.exists():
        print(f"error: path not found: {corpus}", file=sys.stderr)
        return 2
    return cmd_report(nb, corpus, args.text_column, args.categ_column)


if __name__ == "__main__":
    raise SystemExit(main())
