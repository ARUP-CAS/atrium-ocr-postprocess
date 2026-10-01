#!/usr/bin/env python3
"""
tools/build_token_lexicon.py
============================
Build the corpus's own dictionary: a token -> DOCUMENT-FREQUENCY table for the
``rule_short_garbage`` vocabulary signal (issue #30, D14).

Why this exists
---------------
Issue #30 ends in the same place from both sides. ``_has_strong_garbage_evidence()``
is ``False`` on the entire short diacritic-free population, so the merged gate
suspended the rule there rather than narrowing it; ``_has_shape_garbage_evidence()``
narrows it back, but only for the half that is separable BY SHAPE. The residue --
``edelite``, ``vfetennl k.`` -- is spelled exactly the way a word is spelled. Both
the issue thread and the in-tree notes conclude the same thing: separating it
needs a lexicon, not another character test.

The obvious lexicons are not usable here. The Czech ones (korektor and friends)
are CC BY-NC-SA, and this pipeline's output would inherit the clause. A Latin
binomial list would cover ``Equus caballus`` and miss ``Reg.Bez.Aussig.``; a German
one the reverse. And every one of them would be wrong in the same way about the
same thing -- none contains the site codes, context labels and excavation
shorthand that make up most of this corpus's short lines.

The corpus is its own best dictionary, and it is already on disk:

    A token that appears in many DISTINCT DOCUMENTS is vocabulary.
    A token that appears in one is what that scan did to some ink.

DOCUMENT frequency, not line frequency: OCR noise repeats freely inside the scan
that produced it -- the same misread header on forty pages -- and almost never
across scans produced years apart by different operators. Counting lines would
let one badly-scanned document vote its own garbage into the dictionary; counting
documents makes that take forty independent documents agreeing.

And it is a property of the DATA, not of the classifier. Nothing here reads
``categ``, ``quality_score`` or any other pipeline output. That is deliberate and
it is the whole point: this issue has already built one circular objective by
scoring trials against the pipeline's own labels, and a lexicon filtered by
"lines we called Clear" would be the same mistake wearing a different hat.

What it is not
--------------
Not a word list, and it must not be read as one. It is an attestation count.
``Poaceae`` scoring 40 does not mean ``Poaceae`` is Czech; it means forty separate
archaeobotanical reports contain that string, which is the only claim the witness
needs in order to stop convicting it. Symmetrically, absence is weak evidence and
is treated as such -- see ``SHORT_GARBAGE_LEXICON_CONVICT``, which ships false.

Output
------
A TSV with a ``#`` provenance header, then ``token<TAB>document_frequency``, sorted
by descending frequency then token. Tokens only -- no line text, no document
names, no counts per document -- so the artefact can be shared and diffed without
a redaction pass, the same discipline ``tools/issue30_perplex_report.py`` follows.

Tokenisation is imported from ``text_util``, never reimplemented. Four separate
harness divergences in this repository came from a tool carrying its own copy of
logic that then moved; the table has to be built with the same ``_witness_subtokens``
and ``_STRIP_CHARS`` the predicate looks tokens up with, or lookups silently miss.
(#30 D46, 2026-10-01: the witness now also splits on joining punctuation such as
``,`` and ``/``, so a table built before D46 keys comma-fused strings the predicate
no longer looks up. Rebuild it.)

Usage
-----
    # Build over a DOC_LINE_CATEG corpus
    python tools/build_token_lexicon.py /path/to/DOC_LINE_CATEG -o tools/gold/token_df.tsv

    # Then point the predicate at it (setup/config.txt, [TEXT_UTILS])
    SHORT_GARBAGE_LEXICON_PATH   = tools/gold/token_df.tsv
    SHORT_GARBAGE_LEXICON_MIN_DF = 3

    # Inspect what a specific token scored without loading the whole table
    python tools/build_token_lexicon.py <corpus> --lookup malakofauna oueussd Poaceae

Exit codes
----------
  0  Table written (or lookup completed).
  2  Bad arguments / missing path.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402

# A long OCR line can exceed the csv module's default field limit, and dying
# halfway through a multi-million-line corpus with an opaque error helps nobody.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TOOL_VERSION = "1.0"

#: Minimum alphabetic characters for a token to be counted at all. Matched to
#: SHORT_GARBAGE_WITNESS_MIN_ALPHA: the predicate never looks up anything shorter,
#: so storing them would only inflate the table.
DEFAULT_MIN_ALPHA = 4


def iter_tokens(text: str):
    """Yield lookup-normalised tokens from one line, exactly as the predicate does.

    Mirrors ``text_util.shape_garbage_clauses``: split on whitespace, then via
    ``_witness_subtokens`` (joining punctuation, then ``. - –``), strip
    ``_STRIP_CHARS``, lowercase. Any
    divergence here is a table whose keys the predicate cannot find, which fails
    silently as "no vocabulary support" -- i.e. as a witness that convicts more,
    which is the direction that costs real lines.
    """
    for word in text.split():
        for sub in tu._witness_subtokens(word):
            core = sub.strip(tu._STRIP_CHARS)
            if core:
                yield core.lower()


def build(
    collections: "Sequence[tuple[str, Path]]",
    min_alpha: int = DEFAULT_MIN_ALPHA,
    text_column: str = "text",
    progress_every: int = 250_000,
    quiet: bool = False,
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
    """Count distinct documents per token across one or more named collections.

    ``collections`` is a sequence of ``(name, path)``. A single unnamed corpus is
    ``[("", path)]`` and behaves exactly as this function did before.

    Returns ``(df_total, df_by_collection, stats)``.

    **Totals are sums.** Document frequency is the unit and a document belongs to
    exactly one collection, so a token's total is the sum of its per-collection
    counts with no double counting -- which is what makes the split safe to read
    and safe to add up again.

    The split is worth carrying because it shows WHERE a token lives, which a
    single total cannot. It does not show WHAT the token is. `ppole` is the
    worked example: 3 ARUP / 226 ARUB documents (stage 08a), because it is one
    institution's form convention -- the standard abbreviation of *popelnicová
    pole* (@david-spacil, 2026-09-19), not a misread. A token strong in one
    collection and absent from the other may be that collection's own vocabulary
    or its systematic scanning artefact, and the columns cannot tell which.

    Memory: one ``set`` of tokens per document, discarded at the end of that
    document, plus the running counters. A 12.7M-line corpus builds in a few
    minutes on one core with no GPU and no model.
    """
    df_total: dict[str, int] = defaultdict(int)
    df_by_collection: dict[str, dict[str, int]] = {name: defaultdict(int) for name, _ in collections}
    stats = {"documents": 0, "lines": 0, "tokens_seen": 0, "files_unreadable": 0}

    for name, corpus in collections:
        files = sorted(corpus.glob("*.csv")) if corpus.is_dir() else [corpus]
        if not files:
            raise FileNotFoundError(f"no CSV files found in {corpus}")
        per = df_by_collection[name]

        for n, path in enumerate(files, 1):
            seen_in_doc: set[str] = set()
            try:
                with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames is None or text_column not in reader.fieldnames:
                        stats["files_unreadable"] += 1
                        continue
                    for row in reader:
                        stats["lines"] += 1
                        for token in iter_tokens(str(row.get(text_column) or "")):
                            stats["tokens_seen"] += 1
                            if sum(c.isalpha() for c in token) < min_alpha:
                                continue
                            seen_in_doc.add(token)
            except OSError:
                stats["files_unreadable"] += 1
                continue

            # One document, one vote per token, however many times it occurs in it.
            for token in seen_in_doc:
                df_total[token] += 1
                per[token] += 1
            stats["documents"] += 1

            if not quiet and progress_every and stats["lines"] >= progress_every and n % 50 == 0:
                label = f"{name}: " if name else ""
                print(
                    f"  … {label}{n:,}/{len(files):,} documents, {stats['lines']:,} lines, "
                    f"{len(df_total):,} distinct tokens",
                    file=sys.stderr,
                )

    return dict(df_total), {k: dict(v) for k, v in df_by_collection.items()}, stats


def write_table(
    df_counts: dict[str, int],
    out_path: Path,
    corpus: "str | Path",
    stats: dict[str, int],
    min_alpha: int,
    min_df_note: int,
    df_by_collection: "dict[str, dict[str, int]] | None" = None,
) -> int:
    """Write the TSV. Returns the number of rows written.

    With one unnamed collection the output is ``token<TAB>document_frequency``,
    byte-for-byte the format this tool has always written. With several, the
    per-collection counts follow the total in the declared order.

    **The total stays in field 1 in both cases**, because that is the only field
    ``text_util._read_token_lexicon`` reads. Putting a collection there, or
    re-ordering, would leave the predicate reading one collection's count while
    the header said otherwise.
    """
    names = [n for n in (df_by_collection or {}) if n]
    rows = sorted(df_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# token document-frequency table — tools/build_token_lexicon.py v{TOOL_VERSION}\n")
        fh.write(f"# built: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        fh.write(f"# corpus: {corpus}\n")
        fh.write(f"# documents: {stats['documents']}  lines: {stats['lines']}\n")
        fh.write(f"# min_alpha: {min_alpha}  distinct_tokens: {len(rows)}\n")
        fh.write(
            f"# intended SHORT_GARBAGE_LEXICON_MIN_DF: {min_df_note} "
            f"(rows at or above it: {sum(1 for _, v in rows if v >= min_df_note)})\n"
        )
        if names:
            fh.write(f"# columns: token<TAB>document_frequency<TAB>{'<TAB>'.join(names)}\n")
            fh.write("# The per-collection counts sum to the total: a document belongs to one collection.\n")
            fh.write("# A token strong in one collection and absent from another may be that\n")
            fh.write("# collection's own convention (`ppole`, an abbreviation) or its systematic\n")
            fh.write("# scanning artefact. The split shows where a token lives, not which it is.\n")
        else:
            fh.write("# columns: token<TAB>document_frequency\n")
        fh.write("#\n")
        fh.write("# Attestation counts, NOT a claim that any row is a word of any language.\n")
        fh.write("# Contains no line text and no document identifiers.\n")
        for token, count in rows:
            if names:
                per = "\t".join(str(df_by_collection[n].get(token, 0)) for n in names)
                fh.write(f"{token}\t{count}\t{per}\n")
            else:
                fh.write(f"{token}\t{count}\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_token_lexicon.py",
        description="Build a token/document-frequency table for the #30 vocabulary signal.",
        epilog="Emits tokens and counts only — no line text, no document names.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "corpus",
        metavar="PATH",
        nargs="?",
        help="DOC_LINE_CATEG directory, or a single CSV. Omit when using --collection.",
    )
    ap.add_argument(
        "--collection",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "A named collection, repeatable: --collection ARUP=../ARUP/DOC_LINE_CATEG_307 "
            "--collection ARUB=../ARUB/DOC_LINE_CATEG_307. Counts are pooled into one table and "
            "also reported per collection, which is how a collection-specific token becomes "
            "visible -- a convention or an artefact alike; the split cannot tell which. Document "
            "frequency is the unit, so the per-collection columns sum to the total."
        ),
    )
    ap.add_argument("-o", "--output", metavar="TSV", help="Where to write the table.")
    ap.add_argument(
        "--min-alpha",
        type=int,
        default=DEFAULT_MIN_ALPHA,
        metavar="N",
        help=(
            f"Skip tokens with fewer than N alphabetic characters (default {DEFAULT_MIN_ALPHA}, "
            "matching SHORT_GARBAGE_WITNESS_MIN_ALPHA — the predicate never looks up anything shorter)."
        ),
    )
    ap.add_argument(
        "--min-df",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Recorded in the header as the intended SHORT_GARBAGE_LEXICON_MIN_DF and used for the "
            "summary. The table itself keeps every count, so the threshold stays a config decision "
            "and re-tuning it does not mean rebuilding (default 3)."
        ),
    )
    ap.add_argument("--text-column", default="text", metavar="COL", help="Column to tokenise (default: text).")
    ap.add_argument(
        "--lookup",
        nargs="+",
        metavar="TOKEN",
        help="Print the document frequency of these tokens instead of writing a table.",
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    return ap


def parse_collections(specs: list[str], positional: "str | None") -> "list[tuple[str, Path]] | str":
    """Resolve CLI inputs to ``[(name, path)]``, or return an error message.

    A bare positional path stays unnamed, which is what keeps the single-corpus
    output format unchanged.
    """
    out: list[tuple[str, Path]] = []
    for spec in specs:
        name, sep, raw = spec.partition("=")
        if not sep or not name.strip() or not raw.strip():
            return f"--collection expects NAME=PATH, got {spec!r}"
        if any(c in name for c in "\t\n"):
            return f"collection name may not contain whitespace control characters: {name!r}"
        out.append((name.strip(), Path(raw.strip())))
    if positional:
        out.append(("", Path(positional)))
    if not out:
        return "give a corpus PATH or at least one --collection NAME=PATH"
    names = [n for n, _ in out if n]
    if len(names) != len(set(names)):
        return f"duplicate collection name in {names}"
    if names and len(names) != len(out):
        return "mixing a bare PATH with --collection is ambiguous; name every collection"
    for _, path in out:
        if not path.exists():
            return f"path not found: {path}"
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = parse_collections(args.collection, args.corpus)
    if isinstance(resolved, str):
        print(f"error: {resolved}", file=sys.stderr)
        return 2
    collections = resolved
    corpus_label = ", ".join(f"{n}={p}" if n else str(p) for n, p in collections)

    if not args.output and not args.lookup:
        print("error: pass -o/--output to write a table, or --lookup to query without writing.", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Building token document-frequency over {corpus_label} …", file=sys.stderr)

    try:
        df_counts, df_by_collection, stats = build(
            collections,
            min_alpha=args.min_alpha,
            text_column=args.text_column,
            quiet=args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"\n=== token lexicon: {stats['documents']:,} document(s), {stats['lines']:,} line(s) ===",
        file=sys.stderr,
    )
    total = len(df_counts)
    print(f"  distinct tokens (>= {args.min_alpha} letters): {total:,}", file=sys.stderr)
    if total:
        at_threshold = sum(1 for v in df_counts.values() if v >= args.min_df)
        hapax = sum(1 for v in df_counts.values() if v == 1)
        print(
            f"  attested in >= {args.min_df} document(s):      {at_threshold:,} ({100.0 * at_threshold / total:.1f}%)",
            file=sys.stderr,
        )
        # The hapax share is the interesting diagnostic: it is roughly the size of
        # the population an unattested-token rule would reach, and a corpus where
        # it is small is a corpus where that rule has little to do.
        print(
            f"  seen in exactly one document:      {hapax:,} ({100.0 * hapax / total:.1f}%)  — where the residue lives",
            file=sys.stderr,
        )
    if stats["files_unreadable"]:
        print(
            f"  ! {stats['files_unreadable']} file(s) skipped (unreadable, or no {args.text_column!r} column)",
            file=sys.stderr,
        )

    names = [n for n, _ in collections if n]
    if len(names) > 1:
        print("\n  per collection:", file=sys.stderr)
        for name in names:
            per = df_by_collection[name]
            only_here = sum(1 for t, v in per.items() if v and df_counts.get(t, 0) == v)
            print(
                f"    {name:<12} {len(per):>9,} distinct tokens, {only_here:,} of them seen nowhere else",
                file=sys.stderr,
            )

    if args.lookup:
        header = f"\n  {'token':<32} {'documents':>10}"
        if names:
            header += "".join(f" {n:>10}" for n in names)
        print(header + f"   verdict at min_df={args.min_df}")
        for raw in args.lookup:
            for token in iter_tokens(raw) or [raw.lower()]:
                count = df_counts.get(token, 0)
                verdict = "attested" if count >= args.min_df else "UNATTESTED"
                split = "".join(f" {df_by_collection[n].get(token, 0):>10}" for n in names)
                print(f"  {token:<32} {count:>10}{split}   {verdict}")
        if len(names) > 1:
            print(
                "\n  A token attested in ONE collection only may be that collection's convention\n"
                "  (`ppole` is an abbreviation) or its systematic scanning artefact — the split\n"
                "  cannot tell which. Read it as where the token lives, not as a verdict."
            )
        return 0

    written = write_table(
        df_counts,
        Path(args.output),
        corpus_label,
        stats,
        min_alpha=args.min_alpha,
        min_df_note=args.min_df,
        df_by_collection=df_by_collection if names else None,
    )
    print(f"\nTable written → {args.output} ({written:,} rows)", file=sys.stderr)
    print(
        "\nTo use it, set in setup/config.txt [TEXT_UTILS]:\n"
        f"  SHORT_GARBAGE_LEXICON_PATH   = {args.output}\n"
        f"  SHORT_GARBAGE_LEXICON_MIN_DF = {args.min_df}\n"
        "The veto is inert until SHORT_GARBAGE_WITNESS_ENABLE is also true.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
