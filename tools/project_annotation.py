#!/usr/bin/env python3
"""Read a RETURNED annotation ask: project it, and join it back onto the corpus.

Issue #30 (renamed repository: atrium-ocr-postprocess#3). ``build_annotation_sample.py``
cuts the ask -- ``census.csv`` (the at-risk head, complete), ``sample.csv`` (a
stratified draw from the tail) and ``frame.json`` (what each sampled row stands
for). This tool is the other end: what the answers say, and how to score the
pipeline against them.

Two subcommands
---------------
``report``
    Text only, seconds, no corpus. Label coverage, the census measured exactly
    (line-weighted by each row's ``categ_current`` mix), the sample's label rates
    per stratum with Wilson 95% intervals, the tail projected through the frame,
    and what arming the shape witness would do to each -- using the production
    ``shape_garbage_clauses``, not a copy of it.

``join``
    Streams DOC_LINE_CATEG CSVs and writes a gold SIDECAR keyed on
    ``(file, page_num, line_num)`` -- the shape ``--gold-sidecar`` reads -- for
    every line carrying an annotated string, plus the list of document CSVs those
    lines live in (``--docs-out``), so the documents can be staged for
    ``ab_constant_eval.py``.

Why the join does not go through the witness report
---------------------------------------------------
``short_garbage_witness_report.py --from-distinct`` projects labels onto the
lines the witness fires on TODAY. A predicate change that stops it firing on a
labelled line (#30 D46 did, on both lines @DanaKriv labelled `Clear`) would drop
that line from the sidecar, and the A/B would never see the change it is meant to
measure. The join here matches on text alone.

How a returned row is matched
-----------------------------
Exact ``text`` first, then every spelling in ``variants`` (split on ``" | "``),
then ``family_key`` -- the same normalisation ``build_annotation_pack`` grouped
the families with, so a row settles the lines it says it settles. A family key
claimed by two rows with different labels is dropped from family matching and
reported; exact spellings still match.

Labels are validated, not passed through: ``clear`` / ``Clear`` / ``CLEAR`` are
accepted, anything outside the five categories stops the run with the row named.
A blank is skipped, never guessed.

What the projection assumes, said where it is printed
-----------------------------------------------------
Blank rows are ignored. That is only an unbiased estimate if blanks are random,
and in the 2026-10-01 return they are not: every blank is a non-Czech line. The
report prints the blank exposure separately so the gap is visible.

Usage
-----
::

    python tools/project_annotation.py report \\
        --census docs/issue30/census.csv \\
        --sample docs/issue30/sample.csv \\
        --frame  docs/issue30/frame.json

    python tools/project_annotation.py join \\
        --census ... --sample ... \\
        --corpus ../ARUP/DOC_LINE_CATEG_307 --corpus ../ARUB/DOC_LINE_CATEG_307 \\
        --out issue30_stage12_out/12b_dana_sidecar.csv --docs-out issue30_stage12_out/12b_docs.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import text_util as tu  # noqa: E402
from tools.build_annotation_pack import family_key  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TOOL_VERSION = "1.0"

#: The five categories, keyed case-insensitively. Strict on purpose: these files
#: come back from people, and a typo read as "not annotated" is a silent drop.
_LABELS = {
    "clear": "Clear",
    "noisy": "Noisy",
    "trash": "Trash",
    "non-text": "Non-text",
    "nontext": "Non-text",
    "empty": "Empty",
}

#: The categories under which the pipeline is "keeping" a line -- the only ones
#: the witness can newly convict. Same tuple build_annotation_sample uses.
KEPT = ("Clear", "Noisy")

SIDECAR_COLUMNS = ("file", "page_num", "line_num", "gold_categ", "gold_source", "tranche", "stratum")

_VARIANT_SEP = " | "


def normalize_label(value: str, where: str) -> str:
    """``""`` for a blank, the canonical category otherwise; raise on anything else."""
    raw = (value or "").strip()
    if not raw:
        return ""
    canon = _LABELS.get(raw.lower().replace("_", "-"))
    if canon is None:
        raise ValueError(f"{where}: gold_categ {raw!r} is not one of Clear, Noisy, Trash, Non-text, Empty")
    return canon


def parse_mix(mix: str) -> dict[str, int]:
    """``Clear:277|Trash:19`` -> ``{"Clear": 277, "Trash": 19}``."""
    out: dict[str, int] = {}
    for part in (mix or "").split("|"):
        name, _, count = part.partition(":")
        try:
            out[name.strip()] = out.get(name.strip(), 0) + int(count)
        except ValueError:
            continue
    return out


@dataclass
class Decision:
    text: str
    label: str
    source: str
    tranche: str
    stratum: str
    lines_settled: int
    mix: dict[str, int]
    variants: list[str] = field(default_factory=list)


def read_ask(path: Path, source: str) -> list[Decision]:
    """Every row of one returned file, blanks included (``label == ""``).

    ``utf-8-sig``: a file saved back from a spreadsheet often starts with a BOM,
    which would otherwise turn the first header into ``\ufefftext``.
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"text", "gold_categ"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        rows = []
        for i, row in enumerate(reader, start=2):
            variants = [v.strip() for v in (row.get("variants") or "").split(_VARIANT_SEP) if v.strip()]
            try:
                settled = int(row.get("lines_settled") or 0)
            except ValueError:
                settled = 0
            rows.append(
                Decision(
                    text=(row.get("text") or "").strip(),
                    label=normalize_label(row.get("gold_categ", ""), f"{path}:{i}"),
                    source=source,
                    tranche=(row.get("tranche") or "").strip(),
                    stratum=(row.get("stratum") or "").strip(),
                    lines_settled=settled,
                    mix=parse_mix(row.get("categ_current", "")),
                    variants=variants,
                )
            )
    return rows


class Decisions:
    """Labelled rows indexed for lookup by a corpus line's text."""

    def __init__(self, decisions: list[Decision]):
        self.labelled = [d for d in decisions if d.label]
        self.by_text: dict[str, Decision] = {}
        by_family: dict[str, list[Decision]] = defaultdict(list)
        for d in self.labelled:
            for spelling in [d.text, *d.variants]:
                self.by_text.setdefault(spelling, d)
            for key in {family_key(s) for s in [d.text, *d.variants]}:
                if key:
                    by_family[key].append(d)
        self.conflicts = sorted(k for k, ds in by_family.items() if len({d.label for d in ds}) > 1)
        self.by_family = {k: ds[0] for k, ds in by_family.items() if k not in self.conflicts}
        self.max_words = max((len(s.split()) for d in self.labelled for s in [d.text, *d.variants]), default=0)

    def lookup(self, text: str) -> Decision | None:
        text = (text or "").strip()
        hit = self.by_text.get(text)
        if hit is not None:
            return hit
        if len(text.split()) > self.max_words + 2:
            return None
        return self.by_family.get(family_key(text))


def read_decisions(paths: list[Path], sources: list[str] | None = None) -> Decisions:
    rows: list[Decision] = []
    for i, path in enumerate(paths):
        rows.extend(read_ask(path, (sources or [])[i] if sources and i < len(sources) else path.stem))
    return Decisions(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, centre - half, centre + half


def witness_fires(text: str) -> bool:
    """The production predicate. The returned rows carry no language, so the
    general vowel-run threshold applies (D44's split needs ``original_lang``)."""
    return bool(tu.shape_garbage_clauses(text))


def outcome(current: str, fires: bool) -> str:
    """What arming the witness does to one line the pipeline currently calls ``current``."""
    return "Trash" if fires and current in KEPT else current


def score_rows(decisions: list[Decision], weight_of=lambda d: 1.0) -> dict[str, dict[str, float]]:
    """Error counts with the witness off and on, line-weighted by ``categ_current``."""
    arms: dict[str, Counter] = {"off": Counter(), "on": Counter()}
    for d in decisions:
        if not d.label:
            continue
        fires = witness_fires(d.text)
        w = weight_of(d)
        for current, n in d.mix.items():
            for arm, new in (("off", current), ("on", outcome(current, fires))):
                m = arms[arm]
                m["lines"] += n * w
                m["errors"] += n * w * (new != d.label)
                m["clear_loss"] += n * w * (d.label == "Clear" and new in ("Trash", "Non-text"))
                m["trash_caught"] += n * w * (d.label == "Trash" and new == "Trash")
                m["gold_trash"] += n * w * (d.label == "Trash")
                m["noisy_to_trash"] += n * w * (d.label == "Noisy" and new == "Trash")
    return {arm: dict(m) for arm, m in arms.items()}


def blank_exposure(decisions: list[Decision]) -> dict[str, int]:
    kept = convicted = 0
    for d in decisions:
        if d.label:
            continue
        n = sum(v for k, v in d.mix.items() if k in KEPT)
        kept += n
        convicted += n if witness_fires(d.text) else 0
    return {"rows": sum(1 for d in decisions if not d.label), "kept_lines": kept, "witness_convicts": convicted}


def project(census: list[Decision], sample: list[Decision], frame: list[dict]) -> dict:
    by_stratum = {f["stratum"]: f for f in frame}
    result: dict = {"tool_version": TOOL_VERSION}

    result["coverage"] = {
        name: {
            "rows": len(rows),
            "labelled": sum(1 for d in rows if d.label),
            "labels": dict(Counter(d.label or "(blank)" for d in rows)),
        }
        for name, rows in (("census", census), ("sample", sample))
    }

    at_risk = [d for d in census if d.stratum.startswith("at_risk")]
    confirms = [d for d in census if not d.stratum.startswith("at_risk")]
    result["census_at_risk"] = {"arms": score_rows(at_risk), "blank": blank_exposure(at_risk)}
    result["census_confirms_trash"] = {"labels": dict(Counter(d.label or "(blank)" for d in confirms))}

    strata = {}
    tail = {"off": Counter(), "on": Counter()}
    projected_labels: Counter = Counter()
    for name, f in sorted(by_stratum.items()):
        rows = [d for d in sample if d.stratum == name]
        labelled = [d for d in rows if d.label]
        counts = Counter(d.label for d in labelled)
        p, lo, hi = wilson(counts.get("Trash", 0), len(labelled))
        labelled_lines = sum(sum(d.mix.values()) for d in labelled)
        factor = f["lines_in_stratum"] / labelled_lines if labelled_lines else 0.0
        arms = score_rows(labelled, weight_of=lambda d, k=factor: k)
        for arm in tail:
            tail[arm].update(arms[arm])
        for d in labelled:
            projected_labels[d.label] += sum(d.mix.values()) * factor
        strata[name] = {
            "sampled": len(rows),
            "labelled": len(labelled),
            "blank": len(rows) - len(labelled),
            "labels": dict(counts),
            "trash_share": p,
            "trash_ci95": [lo, hi],
            "lines_in_stratum": f["lines_in_stratum"],
        }
    result["sample_strata"] = strata
    result["tail_projection"] = {
        "lines": sum(f["lines_in_stratum"] for f in frame),
        "labels": dict(projected_labels),
        "arms": {arm: dict(m) for arm, m in tail.items()},
        "blank": blank_exposure(sample),
    }
    result["witness_convicts_gold_clear"] = sorted(
        d.text
        for d in census + sample
        if d.label == "Clear" and witness_fires(d.text) and any(k in KEPT for k in d.mix)
    )
    return result


def _fmt_arms(arms: dict) -> list[str]:
    out = []
    for arm in ("off", "on"):
        m = arms[arm]
        out.append(
            f"    witness {arm:3}  errors {m.get('errors', 0):>8,.0f}   Clear-loss {m.get('clear_loss', 0):>6,.0f}   "
            f"Trash caught {m.get('trash_caught', 0):>7,.0f} / {m.get('gold_trash', 0):,.0f}   "
            f"Noisy->Trash {m.get('noisy_to_trash', 0):>6,.0f}"
        )
    return out


def render(result: dict) -> str:
    lines = [
        f"project_annotation.py v{TOOL_VERSION} -- witness predicate: production shape_garbage_clauses, no language",
        "",
    ]
    lines.append("== coverage")
    for name, c in result["coverage"].items():
        lines.append(f"    {name:7} {c['labelled']:>4} of {c['rows']:>4} rows labelled   {c['labels']}")
    lines.append(
        "    Blank rows are skipped. If they are not random, the figures below describe the labelled part only."
    )
    lines.append("")
    ca = result["census_at_risk"]
    lines.append(
        f"== census, at-risk rows, exact (line-weighted, {ca['arms']['off'].get('lines', 0):,.0f} labelled lines)"
    )
    lines += _fmt_arms(ca["arms"])
    b = ca["blank"]
    lines.append(
        f"    blank rows: {b['rows']}, {b['kept_lines']:,} kept lines, {b['witness_convicts']:,} of them convicted when armed"
    )
    lines.append(f"== census, confirms_trash rows (already Trash today): {result['census_confirms_trash']['labels']}")
    lines.append("")
    lines.append("== sample, per stratum (Trash share of labelled rows, Wilson 95%)")
    for name, s in result["sample_strata"].items():
        lo, hi = s["trash_ci95"]
        lines.append(
            f"    {name:16} {s['labelled']:>3}/{s['sampled']:<3} labelled  {s['labels']}  "
            f"Trash {s['trash_share']:.1%} [{lo:.1%}, {hi:.1%}]  -> {s['lines_in_stratum']:,} lines"
        )
    tp = result["tail_projection"]
    total = tp["lines"] or 1
    shares = ", ".join(
        f"{k} {v:,.0f} ({v / total:.1%})" for k, v in sorted(tp["labels"].items(), key=lambda kv: -kv[1])
    )
    lines.append(f"== tail projected through the frame, by lines: {shares} of {tp['lines']:,}")
    lines += _fmt_arms(tp["arms"])
    b = tp["blank"]
    lines.append(
        f"    blank rows: {b['rows']} (not projected), {b['kept_lines']:,} kept lines, {b['witness_convicts']:,} convicted when armed"
    )
    lines.append("")
    lines.append(
        f"== gold-Clear rows the armed witness would convict: {result['witness_convicts_gold_clear'] or 'none'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


def _collect(paths: list[Path], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(x for x in p.glob("**/*.csv" if recursive else "*.csv") if x.is_file()))
        elif p.is_file():
            out.append(p)
    return out


def join(decisions: Decisions, csvs: list[Path], out: Path, docs_out: Path | None) -> dict:
    seen: set[tuple[str, str, str]] = set()
    docs: dict[str, Path] = {}
    basenames: dict[str, set[Path]] = defaultdict(set)
    matched_by_row: Counter = Counter()
    duplicates = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SIDECAR_COLUMNS)
        for path in csvs:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    d = decisions.lookup(row.get("text") or "")
                    if d is None:
                        continue
                    key = (
                        (row.get("file") or path.stem).strip(),
                        (row.get("page_num") or "").strip(),
                        (row.get("line_num") or "").strip(),
                    )
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    writer.writerow([*key, d.label, f"{d.source}:{d.tranche}", d.tranche, d.stratum])
                    matched_by_row[id(d)] += 1
                    docs[str(path)] = path
                    basenames[path.name].add(path)
    if docs_out is not None:
        docs_out.parent.mkdir(parents=True, exist_ok=True)
        docs_out.write_text("".join(f"{p}\n" for p in sorted(docs)), encoding="utf-8")
    unmatched = [d.text for d in decisions.labelled if not matched_by_row[id(d)]]
    settled = sum(d.lines_settled for d in decisions.labelled)
    return {
        "csvs_read": len(csvs),
        "lines_joined": len(seen),
        "lines_settled_by_labelled_rows": settled,
        "documents": len(docs),
        "duplicate_keys_skipped": duplicates,
        "basename_collisions": sorted(n for n, ps in basenames.items() if len(ps) > 1),
        "labelled_rows": len(decisions.labelled),
        "labelled_rows_unmatched": unmatched,
        "family_conflicts": decisions.conflicts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="Project the returned ask; text only.")
    rep.add_argument("--census", required=True, type=Path)
    rep.add_argument("--sample", required=True, type=Path)
    rep.add_argument("--frame", required=True, type=Path)
    rep.add_argument("--json", type=Path, default=None, metavar="PATH", help="Also write the figures as JSON.")

    jn = sub.add_parser("join", help="Write a (file, page_num, line_num) gold sidecar from the returned ask.")
    jn.add_argument("--census", required=True, type=Path)
    jn.add_argument("--sample", required=True, type=Path)
    jn.add_argument(
        "--corpus", required=True, type=Path, action="append", help="DOC_LINE_CATEG dir or CSV; repeatable."
    )
    jn.add_argument("--recursive", action="store_true", help="Walk sub-directories of each --corpus.")
    jn.add_argument("--out", required=True, type=Path, help="The sidecar to write.")
    jn.add_argument("--docs-out", type=Path, default=None, help="Write the CSV paths of every joined document here.")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "report":
            census = read_ask(args.census, "census")
            sample = read_ask(args.sample, "sample")
            frame = json.loads(args.frame.read_text(encoding="utf-8"))
            result = project(census, sample, frame)
            print(render(result))
            if args.json:
                args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return 0

        decisions = read_decisions([args.census, args.sample], ["census", "sample"])
        csvs = _collect(args.corpus, args.recursive)
        if not csvs:
            print(
                "error: no CSVs under --corpus (pass --recursive for an archive with sub-directories)", file=sys.stderr
            )
            return 2
        for c in args.corpus:
            print(f"  corpus {c}: {len(_collect([c], args.recursive)):,} CSVs", file=sys.stderr)
        stats = join(decisions, csvs, args.out, args.docs_out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["lines_joined"] == 0:
        print("error: no line joined -- wrong corpus, or the text column differs", file=sys.stderr)
        return 3
    if stats["basename_collisions"]:
        print(
            "error: two archives share a document file name; staging them into one directory would overwrite",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
