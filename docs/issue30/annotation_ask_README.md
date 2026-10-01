# Issue #30 — the annotation request

> ## ✅ Returned — 2026-10-01
>
> @DanaKriv's answers are in [`answers/2026-10-01/`](answers/2026-10-01/): 249 of the 357 rows
> labelled, every blank a line that is not Czech. What they say is summarised in
> [`README.md`](README.md) § "2026-10-01 — the answers are in"; the figures are reproduced by
> `python tools/project_annotation.py report --census docs/issue30/answers/2026-10-01/census.csv
> --sample docs/issue30/answers/2026-10-01/sample.csv --frame docs/issue30/answers/2026-10-01/frame.json`
> from the repository root.
> The files below are the request as it was sent and are not edited.

> ## ✅ These files are ready — 2026-09-22
>
> **The earlier hold is lifted.** The previous version of this request was built from the wrong
> set of lines. It has been rebuilt from the right one and replaced. This is the version to work
> on.
>
> It is much smaller: **357 decisions instead of 792**. The reason is the one given in the hold
> notice. The program keeps a dictionary of words that appear across the archive, and anything in
> that dictionary is left alone by the new rule. When we first measured how much text was at risk,
> that dictionary was switched off by mistake. With it switched on, **82% of what we were going to
> ask about cannot be affected at all** — including the single biggest item, `ppole`, and most of
> the rest of the old first page.
>
> **Nothing already done is wasted.** Any answers from the earlier files still join back correctly.

---

## What to do

Fill in **`gold_categ`** only. Write one of: `Clear`, `Noisy`, `Trash`, `Empty`. The fifth
category, `Non-text`, is best left blank: it cannot be told apart from `Trash` without the page
image, which you do not have. That follows @david-spacil's answer of 2026-09-22 and is explained in
[`issue30_annotation_guide.md`](issue30_annotation_guide.md) § 4 — if you disagree with it, say so
(guide § 7.2) rather than working around it.

**If you are not sure, leave it blank.** A blank cell is skipped. It is never treated as a guess.

`confidence` and `note` are free text and optional. Please do not change any other column, and do
not re-sort or delete rows.

One row is one decision. The `lines_settled` column shows how many lines of the archive that one
decision settles.

## The two files

| file         | rows |       lines settled | what it is                                                                                                                                |
|--------------|-----:|--------------------:|-------------------------------------------------------------------------------------------------------------------------------------------|
| `census.csv` |  157 |               1,217 | The most common text at risk, every string the program currently answers in two different ways, and a few checks. Complete, not a sample. |
| `sample.csv` |  200 | 5,646 (represented) | A random selection from the long tail of rare text.                                                                                       |
| `frame.json` |    — |                   — | A small technical file. **Please send it back with the answers.** Without it the 200 rows in `sample.csv` cannot be turned into a number. |

**357 decisions, 516 different spellings.** One row often covers several spellings of the same
thing — punctuation, spacing, a stray mark. The `variants` column lists them. The largest row,
`Dauerleihe`, covers **24** spellings on its own.

## What the request buys

* The full list holds **42,248 different pieces of text**, covering **48,909 lines**.
* Of those, **5,563 pieces of text / 6,714 lines** are *at risk* — meaning the program keeps them
  today, so switching the new rule on would start discarding them. The rest are already discarded,
  where an answer cannot change anything.
* `census.csv` settles **18.1%** of that risk in 157 decisions.
* `sample.csv` estimates the rest to within **±6.9 percentage points**.

Two honest notes on those last two figures, neither of which changes what you are being asked to
do:

* **18.1% counts 149 lines that cannot change anything.** 39 of the 157 census rows are checks:
  text the program already discards, included so we can see whether the new rule agrees with it.
  Counting only the rows where your answer can move a decision, the census settles **15.9%**.
* **The 18.1% and the 6,714 come from the program's own summary**, and recounting them from the
  delivered files gives figures about 2% apart (5,695 pieces of text / 6,668 lines). The difference
  is in how two tools count the same thing and it is recorded rather than hidden. It changes
  nothing about the rows themselves.

## Why the census is small now

The old request could settle most of the risk in a few hundred rows because a handful of pieces of
text repeated tens of thousands of times. Those are exactly the items the dictionary now protects.
What is left is flat: **96% of the text at risk appears exactly once**, and even the 500 most
common pieces of it cover only 22% of the lines at risk.

So `census.csv` is only 60 frequency rows. The rest of the file is 97 pieces of text the program
currently answers two different ways in different places — worth settling for their own sake — and
the 39 checks. The estimate now comes from `sample.csv`.

## The selection record — please keep this file with the answers

Without `frame.json`, `sample.csv` cannot be turned into a number and becomes 200 individual
opinions.

`sample.csv` is drawn from three groups. Each row stands for a number of similar rows that were not
included, and `weight` is how many:

| group             | pieces of text | lines | how many we ask about |  weight | margin of error |
|-------------------|---------------:|------:|----------------------:|--------:|----------------:|
| `at_risk/full`    |          1,310 | 1,370 |                    48 | 27.2917 |    ±14.1 points |
| `at_risk/none`    |          2,668 | 2,766 |                    98 | 27.2245 |     ±9.9 points |
| `at_risk/partial` |          1,467 | 1,510 |                    54 | 27.1667 |    ±13.3 points |

To turn answers into a figure for the whole archive, we work out a rate for each group separately,
then combine the three by their **line** counts — not by their counts of distinct text. The three
groups have different average numbers of lines per entry, so combining them the other way gives the
wrong answer.

One shortcut looks right and is not: multiplying `lines_settled` by `weight` gives 5,717 lines
against the 5,646 recorded above. The weight counts pieces of text, not lines.

Settings used to build these files: `--census 60 --sample 200 --seed 30`.

## One thing worth knowing before you start

The largest single item in this request is **`Dauerleihe`** — German for *permanent loan* — on 286
lines the program currently keeps. It is scanned perfectly correctly. It is flagged because it has
three vowels in a row, which usually means damage, and because the archive's own dictionary has
never seen the word.

`FEUILLETON.` is the same story. So, in a different way, is `Dated=Dated (relatively)`.

This is not a trick, and there is no right answer we are hoping for. If your answer is that
`Dauerleihe` and `FEUILLETON.` are `Clear`, that confirms what @david-spacil told us: a run of three
vowels is evidence of damage in Czech and not in German or French. That change is already built —
the three-vowel test now asks for four vowels on lines detected as German or French — and it ships
inside the new rule, which stays switched off. (Simply requiring four vowels everywhere was measured
and rejected: it lets real scanning errors such as `J. Vysoean` through.) `Dated=Dated (relatively)`
is caught by a different test — too few different letters — which that change does not touch, so
your answer on it is evidence of its own. Please judge what you see.

---

Nothing here suggests an answer, and nothing written in these files changes the program directly.
The answers are joined back separately and looked at first.
