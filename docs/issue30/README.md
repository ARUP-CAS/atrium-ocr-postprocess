# Issue #30 — what is in this folder

**For:** @DanaKriv and @david-spacil, and anyone joining
[issue #30](https://github.com/ufal/atrium-alto-postprocess/issues/30) without having followed it.

The program in this repository reads the scanned text of the archive line by line and sorts every
line into one of five boxes: `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty`. Issue #30 is about one
change to how it decides. This folder holds everything written for the people who provide the data
and use the results, rather than for the people writing the code.

---

## Read in this order

**1. Start here — what the archive looks like to the program**
[`issue30_corpus_profile.md`](issue30_corpus_profile.md) · for both of you · about 15 minutes

Measured across all 113,100 documents. It covers the two collections, why documents from 2010
onwards behave differently, Latin species names, your own company name in page headers, and why
dictionary-style checks keep misfiring on this archive. Sections 0 and 3 were rewritten on
2026-09-22 after the measurement was repeated correctly; § 3c explains why a correctly scanned
German word is the largest thing at risk. **Nothing in it asks you for anything.** If you read one
document, read this one.

**2a. If you are @DanaKriv — how the labelling works**
[`issue30_annotation_guide.md`](issue30_annotation_guide.md) · about two to four hours of work

What each of the five labels means, what to fill in, what to leave alone, and where to start.
**The earlier notice asking you to wait has been withdrawn** — the files were rebuilt and the
request is ready. Section 7.2 contains a question you can answer without the files, and it is
worth reading before you begin.

**2b. If you are @david-spacil — the open questions**
[`issue30_review_request.md`](issue30_review_request.md) · most items need one line each

**✅ Answered in full on 2026-09-22.** Eight items, all closed or waiting on @DanaKriv. Worth
reading even so: two of the answers correct things this folder used to state as fact — three
doubled-letter spellings turn out to be an old spelling of *suť* rather than scanning errors, and two
strings we described as correctly scanned are not. **⏳ § 9 is new**: the follow-up questions your
answers made possible to ask, also listed under "Still open" below. **⏳ § 10 (2026-09-23)** adds
one more, Q8, for @david-spacil alone and optional.

**3. Reference, when you need it**
[`annotation_ask_README.md`](annotation_ask_README.md) — how the two request files are built and
how to fill them in. Read it together with the guide rather than on its own.

---

## The data files here

| file         | what it is                                                                                                                                                                       |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `census.csv` | 157 rows. The most common text at risk, every string the program currently answers in two different ways, and 39 checks. ✅ **Returned with @DanaKriv's answers** (119 labelled). |
| `sample.csv` | 200 rows. A random selection from the long tail of rare text. ✅ **Returned with @DanaKriv's answers** (130 labelled).                                                            |
| `frame.json` | A small technical file recording how `sample.csv` was chosen. Came back unchanged with the answers, which is what lets the 200 rows be turned into a figure.                     |

✅ **2026-10-01:** these are now the returned files (via @david-spacil, `anotace_Dana.zip`); they
replaced the blank request in place, and only `gold_categ` changed. The blank request as sent is in
the repository history.

When the request went out, `census.csv` and `sample.csv` had three empty columns: `gold_categ`, `confidence` and
`note`. Those were the ones to fill in; everything else was there to help decide.

---

## Files elsewhere in the repository

| where                                       | what it is                                                                                                                                                                                                                                                                     | worth opening?                                                                                                                                        |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_samples/`                             | Small demonstration files showing every stage of the pipeline. **Important: this data is invented.** An imaginary site, imaginary researchers, no real records from either collection. It exists so anyone can see the shapes of the files without access to the real archive. | Yes — `data_samples/README.md` first, then `DOC_LINE_CATEG/CTX000000002.csv`, which is one row per line of text with the category the program gave it |
| `tools/gold/sidecars/issue30_gold_2067.csv` | The 2,067 lines that have been labelled by hand — your own work, collected into one file. Every accuracy figure quoted in issue #30 is measured against it.                                                                                                                    | It has no text column, so it is hard to read directly. `tools/gold/GOLD.md` explains what it is for                                                   |
| `tools/gold/GOLD_CLEAR.csv`                 | Nine example lines that must always come out as `Clear`. A one-screen illustration of what a labelled row looks like.                                                                                                                                                          | Yes, as an example                                                                                                                                    |
| `README.md` (repository root)               | Mostly installation and configuration. The useful part for you is the five-category table, roughly a third of the way down.                                                                                                                                                    | That table only                                                                                                                                       |
| `docs/categorization_logic.md`              | The full rule-by-rule description of how a line's category is decided. Accurate and complete, but written for developers.                                                                                                                                                      | Only if you want the detail                                                                                                                           |
| `issue30_gold_ab_findings.md` (this folder) | Technical findings from the measurement runs.                                                                                                                                                                                                                                  | Developers only                                                                                                                                       |
| `agent_dev_logs/digests/30.digest.md`       | The complete working record of issue #30, including every correction we have had to make to our own conclusions. Long, and written for us rather than for you — but nothing is hidden in it.                                                                                   | Only if you want the reasoning                                                                                                                        |

**Two notes on licensing**, because they differ. The code is under the MIT licence. The
demonstration data in `data_samples/` is under Creative Commons BY-NC 4.0, which does not allow
commercial reuse. Neither covers the real archive collections.

**One warning about file names.** `data_samples/arup_page_stats_SHORT.csv` and
`arub_page_stats_SHORT.csv` are named after the two collections, but they contain neither. They are
identical to each other and hold only the invented demonstration rows.

---

## The two open questions — both now answered by @david-spacil

He answered on 2026-09-22, and neither answer came from more computing. Both are recorded in full
in [`issue30_review_request.md`](issue30_review_request.md).

**1. What should `Trash` mean?** ✅ **Illegible.** Anything legible is `Clear`; easily decipherable
is `Noisy` — regardless of how useful the line is. So a correctly scanned web address such as
`http://www.arub.cz` is `Clear`. `Trash` lines should be scanned again rather than deleted, and
`Non-text` cannot be judged without the page image, so it is out of scope for labelling.

**@DanaKriv, this is the one thing worth reading before you start.** It is his view, not yet a
joint decision, and the whole measurement is scored against it. If you disagree, please say so
first. See `issue30_annotation_guide.md` § 4 and § 7.2.

**2. Is the new rule set one notch too tight?** ✅ **It is set to the wrong question.** The test
looks for three vowels in a row, which is good evidence of damage in Czech — Czech has no
three-vowel runs — and simply wrong for German and French, where words like `Dauerleihe` and
`FEUILLETON` have them naturally. His answer is to split the rule by language rather than blunt it.

He also corrected us: `J. Vysoean` and `POSTKRANIAINY SKELET` are **not** correctly scanned, as we
had written. They are real scanning errors, so the three-vowel test catches them rightly. See
`issue30_review_request.md` § 7.

**Measured since then, on 2026-09-22.** Two runs finished and both agree with him:

* **Simply requiring four vowels instead of three is not worth doing.** Against the 2,064 labelled
  lines it changes nothing that matters, and it finds less rubbish than the current rule.
* **The new rule as a whole passes its test** — 513 mistakes down to 503, no extra readable lines
  lost, and half as much rubbish found again.
* **The only two lines the rule gets wrong are both caught by the three-vowel test**, and one of
  them is a German sentence. The part he identified is the only part in dispute.

**Settled on 2026-09-22, in the final runs.** The language split is built and measured, and it
ships *inside* the new rule, which is switched off — so switching the rule on switches the split on
with it (`SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS = deu,fra` in `setup/config.txt`; emptying
that key gives back a single threshold). Measured: **12 lines improved against 1 made worse**, total mistakes 503 → 502, readable lines
wrongly discarded 38 → 37. Both numbers move the right way. The line it stops destroying is the
German sentence `Frauenzimmerbad", sämtlic Gesellschastsbäder,`.

Two cautions worth more than the result. **It works partly by luck** — the program guesses a
language for every line, and on short damaged lines that guess is often wrong. It called
`deutendes. Alhimiaal` *Afrikaans* (a person would read it as German), and that mistake is the only
reason the rule still catches it correctly. And **44.5% of labelled lines are not recognised as
Czech, which is misleading**: the list includes Vietnamese, Estonian, Xhosa and Uzbek, which do not
appear in this archive. The share the split actually acts on is **5.8%**. Details in
`issue30_review_request.md` § 7.

---

## A new file you can edit yourselves

`setup/word_lists.txt` is a plain text file listing the words this program should not treat as
damage. Until now those lists lived inside the program code or in a settings file, and adding a
word meant asking a developer.

**The section to use is `[allowed]`.** If the program keeps throwing away a word you know is real —
a German museum term, a Latin species name, an old spelling, a local abbreviation, a place name —
put it there with a short note saying what it is. Nothing you write there can make the program
discard anything.

**What listing a word does today, precisely** — because an earlier summary in the issue thread
(comment 61) said more than the code does:

* **It does:** stop the word counting against the line's quality score. That alone can move a line
  up a whole category, and it means a line made **only** of listed words can no longer be discarded
  by the short-line rule.
* **It also does** (✅ Q5b, 2026-10-01): stop the new rule's shape tests (three vowels in a row, a
  doubled first letter, too few different letters) from reading the word, so a listed word can
  never be the reason a line is discarded. The other words on the line are still read.

✅ **Q5a, 2026-10-01:** @david-spacil switched on the reviewed entries (pull request #7) — the old
spellings and `ppole`, the German museum terms and place names, `malakofauna`, `diapozitiv`, and the
Latin anatomy and taxonomy. The ones left out are still in the file, commented, with his reason
beside each (`Bauerleihe` and `beuern` are scanning damage; `naiade` and `oueste` were not found).

---

## ✅ 2026-10-01 — the answers are in

@david-spacil summarised what he and @DanaKriv agreed after the 30 September meeting, and the 357
decisions came back with it. What they settle, and what is left:

| #       | answer                                                                                                                                                                                                                                                                                                                                                   |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Q1**  | ✅ Agreed: `Trash` = illegible, decipherable = `Noisy`, usefulness does not count.                                                                                                                                                                                                                                                                        |
| **Q2**  | ✅ The de-duplication step stays as it is.                                                                                                                                                                                                                                                                                                                |
| **Q3**  | ✅ Returned: 249 of 357 rows labelled. **Every blank is a line that is not Czech** (German, Latin, English), so the labels say nothing yet about German words such as `Dauerleihe`.                                                                                                                                                                       |
| **Q4**  | ✅ `Clear`. Now set: every recognised web or e-mail address is `Clear`.                                                                                                                                                                                                                                                                                   |
| **Q5a** | ✅ Switched on by @david-spacil (pull request #7): the reviewed entries, with the rest left commented.                                                                                                                                                                                                                                                    |
| **Q5b** | ✅ *Yes, listed means never evidence of damage* (@david-spacil, 2026-10-01). Done: the new rule skips a listed word in every test, so it can never be the reason a line is discarded; the other words on the line are still read. Your Q5a entries take effect through both the score and the new rule.                                                   |
| **Q6**  | ✅ Answered by the labels: **`Trash`**. About a third of what the new rule would discard is decipherable, but nothing the program can see tells those lines apart reliably — a line with one damaged word next to a clean one looked promising in the common text and failed in the rare text. Your fallback was "prefer `Trash`", so that is what stays. |

**What your labels showed about the program**, beyond answering Q6:

* The new rule, switched on, would fix most of what it touches: in the rare text it takes the
  estimated mistakes from about 4,800 lines to about 1,200, and it discards **none** of the lines
  you marked `Clear`.
* That last part needed one correction first. The rule used to read words joined by a comma or a
  slash without a space — `gut erhalten,Siedelungsfund,gefunden` — as one long word, and judge it
  as damage. Your only two `Clear` lines in the at-risk set, `XXX,1937,str. 21` and
  `okraj sekt.III,konec`, were exactly that. It now reads them word by word.
* The rule is still switched off. The last check before switching it on is a re-run on the cluster
  that scores it against your labels on the real lines.

## ⏳ Still open — asked in the issue thread on 2026-09-22 (Q8: 2026-09-23)

> **2026-10-01:** Q1–Q6 are answered (above), and Q5a's entries are switched on (pull request #7).
> The table below is kept as asked.

Posted after [comment 61](https://github.com/ufal/atrium-alto-postprocess/issues/30#issuecomment-5783543756)
and numbered the same way there. Each needs a line; "no opinion" is an answer, and a blank is never
read as a yes. **Nothing changes in the program until they are answered.**

**Status, 2026-09-23.** Q7 is answered, and the re-check behind it found a lost exemption that is
now restored — ✅ **confirmed** on `3b02959`: 336/508, with `Lokolieace: •VIII,` the only break
(review request § 3). @david-spacil is going through Q1–Q6 with @DanaKriv by e-mail. Q8 is new, for
@david-spacil alone, and optional.

| #       | for           | question                                                                                                                                                                                                                                                                    | where it is explained                                                                       |
|---------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Q1**  | @DanaKriv     | Do you agree that `Trash` means illegible — legible is `Clear`, decipherable is `Noisy`, usefulness does not count, and `Non-text` is left blank when labelling?                                                                                                            | [guide § 7.2](issue30_annotation_guide.md), [review request § 6](issue30_review_request.md) |
| **Q2**  | @DanaKriv     | Is the de-duplication trade acceptable — 10 readable lines pulled down, 51 rubbish lines rescued? @david-spacil chose to keep the step.                                                                                                                                     | [guide § 7.1](issue30_annotation_guide.md)                                                  |
| **Q3**  | @DanaKriv     | The 357 decisions (`census.csv`, `sample.csv`, `frame.json`) — roughly when might a first batch come? Partial returns are welcome.                                                                                                                                          | [guide](issue30_annotation_guide.md)                                                        |
| **Q4**  | both          | Once Q1 is agreed: should **every** recognised web or e-mail address be `Clear`, or `Noisy`? The program cannot tell a damaged address from a correct one.                                                                                                                  | [review request § 9](issue30_review_request.md)                                             |
| **Q5a** | both          | Which `[allowed]` entries should be switched on — the four confirmed ones, all the candidates, a named subset, or none yet?                                                                                                                                                 | "A new file you can edit yourselves", above                                                 |
| **Q5b** | both          | Should a listed word also be exempt from the new rule's shape tests, so it can never be the reason a line is discarded?                                                                                                                                                     | "A new file you can edit yourselves", above                                                 |
| **Q6**  | both          | The short-line rule can only answer `Trash`. When it cannot tell decipherable from illegible on 1–3 words, which mistake is better — `Trash` (re-processed) or `Noisy`?                                                                                                     | [review request § 9](issue30_review_request.md)                                             |
| **Q7**  | @david-spacil | ✅ **Answered 2026-09-23: `master`.** Re-check done and confirmed — **336/508** on `3b02959`, 11 fixed / 1 broken / 3 wrong either way, once the lost `S-VIIIb` exemption was restored (335/508 before).                                                                     | [review request § 3](issue30_review_request.md)                                             |
| **Q8**  | @david-spacil | ⏳ *Asked 2026-09-23, optional.* In this archive, are `ä`, `ö`, `ü` and `ß` reliable signs of German, or do damaged Czech scans produce them (`á` read as `ä`, `ů` as `ü`)? The answer decides whether a German-letter exemption is worth measuring; it switches nothing on. | [review request § 10](issue30_review_request.md)                                            |

---

_If anything here is unclear, or a row is impossible to judge without seeing the page it came from,
please say so rather than working around it. We can send the surrounding lines for any row._
