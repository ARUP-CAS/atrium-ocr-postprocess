#!/usr/bin/env python3
"""
text_util.py

Purpose:
Provides the core text-processing utilities for the ALTO OCR post-processing pipeline.
This includes functions for detecting OCR noise, calculating character/symbol density,
scoring word "weirdness", and running text chunks through a GPU-accelerated Perplexity model.

Categories Outputted:
  - Empty     : A blank line.
  - Non-text  : Lines that are too short, lack letters, or are purely numbers/symbols.
  - Trash     : Severe OCR corruption, high symbol density, gibberish, or failed language ID.
  - Noisy     : Partially degraded text (e.g., isolated strange symbols, mid-word uppercase).
  - Clear     : Structurally sound text with low perplexity.

Those five, and only those five, are what this module writes into `lines[].categ`.
They are bound to the CATEG_* label constants below (collected in
CATEGORIES_EMITTED) and cross-checked at import time against the hub registry,
atrium_vocab.LINE_CATEGORY_ORIGINATORS["alto-postprocess"]. The same block's other
authorised originator, digital-convert, emits a DISJOINT set ({Garbage, Inverted});
that is deliberate and is not drift to reconcile.
"""

import configparser
import functools
import itertools
import os
import re
import sys
import unicodedata
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

# The empty lexicon, shared so that "no table configured" and "table unreadable"
# return the same object rather than two equal-but-distinct ones.
_EMPTY_LEXICON: Mapping[str, int] = MappingProxyType({})

# ---------------------------------------------------------------------------
# Line-category labels  (hub registry: atrium_vocab.LINE_CATEGORY_ORIGINATORS)
# ---------------------------------------------------------------------------
# The five strings below are the ONLY values this module ever writes into
# `lines[].categ`. They are spelled out here as literals, not derived from the
# registry, and that direction is deliberate:
#
#   * these strings ARE the emitted contract. Deriving them would make a stale or
#     re-ordered vendored copy of atrium_vocab.py silently change what the
#     pipeline outputs -- the registry is a description of behaviour, not a knob
#     that steers it;
#   * `lines[].categ` has a SECOND authorised originator (`digital-convert`, in
#     atrium-llm-enrich) which emits {Garbage, Inverted}. The two sets are
#     disjoint ON PURPOSE -- one is an OCR verdict over a rendered image, the
#     other a decode-sanity verdict over an embedded text layer -- so "align the
#     two" is never the fix for a disagreement found here.
#
# The registry's role is therefore advisory only: the check below reports a drift
# between this file and the hub declaration and then gets out of the way. See
# defect V-1 in the hub's docs/skos_strategy.md for why a consumer that filters
# this field must handle BOTH sets.
#
# Naming note: the CATEG_*_SCORE_MAX / CATEG_GARBAGE_DENSITY_HIGH constants further
# down are quality-score THRESHOLDS, not labels. Same prefix, different kind. And
# "Process", which pre_filter_line() also returns, is NOT one of these: it is the
# routing sentinel meaning "no verdict yet, score this line" (classify_TEXT.py gates
# on `cat != "Process"`), and it never reaches lines[].categ.
CATEG_EMPTY = "Empty"
CATEG_NON_TEXT = "Non-text"
CATEG_TRASH = "Trash"
CATEG_NOISY = "Noisy"
CATEG_CLEAR = "Clear"

#: Every value `determine_category()` / `categorize_line()` can return, sorted so a
#: comparison against the registry is order-independent. Consumers that need the
#: set (service/text_api.py's `lines[]` projection documents it) should read this
#: rather than re-typing the strings.
CATEGORIES_EMITTED: tuple = tuple(sorted((CATEG_CLEAR, CATEG_EMPTY, CATEG_NOISY, CATEG_NON_TEXT, CATEG_TRASH)))

# Advisory consistency check, NOT a gate. atrium_vocab is a vendored hub-canonical
# file and is legitimately absent in some execution contexts (a bare `text_util.py`
# copied next to a notebook, an image built before the vendor step). A missing
# registry must never break the pipeline, so this follows the house idiom used by
# atrium_document.py's origin check: abstain with a NOTE on stderr, never fatal.
# Silence here means agreement; nothing is printed on the happy path.
try:
    from atrium_vocab import LINE_CATEGORY_ORIGINATORS as _VOCAB_LINE_CATEGORY_ORIGINATORS
except ImportError:  # registry not vendored here - abstain, do not guess
    _VOCAB_LINE_CATEGORY_ORIGINATORS = None
else:
    _declared = tuple(sorted(_VOCAB_LINE_CATEGORY_ORIGINATORS.get("alto-postprocess", ())))
    if _declared and _declared != CATEGORIES_EMITTED:
        print(
            "[text_util] NOTE - line-category drift: this module emits "
            f"{list(CATEGORIES_EMITTED)} but atrium_vocab declares {list(_declared)} for "
            "originator 'alto-postprocess'. Emission is unchanged; reconcile the registry "
            "or this file (see defect V-1 in the hub's docs/skos_strategy.md).",
            file=sys.stderr,
        )
    del _declared

# ---------------------------------------------------------------------------
# Ablation Kill-Switch (Part B)
# Inject rule names here via override_constants to disable them for ablation sweeps.
# ---------------------------------------------------------------------------
DISABLED_RULES: frozenset = frozenset()

# ---------------------------------------------------------------------------
# Rule-Fire Coverage Instrumentation (Increment B5)
# When RULE_FIRE_COUNTS is not None (i.e. inside a rule_fire_capture() block),
# every _fire(name) call increments the counter for that rule. Outside a capture
# block _fire() is a no-op so there is zero overhead during normal production runs.
# ---------------------------------------------------------------------------
RULE_FIRE_COUNTS: dict | None = None


def _fire(name: str) -> None:
    """Register a single rule execution against the active capture context."""
    if RULE_FIRE_COUNTS is not None:
        RULE_FIRE_COUNTS[name] = RULE_FIRE_COUNTS.get(name, 0) + 1


@contextmanager
def rule_fire_capture():
    """Context manager that enables rule-fire counting for the enclosed block.

    Yields the live counts dict so callers can inspect it after (or during)
    the run.
    """
    global RULE_FIRE_COUNTS
    prev, RULE_FIRE_COUNTS = RULE_FIRE_COUNTS, {}
    try:
        yield RULE_FIRE_COUNTS
    finally:
        RULE_FIRE_COUNTS = prev


# ---------------------------------------------------------------------------
# Configuration & Regular Expressions
# ---------------------------------------------------------------------------

# (12-factor III) Config resolution order, highest precedence first:
#
#   1. environment  ATRIUM_<SECTION>_<KEY>   e.g. ATRIUM_TEXT_UTILS_SHORT_PPL_CAP=900
#   2. the INI file named by LANGID_CONFIG   (default: setup/config.txt)
#   3. the in-code default
#
# The env layer exists because the file was previously the ONLY way to change a
# value: LANGID_CONFIG names a *path*, not a value, so a deploy could not move a
# single threshold without editing a file inside its image or bind-mounting a
# replacement. The section is part of the variable name because keys are not
# unique across sections (WORKERS_MAX appears in both EXTRACT and CLASSIFY).
#
# Values are still read at import time, so an override must be set before the
# process starts. Derived constants (ROT_GHOSTLIST, _LANG_DIACRITICS) are built
# once from these and are not rebuilt by override_constants().
ENV_PREFIX = "ATRIUM_"

_config = configparser.RawConfigParser()
# Anchored to this file, NOT to the working directory. The default used to be the
# relative "setup/config.txt", so a process started anywhere but the repo root
# found nothing and ran every constant on its in-code default after one stderr
# line -- while tools/recategorize_from_csv.py resolved the SAME file absolutely
# via `_ROOT`, giving one process two different configurations.
#
# Measured 2026-09-10: 0 of 84 scalar constants currently differ between the file
# and the in-code defaults, so this is behaviour-neutral today. That is exactly
# why it is worth landing now -- the next round of runs exists to CHANGE those
# constants, and the bug goes live the moment setup/config.txt stops being a
# mirror of the defaults. A cluster job launched from a scheduler's working
# directory would then silently score with the old values.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "setup" / "config.txt"
_config_path = Path(os.getenv("LANGID_CONFIG", str(_DEFAULT_CONFIG_PATH)))

if _config_path.exists():
    _config.read(_config_path)
elif os.getenv("LANGID_CONFIG"):
    # Explicitly pointed somewhere that does not exist. Previously this fell
    # through to the in-code defaults in silence, so a typo'd path ran the whole
    # collection on defaults and looked exactly like a successful run.
    raise FileNotFoundError(
        f"LANGID_CONFIG points at {_config_path}, which does not exist. "
        f"Unset it to use the bundled setup/config.txt, or correct the path."
    )
else:
    # Running outside the repo root with no explicit path. Legitimate for a
    # library import, but the operator should know the defaults are in force.
    print(
        f"[config] {_config_path} not found - every constant is using its in-code default. "
        f"Set LANGID_CONFIG to silence this.",
        file=sys.stderr,
    )


def _env_override(section, key):
    """Return the raw env value for a section/key, or None if unset."""
    return os.getenv(f"{ENV_PREFIX}{section}_{key}")


def _get_float(section, key, default):
    raw = _env_override(section, key)
    if raw is not None:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{section}_{key}={raw!r} is not a float") from exc
    return _config.getfloat(section, key, fallback=default) if _config.has_section(section) else default


def _get_str(section, key, default):
    raw = _env_override(section, key)
    if raw is not None:
        return raw
    return _config.get(section, key, fallback=default) if _config.has_section(section) else default


def _get_int(section, key, default):
    raw = _env_override(section, key)
    if raw is not None:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{section}_{key}={raw!r} is not an int") from exc
    return _config.getint(section, key, fallback=default) if _config.has_section(section) else default


def _get_csv_set(section, key, default):
    """Parse a comma-separated config value into a frozenset of stripped tokens."""
    raw = _get_str(section, key, default)
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


# ---------------------------------------------------------------------------
# The hand-maintained word lists (#30)
# ---------------------------------------------------------------------------
#
# Defined HERE, high in the module, because the constants that read it are
# themselves module-level and are evaluated in source order. Everything below
# needs only `_get_str`, `os`, `Path`, `functools` and `MappingProxyType`, all of
# which exist by this point.

# (#30) Where the hand-maintained word lists live. This is the one file in the
# repository meant to be edited by the people who know the material rather than
# by the people who deploy the code: units, reference labels, section headings,
# and -- the section that did not exist before -- ordinary open-class words this
# archive uses that the program keeps getting wrong.
#
# Every section is a VETO. Nothing in that file can make the program convict a
# line; it can only stop it. Adding a word that was never at risk does nothing.
#
# EMPTY = fall back to the values compiled in below, which is exactly the
# behaviour that shipped before the file existed. A file that omits a section
# falls back for that section alone, so deleting a section cannot silently empty
# a list production depends on.
WORD_LISTS_PATH = _get_str("TEXT_UTILS", "WORD_LISTS_PATH", "setup/word_lists.txt").strip()
if WORD_LISTS_PATH and not os.path.isabs(WORD_LISTS_PATH):
    # Anchored to this module, not to the working directory -- the same fix
    # 30.plan.md records for the config path itself, made here before it bites.
    WORD_LISTS_PATH = str(Path(__file__).resolve().parent / WORD_LISTS_PATH)

#: The sections `setup/word_lists.txt` may define, and the in-code fallback for
#: each. The fallback is what the module used before the file existed, so an empty
#: `WORD_LISTS_PATH` — or a file that simply omits a section — is byte-identical to
#: the behaviour that shipped before 2026-09-22.
#:
#: `allowed` has no fallback and defaults to empty: it is the new open-class layer
#: (#30), and an archive that has not written one has none.
_WORD_LIST_SECTIONS: tuple[str, ...] = ("allowed", "neutral", "notation_labels", "header_labels")


@functools.lru_cache(maxsize=4)
def _read_word_lists(path: str, mtime: float) -> Mapping[str, frozenset]:
    """Parse the hand-maintained section file, keyed on path+mtime so edits are seen.

    Format, and it is deliberately the plainest thing that can hold several lists:
    ``[section]`` opens a section, one token per line, ``#`` starts a comment
    anywhere on a line, blanks are skipped, and everything is case-folded. A
    section that is absent or empty is simply absent from the result, and the
    caller falls back to its in-code default — so deleting a section cannot
    silently empty a list that production depends on.

    A file that cannot be read degrades to ``{}`` rather than raising. This is
    configuration, not input: a missing file must not stop the pipeline, and the
    fallbacks are the values that shipped for a year before the file existed.

    Cached on (path, mtime) rather than zero-argument, like ``_read_token_lexicon``
    above and for the same reason — a zero-argument cache would freeze a flag and
    is exactly the class of bug ``_CACHES_FROM_FLAG`` exists to close (#30 D28).
    """
    out: dict[str, set] = {}
    section: str | None = None
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip().lower()
                    out.setdefault(section, set())
                    continue
                if section is None:
                    continue
                out[section].add(line.lower())
    except OSError:
        return MappingProxyType({})
    return MappingProxyType({k: frozenset(v) for k, v in out.items() if v})


def word_list(section: str, fallback: frozenset, override: str = "") -> frozenset:
    """One section of the hand-maintained file, with the config key still on top.

    PRECEDENCE, and the middle layer is the new one:

        1. an explicit config/env value  -- `ATRIUM_TEXT_UTILS_<KEY>`, or the key
           in whatever `LANGID_CONFIG` points at;
        2. the `[section]` in `setup/word_lists.txt`;
        3. the in-code default.

    Layer 1 exists because moving a list into the file would otherwise TAKE AWAY
    an operator's ability to override it -- `tests/test_config_constants.py`'s
    tier-1 round-trip caught exactly that, by pointing `LANGID_CONFIG` at an
    alternate config and finding `ROT_WHITELIST` no longer followed it. The keys
    therefore stay in `setup/config.txt`, but EMPTY: empty means "not overridden,
    use the file", so the member list lives in one place while the override path
    stays open.

    Read at CALL time, not at import, so an operator editing the file does not
    have to restart a long-running service to see the change -- the mtime key on
    the cache above is what makes that cheap.
    """
    override = (override or "").strip()
    if override:
        return frozenset(t.strip() for t in override.split(",") if t.strip())
    path = (WORD_LISTS_PATH or "").strip()
    if not path:
        return fallback
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return fallback
    return _read_word_lists(path, mtime).get(section.lower(), fallback)


# (#30) The two [CLASSIFY] language fallbacks, named once.
#
# They used to be spelled out at three call sites -- here, classify_TEXT.main()
# and tools/recategorize_from_csv._load_lang_config() -- and they drifted: the
# offline copy was missing `slk` after the shipped config gained it, so a
# Slovak line reached the guards at TRUST_TIER_UNKNOWN (0.50) offline and
# TRUST_TIER_TRUSTED (0.85) in production. That is the same trust-tier class of
# divergence already fixed three times in this repository's test harnesses.
# Import these rather than retyping the strings.
DEFAULT_EXPECTED_LANGS = "ces,deu,eng"
DEFAULT_TRUSTED_FOREIGN_LANGS = "deu,eng,fra,pol,ita,slk"

COMMON_LANGS = [lang.strip() for lang in DEFAULT_EXPECTED_LANGS.split(",") if lang.strip()]
if _config.has_section("CLASSIFY") and _config.has_option("CLASSIFY", "EXPECTED_LANGS"):
    COMMON_LANGS = [lang.strip() for lang in _config.get("CLASSIFY", "EXPECTED_LANGS").split(",") if lang.strip()]

_TRUSTED_FOREIGN_LANG_BASES: frozenset = frozenset(
    lang.strip()
    for lang in _get_str("CLASSIFY", "TRUSTED_FOREIGN_LANGS", DEFAULT_TRUSTED_FOREIGN_LANGS).split(",")
    if lang.strip()
)


def _lang_base(lang_code: str) -> str:
    return lang_code.split("_")[0]


CZ_DIACS = frozenset(_get_str("TEXT_UTILS", "CZ_DIACS", "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"))

METADATA_MARKERS = frozenset(
    _get_str("TEXT_UTILS", "METADATA_MARKERS", "Tb.,č.neg,neg.,obr.,obr ,neg ,Tb ,č. neg,č.neg.,č.,str.,Datum").split(
        ","
    )
)

VOWEL_CHARS = frozenset(_get_str("TEXT_UTILS", "VOWEL_CHARS", "aeiouyáéíóúýěůäöüAEIOUYÁÉÍÓÚÝĚŮÄÖÜ"))
ROTATABLE_CHARS = frozenset(_get_str("TEXT_UTILS", "ROTATABLE_CHARS", "pbqdnuwmoxszeyv"))
WQX_CHARS = frozenset(_get_str("TEXT_UTILS", "WQX_CHARS", "wqxWQX"))
NONTEXT_MARKERS = _get_csv_set("TEXT_UTILS", "NONTEXT_MARKERS", "IVerc")
REMAP_KEEP_SCORE_LANGS = _get_csv_set("CLASSIFY", "REMAP_KEEP_SCORE_LANGS", "slk")


def has_cz_diacs(text: str) -> bool:
    """True if *text* contains at least one Czech diacritic glyph."""
    return any(ch in CZ_DIACS for ch in text)


_EXPECTED_LANGS_BASES: frozenset = frozenset(_lang_base(lng) for lng in COMMON_LANGS)

PERPLEXITY_THRESHOLD_MAX = _get_float("TEXT_UTILS", "PERPLEXITY_THRESHOLD_MAX", 1000.0)
SHORT_PPL_CAP = _get_float("TEXT_UTILS", "SHORT_PPL_CAP", 850.0)

LANG_SCORE_ROUGH = _get_float("TEXT_UTILS", "LANG_SCORE_ROUGH", 0.45)

# Core signal weights
QS_WEIGHT_VOWEL = _get_float("TEXT_UTILS", "QS_WEIGHT_VOWEL", 0.07)
QS_WEIGHT_LANG = _get_float("TEXT_UTILS", "QS_WEIGHT_LANG", 0.05)
QS_WEIGHT_GIBBERISH = _get_float("TEXT_UTILS", "QS_WEIGHT_GIBBERISH", 0.04)
QS_WEIGHT_FUSED = _get_float("TEXT_UTILS", "QS_WEIGHT_FUSED", 0.03)
QS_LENGTH_MAX = _get_float("TEXT_UTILS", "QS_LENGTH_MAX", 100.0)
QS_WEIGHT_VALID_WORD = _get_float("TEXT_UTILS", "QS_WEIGHT_VALID_WORD", 0.35)
QS_WEIGHT_WEIRD = _get_float("TEXT_UTILS", "QS_WEIGHT_WEIRD", 0.18)
QS_WEIGHT_PERPLEXITY = _get_float("TEXT_UTILS", "QS_WEIGHT_PERPLEXITY", 0.08)
QS_WEIGHT_LENGTH = _get_float("TEXT_UTILS", "QS_WEIGHT_LENGTH", 0.02)
QS_WEIGHT_GARBAGE = _get_float("TEXT_UTILS", "QS_WEIGHT_GARBAGE", 0.18)

CATEG_TRASH_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_TRASH_SCORE_MAX", 0.55)
CATEG_NOISY_SCORE_MAX = _get_float("TEXT_UTILS", "CATEG_NOISY_SCORE_MAX", 0.80)
CATEG_GARBAGE_DENSITY_HIGH = _get_float("TEXT_UTILS", "CATEG_GARBAGE_DENSITY_HIGH", 0.35)
QS_GARBAGE_NORM_MAX = _get_float("TEXT_UTILS", "QS_GARBAGE_NORM_MAX", 0.35)

# Inverted / 180°-rotated scan detection
ROT_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "ROT_RATIO_INVERTED_MIN", 0.55)
WEIRD_RATIO_INVERTED_MIN = _get_float("TEXT_UTILS", "WEIRD_RATIO_INVERTED_MIN", 0.35)
PPL_INVERTED_MIN = _get_float("TEXT_UTILS", "PPL_INVERTED_MIN", 200.0)
ROT_HIGH_LANG_CONF = _get_float("TEXT_UTILS", "ROT_HIGH_LANG_CONF", 0.90)

LOWPPL_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CLEAR_MAX", 50.0)
HARD_SWEEP_LANG_MAX = _get_float("TEXT_UTILS", "HARD_SWEEP_LANG_MAX", 0.45)
HARD_SWEEP_PPL_MIN = _get_float("TEXT_UTILS", "HARD_SWEEP_PPL_MIN", 1000.0)
GHOST_DOMINATED_MIN_RATIO = _get_float("TEXT_UTILS", "GHOST_DOMINATED_MIN_RATIO", 0.5)
WORD_W_PENALTY = _get_float("TEXT_UTILS", "WORD_W_PENALTY", 0.20)

INVERTED_RUN_MIN = _get_int("TEXT_UTILS", "INVERTED_RUN_MIN", 4)
INVERTED_PAGE_MAJORITY = _get_float("TEXT_UTILS", "INVERTED_PAGE_MAJORITY", 0.60)

SURROUNDED_TRASH_QS_MARGIN = _get_float("TEXT_UTILS", "SURROUNDED_TRASH_QS_MARGIN", 0.15)
PAGE_GARBAGE_CLEAR_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_CLEAR_MAX", 0.05)
PAGE_GARBAGE_LANG_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_LANG_MAX", 0.50)
PAGE_GARBAGE_MEDIAN_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_MEDIAN_QS_MAX", 0.55)
PAGE_GARBAGE_NOISY_QS_MAX = _get_float("TEXT_UTILS", "PAGE_GARBAGE_NOISY_QS_MAX", 0.80)
PAGE_CLEAN_CLEAR_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_CLEAR_MIN", 0.60)
PAGE_CLEAN_MEDIAN_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_MEDIAN_QS_MIN", 0.80)
PAGE_CLEAN_RECOVER_QS_MIN = _get_float("TEXT_UTILS", "PAGE_CLEAN_RECOVER_QS_MIN", 0.45)

TRASH_REASONS = frozenset({"trash_threshold", "trash_hard_sweep", "trash_inverted"})

MOSTLY_READABLE_VALID_MIN = _get_float("TEXT_UTILS", "MOSTLY_READABLE_VALID_MIN", 0.85)
SHORT_NOISY_QS_PENALTY = _get_float("TEXT_UTILS", "SHORT_NOISY_QS_PENALTY", 0.20)

LANG_SCORE_REMAP = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP", 0.75)
LANG_SCORE_REMAP_FAR = _get_float("TEXT_UTILS", "LANG_SCORE_REMAP_FAR", 0.50)
LANG_REMAP_ALWAYS = _get_str("TEXT_UTILS", "LANG_REMAP_ALWAYS", "true").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
SINGLE_CHAR_ALLOWED = _get_str("TEXT_UTILS", "SINGLE_CHAR_ALLOWED", "aAiIuUvVzZkKsS")
# ── Page-relative perplexity blend (default OFF) ────────────────────────────
# SHORT_PPL_CAP flattens perplexity to a constant for wc <= 2, and the capped
# value is what is both scored AND stored -- the raw LM number is discarded and
# is unrecoverable from a delivered CSV. That leaves short lines with no usable
# perplexity at all: 850 is below HARD_SWEEP_PPL_MIN (1000), PPL_EXTREME_MIN
# (3000) and PPL_GARBAGE_ABSOLUTE (30000), so no perplexity rule can fire on a
# one- or two-token line.
#
# Simply uncapping does not fix it: both surviving perplexity routes gate on LOW
# language-ID confidence, while garbage tokens score high (oueussd at 0.9163)
# and real domain words score low (malakofauna at 0.56). Uncapped, real notation
# is demoted around perplexity 3000 while oueussd survives to 30000.
#
# The blend instead reads a short line's perplexity RELATIVE to the long lines on
# its own page, which is the comparison a human makes. It is a page-consistency
# prior, not a garbage detector: on a mixed page it pulls a garbage token toward
# its clean neighbours. Off by default until calibrated on a real ARUP/ARUB run.
PAGE_PPL_BLEND_ENABLE = _get_str("TEXT_UTILS", "PAGE_PPL_BLEND_ENABLE", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
# Geometric blend weight on the line's own perplexity. Log-space is where the
# average has to happen: perplexities span six orders of magnitude here (II/C
# measures ~6e7), so an arithmetic blend is dominated by the tail.
PAGE_PPL_BLEND_WEIGHT = _get_float("TEXT_UTILS", "PAGE_PPL_BLEND_WEIGHT", 0.5)
# Minimum word count for a line to count toward the page reference.
PAGE_PPL_LONG_MIN_WC = _get_int("TEXT_UTILS", "PAGE_PPL_LONG_MIN_WC", 4)
# Minimum number of such lines before a page reference is trusted at all.
PAGE_PPL_MIN_LONG_LINES = _get_int("TEXT_UTILS", "PAGE_PPL_MIN_LONG_LINES", 3)

SHORT_VALID_WORDS_DEFAULT = frozenset(
    "a,i,k,o,s,u,v,z,se,si,po,na,za,ze,do,od,ke,ku,ve,ní,mi,ti,by,je,to,co,ač,my,ty,on,ji,jí,už,až".split(",")
)
#: Hand-editable in setup/word_lists.txt [short_valid] since 2026-09-22.
SHORT_VALID_WORDS = word_list("short_valid", SHORT_VALID_WORDS_DEFAULT, _get_str("TEXT_UTILS", "SHORT_VALID_WORDS", ""))

REPEAT_ALLOWED_CHARS = _get_str("TEXT_UTILS", "REPEAT_ALLOWED_CHARS", "oOuU")
REPEATED_DOUBLE_MIN = _get_int("TEXT_UTILS", "REPEATED_DOUBLE_MIN", 2)
VOWEL_RATIO_LOW = _get_float("TEXT_UTILS", "VOWEL_RATIO_LOW", 0.20)
VOWEL_RATIO_HIGH = _get_float("TEXT_UTILS", "VOWEL_RATIO_HIGH", 0.70)
ACADEMIC_TITLES = _get_csv_set(
    "TEXT_UTILS",
    "ACADEMIC_TITLES",
    "PhDr,MUDr,JUDr,MVDr,RNDr,PaedDr,CSc,DrSc,Ing,Mgr,Bc,PhD,DiS,prof,doc",
)

LDL_ALLOWED_FOLLOW = frozenset(_get_str("TEXT_UTILS", "LDL_ALLOWED_FOLLOW", ".,/:%-;?)="))
LDL_UNITS_DEFAULT: frozenset = frozenset("m,cm,mm,g,kg,km,ha,l,ml".split(","))
#: Hand-editable in setup/word_lists.txt [ldl_units] since 2026-09-22.
LDL_UNITS: frozenset = word_list("ldl_units", LDL_UNITS_DEFAULT, _get_str("TEXT_UTILS", "LDL_UNITS", ""))

SHORT_EXCEPTION_TOKENS_DEFAULT = frozenset("mm,cm,m,g,kg,km,ha,l,ml,tb,neg,obr,str,č,čneg".split(","))
#: Hand-editable in setup/word_lists.txt [short_exception] since 2026-09-22.
SHORT_EXCEPTION_TOKENS = word_list(
    "short_exception", SHORT_EXCEPTION_TOKENS_DEFAULT, _get_str("TEXT_UTILS", "SHORT_EXCEPTION_TOKENS", "")
)
HEADLINE_MAX_WORDS = _get_int("TEXT_UTILS", "HEADLINE_MAX_WORDS", 8)
HEADLINE_MAX_DIGITS = _get_int("TEXT_UTILS", "HEADLINE_MAX_DIGITS", 2)

GARBAGE_KEEP_CHARS = frozenset(_get_str("TEXT_UTILS", "GARBAGE_KEEP_CHARS", "")) | {" "}
FUSED_VOWEL_RUN_MIN = _get_int("TEXT_UTILS", "FUSED_VOWEL_RUN_MIN", 3)
WX_REPEAT_MIN = _get_int("TEXT_UTILS", "WX_REPEAT_MIN", 2)

ISOLATED_CHAR_RATIO_MAX = _get_float("TEXT_UTILS", "ISOLATED_CHAR_RATIO_MAX", 0.40)
ISOLATED_CHAR_MIN_TOKENS = _get_int("TEXT_UTILS", "ISOLATED_CHAR_MIN_TOKENS", 3)

# ── rule_short_garbage shape witness (issue #30) ────────────────────────────
# OFF by default, like PAGE_PPL_BLEND_ENABLE above: the predicate ships defined,
# unit-tested and measurable, and changes no category until someone turns it on
# against a gold set. See _has_shape_garbage_evidence() for what it tests and,
# just as importantly, what it deliberately does not.
SHORT_GARBAGE_WITNESS_ENABLE = _get_str("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_ENABLE", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)

# (#30 D35) Rules whose `_fire()` is gated by a CONFIG FLAG rather than by their
# own predicate, and the flag that gates each. A rule listed here reports
# `fire_count == 0` on every corpus while its flag is off -- not because it has
# no population, but because the branch is unreachable by configuration.
#
# This exists because `tools/rule_coverage_report.py` could not tell the two
# apart. Its `_classify()` reads `fire_count == 0` and returns DEAD, whose own
# docstring says the rule "is unreachable dead code and can be permanently
# deleted ... because deletion provably changes nothing". For
# `rule_short_garbage_witness` that is false in the most expensive possible way:
# the 2026-09-21 stage-6 sweep classified it DEAD and recommended retirement,
# while stage 08f had measured the same predicate reaching 100,824 lines across
# both collections and stage 08b had flipped the flag and passed the adoption
# gate (McNemar p = 0.01294). `RULE_COVERAGE.md` called `fire_count == 0` the
# "config-independent" retirement criterion; it is exactly not that.
#
# The taxonomy is not new -- `tests/test_pipeline_parity.py::UNREACHABLE_RULES`
# already separates "unreachable BY CONFIGURATION" from gate shadowing
# (`rule_mid_uppercase`, shadowed by gate 7). It lived only in a test, so no
# tool could read it. This is that distinction, sited next to the flag it is
# about, so a future flag-gated rule has one obvious place to declare itself.
#
# Values are the NAME of the module-level flag, resolved at read time rather
# than captured, so `override_constants()` is visible to readers of this map.
CONFIG_GATED_RULES: dict[str, str] = {
    "rule_short_garbage_witness": "SHORT_GARBAGE_WITNESS_ENABLE",
    "rule_domain_notation_categ": "DOMAIN_NOTATION_CATEG",
}


def rule_is_config_gated_off(rule: str) -> bool:
    """Is ``rule``'s fire site currently unreachable because its flag is off?

    Veto only, and deliberately narrow: it answers "can this rule fire at all in
    the configuration now in force", never "does this rule matter". A rule that
    is not in ``CONFIG_GATED_RULES`` always returns False, so the default answer
    is the honest one.
    """
    flag = CONFIG_GATED_RULES.get(rule)
    if flag is None:
        return False
    return not bool(globals().get(flag, False))


SHORT_GARBAGE_WITNESS_MIN_ALPHA = _get_int("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_MIN_ALPHA", 4)
SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA = _get_int("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA", 7)
SHORT_GARBAGE_WITNESS_VARIETY_MAX = _get_float("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_VARIETY_MAX", 0.50)
SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA = _get_int("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA", 8)
# (#30) The witness's OWN vowel-run length, decoupled from FUSED_VOWEL_RUN_MIN.
# It defaults to that value, so the shipped behaviour is unchanged -- but
# detect_fused_words() feeds `fused_ratio` in the quality score and the
# `fused_words` CSV column, so the two knobs steering each other meant the only
# clause that discriminates on the #30 population could not be tuned without
# moving scores on every line in the corpus. The trade at 3 vs 4 is measured on
# gold (stage 10d): errors 503 = 503, `Clear`-loss 38 = 38, fixes 2 / breaks 2,
# p = 1, `Trash`-recall 34 -> 32/180. A global 4 buys nothing and costs recall,
# so this stays 3 and the D44 split below is the answer.
SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN = _get_int("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN", FUSED_VOWEL_RUN_MIN)
# (#30 D44) The vowel-run clause is a fact about CZECH PHONOTACTICS, not about
# scanning. Czech has no triphthongs, so three vowels in a row is good evidence of
# damage -- in Czech. German and French have them natively, which is why the
# clause reaches `Dauerleihe` (*aue*, permanent loan) and `FEUILLETON` (*eui*),
# both scanned perfectly correctly, and why @david-spacil's answer on 2026-09-22
# was to split by language rather than to blunt the threshold:
#
#   "For Czech, 3+ vowels in a row is a good rule -- Czech has no triphthongs.
#    For German and French it does damage. So either split by language, or try 4+."
#
# Blunting it to 4 everywhere was measured and is the worse trade: it releases
# `J. Vysoean` (*Vysočan*), `POSTKRANIAINY SKELET`, `lenaye` and `noienm k.`,
# which are damage, to spare two German words -- 3.5 lines that currently agree
# with `Trash` given up per at-risk line spared (stage 10b).
#
# So: 3 vowels convict in any language EXCEPT these, and 4 convict everywhere,
# including these. Set EXEMPT_LANGS empty to get a single global threshold back.
# The language is the RAW FastText label, not the stored `lang` column -- see the
# note on `determine_category`'s `lang` parameter.
SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS: frozenset = _get_csv_set(
    "TEXT_UTILS", "SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS", "deu,fra"
)
# The run length required in an exempt language. 4 rather than "never": a German
# line can still be scanned into `oueussd`, and four vowels in a row is not a word
# in any of these languages either.
SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT = _get_int("TEXT_UTILS", "SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT", 4)
# (#30 D43) What category a recognised web or e-mail address gets.
#
# @david-spacil, 2026-09-22, on what the five categories mean: "`Trash` =
# illegible. Anything legible is `Clear`, easily decipherable is `Noisy` --
# regardless of how useful the line is to us." `http://www.arub.cz` is scanned
# perfectly correctly on 5,309 lines, so by that definition it is `Clear`. D33
# moved it Trash -> Noisy, the right direction and one step short.
#
# This is a CATEGORY NAME rather than an on/off switch on purpose. The question
# "what is a legible URL worth?" has had three different answers in this issue --
# `Trash` (before D33), `Noisy` (D33), `Clear` (W4) -- and a fourth is arguable
# (`Non-text`, since the README says that category "may be checked for
# identifiers of finds/sites"). Which one the archive wants is @DanaKriv's and
# @david-spacil's to settle, not a code change per answer. Set the key, re-score,
# compare.
#
# SHIPS EMPTY = the route is off and behaviour is exactly as before: the address
# falls through the cascade and lands wherever the ordinary rules put it.
# Any of the five category names arms it.
DOMAIN_NOTATION_CATEG = _get_str("TEXT_UTILS", "DOMAIN_NOTATION_CATEG", "").strip()
#: The `reason` each category is reported under, so the three threshold columns in
#: DOC_LINE_CATEG keep meaning what they say instead of all reading False on a
#: novel reason string. `Non-text` and `Empty` have no threshold column, and their
#: reasons are the ones the pre-filter already uses.
_DOMAIN_NOTATION_REASON: dict[str, str] = {
    CATEG_CLEAR: "clear_threshold",
    CATEG_NOISY: "noisy_threshold",
    CATEG_TRASH: "trash_threshold",
    CATEG_NON_TEXT: "non_text",
    CATEG_EMPTY: "empty",
}
if DOMAIN_NOTATION_CATEG and DOMAIN_NOTATION_CATEG not in _DOMAIN_NOTATION_REASON:
    raise ValueError(
        f"DOMAIN_NOTATION_CATEG={DOMAIN_NOTATION_CATEG!r} is not one of "
        f"{sorted(_DOMAIN_NOTATION_REASON)}. Leave it empty to keep the shipped behaviour, in "
        "which a recognised address is categorised by the ordinary cascade."
    )
# (#30 D14) The lexical signal for the residue. A path to a token/document-frequency
# table built by tools/build_token_lexicon.py. EMPTY BY DEFAULT: with no table the
# veto is inert and the predicate is byte-identical to the shape-only version, so
# this key changes nothing until an operator points it at a built table.
SHORT_GARBAGE_LEXICON_PATH = _get_str("TEXT_UTILS", "SHORT_GARBAGE_LEXICON_PATH", "").strip()
SHORT_GARBAGE_LEXICON_MIN_DF = _get_int("TEXT_UTILS", "SHORT_GARBAGE_LEXICON_MIN_DF", 3)
# (#30 D40, removed 2026-09-22) The de-gemination guard used to live here --
# SHORT_GARBAGE_LEXICON_GEMINATE_RATIO and _MAX_DF, a pair that let the witness
# convict a doubled-initial token ALTHOUGH the lexicon attested it. Removed on
# @david-spacil's answer ("if the dictionary already covers all eight, it looks
# redundant"), and because three of the eight tokens it was fitted to -- `ssuti`,
# `ssutí`, `ssutě` -- turned out to be an old spelling of `suť`, so the gap the
# ratio sat in had real language on BOTH sides of it. Zero of 42,853 rows in the
# full-scale witness queue carried any of the eight.
# (#30, 2026-09-19) Feed the corpus lexicon to `compute_valid_ratio` as its
# `word_set`. SHIPS FALSE: `valid_word_ratio` feeds `compute_quality_score` and
# every threshold under it, so this moves scores on every line in the corpus --
# the largest change this issue has proposed. Off, it is byte-identical.
QUALITY_VOCABULARY_ENABLE = _get_str("TEXT_UTILS", "QUALITY_VOCABULARY_ENABLE", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
# (#30, 2026-09-19) Strip bullet and marker glyphs alongside punctuation. The full
# table ends in `♦zkoumaná`, `✓stopy`, `•nevelká` -- a marker fused to a real
# word, which corrupts the lexicon at build time AND hides the real word at
# lookup time. SHIPS FALSE because it changes tokenisation for every line.
STRIP_SYMBOL_GLYPHS = _get_str("TEXT_UTILS", "STRIP_SYMBOL_GLYPHS", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)
#: The glyphs STRIP_SYMBOL_GLYPHS adds. Markers and bullets only -- no character
#: that can carry meaning inside an excavation code.
_SYMBOL_GLYPHS = "♦✓■□●○•▪▫★☆†‡§¶–—―‹›«»„“”‘’"
# (#30 D14) The lexicon used as EVIDENCE rather than as a veto: a token with no
# attestation anywhere in the collection convicts. This is the only mechanism in
# this module that can reach `edelite` -- the phonotactically legal residue -- and
# it is the only part of the witness that can ADD a conviction, so it is gated
# separately and ships false. Double-gated in practice: it is read only when
# SHORT_GARBAGE_WITNESS_ENABLE is also true, and only when a table is configured.
# Measured on gold as stage 5c and REJECTED: +44 `Trash` catches for +21 `Clear`
# losses, `Clear`-loss 42 -> 63 (docs/issue30/issue30_gold_ab_findings.md § 6). An
# unattested token is not the same thing as a non-word, and rare real vocabulary
# is the failure mode.
SHORT_GARBAGE_LEXICON_CONVICT = _get_str("TEXT_UTILS", "SHORT_GARBAGE_LEXICON_CONVICT", "false").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)


def _warn_uncoupled_witness() -> None:
    """Emit the configuration advisories once at import.

    Defined here, evaluated after the keys exist. Same house idiom as the
    atrium_vocab check at the top of this module: a NOTE on stderr, never fatal,
    and silence on the happy path -- which includes the shipped configuration,
    where the witness flag is false and no lexicon is configured.

    A tuple of one, deliberately: it held two until the de-gemination guard's
    scale advisory went with the guard (#30 D40), and the next configuration
    advisory belongs in it rather than in a second bespoke call site.
    """
    for warning in (uncoupled_witness_warning,):
        message = warning()
        if message:
            print(f"[text_util] NOTE - {message}", file=sys.stderr)


SYM_LET_DIG_NONTEXT = _get_str("TEXT_UTILS", "SYM_LET_DIG_NONTEXT", "true").strip().lower() in (
    "true",
    "1",
    "yes",
    "on",
)

ALLOWED_INTERNAL: frozenset = frozenset(_get_str("TEXT_UTILS", "ALLOWED_INTERNAL", ".-,+()\"'/—–:%;?!/"))
_STRIP_CHARS: str = _get_str("TEXT_UTILS", "STRIP_CHARS", ".,;:!?()[]\"'/\\")
# (#30 D27) Extended here rather than at the 28 call sites, so the builder and the
# predicate cannot drift apart -- the divergence this repository has been bitten by
# four times. Off by default: with the flag false this line is a no-op and
# tokenisation is byte-identical to every table already built.
if STRIP_SYMBOL_GLYPHS:
    _STRIP_CHARS = _STRIP_CHARS + _SYMBOL_GLYPHS

RE_TRASH_MULTI_SYMBOL: re.Pattern = re.compile(r"[^\w\s]{2,}")
RE_TRASH_LDL: re.Pattern = re.compile(r"[a-zA-Z][^a-zA-Z\s]+[a-zA-Z]")
RE_NON_TEXT: re.Pattern = re.compile(r"^[\d\s\-\u2013\u2014/:.,()%]+$")
RE_GARBAGE_CLUSTERS: re.Pattern = re.compile(r"[~=]|[\u00C0-\u017F]{2,}|[A-Z]=[A-Z]")
RE_ROMAN_NUMERAL: re.Pattern = re.compile(r"^[IVXLCDMivxlcdm]+\.?$")
RE_STAMP: re.Pattern = re.compile(r"^(?:[A-Za-z]+)?[\W_]*\d{2,4}\s*/\s*\d{2,4}[\W_]*$")
RE_ARCHIVE_CODE: re.Pattern = re.compile(r"^[A-Za-z]{1,3}\d{3,}(?:/\d+)?$")
RE_ALPHANUM_TOKEN: re.Pattern = re.compile(r"^[A-Za-z0-9]{5,}$")
RE_ARCHIVE_REF_SPACED: re.Pattern = re.compile(r"^[A-Za-záčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,5}[\s.\-]+\d{1,}")

_RE_SPACED_CAPS: re.Pattern = re.compile(
    r"(?<!\S)"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ] ){3,}"
    r"[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]"
    r"(?!\S)"
)


def _collapse_spaced_caps(m: re.Match) -> str:
    letters = m.group(0).replace(" ", "")
    return letters[0].upper() + letters[1:].lower()


DEU_DIACS = frozenset(_get_str("TEXT_UTILS", "DEU_DIACS", "äöüßÄÖÜ"))

_LANG_DIACRITICS: dict[str, frozenset] = {
    "ces": CZ_DIACS,
    "deu": DEU_DIACS,
}

DIACRITIC_INFER_THRESHOLD = _get_float("TEXT_UTILS", "DIACRITIC_INFER_THRESHOLD", 0.07)

PPL_EXTREME_MIN = _get_float("TEXT_UTILS", "PPL_EXTREME_MIN", 3000.0)
EXTREME_LANG_CONF = _get_float("TEXT_UTILS", "EXTREME_LANG_CONF", 0.85)
LOWPPL_CZECH_CLEAR_MAX = _get_float("TEXT_UTILS", "LOWPPL_CZECH_CLEAR_MAX", 180.0)
CZECH_CLEAR_GARBAGE_MAX = _get_float("TEXT_UTILS", "CZECH_CLEAR_GARBAGE_MAX", 0.15)

ANCHOR_MIN_WORDS = _get_int("TEXT_UTILS", "ANCHOR_MIN_WORDS", 2)
ANCHOR_WORD_LEN = _get_int("TEXT_UTILS", "ANCHOR_WORD_LEN", 3)
ANCHOR_VOWEL_RATIO = _get_float("TEXT_UTILS", "ANCHOR_VOWEL_RATIO", 0.10)

SUSPICIOUS_ROT_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_ROT_RATIO", 0.65)
SUSPICIOUS_WQX_RATIO = _get_float("TEXT_UTILS", "SUSPICIOUS_WQX_RATIO", 0.15)
INVERTED_WEIRD_PENALTY = _get_float("TEXT_UTILS", "INVERTED_WEIRD_PENALTY", 0.45)

PPL_GARBAGE_ABSOLUTE = _get_float("TEXT_UTILS", "PPL_GARBAGE_ABSOLUTE", 30000.0)
GHOST_HITS_INVERTED_MIN = _get_int("TEXT_UTILS", "GHOST_HITS_INVERTED_MIN", 1)
TRAILING_FILL_CHARS = (
    _get_str("TEXT_UTILS", "TRAILING_FILL_CHARS", "\\x20._:-<\\u2013\\u2014")
    .encode("latin-1", "backslashreplace")
    .decode("unicode_escape")
)

# ---------------------------------------------------------------------------
# Lexicon Integration for Rotation/Inversion detection
# ---------------------------------------------------------------------------

_MIRROR_GLYPH = {
    "b": "d",
    "d": "b",
    "p": "q",
    "q": "p",
    "a": "a",
    "e": "e",
    "i": "i",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
}
_ROTATE_GLYPH = {
    "b": "q",
    "q": "b",
    "d": "p",
    "p": "d",
    "h": "y",
    "n": "u",
    "u": "n",
    "m": "w",
    "w": "m",
    "y": "h",
    "a": "e",
    "e": "a",
    "i": "!",
    "l": "l",
    "o": "o",
    "s": "s",
    "x": "x",
    "z": "z",
}


def _transform_word(w: str, glyph_map: dict) -> str | None:
    out = []
    for ch in w:
        img = glyph_map.get(ch)
        if img is None:
            return None
        out.append(img)
    return "".join(reversed(out))


ROT_WHITELIST_DEFAULT: frozenset = frozenset(
    "po,pod,do,od,on,ony,by,bez,ne,nebo,ven,den,zde,se,ve,mez,pouze,bude".split(",")
)
#: Hand-editable in setup/word_lists.txt [rot_whitelist] since 2026-09-22.
ROT_WHITELIST: frozenset = word_list(
    "rot_whitelist", ROT_WHITELIST_DEFAULT, _get_str("TEXT_UTILS", "ROT_WHITELIST", "")
)
_GHOST_REAL_WORD_COLLISIONS_DEFAULT: frozenset = frozenset({"no", "bo"})
#: Hand-editable in setup/word_lists.txt [ghost_collisions] since 2026-09-22.
_GHOST_REAL_WORD_COLLISIONS: frozenset = word_list(
    "ghost_collisions", _GHOST_REAL_WORD_COLLISIONS_DEFAULT, _get_str("TEXT_UTILS", "GHOST_WORD_COLLISIONS", "")
)


def _build_ghostlist() -> frozenset:
    ghosts = set()
    for w in ROT_WHITELIST:
        for img in (_transform_word(w, _MIRROR_GLYPH), _transform_word(w, _ROTATE_GLYPH)):
            if img:
                ghosts.add(img)
    return frozenset(ghosts - ROT_WHITELIST - _GHOST_REAL_WORD_COLLISIONS)


ROT_GHOSTLIST: frozenset = _build_ghostlist()


def analyze_rotation_signals(text: str) -> tuple[bool, bool]:
    words = [w.lower() for w in re.split(r"\W+", text) if w]
    if not words:
        return has_cz_diacs(text), False

    real_hits = sum(1 for w in words if w in ROT_WHITELIST)
    ghost_hits = sum(1 for w in words if w in ROT_GHOSTLIST)

    is_upright_czech = has_cz_diacs(text) or real_hits > 0

    ghost_share = ghost_hits / len(words)
    ghost_dominated = ghost_hits > 0 and ghost_share >= GHOST_DOMINATED_MIN_RATIO
    return is_upright_czech, ghost_dominated


def ghost_word_share(text: str) -> tuple[int, float]:
    words = [w.lower() for w in re.split(r"\W+", text) if w]
    if not words:
        return 0, 0.0
    ghost_hits = sum(1 for w in words if w in ROT_GHOSTLIST)
    return ghost_hits, ghost_hits / len(words)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_mid_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper():
        return False
    if core.rstrip(".") in ACADEMIC_TITLES:
        return False

    caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
    if caps_run >= 2 and any(c.islower() for c in core[caps_run:]):
        return True

    for i in range(1, len(core)):
        if core[i].isupper() and core[i - 1].islower():
            return True

    return False


def _has_starting_uppercase(core: str) -> bool:
    if len(core) < 2 or core.isupper():
        return False
    if core.rstrip(".") in ACADEMIC_TITLES:
        return False
    return core[0].isupper() and core[1].isupper()


def _split_subtokens(word: str) -> list[str]:
    return [p for p in re.split(r"[.\-\u2013]", word) if p]


def remap_lang(
    label: str, score: float, known_bases: frozenset, default_lang: str, remap_floor: float = LANG_SCORE_REMAP
) -> tuple[str, float]:
    base = _lang_base(label)
    if base in known_bases:
        return label, score
    suffix = label[len(base) :]
    new_label = default_lang + suffix
    if base in REMAP_KEEP_SCORE_LANGS:
        return new_label, score
    cap = remap_floor if suffix == "_Latn" else LANG_SCORE_REMAP_FAR
    if LANG_REMAP_ALWAYS or score > cap:
        return new_label, cap
    return new_label, score


def compute_garbage_density(text: str) -> float:
    if not text:
        return 0.0
    noise_chars = sum(1 for c in text if not c.isalnum() and c not in GARBAGE_KEEP_CHARS)
    return noise_chars / len(text)


def _has_repeated_run(core: str) -> bool:
    if len(core) < 4:
        return False
    for ch in set(core):
        if ch.isdigit():
            continue
        if ch * 3 in core:
            return True
        if ch in REPEAT_ALLOWED_CHARS:
            continue
        if ch * 2 in core and core.count(ch) >= REPEATED_DOUBLE_MIN:
            return True
        if (core.count(ch) / len(core) >= 0.30) and core.count(ch) >= 3:
            return True
    return False


def _trailing_alpha_run(token: str, start: int) -> str:
    j = start
    while j < len(token) and token[j].isalpha():
        j += 1
    return token[start:j]


def has_symbol_letter_digit(word: str) -> bool:
    has_letter = any(c.isalpha() for c in word)
    has_digit = any(c.isdigit() for c in word)
    has_symbol = any((not c.isalnum()) and not c.isspace() and c not in ALLOWED_INTERNAL for c in word)
    return has_letter and has_digit and has_symbol


# ---------------------------------------------------------------------------
# Structural Text-Quality Detectors
# ---------------------------------------------------------------------------


def infer_lang_from_diacritics(text: str, expected_bases: frozenset, threshold: float | None = None) -> str | None:
    if threshold is None:
        threshold = DIACRITIC_INFER_THRESHOLD
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return None
    for lang_code, diacs in _LANG_DIACRITICS.items():
        if lang_code not in expected_bases:
            continue
        ratio = sum(1 for c in alpha if c in diacs) / len(alpha)
        if ratio >= threshold:
            return lang_code
    return None


def compute_rotatable_ratio(text: str) -> float:
    alpha_chars = [c.lower() for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    rotatable_count = sum(1 for c in alpha_chars if c in ROTATABLE_CHARS)
    return rotatable_count / len(alpha_chars)


def detect_strange_symbols(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        count += sum(1 for ch in core if not ch.isalnum() and ch not in ALLOWED_INTERNAL)
    return count


def detect_repeated_chars(text: str) -> int:
    count = 0
    for word in text.split():
        if any(_has_repeated_run(sub.strip(_STRIP_CHARS)) for sub in _split_subtokens(word)):
            count += 1
    return count


def compute_vowel_ratio(text: str) -> float:
    denom = [c for c in text if c.isalpha() or ((not c.isalnum()) and not c.isspace())]
    if not denom:
        return 0.0
    return sum(1 for c in denom if c in VOWEL_CHARS) / len(denom)


def _is_allowed_token(core: str) -> bool:
    """Is this token on the archive's hand-maintained allow list? (#30)

    Reads `setup/word_lists.txt` `[allowed]`. It shipped EMPTY until 2026-10-01;
    since then it carries @david-spacil's reviewed entries (#30 Q5a, PR #7), so
    every caller below is live for those words and for nothing else.

    WHAT IT MEANS, TWICE OVER. (1) The shape witness never reads it as evidence
    of damage: `shape_garbage_clauses()` skips it (#30 Q5b, answered yes on
    2026-10-01). (2) The token contributes NOTHING to the line's quality score. It
    stops counting toward the invalid-word, weird-word, gibberish and fused-word
    measurements -- which together are 0.60 of the score's weight. It does not
    count as a GOOD word either: `compute_valid_ratio` treats it as non-evaluable,
    the way `_is_neutral_token` already treats a unit. "Not debuffed" is the ask;
    actively raising the ratio would be more than the ask.

    MATCHING IS EXACT, case-folded and `_STRIP_CHARS`-stripped, and deliberately
    does NOT fold diacritics -- unlike `_NOTATION_LABELS_FOLDED`, which does.
    Folding here would mean that listing `jáma` also excuses `jama`, i.e. the
    accent-stripped form OCR produces when it fails. That is the damaged reading,
    and a list of words the archive says are real should not quietly also cover
    the ways they come out wrong. An archive that wants both lists both.
    """
    lst = word_list("allowed", frozenset())
    if not lst:
        return False
    return core.strip(_STRIP_CHARS).lower() in lst


def detect_gibberish_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if _is_allowed_token(core):
                continue
            if len(core) < 4 or core.isupper():
                continue
            numeric_chars = sum(1 for c in core if c.isdigit() or c in "-./,;:")
            if numeric_chars / len(core) >= 0.6:
                continue
            letters = [c for c in core if c.isalpha()]
            if not letters:
                continue
            if sum(1 for c in letters if c in VOWEL_CHARS) / len(letters) > VOWEL_RATIO_HIGH:
                flagged = True
                break
        if flagged:
            count += 1
    return count


def _has_ldl(token: str) -> bool:
    n = len(token)
    for i, ch in enumerate(token):
        if not ch.isdigit():
            continue
        nxt = token[i + 1] if i + 1 < n else ""
        prev = token[i - 1] if i > 0 else ""
        if nxt and not nxt.isspace() and not nxt.isdigit() and nxt not in LDL_ALLOWED_FOLLOW:
            if nxt.isalpha():
                run = _trailing_alpha_run(token, i + 1)
                if run.lower() in LDL_UNITS:
                    continue
            return True
        if prev.isalpha():
            return True
    return False


def detect_letter_digit_letter(text: str) -> int:
    return sum(1 for word in text.split() if _has_ldl(word))


def detect_mid_uppercase(text: str) -> int:
    count = 0
    for word in text.split():
        core = word.strip(".,;:!?()[]\"'-/")
        if _is_mid_uppercase(core):
            count += 1
    return count


def detect_wx_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core or _is_allowed_token(core):
                continue
            if sum(1 for c in core if c in "wW") >= WX_REPEAT_MIN or sum(1 for c in core if c in "xX") >= WX_REPEAT_MIN:
                flagged = True
                break
        if flagged:
            count += 1
    return count


def is_all_caps_line(text: str) -> bool:
    alpha_words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False
    return all(w.isupper() for w in alpha_words)


_RE_FUSED_CONSONANT_RUN: re.Pattern = re.compile(r"[bcčdfghjklmnpqrřsštvwxzž]{5,}", re.IGNORECASE)
_RE_FUSED_VOWEL_RUN: re.Pattern = re.compile(r"[aeiouyáéíóúýěůäöü]{%d,}" % FUSED_VOWEL_RUN_MIN, re.IGNORECASE)


def detect_fused_words(text: str) -> int:
    count = 0
    for word in text.split():
        flagged = False
        for sub in _split_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            if not core or not any(c.isalpha() for c in core) or _is_allowed_token(core):
                continue
            if len(core) > 14 or _RE_FUSED_CONSONANT_RUN.search(core) or _RE_FUSED_VOWEL_RUN.search(core):
                flagged = True
                break
        if flagged:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-filtering & Parsing
# ---------------------------------------------------------------------------


def pre_filter_line(line: str) -> tuple[str, str]:
    clean_text = line.strip()
    if not clean_text:
        return "Empty", ""

    clean_text = re.sub(
        r"(?<=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])1(?=[a-záčďéěíňóřšťůúýžA-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ])", "l", clean_text
    )
    clean_text = re.sub(r"(?<![\d.,])\b2(?=[a-záčďéěíňóřšťůúýž])", "z", clean_text)
    clean_text = _RE_SPACED_CAPS.sub(_collapse_spaced_caps, clean_text)

    if any(marker.lower() in clean_text.lower() for marker in METADATA_MARKERS):
        return "Process", clean_text

    if is_forgiven_headline(clean_text, compute_garbage_density(clean_text)):
        return "Process", clean_text

    if clean_text.startswith('"') and not clean_text.endswith('"'):
        clean_text += '"'
    elif clean_text.endswith('"') and not clean_text.startswith('"'):
        clean_text = '"' + clean_text

    n_chars = len(clean_text)

    if is_non_text(clean_text):
        return "Non-text", clean_text
    if RE_ROMAN_NUMERAL.match(clean_text.strip()):
        return "Non-text", clean_text
    if RE_STAMP.search(clean_text) or any(m in clean_text for m in NONTEXT_MARKERS):
        return "Non-text", clean_text

    tokens = clean_text.split()
    valid_long_words = sum(
        1
        for tok in tokens
        if len(tok.strip(_STRIP_CHARS)) >= ANCHOR_WORD_LEN
        and tok.strip(_STRIP_CHARS).isalpha()
        and compute_vowel_ratio(tok.strip(_STRIP_CHARS)) >= ANCHOR_VOWEL_RATIO
    )
    if valid_long_words >= ANCHOR_MIN_WORDS:
        return "Process", clean_text

    if sum(c.isdigit() for c in clean_text) / n_chars > 0.4:
        return "Process", clean_text

    unique_symbols = set(c for c in clean_text if not c.isspace())
    if n_chars < 4 or len(unique_symbols) < 3:
        return "Non-text", clean_text

    letters = sum(c.isalpha() for c in clean_text)
    if letters / n_chars < 0.3:
        return "Non-text", clean_text

    if SYM_LET_DIG_NONTEXT and len(tokens) == 1 and has_symbol_letter_digit(tokens[0]):
        return "Non-text", clean_text

    if len(tokens) >= ISOLATED_CHAR_MIN_TOKENS:
        alpha_tokens = [tok for tok in tokens if any(c.isalpha() for c in tok)]
        if alpha_tokens:
            valid_singles = frozenset(SINGLE_CHAR_ALLOWED)
            single_char_tokens = [
                tok for tok in alpha_tokens if len(tok.strip(_STRIP_CHARS)) == 1 and tok.strip(_STRIP_CHARS).isalpha()
            ]
            invalid_singles = [tok for tok in single_char_tokens if tok.strip(_STRIP_CHARS) not in valid_singles]

            is_pure_isolated = len(single_char_tokens) == len(alpha_tokens)
            high_isolated_ratio = (len(invalid_singles) / len(alpha_tokens)) >= ISOLATED_CHAR_RATIO_MAX

            if is_pure_isolated or high_isolated_ratio:
                run_length = 0
                collapsed_spans = []
                current_span = []

                for tok in tokens:
                    core = tok.strip(_STRIP_CHARS)
                    if len(core) == 1 and core.isalpha():
                        run_length += 1
                        current_span.append(core)
                    else:
                        if run_length >= 3:
                            collapsed_spans.append("".join(current_span))
                        run_length = 0
                        current_span = []
                if run_length >= 3:
                    collapsed_spans.append("".join(current_span))

                rescued = False
                for span in collapsed_spans:
                    if compute_vowel_ratio(span) > 0.15 and compute_garbage_density(span) < 0.20:
                        rescued = True
                        break

                if not rescued:
                    return "Non-text", clean_text

    return "Process", clean_text


def parse_line_splits(line_text: str) -> tuple[str, str, str]:
    clean_line = line_text.strip()
    pattern = r"(\S+)(?:-|­|\xad)\s*\{([^}]+)\}"
    matches = list(re.finditer(pattern, clean_line))
    if not matches:
        return clean_line, "", ""
    last_prefix = last_suffix = ""

    def replace_match(match):
        nonlocal last_prefix, last_suffix
        prefix = match.group(1)
        content = match.group(2)
        last_prefix = prefix
        last_suffix = content[len(prefix) :] if content.startswith(prefix) else ""
        return content

    merged_text = re.sub(pattern, replace_match, clean_line)
    return merged_text, last_prefix, last_suffix


# ---------------------------------------------------------------------------
# Per-Word Weirdness Scoring
# ---------------------------------------------------------------------------


def score_word(word: str) -> float:
    core = word.strip(_STRIP_CHARS)
    # (#30) The archive's own allow list, before any shape test. A word a person
    # has vouched for carries no weirdness, whatever it looks like -- which is the
    # point, since every word on that list is there BECAUSE it looks wrong.
    if _is_allowed_token(core):
        return 0.0
    if len(core) == 1:
        if core in SINGLE_CHAR_ALLOWED or "." in word:
            return 0.0
        if core.isdigit():
            return 0.25
        if not core.isalpha():
            return 0.0
        return 0.85
    if len(core) < 2:
        return 0.0

    has_strange = any(not ch.isalnum() and ch not in ALLOWED_INTERNAL for ch in core)
    has_rep = _has_repeated_run(core)
    has_ldl = _has_ldl(core)
    has_uppercase = _is_mid_uppercase(core)
    has_wqx = any(c in WQX_CHARS for c in core)

    has_caps_prefix = False
    if len(core) >= 4 and not core.isupper() and core.rstrip(".") not in ACADEMIC_TITLES:
        caps_run = sum(1 for _ in itertools.takewhile(str.isupper, core))
        if caps_run >= 2 and any(c.islower() for c in core[caps_run:]):
            has_caps_prefix = True

    alpha_chars = [c for c in core if c.isalpha()]
    is_vowelless_long = (
        len(alpha_chars) >= 3
        and not any(c in VOWEL_CHARS for c in alpha_chars)
        and core.rstrip(".") not in ACADEMIC_TITLES
    )

    return min(
        1.0,
        0.40 * has_strange
        + 0.35 * has_rep
        + 0.15 * has_ldl
        + 0.25 * has_uppercase
        + 0.20 * has_caps_prefix
        + WORD_W_PENALTY * has_wqx
        + 0.50 * is_vowelless_long,
    )


def score_words_in_line(text: str) -> list[tuple[str, float]]:
    is_upright, ghost_dom = analyze_rotation_signals(text)
    rot_ratio = compute_rotatable_ratio(text)

    words = text.split()
    wqx_words = sum(1 for w in words if any(c in WQX_CHARS for c in w))
    wqx_ratio = wqx_words / len(words) if words else 0.0

    is_suspicious_rot = rot_ratio > SUSPICIOUS_ROT_RATIO and wqx_ratio >= SUSPICIOUS_WQX_RATIO and not is_upright

    frag_count = sum(1 for w in words if w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2)
    frag_ratio = frag_count / len(words) if words else 0.0

    is_highly_fragmented = frag_ratio > 0.60 and len(words) >= 4

    results = []
    for w in words:
        s = score_word(w)
        if (ghost_dom or is_suspicious_rot) and not is_upright:
            s = min(1.0, s + INVERTED_WEIRD_PENALTY)

        if is_highly_fragmented:
            core = w.strip(_STRIP_CHARS)
            if core.isdigit() or len(core) <= 2:
                s = min(1.0, s + 0.35)

        results.append((w, s))

    return results


def compute_word_weird_ratio(word_scores: list[tuple[str, float]]) -> float:
    if not word_scores:
        return 0.0
    return sum(s for _, s in word_scores) / len(word_scores)


# ---------------------------------------------------------------------------
# Perplexity (GPU batch)
# ---------------------------------------------------------------------------


def calculate_perplexity_batch(texts: list[str], model, tokenizer, device) -> list[float]:
    import torch
    from torch import nn

    if not texts:
        return []
    try:
        max_length = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024))
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        target_ids = input_ids.clone()
        target_ids[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(target_ids.size(0), -1)
            non_masked = shift_labels != -100
            seq_loss = (loss * non_masked).sum(dim=1)
            num_tokens = non_masked.sum(dim=1).clamp(min=1)
            ppl = torch.exp(seq_loss / num_tokens)
            return ppl.tolist()
    except Exception as e:
        print(f"[Error] Batch PPL ({len(texts)} lines) failed: {e}", file=sys.stderr, flush=True)
        return [99999.0] * len(texts)


# ---------------------------------------------------------------------------
# Categorisation & Clamping
# ---------------------------------------------------------------------------


def _lm_confident_czech(is_upright_czech, ppl, garbage_density):
    return is_upright_czech and ppl < LOWPPL_CZECH_CLEAR_MAX and garbage_density < CZECH_CLEAR_GARBAGE_MAX


def _trailing_fill_rescued(text_source: str, valid_word_ratio: float, word_count: int) -> bool:
    if valid_word_ratio <= 0.0:
        return False
    core = text_source.rstrip(TRAILING_FILL_CHARS)
    if not core or core == text_source:
        return False
    if compute_garbage_density(core) >= CATEG_GARBAGE_DENSITY_HIGH:
        return False
    return has_cz_diacs(core) or (word_count <= 4 and len(text_source) <= 25)


def is_forgiven_headline(text: str, garbage_density: float) -> bool:
    """(#3 2026-07-02 calibration) Recognise short numbered headlines/captions
    (``"2, Popis nálezu i - 3"``, ``"Plánek č. 1"``) and bare domain
    abbreviations (``mm``, ``Tb.``, ``č.neg.``) that would otherwise mis-route
    to Trash/Non-text purely because the digits/symbols around one or two real
    words drag ``valid_word_ratio`` down.

    Every token is classified as one of:
      * NUMBERING  — a pure digit (short numbering only, see
        ``HEADLINE_MAX_DIGITS``) or a roman numeral. Supplies *context*.
      * ABBREV     — a known unit/abbreviation (``SHORT_EXCEPTION_TOKENS``), an
        academic title, or a ``METADATA_MARKERS`` marker. Supplies both
        *content* and *context* (a bare ``mm`` line qualifies on its own).
      * FUNCTION   — a whitelisted short Czech word (``SHORT_VALID_WORDS`` /
        ``SINGLE_CHAR_ALLOWED``). Real *content*, but no context by itself.
      * CLEAN WORD — passes the same acceptance test as ``compute_valid_ratio``'s
        inner branch, plus a vowel-bearing check. Real *content*, no context.
        Multi-token lines only: a single bare "clean-looking" word is exactly the
        profile of an inverted-scan / short-garbage token (``oueussd``, ``olie``)
        that rule_inverted / rule_short_garbage exist to catch.
      * STRUCTURAL — pure punctuation: no information either way.
      * GARBAGE    — anything else, and disqualifies the whole line.

    A line is forgiven only when it carries BOTH real *content* (a clean word,
    abbreviation, or function word) AND genuine numbering/abbreviation *context*
    (a digit, roman numeral, or domain abbreviation). Requiring the context term
    is what keeps a bare short prose fragment (``"popel dřevo kůstky"``) — no
    numbering, no abbreviation — out of the forgiveness path; those must route on
    their own quality score, exactly as before this pass. Every DanaKriv example
    carries such context (``2, ...``, ``4. ...``, ``Plánek č. 1``, ``mm``).

    Deliberately tight: a single OCR-mangled token (``oAOrt``, ``vyt1ačená``)
    or an over-long digit run (an archive/stamp code, not a caption number)
    disqualifies the line, so genuine garbage is never rescued.
    """
    tokens = text.split()
    if not tokens or len(tokens) > HEADLINE_MAX_WORDS:
        return False
    if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
        return False

    multi_token = len(tokens) >= 2
    has_content = False  # a clean word, abbreviation, or function word
    has_context = False  # numbering (digit / roman) or a domain abbreviation
    for tok in tokens:
        core = tok.strip(_STRIP_CHARS)

        # STRUCTURAL — pure punctuation (no alnum at all) carries no
        # information either way.
        if not core or not any(c.isalnum() for c in core):
            continue
        # NUMBERING — short numbering only; longer digit runs are archive/stamp
        # codes, not caption numbers.
        if core.isdigit():
            if len(core) > HEADLINE_MAX_DIGITS:
                return False
            has_context = True
            continue

        normalized = core.lower().replace(".", "").replace(",", "")

        # ABBREV — a domain unit/marker/title supplies both content and context,
        # so a bare "mm" / "Tb." / "č.neg." line qualifies on its own.
        if (
            normalized in SHORT_EXCEPTION_TOKENS
            or core.rstrip(".") in ACADEMIC_TITLES
            or any(marker.lower() in tok.lower() for marker in METADATA_MARKERS)
        ):
            has_content = True
            has_context = True
            continue

        # NUMBERING — roman numeral (checked after ABBREV so real abbreviations
        # built only of I/V/X/L/C/D/M aren't misread as numbering). A lone
        # ambiguous glyph ("v", "i", "l", ...) is a Czech function word, not a
        # numeral, so genuine roman numbering needs at least two glyphs.
        if len(core.rstrip(".")) >= 2 and RE_ROMAN_NUMERAL.match(core):
            has_context = True
            continue

        # FUNCTION — a whitelisted short Czech word / single char is real
        # content, but is NOT numbering/abbreviation context on its own.
        if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
            has_content = True
            continue

        # CLEAN WORD — multi-token lines only (see docstring).
        if multi_token:
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if (
                len(core) >= 3
                and alpha / len(core) >= 0.70
                and not has_strange
                and not _is_mid_uppercase(core)
                and compute_vowel_ratio(core) > 0.0
            ):
                has_content = True
                continue

        # GARBAGE
        return False

    return has_content and has_context


def inspect_short_line_telemetry(
    text_source: str,
    word_count: int,
    valid_word_ratio: float,
    lang_score: float,
    perplexity: float,
    weird_ratio: float = 0.0,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
) -> dict:
    """
    Step 4 Telemetry Helper: Audit short lines (word_count <= 2) to log properties
    and identify potential false Clear/Noisy promotions.
    """
    structured = is_structured_line(text_source)
    damaged = count_damaged_tokens(text_source) > 0
    forgiven = is_forgiven_headline(text_source, garbage_density)

    category, score, reason = categorize_line(
        qs=compute_quality_score(
            valid_word_ratio=valid_word_ratio,
            perplexity=perplexity,
            text_length=len(text_source),
            weird_ratio=weird_ratio,
            garbage_density=garbage_density,
            lang_score=lang_score,
            is_upright_czech=is_upright_czech,
        ),
        txt=text_source,
        wc=word_count,
        vowel_ratio=compute_vowel_ratio(text_source),
        perplexity=perplexity,
        weird_ratio=weird_ratio,
        valid_word_ratio=valid_word_ratio,
        lang_score=lang_score,
        garbage_density=garbage_density,
        is_upright_czech=is_upright_czech,
        return_reason=True,
    )

    return {
        "text": text_source,
        "word_count": word_count,
        "valid_word_ratio": valid_word_ratio,
        "lang_score": lang_score,
        "perplexity": perplexity,
        "weird_ratio": weird_ratio,
        "garbage_density": garbage_density,
        "structured": structured,
        "damaged": damaged,
        "forgiven_headline": forgiven,
        "is_upright_czech": is_upright_czech,
        "final_category": category,
        "quality_score": score,
        "route_reason": reason,
    }


def determine_category(
    qs: float,
    text_source: str,
    word_count: int,
    vr: float,
    ppl: float,
    weird_ratio: float = 0.0,
    valid_word_ratio: float = 1.0,
    lang_score: float = 1.0,
    orig_lang_score: float = 1.0,
    gibberish_present: bool = False,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
    ghost_dominated: bool = False,
    lang: str | None = None,
) -> tuple[str, str]:
    """`lang` is the RAW detected language label, and only the vowel-run clause reads it.

    (#30 D44.) Pass `original_lang` -- what FastText actually said -- and not the
    stored `lang` column. That column has been through `remap_lang()`, which
    rewrites any base outside EXPECTED_LANGS + TRUSTED_FOREIGN_LANGS to Czech, so
    feeding it here would let a remap decide a phonotactic question. None is the
    honest default for a caller that does not know, and it applies the general
    threshold rather than an exemption.
    """
    if word_count == 0 or not text_source.strip():
        return "Empty", "empty"

    stripped = text_source.strip()
    rot_ratio = compute_rotatable_ratio(text_source)
    words = text_source.split()

    structured = is_structured_line(text_source)

    # (#30) Notation is exempt from the two PERPLEXITY-ONLY Trash routes below.
    # For this class the LM score is not a quality signal at all, and is in fact
    # inverted: `II/C` measures ~6e7 perplexity and is correct, `oueussd` ~4600
    # and is garbage. Leaving the routes blind to it meant that the moment
    # perplexity became real -- SHORT_PPL_CAP raised, or the page blend enabled
    # -- section 1 trashed every grid reference before `rule_short_garbage` was
    # reached. Computed once here and reused at gate 6.
    #
    # `rule_hard_sweep` is deliberately NOT exempted. It is the only one of the
    # three that requires a second, independent witness: `orig_lang_score <
    # HARD_SWEEP_LANG_MAX` means FastText also failed to place the line. Exempt
    # the routes whose sole evidence we have just disowned; keep the one that
    # corroborates. Measured on dot-mutated garbage with the cap off, exempting
    # all three let 65% escape to Noisy/Clear, while leaving hard sweep armed
    # trashed 600 of 600 and cost nothing on the pinned notation shapes.
    notation = "rule_domain_notation" not in DISABLED_RULES and is_domain_notation(text_source)

    # ------------------------------------------------------------
    # 0b. Web and e-mail addresses -- the category is chosen in config
    # ------------------------------------------------------------
    # (#30 D43) SHIPS EMPTY = this block is skipped and nothing changes.
    #
    # A recognised address gets whatever category `DOMAIN_NOTATION_CATEG` names,
    # and EVERY address the pattern catches gets it -- not only the short ones.
    # That is the point: `http://www.arub.cz` and
    # `roku 1820 (http://www.hrady.cz/index.php?OID=1291).` are the same kind of
    # thing to a reader, and today they land in different categories for reasons
    # that are about token counts rather than about the address.
    #
    # WHY IT IS FIRST, above even the hard sweep. Every signal below this line
    # measures how word-like a string is, and an address is not trying to be a
    # word: `weird_ratio` is 1.00 on `http://www.arub.cz` because every character
    # a URL needs past the letters is a symbol, `detect_fused_words` fires on the
    # address shape rather than on a fusion, an `@` halves `valid_word_ratio`, and
    # perplexity on a domain name is noise. Letting those decide and then
    # correcting them afterwards is how this one line has had three different
    # answers in this issue. If the archive has a rule for addresses, the rule is
    # the answer.
    #
    # The `notation` exemption at gates 1a/1b is narrower and stays: it covers
    # every notation shape (sigla, grid references, labels), where the perplexity
    # routes are wrong but the rest of the cascade is not, and it deliberately
    # does NOT exempt `rule_hard_sweep`. This route is only the URL/e-mail
    # sub-shape, and only once an operator has said what that shape is worth.
    #
    # KNOWN LIMIT, measured rather than assumed: nothing here separates a
    # correctly scanned address from a mis-scanned one. `e-mail: officeauappmost.cz`
    # -- the same label with its `@` lost to the scanner -- already reads `Clear`
    # today without this route. A configured category therefore applies to both,
    # and `Noisy` is the honest setting for an archive that minds the difference.
    if (
        DOMAIN_NOTATION_CATEG
        and "rule_domain_notation_categ" not in DISABLED_RULES
        and _RE_NOTATION_URL.search(stripped)
    ):
        _fire("rule_domain_notation_categ")
        return DOMAIN_NOTATION_CATEG, _DOMAIN_NOTATION_REASON[DOMAIN_NOTATION_CATEG]

    # ------------------------------------------------------------
    # 1. Hard sweep
    # ------------------------------------------------------------
    if "rule_hard_sweep" not in DISABLED_RULES:
        if orig_lang_score < HARD_SWEEP_LANG_MAX and ppl > HARD_SWEEP_PPL_MIN:
            _fire("rule_hard_sweep")
            return "Trash", "trash_hard_sweep"

    if "rule_extreme_ppl" not in DISABLED_RULES and not notation:
        if ppl >= PPL_EXTREME_MIN and orig_lang_score < EXTREME_LANG_CONF:
            _fire("rule_extreme_ppl")
            return "Trash", "trash_hard_sweep"

    if "rule_absolute_ppl" not in DISABLED_RULES and not notation:
        if ppl >= PPL_GARBAGE_ABSOLUTE and not is_upright_czech:
            _fire("rule_absolute_ppl")
            return "Trash", "trash_hard_sweep"

    # ------------------------------------------------------------
    # 2. Inverted / mirrored scan
    # ------------------------------------------------------------
    if "rule_inverted" not in DISABLED_RULES:
        if not is_upright_czech and (
            ghost_dominated
            or (
                not has_cz_diacs(text_source)
                and rot_ratio >= SUSPICIOUS_ROT_RATIO
                and ppl >= PPL_INVERTED_MIN
                and ghost_word_share(text_source)[0] >= GHOST_HITS_INVERTED_MIN
            )
        ):
            _fire("rule_inverted")
            return "Trash", "trash_inverted"

    # ------------------------------------------------------------
    # 3. All-caps vowel-less scramble
    # ------------------------------------------------------------
    if "rule_allcaps" not in DISABLED_RULES:
        if vr < 0.10 and is_all_caps_line(text_source) and not structured:
            _fire("rule_allcaps")
            return "Trash", "allcaps_novowel"

    # ------------------------------------------------------------
    # 4. Extreme garbage density
    # ------------------------------------------------------------
    if "rule_garbage_density" not in DISABLED_RULES:
        if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
            is_siglum = word_count <= 2 and _RE_SIGLUM.match(stripped)

            if structured or is_siglum:
                pass

            elif "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
                text_source,
                valid_word_ratio,
                word_count,
            ):
                pass

            else:
                _fire("rule_garbage_density")
                return "Trash", "trash_threshold"

    # ------------------------------------------------------------
    # 5. Determine structural state
    # ------------------------------------------------------------
    forgiven = "rule_forgiven_headline" not in DISABLED_RULES and is_forgiven_headline(
        text_source,
        garbage_density,
    )

    damaged = "rule_damaged_token" not in DISABLED_RULES and word_count >= 3 and count_damaged_tokens(text_source) > 0

    # ------------------------------------------------------------
    # 5b. Zero-alphabetic content
    # ------------------------------------------------------------
    if (
        "rule_zero_alpha" not in DISABLED_RULES
        and not structured
        and not is_upright_czech
        and not any(c.isalpha() for c in text_source)
    ):
        _fire("rule_zero_alpha")

        if forgiven:
            return "Noisy", "noisy_threshold"

        return "Trash", "trash_threshold"

    # ------------------------------------------------------------
    # 6. Short-line garbage
    # ------------------------------------------------------------
    if "rule_short_garbage" not in DISABLED_RULES and not forgiven and not structured and not notation:
        if (
            word_count <= ISOLATED_CHAR_MIN_TOKENS
            and not has_cz_diacs(text_source)
            and (lang_score <= LANG_SCORE_REMAP or rot_ratio >= SUSPICIOUS_ROT_RATIO)
            and (gibberish_present or weird_ratio > 0.0)
        ):
            _fire("rule_short_garbage")
            # (#30 D15) The second witness.
            #
            # `_has_strong_garbage_evidence()` is False on the ENTIRE disputed
            # population -- pinned on production vectors by
            # tests/test_calibration.py::test_strong_evidence_is_false_on_the_
            # entire_disputed_population -- so on these lines the merged gate is a
            # suspension of the rule rather than a narrowing of it. That is the
            # accepted debt: roughly 26,000 garbage lines reach `Clear`.
            #
            # `_has_shape_garbage_evidence()` is the narrowing. It is a function
            # of the text -- plus, since D44, the raw language label for the
            # vowel-run clause -- and it separates the half that IS separable
            # (`oueussd` from `malakofauna`); the phonotactically legal residue
            # (`edelite`) still needs a lexicon and is out of scope by design.
            #
            # This is the ONLY site in the short-line path that returns `Trash`.
            # Section 7's `damage` branch returns `Noisy`, so a witness placed
            # there could improve `Clear` -> `Noisy` while never restoring a
            # `Trash` verdict -- and the strict xfail that tracks the debt would
            # stay green forever. That mistake was made once already; see the
            # plan's "Did you run it, or read it?" note.
            #
            # SHIPS OFF. `SHORT_GARBAGE_WITNESS_ENABLE` defaults to false, so the
            # disjunct below cannot change any outcome until the flag is flipped,
            # and the flag must not be flipped until the witness is measured
            # against a gold set (tools/gold/GOLD.md). Wiring and enabling are
            # deliberately separate commits.
            _shape_witness = SHORT_GARBAGE_WITNESS_ENABLE and _has_shape_garbage_evidence(text_source, lang)
            if _shape_witness:
                _fire("rule_short_garbage_witness")
            if qs < CATEG_TRASH_SCORE_MAX + 0.35 and (
                _has_strong_garbage_evidence(
                    text_source,
                    valid_word_ratio=valid_word_ratio,
                    lang_score=lang_score,
                    orig_lang_score=orig_lang_score,
                    gibberish_present=gibberish_present,
                    garbage_density=garbage_density,
                    weird_ratio=weird_ratio,
                    is_upright_czech=is_upright_czech,
                )
                or _shape_witness
            ):
                return "Trash", "trash_threshold"

    elif "rule_short_garbage" not in DISABLED_RULES and not forgiven and not structured and notation:
        # Reached only when the notation predicate is the DECIDING term — the
        # other three would have let rule_short_garbage run. Recorded so coverage
        # and ablation see the suppression rather than just its absence.
        _fire("rule_domain_notation")

    # ------------------------------------------------------------
    # 7. Short lines (1-2 words)
    # ------------------------------------------------------------
    if "rule_short_line" not in DISABLED_RULES and word_count <= 2:
        _fire("rule_short_line")

        if _RE_SIGLUM.match(stripped) and sum(c.isalpha() for c in stripped) >= 2:
            return "Clear", "clear_threshold"

        if word_count == 1:
            solitary = stripped.strip(_STRIP_CHARS)

            if len(solitary) == 1 and solitary.isalpha() and "." not in stripped:
                return "Trash", "trash_threshold"

        if any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words):
            if _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return "Trash", "trash_threshold"

        structurally_clean = valid_word_ratio >= 1.0

        damage = (
            weird_ratio >= 0.40
            or (any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words) and not structured)
            or ((gibberish_present or detect_fused_words(text_source) > 0) and not structurally_clean)
            or (
                garbage_density >= CATEG_GARBAGE_DENSITY_HIGH
                and not _trailing_fill_rescued(
                    text_source,
                    valid_word_ratio,
                    word_count,
                )
                and not structured
            )
        )

        if damage:
            return "Noisy", "noisy_threshold"

        if valid_word_ratio <= 0.0 and not is_upright_czech and not structured:
            if forgiven:
                return "Noisy", "noisy_threshold"

            return "Trash", "trash_threshold"

        if is_upright_czech or valid_word_ratio >= 1.0 or structured:
            if count_damaged_tokens(text_source) > 0 or (not is_upright_czech and weird_ratio >= 0.40):
                return "Noisy", "noisy_threshold"

            return "Clear", "clear_threshold"

        return "Noisy", "noisy_threshold"

    # ------------------------------------------------------------
    # 8. High-confidence LM override
    # ------------------------------------------------------------
    if "rule_lowppl_clear" not in DISABLED_RULES:
        if ppl < LOWPPL_CLEAR_MAX and word_count >= 3:
            if valid_word_ratio < MOSTLY_READABLE_VALID_MIN:
                _fire("rule_lowppl_clear")
                return "Noisy", "noisy_threshold"

            if damaged:
                _fire("rule_damaged_token")
                return "Noisy", "noisy_threshold"

            _fire("rule_lowppl_clear")
            return "Clear", "lowppl_clear"

    # ------------------------------------------------------------
    # 9. Explicit diagnostic hard gates [STEP 5]
    # ------------------------------------------------------------
    thresh_trash = CATEG_TRASH_SCORE_MAX + 0.35

    def check_rescues() -> tuple[str, str]:
        if "rule_trailing_fill_rescue" not in DISABLED_RULES and _trailing_fill_rescued(
            text_source,
            valid_word_ratio,
            word_count,
        ):
            _fire("rule_trailing_fill_rescue")
            return "Noisy", "noisy_threshold"

        if forgiven:
            _fire("rule_forgiven_headline")
            return "Noisy", "noisy_threshold"

        if "rule_reference_floor" not in DISABLED_RULES and is_clean_reference(text_source):
            _fire("rule_reference_floor")
            return "Noisy", "noisy_threshold"

        return "Trash", "trash_threshold"

    # 9a. WQX / rotation
    if "rule_wqx_rot" not in DISABLED_RULES:
        wqx_ratio = sum(1 for w in words if any(c in WQX_CHARS for c in w)) / max(word_count, 1)

        if (rot_ratio > 0.50 or wqx_ratio > 0.10) and orig_lang_score < 0.75 and not is_upright_czech:
            _fire("rule_wqx_rot")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9b. Vowelless / all-caps
    if "rule_vowelless" not in DISABLED_RULES:
        if word_count <= 3 and vr < 0.30 and not is_upright_czech and is_all_caps_line(text_source) and not structured:
            _fire("rule_vowelless")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9c. Ledger fragmentation
    if "rule_ledger_fragmentation" not in DISABLED_RULES:
        if len(words) >= 4:
            frag_count = sum(1 for w in words if (w.strip(_STRIP_CHARS).isdigit() or len(w.strip(_STRIP_CHARS)) <= 2))

            if (frag_count / len(words)) > 0.60:
                _fire("rule_ledger_fragmentation")

                if qs < thresh_trash and _has_strong_garbage_evidence(
                    text_source,
                    valid_word_ratio=valid_word_ratio,
                    lang_score=lang_score,
                    orig_lang_score=orig_lang_score,
                    gibberish_present=gibberish_present,
                    garbage_density=garbage_density,
                    weird_ratio=weird_ratio,
                    is_upright_czech=is_upright_czech,
                ):
                    return check_rescues()

    # 9d. Mid-uppercase
    if "rule_mid_uppercase" not in DISABLED_RULES:
        if word_count <= 2 and any(_is_mid_uppercase(w.strip(_STRIP_CHARS)) for w in words) and not structured:
            _fire("rule_mid_uppercase")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9e. Bigram run
    if "rule_bigram_run" not in DISABLED_RULES:
        has_bigram_run = any(_RE_BIGRAM_RUN.search(w.strip(_STRIP_CHARS)) for w in words)

        if has_bigram_run:
            _fire("rule_bigram_run")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # 9f. Fragment tokens
    if "rule_fragment_tokens" not in DISABLED_RULES:
        lengths = [
            len(core) + 1 if w.endswith(".") else len(core) for w in words for core in [w.strip(_STRIP_CHARS)] if core
        ]

        fragment_like = bool(lengths and (sum(lengths) / len(lengths)) < 2.0)

        if fragment_like:
            _fire("rule_fragment_tokens")

            if qs < thresh_trash and _has_strong_garbage_evidence(
                text_source,
                valid_word_ratio=valid_word_ratio,
                lang_score=lang_score,
                orig_lang_score=orig_lang_score,
                gibberish_present=gibberish_present,
                garbage_density=garbage_density,
                weird_ratio=weird_ratio,
                is_upright_czech=is_upright_czech,
            ):
                return check_rescues()

    # ------------------------------------------------------------
    # 10. Quality-score band routing
    # ------------------------------------------------------------
    if qs < CATEG_TRASH_SCORE_MAX:
        return check_rescues()

    if "rule_mostly_readable_noisy" not in DISABLED_RULES:
        if valid_word_ratio < MOSTLY_READABLE_VALID_MIN and not _lm_confident_czech(
            is_upright_czech,
            ppl,
            garbage_density,
        ):
            _fire("rule_mostly_readable_noisy")
            return "Noisy", "noisy_threshold"

    # [STEP 2] Restored character-level damage invariant capping route to Noisy
    if damaged:
        _fire("rule_damaged_token")
        return "Noisy", "noisy_threshold"

    return "Clear", "clear_threshold"


def categorize_line(
    qs: float,
    txt: str,
    wc: int,
    vowel_ratio: float,
    perplexity: float,
    weird_ratio: float = 0.0,
    return_reason: bool = False,
    valid_word_ratio: float = 1.0,
    lang_score: float = 1.0,
    orig_lang_score: float = 1.0,
    gibberish_present: bool = False,
    garbage_density: float = 0.0,
    is_upright_czech: bool = False,
    ghost_dominated: bool = False,
    lang: str | None = None,
) -> tuple[str, float] | tuple[str, float, str]:
    categ, reason = determine_category(
        qs,
        txt,
        wc,
        vowel_ratio,
        perplexity,
        weird_ratio,
        valid_word_ratio,
        lang_score,
        orig_lang_score,
        gibberish_present,
        garbage_density,
        is_upright_czech,
        ghost_dominated,
        lang,
    )

    # Label constants, not literals, on the three sites where a category name is
    # COMPARED against rather than emitted: the pairing of CATEG_TRASH with
    # CATEG_TRASH_SCORE_MAX is the whole point of this block, and spelling both
    # halves the same way keeps that visible. The `return` sites in
    # determine_category() are deliberately left as literals - see the note at the
    # head of this file.
    if categ == CATEG_TRASH:
        aligned_score = min(qs, CATEG_TRASH_SCORE_MAX - 0.0001)
    elif categ == CATEG_NOISY:
        aligned_score = max(qs, CATEG_TRASH_SCORE_MAX)
        aligned_score = min(aligned_score, CATEG_NOISY_SCORE_MAX - 0.0001)
    elif categ == CATEG_CLEAR:
        aligned_score = max(qs, CATEG_NOISY_SCORE_MAX)
    else:
        aligned_score = qs

    if return_reason:
        return categ, aligned_score, reason
    return categ, aligned_score


def _has_strong_garbage_evidence(
    text_source: str,
    *,
    valid_word_ratio: float,
    lang_score: float,
    orig_lang_score: float,
    gibberish_present: bool,
    garbage_density: float,
    weird_ratio: float,
    is_upright_czech: bool,
) -> bool:
    if is_structured_line(text_source):
        return False

    if gibberish_present:
        return True

    if valid_word_ratio <= 0.20:
        return True

    if lang_score <= 0.20 and orig_lang_score <= 0.50:
        return True

    if garbage_density >= CATEG_GARBAGE_DENSITY_HIGH:
        return True

    if weird_ratio >= 0.75:
        return True

    if not is_upright_czech and lang_score <= 0.40 and weird_ratio >= 0.40:
        return True

    return False


# ── rule_short_garbage shape witness (issue #30) ────────────────────────────
# A SECOND WITNESS for the short-line garbage route, alongside
# _has_strong_garbage_evidence(). It exists because that predicate returns False
# on the ENTIRE population issue #30 is about -- valid_word_ratio 1.0 (because
# compute_valid_ratio is shape-only), weird_ratio ~0.28-0.35, gibberish 0,
# garbage_density 0.0, and a lang clause that fails on its FIRST conjunct at
# trust-tier scores of 0.28-0.92. Pinned by
# tests/test_calibration.py::test_strong_evidence_is_false_on_the_entire_disputed_population.
# So on those lines "require a second witness" is not a narrowing of the rule;
# there is no second witness to consult, and the rule simply stops convicting.
#
# SHAPE, NOT IDENTITY. It answers "is this spelled the way no European word is
# spelled?", which characters can decide. It does not answer "is this a word?",
# which they cannot: `edelite` is phonotactically legal and stays out of reach.
# That residue is the part of #30 that genuinely needs a lexicon (D14).
#
# What is REUSED from detect_fused_words(): the vowel-run test and its vowel
# class, and nothing shared any more. It used to be _RE_FUSED_VOWEL_RUN by
# reference, so FUSED_VOWEL_RUN_MIN steered both; the witness now compiles its
# own (_compile_vowel_run), and the run length comes from
# SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN and the D44 language split
# (_vowel_run_min_for). FUSED_VOWEL_RUN_MIN only supplies the default.
#
# What is deliberately NOT reused, and why -- each of these was measured, and
# each has a test naming the counterexample:
#
#   * `len(core) > 14`, detect_fused_words' third clause. Flags
#     `Skelettmaterial`. Length is not evidence of garbage in a compounding
#     language. See test_long_compound_is_not_witnessed.
#
#   * _RE_FUSED_CONSONANT_RUN (5+ consonants). Flags `vrstva`, `vrstvy`,
#     `vrstvou`, `vrstvami`, `ctvrtek`, `ctvrt`, `zmrzl`, `scvrkl` -- ordinary
#     diacritic-free Czech, and `vrstva` ("layer") is the commonest noun in
#     archaeological field documentation. `vrstva 3` satisfies every condition
#     of the short-garbage route at production signals, so a witness carrying
#     this clause would Trash it. Nothing is lost by dropping it: `sektlll` and
#     `Tthts I` are reached by the triple and geminate tests instead. See
#     test_consonant_run_czech_is_not_witnessed.
#
#   * _has_repeated_run() / detect_repeated_chars(). Looks like the same idea
#     and is not: it fires on `Kaaden` (aa) and `Pinii` (ii), both #30 fixtures
#     that must survive. Unifying the two would either widen this predicate or
#     narrow a detector that feeds compute_quality_score corpus-wide.
#
# detect_fused_words() itself is untouched for that last reason -- it feeds the
# `fused_ratio` term of the quality score and the `fused_words` CSV column, so
# narrowing it would move scores on every line in the corpus, not just here.
#
# KNOWN false positives of the vowel-run clause: Latin/French/German loans with
# a 3+ vowel run (`Poaceae`, `Naiade`, `Beaune`, `Radiouhlik`,
# `Sauerstoffflasche`). Most carry weird_ratio 0.0 and so never reach the route
# at all -- a real but THIN margin, since it depends on a signal outside this
# predicate. Measuring that class against annotated lines is a precondition for
# enabling the flag, not a follow-up.
# WIRED, BUT OFF (#30 D15). PR #48 merged as `070620f`, creating the conditional
# this predicate joins, and the witness is now read at gate 6 of
# `determine_category()` as a second disjunct beside `_has_strong_garbage_evidence()`.
# Gate 6 is the only site in the short-line path that returns `Trash`; section 7's
# `damage` branch returns `Noisy`, so a witness placed there could never restore a
# `Trash` verdict.
#
# `SHORT_GARBAGE_WITNESS_ENABLE` still defaults to false, so the disjunct cannot
# change any outcome. Wiring and enabling are separate on purpose: the flag must
# not be flipped until the witness is measured against a GOLD set.
#
# UPDATE 2026-09-10 -- it has been measured once, and it did not pass. On the 508
# annotated lines, flag-on moved 26: 12 fixed, 12 BROKEN, 2 borderline. Every
# break carried a roman numeral, which the exemption in the predicate below now
# clears. That exemption is a narrowing, not a green light: the 508 have NOT been
# re-scored against it (that needs the delivered batch, which is not in the
# tree), so the flag stays false.
#
# UPDATE 2026-09-23 -- the gold A/B exists now. On the 2,064-row sidecar the flag
# passes the adoption gate (stage 10e: errors 513 -> 503, `Clear`-loss unchanged,
# p = 0.013; with the D44 split, 12 fixes / 1 break), and @david-spacil's re-score
# of the 508 on `3b02959` reads 326 -> 336 with one break. The flag still ships
# false: those labels reach 23 of the ~20k lines the witness fires on, so it now
# waits on the annotation ask (docs/issue30/census.csv + sample.csv) and the open
# questions in docs/issue30/README.md, not on code.
#
# UPDATE 2026-10-01 -- the annotation is back (docs/issue30/census.csv + sample.csv,
# 249 of 357 rows labelled, every blank non-Czech). Text-only with this
# predicate and no lexicon: projected tail errors 4,844 -> 1,180, census head
# 339 -> 358 (decipherable damage, which this route answers `Trash`; Q6 keeps
# that), `Clear`-loss 0 -- after D46 below, which her two `Clear` labels found.
# The flag stays false until stage 12 scores the same labels on the real lines
# with the lexicon armed (tools/project_annotation.py join).
#
# The annotations themselves are now here -- tools/gold/sidecars/, joined onto a
# delivered batch with `--gold-sidecar`; see tools/gold/GOLD.md, including its
# provenance note on the 55 labels that changed when the 508 were re-annotated.
# Flipping the flag also means moving the four SHORT_GARBAGE_WITNESS_* constants
# out of `_DELIBERATELY_NOT_TUNABLE` and into `_THRESHOLD_NAMES` + `SEARCH_SPACE`,
# in the same commit.
#
# Covered by tests/test_text_utils.py::TestShapeGarbageWitness (the predicate),
# test_the_disjunction_the_gate_will_evaluate in tests/test_calibration.py (the
# composed condition on production vectors), and
# tests/test_short_garbage_witness_wiring.py (the call site, both flag states).
_RE_TRIPLE_ALPHA_RUN: re.Pattern = re.compile(r"([^\W\d_])\1\1", re.IGNORECASE)
_RE_INITIAL_CONSONANT_GEMINATE: re.Pattern = re.compile(r"^([bcdfghjklmnpqrstvwxz])\1", re.IGNORECASE)

# (#30) Latin taxonomic endings. Closed, and deliberately only the ones that
# CONTAIN a 3+ vowel run -- this exists to stop one clause convicting one class,
# not to hand every Latinate word an exemption. `-aceae` (and `-oideae`) are the
# botanical FAMILY suffix: every family name ends in it, so the vowel-run clause
# was not making an occasional mistake on this class, it was convicting all of
# it. Measured 2026-09-17 on the shipped predicate, 10 of 10: Poaceae, Rosaceae,
# Fabaceae, Brassicaceae, Cyperaceae, Chenopodiaceae, Asteraceae, Betulaceae,
# Fagaceae, Polygonaceae.
#
# `-idae` / `-inae` are the zoological family and subfamily suffixes; they carry
# only a 2-vowel run today, so they are listed for the case where
# SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN is ever lowered, not because they fire now.
_RE_TAXONOMIC_SUFFIX: re.Pattern = re.compile(r"(?:aceae|oideae|eae|iae|aea|oidea|idae|inae)$", re.IGNORECASE)
# Below this many letters a suffix match is a coincidence, not a taxon.
_TAXONOMIC_SUFFIX_MIN_ALPHA: int = 5

# (#30) A grid/context reference whose trailing sub-letter is FUSED to the roman
# segment: `S-VIIIb`, `K-VIIIc`, `AA-VIIIb`. `is_domain_notation()` already
# recognises the hyphenated form -- `S-VIII-b` and `B-XII-c` both return True --
# and the unhyphenated twin differs from it by one character, so this is a
# near-miss in that predicate rather than a new class. The roman-numeral exemption
# does not reach it either: `_split_subtokens` yields `VIIIb`, and the fused
# lowercase letter keeps `_RE_ROMAN_TOKEN` from matching, so `vowel_run` (`III`)
# and `triple` convict it.
#
# Exempted HERE, witness-locally, and NOT by widening `is_domain_notation()`,
# which is also read by `rule_short_garbage`'s outer guard and by the two
# perplexity-only routes. Widening it there would change production behaviour for
# these strings while the witness flag is still false; here it cannot change
# anything until the flag flips -- the discipline the roman-numeral and taxonomy
# exemptions follow.
#
# Measured on the 822-document corpus: seven distinct strings, one excavation's
# grid series -- AA-VIIIb, E-VIIIb, F-VIIIb, J-VIIIb, K-VIIIc, L-VIIIb, S-VIIIb --
# across 7 witnessed lines, 6 of them currently Clear. `S-VIIIb` is annotated
# `Clear` in tools/gold/sidecars/issue30_gold_2067.csv, and on the gold surface
# this guard vetoes exactly that line and none of the gold-Trash catches.
#
# HISTORY, because this guard has already been lost once. Added 2026-09-18
# (`9bc218b`) and measured in the round-2 stage-5a re-run; deleted the same
# evening by `cc4990e`, a lexicon commit whose diff it was not otherwise part of,
# together with the German-diacritic veto `has_expected_lang_diacs()`. No test
# pinned either, so nothing noticed until @david-spacil's 508-line re-check on
# 2026-09-23 found `S-VIIIb` among the witness's two breaks on the default config.
# With a lexicon configured `viiib` is attested and survives anyway; the shipped
# config has none, which is why the guard matters. Restored on his report, and
# now pinned by tests/test_shape_witness_vocabulary.py. The German veto was NOT
# restored with it: the D44 language split now overlaps it, and it needs its own
# measurement (30.plan.md D45).
_RE_FUSED_GRID_REF: re.Pattern = re.compile(r"^[A-Za-z]{1,3}[-/][IVXLCDM]{1,7}[a-z]?$")

# (#30 D46) Punctuation that joins two words without a space. `_split_subtokens`
# splits on `. - –` only, so the witness used to read a comma-list such as
# `erhalten,Siedelungsfund,gefunden` as ONE 30-letter token. On a token that long
# `low_variety` fires on alphabet saturation alone -- a long string simply runs
# out of new letters -- and a roman numeral fused by a comma (`XXX,1937,str`,
# `III,konec`) no longer looks like a roman numeral to `_RE_ROMAN_TOKEN`.
#
# Found in @DanaKriv's 357 decisions (2026-10-01, docs/issue30/census.csv): the only
# two at-risk lines she labelled `Clear` are exactly these, `XXX,1937,str. 21`
# and `okraj sekt.III,konec`, and 41 of the rows she left blank -- readable German
# find descriptions -- fired only because of a fused comma or slash. Splitting
# here, measured on her labels with this predicate: projected tail errors
# 1,282 -> 1,180, projected `Clear`-loss 102 -> 0, one gold-`Trash` row lost
# (`MZMRISCH,IEU Stxdter`, whose `IEU` drops below MIN_ALPHA once separated).
#
# WITNESS-LOCAL, like the exemptions above: `_split_subtokens` also feeds the
# quality metrics, so changing it would move `quality_score` on every line in the
# corpus. The lexicon builder and `tools/ocr_neighbours.py` use this splitter too,
# because a table keyed differently from the lookup misses silently.
_RE_WITNESS_JOINERS: re.Pattern = re.compile(r"[,/;:()\[\]{}<>\"„“”«»=_?!*#]+")


def _witness_subtokens(word: str) -> list[str]:
    """Split one whitespace token the way the shape witness reads it (#30 D46).

    Joining punctuation first (`_RE_WITNESS_JOINERS`), then `_split_subtokens`'s
    `. - –`. Empty pieces are dropped.
    """
    return [sub for piece in _RE_WITNESS_JOINERS.split(word) for sub in _split_subtokens(piece)]


@functools.lru_cache(maxsize=8)
def _compile_vowel_run(min_run: int) -> re.Pattern:
    """The vowel-run pattern at an arbitrary length, cached per length.

    ``_RE_FUSED_VOWEL_RUN`` is compiled once at import from ``FUSED_VOWEL_RUN_MIN``
    and therefore does not respond to ``override_constants``. The witness needs a
    knob the sweep can actually move, so it compiles its own.
    """
    return re.compile(r"[aeiouyáéíóúýěůäöü]{%d,}" % max(2, int(min_run)), re.IGNORECASE)


@functools.lru_cache(maxsize=4)
def _read_token_lexicon(path: str, mtime: float, min_df: int) -> Mapping[str, int]:
    """Load a token/document-frequency table, keyed on path+mtime so edits are seen.

    Returns a READ-ONLY MAPPING token -> document frequency, restricted to tokens
    at or above ``min_df``. It was a bare set until the de-gemination guard needed
    to compare one token's frequency against another's; that guard is gone
    (#30 D40, 2026-09-22) but the mapping stays, because the frequencies are what
    ``min_df`` filters on here and what ``build_token_lexicon.py``'s provenance
    header and per-collection columns are for. A set plus a parallel frequency
    dict would be a second copy of this vocabulary and so a second thing to
    drift, which is the argument already made for SHAPE_GARBAGE_CLAUSES.
    Membership tests and ``bool()`` read identically on a mapping, so every
    existing caller is unaffected. The proxy is because the result is cached: a
    caller that mutated it would poison every later lookup in the process.

    Format, as written by ``tools/build_token_lexicon.py``: ``#``-prefixed
    provenance header, then ``token<TAB>document_frequency`` per line. ``mtime``
    is a cache key and nothing else -- it is what lets a rebuilt table be picked
    up inside one process, which the test suite relies on.

    A comment is a line that starts with ``#`` AND CARRIES NO TAB. Both halves
    are load-bearing: ``#`` is not in ``_STRIP_CHARS``, so the builder emits
    tokens like ``#rdisico`` verbatim, and 27 of them appear in the first real
    822-document table. Skipping every ``#``-initial line dropped them at load
    with no symptom other than a count being quietly wrong. No escaping scheme is
    needed to tell the two apart -- a header line has no tab (``# columns:
    token<TAB>document_frequency`` is literal text), a data line always does.

    A malformed or unreadable table yields an EMPTY set rather than an exception.
    This predicate is consulted per sub-token inside the categoriser; a corrupt
    optional file must degrade to "no vocabulary signal", never take the pipeline
    down mid-corpus.
    """
    del mtime  # cache key only
    try:
        out: dict[str, int] = {}
        # errors="replace" rather than strict: a table is operator-supplied and
        # may have been moved through a Windows editor or a lossy transfer. A
        # mangled byte should cost that one token, not the corpus run.
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                token, sep, rest = stripped.partition("\t")
                # Comment == leading '#' and no tab. See the docstring: a token
                # may legitimately begin with '#'.
                if not sep or (not token and stripped.startswith("#")):
                    continue
                if not token:
                    continue
                # Field 1 is the TOTAL document frequency and the only column this
                # predicate reads. A multi-collection table adds per-collection
                # columns after it (`token<TAB>total<TAB>ARUP<TAB>ARUB`), and
                # int() on the whole remainder would raise on every data row --
                # dropping the entire table and leaving the veto inert with no
                # symptom. Take the field, not the rest of the line.
                try:
                    freq = int(rest.partition("\t")[0])
                except ValueError:
                    continue
                if freq < min_df:
                    continue
                out[token] = freq
        return MappingProxyType(out)
    except (OSError, ValueError, UnicodeError):
        return _EMPTY_LEXICON


def token_lexicon() -> Mapping[str, int]:
    """The vocabulary table currently in force, or an empty mapping when none is configured.

    Maps token -> document frequency. Falsy when no table is configured, which is
    the shipped state and what every caller gates on.
    """
    path = (SHORT_GARBAGE_LEXICON_PATH or "").strip()
    if not path:
        return _EMPTY_LEXICON
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _EMPTY_LEXICON
    return _read_token_lexicon(path, mtime, int(SHORT_GARBAGE_LEXICON_MIN_DF))


def uncoupled_witness_warning(
    witness_enabled: bool | None = None,
    lexicon_path: str | None = None,
) -> str | None:
    """The shape witness armed with no vocabulary table: reach it on purpose or not at all.

    Returns the advisory text, or ``None`` when the configuration is fine.

    The reason is CORPUS EXPOSURE, measured over both full collections (#30). With
    no table the witness would newly convict **8,529 strings / 37,555 lines** the
    pipeline currently keeps as ``Clear`` or ``Noisy`` (stage 08f). With the
    113,100-document table ``tools/build_token_lexicon.py`` builds, that falls to
    **5,563 / 6,714** (stage 9b), an 82.1% drop in lines. What the table spares is
    led by real words, not by damage: ``ppole`` (an abbreviation), ``ARCHAIA`` (the
    excavator's name), ``Lepus europaeus``, ``Triticum monococcum``. Caveat: 9b ran
    after D33 took URL and e-mail citations out of the witness's scope, so the
    82.1% is the table and D33 together, not the table alone.

    Gold cannot show this, and should not be read as if it could. The 2,064-row
    sidecar labels 23 of the ~20k lines the witness reaches in the 822-document
    gold corpus, and on those shape-only and table-armed score alike: round-2
    stage 5a gave 500 errors, 5a-bis (with a table) 502, and @david-spacil's
    508-line re-check with no table went 326 -> 336 (2026-09-23).

    History: until 2026-09-23 this quoted 15,217 convictions against 5,107
    confirmations ("1:3 against"), scored against ``categ`` rather than gold, and
    90.5% of that gap was ``ppole``. The coupling survived; that arithmetic did not.

    ADVISORY, NOT A GATE -- deliberately, and in the same spirit as the
    ``atrium_vocab`` consistency check at the top of this module. The shape-only
    configuration is measured on purpose -- every gold A/B from stage 5a on, and
    the 508 re-check -- and a refusal would make those measurements impossible.
    What must not happen is someone reaching it by accident.
    """
    if witness_enabled is None:
        witness_enabled = SHORT_GARBAGE_WITNESS_ENABLE
    if lexicon_path is None:
        lexicon_path = SHORT_GARBAGE_LEXICON_PATH
    if not witness_enabled or (lexicon_path or "").strip():
        return None
    return (
        "SHORT_GARBAGE_WITNESS_ENABLE is true with no SHORT_GARBAGE_LEXICON_PATH. "
        "Over both collections the witness would then newly convict 37,555 lines the "
        "pipeline currently keeps; a 113,100-document table cuts that to 6,714, and "
        "what it spares is led by real words (ppole, ARCHAIA, Lepus europaeus). "
        "Unless you are deliberately measuring the shape-only configuration, build a "
        "table with tools/build_token_lexicon.py."
    )


def lexicon_document_count(path: str | None = None) -> int | None:
    """How many documents the configured token table was built over, or None.

    Read from the ``# documents: N  lines: M`` provenance line
    ``tools/build_token_lexicon.py`` writes. Cheap: it stops at the first
    non-comment row rather than parsing the table.
    """
    if path is None:
        path = SHORT_GARBAGE_LEXICON_PATH
    path = (path or "").strip()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                m = re.search(r"#\s*documents:\s*([\d,]+)", line)
                if m:
                    return int(m.group(1).replace(",", ""))
    except OSError:
        return None
    return None


def _has_vocabulary_support(token: str) -> bool:
    """Is this token attested as vocabulary elsewhere in the collection?

    (#30 D14.) The residue the shape witness cannot reach -- `edelite`,
    `vfetennl k.` -- is phonotactically legal, so no character-level test
    separates it from `malakofauna`. Both sides of the issue thread concluded the
    way out is a lexicon; the external Czech ones are CC BY-NC-SA, which would
    change this pipeline's output-licence story, so this uses the corpus as its
    own dictionary.

    The signal is DOCUMENT frequency, not line frequency and not category: a
    token that appears in many distinct documents is vocabulary, because OCR
    noise is idiosyncratic to the scan that produced it. Document frequency is a
    property of the data, so consulting it does not re-introduce the circularity
    of scoring against the pipeline's own labels -- which is the mistake this
    issue has already made once, in the sweep objective.

    Veto only: it can withdraw a conviction, never add one.

    KNOWN LIMIT, and it is now accepted rather than carved out (#30 D40,
    2026-09-22). The premise above -- that OCR noise is idiosyncratic to its
    scan -- FAILS for a pre-printed form scanned across the collection: there the
    error is systematic, reproduces once per document and accrues document
    frequency exactly like a word. A de-gemination guard used to carve that shape
    out, letting the witness convict a doubled-initial token although the table
    attested it. It is gone, on the data provider's answer and on measurement:

    * `ppole` (`KULTURA: ppole`) is not the artefact the guard was built for --
      it is *popelnicová pole*, the standard abbreviation, `pp` doubled for a
      plural exactly as in `pp.` for pages (@david-spacil, 2026-09-19).
    * `ssuti` / `ssutí` / `ssutě` are not artefacts either -- they are an old
      spelling of `suť` (@david-spacil, 2026-09-22). So of the eight tokens the
      guard was fitted to, FOUR are real language and four are damage, and the
      frequency gap the threshold sat in had vocabulary on both sides of it.
    * At full scale the table attests all eight, and zero of the 42,853 rows in
      the witness queue carried any of them. The guard fired on nothing.

    So attestation still does not by itself prove vocabulary, and a templated
    artefact can still be wrongly exempted here. That is a recorded, accepted
    limit with no known separating signal -- three were tried and none survived
    the full-collection table -- rather than a gap with a patch waiting.

    What this function does at full scale, measured directly rather than inferred
    from this docstring: `Mammalia`, `Triticum`, `Lepus`, `Linum`, `Arvicola` and
    their common epithets are all attested and exempt through this path once a
    lexicon is configured -- the taxonomic false-positive class this issue spent
    time on (see the corpus profile) turned out to be mostly this mechanism
    working as designed, not a predicate gap.
    """
    lex = token_lexicon()
    if not lex:
        return False
    lowered = token.lower()
    return lowered in lex


# The clause names, in report order. Canonical here rather than in the reporting
# tool, because a second copy of this vocabulary is a second thing to drift.
SHAPE_GARBAGE_CLAUSES: tuple[str, ...] = ("vowel_run", "triple", "initial_geminate", "low_variety", "no_vocabulary")


def _vowel_run_min_for(lang: str | None) -> int:
    """How many consecutive vowels convict, given the detected language (#30 D44).

    `lang` is a FastText label (`deu_Latn`) or a bare base (`deu`); anything
    unrecognised, including None, gets the general threshold. That default is
    deliberate: not knowing the language must not silently exempt a line.
    """
    if lang and _lang_base(lang) in SHORT_GARBAGE_WITNESS_VOWEL_RUN_EXEMPT_LANGS:
        return SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN_EXEMPT
    return SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN


def _has_shape_garbage_evidence(text_source: str, lang: str | None = None) -> bool:
    """Phonotactic evidence that a short line is OCR garbage, not rare vocabulary.

    Read only when ``SHORT_GARBAGE_WITNESS_ENABLE`` is set. See the block above
    for what it tests, what it refuses to test, and the counterexamples behind
    each refusal.

    The verdict is ``bool()`` of the clause list, so the predicate and the
    diagnosis cannot disagree -- see ``shape_garbage_clauses()``.
    """
    return bool(shape_garbage_clauses(text_source, lang))


def shape_garbage_clauses(text_source: str, lang: str | None = None) -> list[str]:
    """Which witness clauses ``text_source`` satisfies, in ``SHAPE_GARBAGE_CLAUSES`` order.

    THE single implementation of the witness. It exists because there used to be
    two: ``tools/short_garbage_witness_report.py`` carried its own copy of these
    four tests so it could name the clause that fired, guarded by an assertion
    that the two agreed. The roman-numeral exemption was added here and not
    there, and the guard did exactly what it was written to do -- it raised, on
    0.74% of real lines, which is every line carrying a roman numeral. That is
    the fourth harness divergence in this repository (see the digest's
    "Harness divergence" section); the structural fix is that the reporting tool
    no longer has an implementation to drift.

    Returns every clause the line satisfies, not just the first. The predicate
    short-circuited on the first hit; this does not, because the report needs the
    full breakdown. The flag ships false, so nothing in production pays for it.
    """
    # Line-level vetoes. `_RE_FUSED_GRID_REF` is the unhyphenated twin of a shape
    # `is_domain_notation()` already accepts -- see its comment for why it lives
    # here and how it was once lost.
    if (
        has_cz_diacs(text_source)
        or is_structured_line(text_source)
        or is_domain_notation(text_source)
        or _RE_FUSED_GRID_REF.match(text_source.strip())
    ):
        return []

    found: set[str] = set()
    for word in text_source.split():
        for sub in _witness_subtokens(word):
            core = sub.strip(_STRIP_CHARS)
            letters = [c for c in core if c.isalpha()]

            # Too short to have a phonotactic shape at all. This is also what
            # keeps every SHORT_VALID_WORDS / _NEUTRAL_LEXICON member out by
            # construction rather than by enumeration.
            if len(letters) < SHORT_GARBAGE_WITNESS_MIN_ALPHA:
                continue

            lowered = core.lower()
            if lowered in _NEUTRAL_LEXICON or lowered in SHORT_EXCEPTION_TOKENS or lowered in SHORT_VALID_WORDS:
                continue

            # ROMAN NUMERALS are exempt, and the exemption sits above every
            # clause below rather than inside one of them. Measured on the 508
            # annotated lines (#30, 2026-09-10): flag-on moved 26 lines, and all
            # 12 it broke carried a roman numeral -- `Sonda VIII/3`,
            # `12.VIII.1977,`, `CCV. CCVI.`, `205; CCLXII).`, `166. Hr.XLIII.1.`,
            # `Lokalisace: I-VIII-eneol.II`, `w XVIII.`.
            #
            # Placement is the whole point: `III` is BOTH a triple-character run
            # and a 3-vowel run (`I` is a vowel), and `CC` opens a consonant
            # geminate, so exempting any single clause leaves the others to
            # convict the same line. Matching on the letters rather than `core`
            # is also deliberate -- `_split_subtokens` yields `VIII/3` whole.
            #
            # `_RE_ROMAN_TOKEN` and not `RE_ROMAN_NUMERAL`: the latter accepts
            # lowercase, and every letter of `lllll` is a numeral glyph, so it
            # would exempt real garbage. Uppercase-only and capped at 7 keeps
            # `IDIDIDIDIDIDUOID` convicted. Known cost: a 7-glyph all-numeral
            # stutter such as `DIDIDID` is now exempt too.
            #
            # This NARROWS the witness; it does not enable it.
            # `SHORT_GARBAGE_WITNESS_ENABLE` still defaults to false. Refusing to
            # convict a roman numeral is not the predicate claiming the line is
            # clean -- that asymmetry is why this does not contradict D2, which
            # kept roman numerals OUT of `is_domain_notation()`'s label lexicon.
            if _RE_ROMAN_TOKEN.match("".join(letters)):
                continue

            # LATIN TAXONOMY is exempt, and like the roman-numeral exemption it
            # sits above every clause rather than inside the one that fires
            # today. Same reason: `-aceae` is a vowel run now, and on a longer
            # name it is one clause away from the others.
            #
            # The block above this function called this class "KNOWN false
            # positives ... a real but THIN margin, since it depends on a signal
            # outside this predicate" -- the margin being that most such lines
            # carry weird_ratio 0.0 and never reach the route. Measuring the
            # class was recorded as a PRECONDITION for enabling the flag. It has
            # now been measured against the predicate, and the predicate convicts
            # the whole botanical family suffix, 10 of 10 (see
            # _RE_TAXONOMIC_SUFFIX). Leaning on weird_ratio to keep taxonomy out
            # of Trash is leaning on a signal this predicate does not control, in
            # a corpus whose archaeobotany and osteology reports are exactly
            # where `-aceae` lives.
            #
            # NOT covered here, and left to _has_vocabulary_support() on purpose:
            # `Naiade` (aia), `Beuern` (eue), `Oueste` (Oue). Those are name-like
            # rather than suffixed, and inventing a "legal vowel sequence" list to
            # catch them is the same guessing that cost 12 lines the last time
            # this predicate was widened by reading rather than by measuring.
            if len(letters) >= _TAXONOMIC_SUFFIX_MIN_ALPHA and _RE_TAXONOMIC_SUFFIX.search("".join(letters)):
                continue

            # LISTED BY THE ARCHIVE is exempt (#30 Q5b). @david-spacil, for
            # himself and @DanaKriv, 2026-10-01: "yes, listed means never evidence
            # of damage". A token in setup/word_lists.txt `[allowed]` is skipped by
            # every clause below, `no_vocabulary` included, so it can never be the
            # reason a line is discarded -- the step the list's header used to call
            # deliberately not taken. Still a veto only: a listed word cannot
            # convict anything, and an unlisted word on the same line is judged as
            # before (`ssuti vfetennl` still fires, on `vfetennl` alone if at all).
            # Live since @david-spacil's reviewed entries landed (Q5a, PR #7).
            if _is_allowed_token(core):
                continue

            # ATTESTED VOCABULARY is exempt (#30 D14). Inert unless
            # SHORT_GARBAGE_LEXICON_PATH points at a built table, so this is a
            # no-op in the shipped configuration. This is what reaches the loans
            # the suffix rule above deliberately does not -- `Naiade`, `Beuern`,
            # `Oueste` -- without anybody guessing at which vowel sequences a
            # European language is allowed to contain.
            if _has_vocabulary_support(core):
                continue

            # THE RESIDUE, and the only clause here that can ADD a conviction.
            # `edelite` and `vfetennl k.` are spelled the way words are spelled,
            # so no shape test reaches them; what they lack is attestation. A
            # token absent from a document-frequency table built over the whole
            # collection appeared in no other document, which is what OCR noise
            # looks like and what vocabulary does not.
            #
            # Off by default and gated on its own key, because "unattested" is
            # not "not a word": a genuinely rare term, a personal name, or a
            # token the table was simply built too narrowly to contain all land
            # here. That is a measurement, not a reading: stage 5c on gold,
            # `Clear`-loss 42 -> 63, rejected (issue30_gold_ab_findings.md § 6).
            if (
                SHORT_GARBAGE_LEXICON_CONVICT
                and token_lexicon()
                and not _has_vocabulary_support(core)
                and "".join(letters).isalpha()
            ):
                found.add("no_vocabulary")

            # Consecutive vowels: `oueussd`, `cuxoaid`, `IDIDIDIDIDIDUOID`.
            #
            # THE LENGTH DEPENDS ON THE LANGUAGE (#30 D44). Three in a row is
            # evidence of damage because CZECH has no triphthongs; German and
            # French have them natively, so in those languages the same run means
            # nothing and the clause was convicting `Dauerleihe` (*aue*) and
            # `FEUILLETON` (*eui*), both scanned correctly. Four is required
            # there, and four still convicts in every language -- a German line
            # can be scanned into `oueussd` as easily as a Czech one.
            #
            # `lang` is None for every caller that does not know the language,
            # and then the general threshold applies, which is the behaviour this
            # clause had before the split.
            if _compile_vowel_run(_vowel_run_min_for(lang)).search(core):
                found.add("vowel_run")

            # The same character three times: `sektlll`, `NINNNIC`. Capped by
            # length -- a long compound reaching three is `Schifffahrt`, a word.
            if len(letters) <= SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA and _RE_TRIPLE_ALPHA_RUN.search(core):
                found.add("triple")

            # A doubled CONSONANT in first position: `Tthts`, `rragment`. Rare in
            # European orthography but NOT impossible: abbreviations and old
            # spellings open that way -- `ppole` (*popelnicová pole*) and `ssuti`
            # (old *suť*) are real words this clause convicts (#30 W2). An
            # attesting table spares them, and since Q5b so does an `[allowed]`
            # entry (the veto above). A bare `^(.)\1` would also take `Aachen`.
            if _RE_INITIAL_CONSONANT_GEMINATE.match(core):
                found.add("initial_geminate")

            # Too few distinct letters for the length: `vansasaasasa`.
            if len(letters) >= SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA and (
                len({c.lower() for c in letters}) / len(letters) <= SHORT_GARBAGE_WITNESS_VARIETY_MAX
            ):
                found.add("low_variety")

    return [c for c in SHAPE_GARBAGE_CLAUSES if c in found]


def _looks_like_measurement(text_source: str) -> bool:
    """
    Detect structured archaeological measurement lines.

    The goal is to protect lines that may be noisy OCR but still contain
    meaningful measurements, dimensions, quantities, or physical descriptions.

    This predicate is deliberately conservative. It requires either:
      - an explicit measurement keyword with numeric context, or
      - multi-character measurement units (mm, cm, km, kg, ml, ha), or a
        bare metre unit glued directly to its number (0,4m), or
      - explicit measurement separator structures (e.g. v - 112, pr.okraje - 145).

    It strictly avoids classifying arbitrary digit-containing OCR or weak
    single-letter substrings/units as structured.

    Keyword/descriptor entries ending in a literal "." (rozm., pr., hl.) are
    matched WITHOUT a trailing \\b. A "." is a non-word character, so a
    trailing \\b can only hold when the abbreviation happens to be glued to a
    following word/digit character ("pr.okraje") — it silently fails whenever
    OCR (correctly) leaves a space or line-end after the dot ("pr. okraje",
    "rozm. 12", "pr. dna - 7"). The leading \\b is unaffected and still keeps
    these from matching mid-word; a lone hit still isn't enough on its own
    ("clouCelRa pr. 4" stays unmatched — one descriptor hit is below the
    corroboration threshold below).
    """
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    lowered = stripped.lower()

    # [STEP 3] Token-bounded full Czech measurement keywords.
    # Word-form entries keep a trailing boundary (avoids matching inside a
    # longer word); dot-terminated abbreviations must NOT have one — see
    # the docstring note on why \b cannot follow a literal "." into
    # whitespace or line-end.
    full_measurement_keywords = r"\brozm\." r"|\b(?:rozměry?|výška|šířka|délka|hloubka|průměr)\b"
    has_full_keyword = bool(re.search(full_measurement_keywords, lowered))

    # Secondary measurement descriptors (require multiple hits or numeric context).
    # Same dotted/word split as above, same reason.
    descriptor_keywords = r"\b(?:pr|hl)\." r"|\b(?:dna|hrdla|okraje)\b"
    descriptor_hits = len(re.findall(descriptor_keywords, lowered))

    # [STEP 3] Multi-character measurement units, or a bare metre unit glued
    # directly to its number (0,4m / 145-167m). A *spaced* bare unit ("3 m",
    # "o 5 m") stays unmatched — too weak a signal alone — but a glued bare
    # "m" is this corpus's single most common unit (far ahead of "mm"), so
    # excluding it outright cost real protection on genuine depth/height
    # readings; only the multi-character units are excluded from the
    # space-tolerant branch.
    has_unit = bool(
        re.search(
            r"\b\d+(?:[.,]\d+)?(?:\s*(?:mm|cm|km|kg|ml|ha)\b|m\b)",
            lowered,
        )
    )

    # Common measurement notation with explicit separators:
    #   v - 112mm
    #   pr.okraje - 145
    #   pr. dna - 7
    #   v: 144 mm
    has_measurement_separator = bool(
        re.search(
            r"\b(?:v|š|s|d|hl|pr|prům|výš|šíř|dél|hloub)"
            r"\.?\s*[:=\-]\s*\d",
            lowered,
        )
    )

    has_digits = any(char.isdigit() for char in stripped)

    if has_full_keyword and has_digits:
        return True

    if descriptor_hits >= 2 and has_digits:
        return True

    if has_measurement_separator:
        numeric_count = sum(char.isdigit() for char in stripped)
        if numeric_count >= 2:
            return True

    if has_unit:
        return True

    return False


# ---------------------------------------------------------------------------
# DORMANT candidate refinement to `_looks_like_measurement`'s `has_unit` branch,
# for the metre-spacing gap observed in issue #30. NOT WIRED INTO THE LIVE
# DETECTOR -- it exists only as a probe, exposed through
# `probe_spaced_decimal_metre_candidate()` and measured by
# `tools/recategorize_from_csv.py --probe-metre-candidate`.
#
# `has_unit` currently accepts a bare "m" only when glued directly to its
# number (`0,46m`), not when spaced (`3 m`) -- the latter stayed unmatched
# because a bare spaced unit alone is too weak a signal (see
# test_single_letter_units_and_probe_noise_rejected: "3 m", "o 5 m").
# Measurement against the 273-doc corpus found that restriction is now
# costing real coverage: 1,155 spaced-metre lines are unrecovered, and the
# quoted examples ("2,10 m", "215,5 193,120 m", "Z /193,445 m/",
# "Rovina profilu Z-V 214 192,740 m") are levelling/elevation readings, not
# noise. All of them carry a decimal separator in the number itself, which
# the rejected probes never do -- so a spaced bare "m" can be allowed
# specifically when the number is decimal-marked, without reopening the
# bare-integer case the pinned test guards against.
#
# Verified here to preserve every existing _looks_like_measurement assertion
# (glued forms, both pinned "3 m"/"o 5 m" rejections) and to recover the four
# quoted corpus lines, but the 1,155 figure above describes the *unrecovered*
# count under the current, narrower regex -- how much of it is actually
# decimal-marked (vs. some other spaced shape) hasn't been measured. Promoting
# this means replacing `has_unit`'s regex with the one below and re-running
# `tools/recategorize_from_csv.py` against the full local corpus to confirm the
# recovery and check for new false positives.
# ---------------------------------------------------------------------------
_RE_UNIT_CANDIDATE = re.compile(
    r"\b\d+(?:[.,]\d+\s*m\b|\s*(?:mm|cm|km|kg|ml|ha)\b|m\b)",
)


def _has_unit_with_spaced_decimal_metre_candidate(lowered: str) -> bool:
    """Probe for the dormant metre-spacing candidate on already-lowercased text."""
    return bool(_RE_UNIT_CANDIDATE.search(lowered))


def probe_spaced_decimal_metre_candidate(text_source: str) -> bool:
    """Return True when `text_source` matches the dormant metre-spacing probe."""
    return _has_unit_with_spaced_decimal_metre_candidate(text_source.lower())


# ---------------------------------------------------------------------------
# Rule for issue #30's "sonda: XIV." finding.
#
# `641f936`'s leading-\b fix on `_looks_like_measurement` correctly closed
# an accidental match — a bare "v." token used to match inside the Roman
# numeral of lines like "Plocha: 3; sonda: XIV." purely because keyword
# matching had no leading word-boundary yet.
#
# This rule explicitly recovers those plot/probe headers. It is now wired
# into `_looks_like_catalogue_reference` to protect these headers.
# ---------------------------------------------------------------------------
_RE_PLOT_PROBE_HEADER = re.compile(
    r"^plocha\s*:\s*\d+\s*;\s*sonda\s*:\s*[a-z]?[ivxlcdm]+\.$",
    flags=re.IGNORECASE,
)


def _looks_like_plot_probe_header(text_source: str) -> bool:
    """Match archaeological plot/probe headers: "Plocha: <n>; sonda: <roman>."."""
    stripped = " ".join(text_source.split())
    if not stripped:
        return False
    return bool(_RE_PLOT_PROBE_HEADER.match(stripped))


def _looks_like_catalogue_reference(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    if is_clean_reference(stripped):
        return True

    if _RE_SIGLUM.match(stripped):
        return True

    if _looks_like_plot_probe_header(stripped):
        return True

    identifier_pattern = re.compile(
        r"^[A-ZČŠŽĚŘÁÉÍÓÚŮÝ]{1,5}"
        r"\d{1,6}"
        r"(?:[/-][A-Z0-9ČŠŽĚŘÁÉÍÓÚŮÝ]{1,12}){1,8}"
        r"[.]?$",
        flags=re.IGNORECASE,
    )

    if identifier_pattern.match(stripped):
        return True

    table_reference_pattern = re.compile(
        r"\b(?:tb|tab|tabul)[.]?\s*"
        r"[IVXLCDM0-9]+"
        r".{0,20}"
        r"\b(?:č|čís|neg|negativ)[.]?"
        r"\s*\d+",
        flags=re.IGNORECASE,
    )

    if table_reference_pattern.search(stripped):
        return True

    reference_pattern = re.compile(
        r"(?:č[.]?\s*j[.]?|"
        r"č[.]?\s*neg[.]?|"
        r"čís[.]?|"
        r"inv[.]?|"
        r"kat[.]?|"
        r"\bčp[.]?|"
        r"ref[.]?)"
        r"\s*[A-Z0-9/-]*\d",
        flags=re.IGNORECASE,
    )

    if reference_pattern.search(stripped):
        return True

    return False


def _looks_like_date_or_document_reference(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    lowered = stripped.lower()

    has_date = bool(
        re.search(
            r"\b\d{1,2}\s*[./-]\s*\d{1,2}"
            r"(?:\s*[./-]\s*\d{2,4})?\b",
            stripped,
        )
    )

    has_year = bool(
        re.search(
            r"\b(?:18|19|20)\d{2}\b",
            stripped,
        )
    )

    document_marker = bool(
        re.search(
            r"\b(?:datum|date|č[.]?\s*j[.]?|"
            r"podpis|sign|spis|číslo|cislo|ref[.]?)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )

    if has_date and has_year:
        return True

    if document_marker and any(char.isdigit() for char in stripped):
        return True

    return False


_RE_TOC_ENTRY = re.compile(r"^\d{1,2}[.,]\s+\S.*\s(?:\d{1,3}|\S{1,2}\s*[-–]\s*\d{1,3})$")

_DOCUMENT_HEADER_LABELS_DEFAULT = frozenset(
    {
        "obsah",
        "úvod",
        "závěr",
        "literatura",
        "seznam",
        "přílohy",
        "příloha",
        "poznámka",
        "shrnutí",
        "resumé",
    }
)
#: Hand-editable in setup/word_lists.txt [header_labels] since 2026-09-22.
_DOCUMENT_HEADER_LABELS = word_list("header_labels", _DOCUMENT_HEADER_LABELS_DEFAULT)


def _looks_like_document_structure_label(text_source: str) -> bool:
    stripped = " ".join(text_source.split())
    if not stripped:
        return False
    core = stripped.rstrip(" :;.-–—")
    return core.lower() in _DOCUMENT_HEADER_LABELS


def is_domain_notation(text_source: str) -> bool:
    """Archaeological / administrative notation shapes that must escape
    `rule_short_garbage`.

    Consulted at two places in `determine_category()`, both of them narrow: the
    outer guard of `rule_short_garbage`, and the three perplexity routes in
    section 1. The second was added once measurement showed the first was inert
    on its own — with perplexity uncapped, section 1 convicted every grid
    reference before gate 6 was ever reached. It confers no OTHER exemption, and
    that is the whole point of keeping it separate from `is_structured_line()`:
    that predicate is read at eleven places in the rule chain and is an absolute
    veto at the first
    statement of `_has_strong_garbage_evidence()`, so widening it to cover
    notation would also switch off `rule_allcaps`, `rule_garbage_density`,
    `rule_zero_alpha`, `rule_vowelless` and `rule_mid_uppercase`, and could
    promote lines directly to Clear. That is far more blast radius than
    recovering grid references needs.

    Recognises notation, never vocabulary. `II/C`, `Reg.Bez.Aussig.` and `1 ks`
    have a *shape*; `malakofauna` and `Equus caballus` do not, so they stay with
    `rule_short_garbage`.

    This used to add "and separating those from `oueussd` needs a lexicon rather
    than a pattern". That was too strong, and issue #30 has since shown where
    the real boundary sits: `oueussd` is separable by SHAPE -- see
    `_has_shape_garbage_evidence()`, which reaches it and leaves `malakofauna`
    alone. What genuinely needs word knowledge is the phonotactically legal
    residue: separating `malakofauna` from `edelite`, where no character-level
    test can help because nothing about the spelling of either is wrong.
    """
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    # The grid pattern is the loose one — its single-letter segment would
    # otherwise accept all-lowercase OCR mush like `o-e`. Require a capital or a
    # digit there, so `sektlll` and `edelite` stay out. The other four shapes
    # are self-identifying (a colon label, a unit word, a dotted tail), so they
    # do not need the guard and must not be subject to it: `radius prox.sin.` is
    # legitimate notation with no capital in it.
    if _RE_NOTATION_GRID.match(stripped) and _RE_NOTATION_HAS_CODE_CHAR.search(stripped):
        return True

    return bool(
        _is_labelled_notation(stripped)
        or _RE_NOTATION_COUNT.match(stripped)
        or _RE_NOTATION_ABBR.match(stripped)
        or _RE_NOTATION_ABBR_SP.match(stripped)
        or _RE_NOTATION_URL.search(stripped)
    )


def is_structured_line(text_source: str) -> bool:
    stripped = " ".join(text_source.split())

    if not stripped:
        return False

    if is_clean_reference(stripped):
        return True

    if _RE_SIGLUM.match(stripped):
        return True

    if _looks_like_catalogue_reference(stripped):
        return True

    if _looks_like_measurement(stripped):
        return True

    if _looks_like_date_or_document_reference(stripped):
        return True

    if _RE_TOC_ENTRY.match(stripped):
        return True

    if _looks_like_document_structure_label(stripped):
        return True

    return False


# ---------------------------------------------------------------------------


def compute_symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text)


def compute_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


_RE_INITIALS = re.compile(r"^([A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]\.?){1,3}$")
_RE_DOTTED_ABBREV = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ſ]{1,4}(\.[A-Za-zÀ-ÖØ-öø-ſ]{1,4})*$")
#: Units and measurement abbreviations: not judged as words when the program
#: measures how much of a line is real vocabulary. Hand-editable in
#: setup/word_lists.txt [neutral] since 2026-09-22; the literal below is the
#: fallback and is what shipped before the file existed.
_NEUTRAL_LEXICON_DEFAULT = frozenset({"dr", "x", "mm", "cm", "dm", "km", "g", "dkg", "kg", "ha", "hl", "ks", "m", "l"})
_NEUTRAL_LEXICON = word_list("neutral", _NEUTRAL_LEXICON_DEFAULT)
_RE_ROMAN_TOKEN = re.compile(r"^[IVXLCDM]{1,7}$")
_RE_ABBREV_NUM = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ſ]{1,4}[.,]\d+[a-z]?$")


def _is_neutral_token(core: str, raw: str = "", next_core: str = "") -> bool:
    if not any(c.isalnum() for c in core):
        return True
    if sum(c.isdigit() for c in core) / len(core) >= 0.50:
        return True
    if _RE_INITIALS.match(core):
        return True
    if core.lower() in _NEUTRAL_LEXICON:
        return True
    if _RE_ABBREV_NUM.match(core):
        return True
    if raw.rstrip(",;:-–—/)").endswith(".") and _RE_DOTTED_ABBREV.match(core):
        alpha = sum(c.isalpha() for c in core)
        if 2 <= alpha <= 5:
            return True
        if alpha == 1 and next_core and (any(c.isdigit() for c in next_core) or _RE_ROMAN_TOKEN.match(next_core)):
            return True
        if alpha == 1 and raw.rstrip().endswith(":"):
            return True
    return False


_RE_SIGLUM = re.compile(r"^([A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{1,4}\.){1,4}$")

# ── rule_domain_notation ────────────────────────────────────────────────────
# Archaeological / administrative notation that `rule_short_garbage` and the
# section-1 perplexity routes must not trash. Deliberately NARROWER than
# `is_structured_line()`: consulted at those two places only, and conferring no
# other exemption, so it cannot silently disable rule_allcaps,
# rule_garbage_density, rule_zero_alpha, rule_vowelless or rule_mid_uppercase the
# way widening `is_structured_line()` would.
#
# Scope is NOTATION, not vocabulary. Grid/context refs, counts, abbreviation
# chains and labelled refs are shapes a regex can recognise. Latin binomials
# (`Equus caballus`, `Ossa tarsi`) and bare foreign words (`malakofauna`) are
# vocabulary, and are deliberately left to `rule_short_garbage` here.
#
# The scope line used to end "they need a lexicon, not a pattern". Keeping them
# out of THIS predicate is still right -- a notation regex claiming binomials
# would be claiming to solve the harder half of #30 -- but "needs a lexicon" was
# a claim about the whole problem and it does not survive: much of the garbage
# side is separable by shape (`_has_shape_garbage_evidence()`), and only the
# phonotactically legal residue (`edelite` against `malakofauna`) needs word
# knowledge.
#
# Dimensions (`12,5 cm`, `145-167mm`, `0,46m`) are NOT covered: they already
# match `_looks_like_measurement()`, so they never reach rule_short_garbage in
# the first place. A second copy here would only be able to drift from it.

# A grid/context segment: roman numeral, short all-caps run, a single letter, or
# digits with an optional letter suffix. Lowercase word fragments are excluded
# on purpose — that is what keeps `Slaot-o` and `sektlll` out.
_NOTATION_SEGMENT = r"(?:[IVXLCDM]{1,6}|[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]{1,4}|[A-Za-z]|\d{1,3}[a-z]?)"

# `II/C`, `I-VIII-c`, `KK-XIII`, `A/1`, `XIV-2b` — at least one separator.
_RE_NOTATION_GRID = re.compile(rf"^{_NOTATION_SEGMENT}(?:[/\-]{_NOTATION_SEGMENT}){{1,4}}$")


def _fold_diacritics(word: str) -> str:
    """Lowercase and strip combining marks, so `Řez` and `rez` compare equal."""
    lowered = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in lowered if unicodedata.category(c) != "Mn")


# (#30) The label half of a labelled reference is a CLOSED set, not "any word".
# It used to be `[A-Za-z...]{3,20}`, which accepts an OCR-corrupted label just as
# happily as a real one: `Bokalisace: B-XII-c` is `Lokalisace` with B-for-L, and
# it was the single false positive in the reviewer's 30-line sample. A closed set
# is the same technique `_DOCUMENT_HEADER_LABELS` already uses, and it costs
# nothing in recall because the vocabulary of these labels is genuinely small.
_NOTATION_LABELS_DEFAULT = frozenset(
    {
        "lokalisace",
        "lokalizace",
        "sonda",
        "plocha",
        "objekt",
        "vrstva",
        "sektor",
        "kontext",
        "hrob",
        "jáma",
        "čtverec",
        "kvadrant",
        "profil",
        "řez",
        "výkop",
        "nález",
        "inv",
        "kat",
        "sáček",
        "karton",
        "situace",
        "blok",
        "segment",
        "horizont",
        # (#30) Added after the closed lexicon regressed six graded lines to
        # `Trash`, five of which were right before it landed. Measured against
        # `is_domain_notation()` directly:
        #
        #   Orientace: SZ-JV   False -> True   (annotated Clear)
        #   Orientace: SV-JZ   False -> True   (annotated Clear)
        #   Komponenta: H      False -> True   (annotated Noisy)
        #
        # That recovers THREE of the six, not five. The other three are not
        # fixed by adding label words and are deliberately left alone:
        # `XIV: 7` and `XII: 2` satisfy the label shape without being words, so
        # admitting them means admitting Roman numerals as labels -- which is the
        # predicate claiming vocabulary it cannot justify (D2); and
        # `Bokalisace: B-XII-c` is annotated Noisy, so lifting it to Clear was
        # never the right answer either.
        "orientace",
        "komponenta",
    }
)
#: Hand-editable in setup/word_lists.txt [notation_labels] since 2026-09-22. The
#: closure is the point -- see the comment above -- so the file is the place to
#: add a real label, not a reason to reopen the pattern.
_NOTATION_LABELS = word_list("notation_labels", _NOTATION_LABELS_DEFAULT)
_NOTATION_LABELS_FOLDED = frozenset(_fold_diacritics(w) for w in _NOTATION_LABELS)

# `Lokalisace: MM-III`, `sonda: III` — a KNOWN label, then a grid reference.
_RE_NOTATION_LABELLED = re.compile(
    rf"^([A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{{3,20}})\s*:\s*"
    rf"{_NOTATION_SEGMENT}(?:[/\-]{_NOTATION_SEGMENT}){{0,4}}$"
)


def _is_labelled_notation(stripped: str) -> bool:
    """A labelled grid reference whose label is a word we actually recognise."""
    match = _RE_NOTATION_LABELLED.match(stripped)
    return bool(match) and _fold_diacritics(match.group(1)) in _NOTATION_LABELS_FOLDED


# `1 ks`, `2 ks`. Multi-character count units only — a bare single-letter unit
# (`3 m`, `o 5 m`) stays out, matching `_looks_like_measurement`'s own refusal.
_RE_NOTATION_COUNT = re.compile(r"^\d{1,4}\s*(?:ks|kusy|kusů|ex)\.?$", re.IGNORECASE)

# `Reg.Bez.Aussig.` — the same shape as _RE_SIGLUM but with segments up to 8
# characters, so German administrative chains stop failing on "Aussig".
#
# Each segment must START WITH A CAPITAL. Without that this pattern was a hole
# rather than a predicate: `^(?:[A-Za-z]{1,8}\.){2,6}$` puts no constraint on
# CONTENT, so `kfjs.qmwx.zzpl.vvbn.` and `oueussd.nupoy.` matched it as readily
# as `Reg.Bez.Aussig.` — 100% of generated dot-chained garbage was accepted,
# against 0% with the capital required. Spurious periods are among the most
# common OCR artefacts in this corpus, and they concentrate on bad scans, so the
# false-positive rate was highest exactly where a Clear label costs most.
# Lowercase chains are not lost: `č.neg.`, `č.j.`, `s.j.` and `inv.č.` all match
# `_RE_SIGLUM`, so `is_structured_line()` already keeps them out of
# `rule_short_garbage` without this pattern.
_RE_NOTATION_ABBR = re.compile(r"^(?:[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ][A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{0,7}\.){2,6}$")

# `radius prox.sin.` — a word followed by a dotted tail. Requires >= 2 tail
# segments: one is not enough (`vfetennl k.` must stay unmatched).
_RE_NOTATION_ABBR_SP = re.compile(
    r"^[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{2,20}\s+"
    r"(?:[A-Za-zÁČĎÉĚÍŇÓŘŠŤŮÚÝŽáčďéěíňóřšťůúýž]{1,8}\.){2,4}$"
)

_RE_NOTATION_HAS_CODE_CHAR = re.compile(r"[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ0-9]")

# (#30 D33) URL / e-mail. A citation or contact string is NOTATION pointing
# outward, not vocabulary -- and a document-frequency table can never attest a
# citation quoted once. Measured on the stage-8 delivery: with the vocabulary
# lexicon armed, boilerplate URLs that repeat verbatim across many documents
# (`http://www.arub.cz`, 5,309 lines, a page footer) are ALREADY exempt via
# `_has_vocabulary_support()` -- attested by repetition, the same mechanism
# that exempts `ppole`. What survives is the opposite case: a UNIQUE
# bibliographic citation, quoted once or twice, that a document-frequency
# table can never attest by construction. 144 such lines in the stage-8
# at-risk population, none appearing more than 3 times, 0 false positives
# against the other 7,289 at-risk survivors and 0 against the predicate's own
# documented garbage fixtures.
#
# `.search()`, not `.match()` -- unlike the four patterns above. A citation
# sits inside a sentence ("roku 1820 (http://www.hrady.cz/...)."; "3 Zdroj
# https://www.obec-kolicin.cz/historie-obce/"), so anchoring the whole line
# would miss the shape this exists to catch. That is safe specifically
# because the trigger substrings -- a URL scheme, `www.`, an `@`-address, or
# an explicit `e-mail:` label -- essentially never occur in ordinary
# Czech/German archival prose by accident; nothing else in this predicate
# needed that argument, and this is the only shape that relies on it.
# `e-mail:` is matched on the LABEL alone, without requiring a clean address
# after it, because OCR damage sometimes costs the `@` itself
# (`e-mail: officeauappmost.cz`) while leaving the label intact.
_RE_NOTATION_URL = re.compile(
    r"https?://\S+|www\.[\w-]+\.\w{2,4}\S*|\be-?mail\b\s*:|[\w.+-]+@[\w-]+\.[\w.-]+",
    re.IGNORECASE,
)


_DMG_SYMBOLS = frozenset("^»«■□¤§~<>#*@$")
_RE_DMG_APOSTROPHE = re.compile(r"[^\W\d_][’‘][^\W\d_]")
_RE_DMG_DIGIT_IN_WORD = re.compile(r"[a-záčďéěíňóřšťúůýž]{2}\d|\d[a-záčďéěíňóřšťúůýž]{2}")
_RE_DMG_VOWELLESS = re.compile(r"[a-záčďéěíňóřšťúůýž]{3,}$")
_CZ_VOWELS = frozenset("aeiouyáéěíóúůý")
_DMG_SYMBOLS_V8 = frozenset("©®™।")
_RE_DMG_CASE_MIX = re.compile(r"[a-záčďéěíňóřšťúůýž][A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]")


def count_damaged_tokens(text: str) -> int:
    """Count tokens carrying character-level OCR damage."""
    count = 0
    for token in text.split():
        core = token.strip(_STRIP_CHARS)
        if not core:
            continue
        if (
            any(c in _DMG_SYMBOLS for c in core)
            or any(c in _DMG_SYMBOLS_V8 for c in core)
            or _RE_DMG_APOSTROPHE.search(core)
            or _RE_DMG_DIGIT_IN_WORD.search(core)
            or (len(core) >= 4 and not token.endswith(".") and _RE_DMG_CASE_MIX.search(core))
            or (
                not token.rstrip(",;:").endswith(".")
                and _RE_DMG_VOWELLESS.match(core)
                and not (_CZ_VOWELS & set(core))
                and "r" not in core
                and "l" not in core
            )
        ):
            count += 1
    return count


_RE_BIGRAM_RUN = re.compile(r"([^\W\d_]{2})\1\1")

_RE_REF_MARKER = re.compile(
    r"\binv\b|\binv\.|\bkont\b|\bkont\.|\bn[áa]l\b|\bn[áa]l\.|\bobr\b|\bobr\.|"
    r"\btab\b|\btab\.|\bneg\b|\bneg\.|\bmax\.|\bmin\.|"
    r"č\.\s?j|č\.\s?inv|č\.\s?pl|inv\.\s?č|s\.\s?j\b",
    re.IGNORECASE,
)
_RE_MEASUREMENT = re.compile(
    r"\d\s?[.,]?\s?(?:mm|cm|km|kg|ml|ha)\b",
    re.IGNORECASE,
)


def is_clean_reference(text: str) -> bool:
    """True for a catalogue/measurement/inventory line with no OCR damage."""
    if count_damaged_tokens(text) > 0:
        return False
    return bool(_RE_REF_MARKER.search(text) or _RE_MEASUREMENT.search(text))


@functools.lru_cache(maxsize=1)
def quality_word_set() -> "frozenset | None":
    """The vocabulary `compute_valid_ratio` should score against, or None.

    (#30 D26.) `compute_valid_ratio` has always taken a `word_set` and production
    has never passed one, so it falls back to a SHAPE test -- at least 3
    characters, at least 70% alphabetic, nothing strange -- and
    ``compute_valid_ratio("oueussd edelite sektlll")`` returns **1.00**. That
    value feeds `compute_quality_score`, which feeds every threshold in this
    module, so the pipeline's primary "is this text?" signal is fooled precisely
    by the population issue #30 is about.

    Returns None unless ``QUALITY_VOCABULARY_ENABLE`` is set AND a lexicon is
    configured, which keeps the shipped behaviour byte-identical.

    KNOWN INCOMPLETE, and measured before it is extended: this is exact
    attestation only. A damaged-but-readable line scores 0 -- ``1 fraament
    okraie`` is "1 fragment okraje" and every token of it is unattested in its
    damaged form -- so the armed signal punishes recoverable text as hard as
    garbage. `tools/ocr_neighbours.recoverability` draws the distinction this
    lacks. Stage 07b was meant to measure what the gap costs and was void -- this
    function's own cache froze the flag, and D28 fixed the class. Its re-run,
    08d, measured the armed flag as a decisive REJECT on gold: 212 fixes against
    540 breaks, `Clear`-loss 40 -> 180. So the flag stays false, and no neighbour
    index goes into production on the idea alone. Adding the complexity first
    and checking afterwards is how this issue acquired three instrument-level
    errors.
    """
    if not QUALITY_VOCABULARY_ENABLE:
        return None
    lex = token_lexicon()
    return frozenset(lex) if lex else None


def compute_valid_ratio(text: str, word_set: set | None = None) -> float:
    words = text.split()
    if not words:
        return 0.0
    valid = 0
    evaluable = 0
    for wi, word in enumerate(words):
        core = word.strip(_STRIP_CHARS)
        if not core:
            continue
        if word_set is not None:
            evaluable += 1
            if core.lower() in word_set:
                valid += 1
        else:
            next_core = words[wi + 1].strip(_STRIP_CHARS) if wi + 1 < len(words) else ""
            # (#30) Non-evaluable, exactly as a unit is: the token neither helps
            # the ratio nor hurts it. Counting it VALID would raise the score,
            # which is more than "not debuffed".
            if _is_allowed_token(core) or _is_neutral_token(core, word, next_core):
                continue
            evaluable += 1
            if core.lower() in SHORT_VALID_WORDS or core in SINGLE_CHAR_ALLOWED:
                valid += 1
                continue
            alpha = sum(c.isalpha() for c in core)
            has_strange = any(not c.isalnum() and c not in ALLOWED_INTERNAL for c in core)
            if len(core) >= 3 and alpha / len(core) >= 0.70 and not has_strange:
                if _is_mid_uppercase(core):
                    continue
                valid += 1
    if evaluable == 0:
        return 1.0
    return valid / evaluable


def is_non_text(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^\d{3}\s\d{2}\s+[A-ZÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ]", text.strip()):
        return False
    if RE_NON_TEXT.match(text.strip()):
        return True

    stripped = text.strip()
    if " " not in stripped:
        if RE_ARCHIVE_CODE.match(stripped):
            return True
        if RE_ALPHANUM_TOKEN.match(stripped):
            if any(c.isdigit() for c in stripped):
                return True
            if stripped.isupper():
                if "X" in stripped:
                    return True
                if len(stripped) >= 10 and compute_vowel_ratio(stripped) < VOWEL_RATIO_LOW:
                    return True
    else:
        if len(stripped) <= 20 and any(c.isdigit() for c in stripped):
            if RE_ARCHIVE_REF_SPACED.match(stripped):
                return True

    if len(text) < 15 and compute_digit_ratio(text) > 0.5:
        return True
    return False


#: (#30 D27, 2026-09-19) `STRIP_SYMBOL_GLYPHS` -> `_STRIP_CHARS` is a DERIVED
#: constant: `_STRIP_CHARS` is built once at import time from the flag (see the
#: `if STRIP_SYMBOL_GLYPHS:` line above `_STRIP_CHARS`'s declaration), the same
#: way `ROT_GHOSTLIST` and `_LANG_DIACRITICS` are built once from their own
#: sources -- but unlike those two, it was never added to the "not rebuilt by
#: `override_constants()`" list, so overriding the flag alone was a silent
#: no-op. This is exactly how `tools/ab_constant_eval.py` measured
#: `STRIP_SYMBOL_GLYPHS` true vs false as bit-identical on every one of the
#: 2,064 gold rows (issue #30 stage 07c): the flag moved, `_STRIP_CHARS` did
#: not, and every `.strip(_STRIP_CHARS)` call site downstream never saw the
#: difference. Mapping a flag name to the derived attribute it feeds lets
#: `override_constants()` rebuild it in the same pass, restored the same way
#: everything else here already is.
_DERIVED_FROM_FLAG: dict[str, str] = {
    "STRIP_SYMBOL_GLYPHS": "_STRIP_CHARS",
}


#: (#30 D26/D27, 2026-09-20) The SECOND form of the same bug, and the reason
#: `_DERIVED_FROM_FLAG` above was a fix for one instance rather than for the
#: class. A module-level constant built once at import is not the only way a
#: flag's value gets frozen: a ZERO-ARGUMENT `functools.lru_cache` function
#: freezes it too, on its first call, for the life of the process.
#:
#: `quality_word_set()` is exactly that. It reads `QUALITY_VOCABULARY_ENABLE`,
#: takes no arguments, and is cached at `maxsize=1`. `tools/ab_constant_eval.py`
#: runs both arms of an A/B IN ONE PROCESS, reference value first, so the
#: `False` arm caches `None` and the `True` arm is handed the same `None` back.
#: That is how stage 07b measured `QUALITY_VOCABULARY_ENABLE` true vs false as
#: bit-identical on all 2,064 gold rows with 0 discordant rows -- the identical
#: signature 07c produced, from the identical cause, written up at the time as a
#: population-coverage result rather than as an unarmed flag.
#:
#: Registering the flag against the caches it feeds lets `override_constants()`
#: clear them on the way in AND on the way out, so neither arm inherits the
#: other's answer. `_read_token_lexicon` and `_compile_vowel_run` need no entry
#: here: both are keyed on their arguments, so a changed threshold is a changed
#: cache key and they were never able to go stale.
#:
#: Pinned by `test_no_unregistered_zero_arg_cache_reads_a_flag`, which fails if a
#: new zero-argument cache appears in this module without an entry below.
_CACHES_FROM_FLAG: dict[str, tuple[str, ...]] = {
    "QUALITY_VOCABULARY_ENABLE": ("quality_word_set",),
    # `quality_word_set()` resolves through `token_lexicon()`, so the two keys
    # that decide WHICH table it gets have to clear it as well -- otherwise an
    # A/B over the lexicon path or its threshold reads the first arm's table.
    "SHORT_GARBAGE_LEXICON_PATH": ("quality_word_set",),
    "SHORT_GARBAGE_LEXICON_MIN_DF": ("quality_word_set",),
}


def _clear_flag_caches(mod, name: str) -> None:
    """Drop any cached value that was computed from ``name``'s previous value."""
    for fn_name in _CACHES_FROM_FLAG.get(name, ()):
        fn = getattr(mod, fn_name, None)
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


@contextmanager
def override_constants(values, modules=None):
    if modules is None:
        modules = (sys.modules[__name__],)
    saved: list[tuple[object, str, object]] = []
    touched_caches: list[tuple[object, str]] = []
    try:
        for mod in modules:
            for name, value in values.items():
                if name in _CACHES_FROM_FLAG:
                    _clear_flag_caches(mod, name)
                    touched_caches.append((mod, name))
                if hasattr(mod, name):
                    saved.append((mod, name, getattr(mod, name)))
                    setattr(mod, name, value)
                derived_name = _DERIVED_FROM_FLAG.get(name)
                if derived_name is None or not hasattr(mod, derived_name) or not hasattr(mod, "_SYMBOL_GLYPHS"):
                    continue
                base = getattr(mod, derived_name)
                glyphs = mod._SYMBOL_GLYPHS
                extended = base.endswith(glyphs) and glyphs != ""
                if value and not extended:
                    saved.append((mod, derived_name, base))
                    setattr(mod, derived_name, base + glyphs)
                elif not value and extended:
                    saved.append((mod, derived_name, base))
                    setattr(mod, derived_name, base[: -len(glyphs)])
        yield
    finally:
        for mod, name, old in reversed(saved):
            setattr(mod, name, old)
        # Clear again on the way out. Anything computed while the override was
        # active was computed from the overridden value, and the caller is now
        # back on the shipped one -- leaving it cached would leak the arm's
        # answer into whatever runs next, which is the same defect one step
        # later.
        for mod, name in touched_caches:
            _clear_flag_caches(mod, name)


def compute_quality_score(
    valid_word_ratio: float,
    perplexity: float,
    text_length: int,
    weird_ratio: float,
    vowel_ratio: float = 0.40,
    garbage_density: float = 0.0,
    lang_score: float | None = None,
    gibberish_ratio: float = 0.0,
    fused_ratio: float = 0.0,
    ppl_max: float | None = None,
    length_max: float | None = None,
    is_upright_czech: bool = False,
) -> float:
    if ppl_max is None:
        ppl_max = PERPLEXITY_THRESHOLD_MAX
    if length_max is None:
        length_max = QS_LENGTH_MAX

    total_weight = (
        QS_WEIGHT_VALID_WORD
        + QS_WEIGHT_WEIRD
        + QS_WEIGHT_PERPLEXITY
        + QS_WEIGHT_LENGTH
        + QS_WEIGHT_GARBAGE
        + QS_WEIGHT_VOWEL
        + QS_WEIGHT_LANG
        + QS_WEIGHT_GIBBERISH
        + QS_WEIGHT_FUSED
    )

    if total_weight <= 0.0:
        total_weight = 1.0

    norm_ppl = 1.0 - min(perplexity / ppl_max, 1.0)
    norm_len = min(text_length / length_max, 1.0)
    norm_weird = 1.0 - min(weird_ratio, 1.0)

    active_garbage_weight = QS_WEIGHT_GARBAGE
    if text_length <= 12 and weird_ratio == 0.0 and garbage_density < max(QS_GARBAGE_NORM_MAX, 1e-9):
        active_garbage_weight = active_garbage_weight / 2.0

    norm_garbage = 1.0 - min(garbage_density / max(QS_GARBAGE_NORM_MAX, 1e-9), 1.0)

    vr = vowel_ratio
    if vr < VOWEL_RATIO_LOW:
        norm_vowel = (vr / VOWEL_RATIO_LOW) if VOWEL_RATIO_LOW > 0 else 1.0
    elif vr > VOWEL_RATIO_HIGH:
        span = max(1.0 - VOWEL_RATIO_HIGH, 1e-9)
        norm_vowel = max(0.0, 1.0 - (vr - VOWEL_RATIO_HIGH) / span)
    else:
        norm_vowel = 1.0

    norm_lang = lang_score if lang_score is not None else 0.5
    norm_gibb = 1.0 - min(gibberish_ratio, 1.0)
    norm_fused = 1.0 - min(fused_ratio, 1.0)

    base_score = (
        QS_WEIGHT_VALID_WORD * valid_word_ratio
        + QS_WEIGHT_WEIRD * norm_weird
        + QS_WEIGHT_PERPLEXITY * norm_ppl
        + QS_WEIGHT_LENGTH * norm_len
        + active_garbage_weight * norm_garbage
        + QS_WEIGHT_VOWEL * norm_vowel
        + QS_WEIGHT_LANG * norm_lang
        + QS_WEIGHT_GIBBERISH * norm_gibb
        + QS_WEIGHT_FUSED * norm_fused
    )

    if active_garbage_weight != QS_WEIGHT_GARBAGE:
        base_score += QS_WEIGHT_GARBAGE - active_garbage_weight

    base_score = base_score / total_weight

    short_penalty = 0.0
    if text_length <= 12 and (weird_ratio > 0.0 or garbage_density >= QS_GARBAGE_NORM_MAX):
        short_penalty = SHORT_NOISY_QS_PENALTY

    final_score = max(0.0, base_score - short_penalty)
    return min(1.0, final_score)


# Advisory, evaluated last so both the constants and uncoupled_witness_warning()
# exist. See _warn_uncoupled_witness for the measurement behind it.
_warn_uncoupled_witness()
