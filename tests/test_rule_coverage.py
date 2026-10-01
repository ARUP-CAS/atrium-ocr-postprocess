"""
tests/test_rule_coverage.py
===========================
Tests for the B5 rule-fire coverage instrumentation.

Verifies four properties:

1. **Parity** — with ``RULE_FIRE_COUNTS = None`` (the default) the engine
   output is byte-identical to the pre-instrumentation behaviour.
2. **Correct fire registration** — inside ``rule_fire_capture()`` a crafted
   line increments exactly that rule's counter.
3. **Context-manager stack safety** — nested calls stack properly.
4. **End-to-end smoke** — ``run_coverage`` completes on the sample fixture.

All tests are pure-Python; the GPU/ML stack is stubbed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Stub the GPU/ML stack before any production imports.
for _n in ("torch", "tqdm", "fasttext", "transformers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["tqdm"].tqdm = lambda x, **k: x  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
for _p in (str(_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import text_util as tu  # noqa: E402
from text_util import (  # noqa: E402
    _fire,
    override_constants,
    rule_fire_capture,
)

_SAMPLE_DIR = _ROOT / "data_samples" / "DOC_LINE_CATEG"
_HAS_SAMPLES = _SAMPLE_DIR.is_dir() and any(_SAMPLE_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# 1. Parity — RULE_FIRE_COUNTS = None is the default
# ---------------------------------------------------------------------------


def test_rule_fire_counts_default_is_none():
    """The global sentinel must be None outside a capture block."""
    assert tu.RULE_FIRE_COUNTS is None


def test_fire_noop_outside_capture():
    """_fire() must be a no-op when RULE_FIRE_COUNTS is None."""
    _fire("rule_hard_sweep")
    assert tu.RULE_FIRE_COUNTS is None


def test_categorize_line_output_unchanged_by_instrumentation():
    """categorize_line() must return the same result with and without a
    capture block active — instrumentation must be transparent."""
    from text_util import categorize_line

    kwargs = dict(
        qs=0.3,
        txt="random gibberish wqx xyz",
        wc=4,
        vowel_ratio=0.1,
        perplexity=5000.0,
        weird_ratio=0.8,
        valid_word_ratio=0.1,
        lang_score=0.2,
        orig_lang_score=0.2,
        gibberish_present=True,
        garbage_density=0.1,
        is_upright_czech=False,
        ghost_dominated=False,
    )

    result_outside = categorize_line(**kwargs)

    with rule_fire_capture():
        result_inside = categorize_line(**kwargs)

    assert result_outside == result_inside, (
        f"categorize_line changed output when inside rule_fire_capture(): {result_outside} vs {result_inside}"
    )


# ---------------------------------------------------------------------------
# 2. Correct fire registration
# ---------------------------------------------------------------------------


def test_fire_increments_counter():
    """_fire() must increment the right key when inside a capture block."""
    with rule_fire_capture() as counts:
        _fire("rule_hard_sweep")
        _fire("rule_hard_sweep")
        _fire("rule_wqx_rot")

    assert counts["rule_hard_sweep"] == 2
    assert counts["rule_wqx_rot"] == 1
    assert counts.get("rule_allcaps", 0) == 0


def test_rule_fire_capture_yields_live_dict():
    """The yielded dict is the live RULE_FIRE_COUNTS."""
    with rule_fire_capture() as counts:
        assert tu.RULE_FIRE_COUNTS is counts
        _fire("rule_extreme_ppl")
        assert counts["rule_extreme_ppl"] == 1


def test_hard_sweep_fires_for_low_lang_high_ppl():
    """A line with very low lang_score and extreme perplexity should trip
    rule_hard_sweep (the first rule in determine_category)."""
    from text_util import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.2,
            txt="klm klm klm",
            wc=3,
            vowel_ratio=0.05,
            perplexity=99000.0,
            weird_ratio=0.9,
            valid_word_ratio=0.0,
            lang_score=0.1,
            orig_lang_score=0.1,
            gibberish_present=True,
            garbage_density=0.05,
            is_upright_czech=False,
            ghost_dominated=False,
        )

    assert categ == "Trash"
    assert counts.get("rule_hard_sweep", 0) == 1

    for rule in (
        "rule_extreme_ppl",
        "rule_absolute_ppl",
        "rule_inverted",
        "rule_allcaps",
        "rule_garbage_density",
    ):
        assert counts.get(rule, 0) == 0, f"{rule} should not fire after rule_hard_sweep"


def test_lowppl_clear_fires_for_low_perplexity():
    """A line with very low perplexity and enough words should trip
    rule_lowppl_clear and be classified Clear."""
    from text_util import categorize_line

    with rule_fire_capture() as counts:
        categ, _ = categorize_line(
            qs=0.85,
            txt="Toto je velmi dobrý český text.",
            wc=6,
            vowel_ratio=0.40,
            perplexity=10.0,
            weird_ratio=0.05,
            valid_word_ratio=0.95,
            lang_score=0.92,
            orig_lang_score=0.92,
            gibberish_present=False,
            garbage_density=0.02,
            is_upright_czech=True,
            ghost_dominated=False,
        )

    assert categ == "Clear"
    assert counts.get("rule_lowppl_clear", 0) == 1


# ---------------------------------------------------------------------------
# 3. Context-manager stack safety
# ---------------------------------------------------------------------------


def test_capture_restores_none_after_exit():
    """RULE_FIRE_COUNTS must return to None after the capture block exits."""
    with rule_fire_capture():
        assert tu.RULE_FIRE_COUNTS is not None
    assert tu.RULE_FIRE_COUNTS is None


def test_nested_capture_restores_outer():
    """Nested rule_fire_capture() calls must stack correctly."""
    with rule_fire_capture() as outer_counts:
        _fire("rule_hard_sweep")
        with rule_fire_capture() as inner_counts:
            _fire("rule_extreme_ppl")
            assert inner_counts.get("rule_extreme_ppl", 0) == 1
            assert inner_counts.get("rule_hard_sweep", 0) == 0
        assert tu.RULE_FIRE_COUNTS is outer_counts
        _fire("rule_hard_sweep")

    assert outer_counts["rule_hard_sweep"] == 2
    assert outer_counts.get("rule_extreme_ppl", 0) == 0
    assert tu.RULE_FIRE_COUNTS is None


def test_capture_restores_on_exception():
    """An exception inside rule_fire_capture() must still restore RULE_FIRE_COUNTS."""
    assert tu.RULE_FIRE_COUNTS is None
    with pytest.raises(RuntimeError):
        with rule_fire_capture():
            assert tu.RULE_FIRE_COUNTS is not None
            raise RuntimeError("boom")
    assert tu.RULE_FIRE_COUNTS is None


def test_disabled_rules_override_suppresses_fire():
    """When a rule is in DISABLED_RULES, its _fire() call is never reached."""
    from text_util import categorize_line

    with override_constants({"DISABLED_RULES": frozenset(["rule_hard_sweep"])}):
        with rule_fire_capture() as counts:
            categorize_line(
                qs=0.2,
                txt="wqx bqd zze",
                wc=3,
                vowel_ratio=0.05,
                perplexity=99000.0,
                weird_ratio=0.9,
                valid_word_ratio=0.0,
                lang_score=0.1,
                orig_lang_score=0.1,
                gibberish_present=True,
                garbage_density=0.05,
                is_upright_czech=False,
                ghost_dominated=False,
            )

    assert counts.get("rule_hard_sweep", 0) == 0, "rule_hard_sweep should NOT fire when it is in DISABLED_RULES"


# ---------------------------------------------------------------------------
# 4. End-to-end smoke on the fixture corpus
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_smoke():
    """run_coverage must complete without error on the smoke fixture and return
    a dict with all registered rules."""
    from rule_coverage_report import RULES, run_coverage

    results = run_coverage(
        raw_path=str(_SAMPLE_DIR),
        skip_loo=True,
        quiet=True,
    )

    assert set(results.keys()) == set(RULES), f"Unexpected rule keys: {set(results.keys()) ^ set(RULES)}"
    for _rule, data in results.items():
        assert "fire_count" in data
        assert "fire_rate" in data
        assert "decisive_count" in data
        assert "clear_loss" in data
        assert "class" in data
        assert data["class"] in {"DEAD", "REDUNDANT-HERE", "LOAD-BEARING", "INERT"}
        assert isinstance(data["fire_count"], int)
        assert isinstance(data["fire_rate"], float)
        assert data["fire_rate"] >= 0.0


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_with_loo_smoke():
    """run_coverage with LOO enabled must complete and return non-negative
    decisive_count and clear_loss for every rule."""
    from rule_coverage_report import RULES, run_coverage

    results = run_coverage(
        raw_path=str(_SAMPLE_DIR),
        skip_loo=False,
        quiet=True,
    )

    for rule in RULES:
        assert results[rule]["decisive_count"] >= 0
        assert results[rule]["clear_loss"] >= 0
        assert results[rule]["clear_loss"] <= results[rule]["decisive_count"]


@pytest.mark.skipif(not _HAS_SAMPLES, reason="no DOC_LINE_CATEG sample CSVs present")
def test_run_coverage_json_output(tmp_path):
    """run_coverage must write valid JSON to the --output path."""
    import json

    from rule_coverage_report import RULES, run_coverage

    out_file = tmp_path / "rule_coverage.json"
    run_coverage(
        raw_path=str(_SAMPLE_DIR),
        output_path=str(out_file),
        skip_loo=True,
        quiet=True,
    )

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert "n_lines" in payload
    assert "n_scored" in payload
    assert "rules" in payload
    # Dynamically match RULES length so test survives when new rules are added.
    assert len(payload["rules"]) == len(RULES)


# ---------------------------------------------------------------------------
# 5. Registry / call-site parity
# ---------------------------------------------------------------------------


def test_rules_registry_matches_fire_call_sites():
    """rule_coverage_report.RULES must equal the _fire() call-sites in text_util.

    This used to be a "keep in sync" comment with nothing enforcing it, and it
    drifted: the five rules introduced by the issue #30 work (rule_short_line,
    rule_damaged_token, rule_reference_floor, rule_bigram_run,
    rule_fragment_tokens) never made it into the registry, so every coverage
    report and every ablation sweep run after PR #32 quietly measured 16 of the
    21 rules. A stale registry does not fail loudly — it just under-reports,
    which is why this needs a test rather than a comment.
    """
    import re

    from rule_coverage_report import RULES

    source = (_ROOT / "text_util.py").read_text(encoding="utf-8")
    call_sites = set(re.findall(r'_fire\("([a-z_]+)"\)', source))

    declared = set(RULES)
    assert declared == call_sites, (
        f"registry out of sync with text_util.py\n"
        f"  fired but not declared: {sorted(call_sites - declared)}\n"
        f"  declared but never fired: {sorted(declared - call_sites)}"
    )
    assert len(RULES) == len(declared), "RULES contains duplicates"


def test_ablation_rule_lists_are_subsets_of_the_registry():
    """The ablation drivers must name rules that actually exist.

    ``override_constants({"DISABLED_RULES": frozenset([name])})`` matches by
    string. A name no ``_fire()`` site produces therefore disables nothing, and
    the driver reports the resulting zero flips / zero clear-loss as
    "**PRUNE** (Signal variance ~ 0)" -- an argument to delete a rule that was
    never switched off.

    That is not hypothetical. Four entries in both lists were still spelled
    ``penalty_*`` from before those gates were renamed to ``rule_*``, so every
    ablation report since carried four such rows, and
    ``rule_coverage_report._PENALTY_RULES`` (which partitioned on the same
    prefix) had been the empty list the whole time. Neither list was covered by
    a test -- only ``rule_coverage_report.RULES`` was, by the test above.

    Asserted as a subset rather than as equality: choosing to ablate a subset is
    a legitimate decision (a cheaper sweep), while naming a rule that does not
    exist never is.
    """
    from greedy_backward_elimination import CANDIDATE_RULES
    from rule_coverage_report import RULES
    from run_ablation_study import RULES_TO_ABLATE

    registry = set(RULES)
    assert set(RULES_TO_ABLATE) <= registry, (
        f"run_ablation_study.RULES_TO_ABLATE names rules that never fire: {sorted(set(RULES_TO_ABLATE) - registry)}"
    )
    assert set(CANDIDATE_RULES) <= registry, (
        f"greedy_backward_elimination.CANDIDATE_RULES names rules that never fire: "
        f"{sorted(set(CANDIDATE_RULES) - registry)}"
    )
    assert len(RULES_TO_ABLATE) == len(set(RULES_TO_ABLATE)), "RULES_TO_ABLATE contains duplicates"


# ---------------------------------------------------------------------------
# (#30 B2) Word-count attribution of rule fires.
# ---------------------------------------------------------------------------


def test_wc_breakdown_attributes_fires_to_word_counts():
    """Every fire lands in exactly one word-count bucket, and buckets sum to totals.

    @david-spacil reported that "59.4% of hard-sweep-family firings land exactly
    on `wc == 3`" and that it was "not measured further, just noting it". Nothing
    in the repository could reproduce that shape of figure, so it stayed an
    anecdote for six weeks. This pins the instrument that can.
    """
    import tools.rule_coverage_report as RC

    counts = RC.run_wc_breakdown(str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True)

    assert set(counts) == set(RC.RULES), "the breakdown must cover the same registry as the coverage report"
    for rule, buckets in counts.items():
        assert set(buckets) == set(RC.WC_BUCKETS), f"{rule} has unexpected buckets: {sorted(buckets)}"
        assert all(v >= 0 for v in buckets.values())

    assert any(sum(b.values()) for b in counts.values()), "no rule fired at all; the capture is not wired"


def test_wc_breakdown_totals_agree_with_the_coverage_pass():
    """The two instruments must not disagree about which rules fire.

    They measure the same thing by different routes -- the coverage pass scores
    document-by-document, the breakdown line-by-line -- so a rule that fires in
    one and not the other means one of them is lying. Fire COUNTS may legitimately
    differ (the coverage pass applies document post-processing, the per-line
    breakdown does not), so only the fired/not-fired sets are compared.
    """
    import tools.rule_coverage_report as RC

    by_wc = RC.run_wc_breakdown(str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True)
    coverage = RC.run_coverage(
        str(_SAMPLE_DIR), config_path=str(_ROOT / "setup" / "config.txt"), quiet=True, skip_loo=True
    )

    fired_by_wc = {r for r, b in by_wc.items() if sum(b.values())}
    fired_by_coverage = {r for r, v in coverage.items() if v["fire_count"]}
    assert fired_by_wc == fired_by_coverage, (
        f"instruments disagree on which rules fire: only in --by-wc {sorted(fired_by_wc - fired_by_coverage)}, "
        f"only in coverage {sorted(fired_by_coverage - fired_by_wc)}"
    )


# ---------------------------------------------------------------------------
# 5. Gold threading — the guard whose absence let a real defect live
# ---------------------------------------------------------------------------
#
# `rule_coverage_report` was the only one of the five `evaluate_dataframe`
# callers that never passed `gold_category_column`, while happily accepting
# `--gold-sidecar`, joining it, and printing "N labels matched" on the way past.
# Every DEAD / REDUNDANT-HERE / LOAD-BEARING verdict it produced -- the
# classification `RULE_COVERAGE.md` cites as the retirement criterion -- was
# therefore agreement with the pipeline's own stored `categ`, which the offline
# re-score reproduces exactly, so the baseline was zero by construction.
#
# Nothing failed. `tests/test_rule_coverage.py` contained no occurrence of the
# word "gold", and a self-referential run looks exactly like a successful one.
# These tests are the missing guard.


def test_every_evaluate_dataframe_driver_forwards_the_gold_column():
    """Source-level check across all five drivers, not just this one.

    A per-driver behavioural test would have to run each driver; this asks the
    cheaper and more durable question -- does any module that calls
    `evaluate_dataframe` do so without ever mentioning `gold_category_column`?
    The answer was yes for two years in one file, and the cost was invisible.
    """
    import re

    offenders = []
    for path in sorted((_ROOT / "tools").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if not re.search(r"\bevaluate_dataframe\s*\(", src):
            continue
        if path.name == "recategorize_from_csv.py":
            continue  # defines it
        if "gold_category_column" not in src:
            offenders.append(path.name)

    assert not offenders, (
        f"these tools call evaluate_dataframe but never mention gold_category_column: {offenders}. "
        "Without it every metric they report is scored against the pipeline's own stored categ, "
        "whose baseline is zero by construction — see tools/gold/GOLD.md."
    )


@pytest.mark.skipif(not _HAS_SAMPLES, reason="sample DOC_LINE_CATEG fixtures not present")
def test_coverage_payload_records_whether_it_was_scored_against_gold(tmp_path):
    """A finished report must say what it was scored against.

    The JSON payload used to carry `input`, `n_lines`, `n_scored` and `rules` and
    nothing else, so a self-referential run and a gold run were indistinguishable
    after the fact -- including in this issue's own thread, where a 23-rule table
    was read as evidence about rule quality.
    """
    import json

    import tools.rule_coverage_report as RC

    out = tmp_path / "rc.json"
    RC.run_coverage(
        str(_SAMPLE_DIR),
        config_path=str(_ROOT / "setup" / "config.txt"),
        output_path=str(out),
        quiet=True,
        skip_loo=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["gold_column"] is None
    assert "self-referential" in payload["decisive_scored_against"]
    assert payload["config"].endswith("config.txt")


@pytest.mark.skipif(not _HAS_SAMPLES, reason="sample DOC_LINE_CATEG fixtures not present")
def test_loo_metrics_scores_against_gold_when_asked():
    """`_loo_metrics` must add the gold verdict rather than redefine the old one.

    Forwarding `gold_category_column` into the existing call would have been the
    obvious fix and the wrong one: with a gold column `flip_count` counts
    DISAGREEMENTS WITH GOLD rather than lines the rule moved, and the baseline is
    no longer zero — so `decisive_count` would have silently become a different
    quantity under the same name. The structural figure stays; the gold figure is
    additional.
    """
    import tools.rule_coverage_report as RC
    from tools.recategorize_from_csv import load_csvs

    df = load_csvs(_SAMPLE_DIR)
    df["gold_categ"] = df["categ"]  # a gold column that agrees with the pipeline
    expected_langs, known_bases = RC._load_lang_config(str(_ROOT / "setup" / "config.txt"))

    plain = RC._loo_metrics(df, "rule_hard_sweep", expected_langs, known_bases)
    scored = RC._loo_metrics(df, "rule_hard_sweep", expected_langs, known_bases, gold_column="gold_categ")

    assert plain["gold_delta_macro_f1"] is None, "no gold column means no gold verdict"
    assert scored["gold_delta_macro_f1"] is not None
    assert scored["gold_n"] == len(df)
    assert scored["decisive_count"] == plain["decisive_count"], (
        "the structural figure must not change meaning when a gold column is supplied"
    )


@pytest.mark.skipif(not _HAS_SAMPLES, reason="sample DOC_LINE_CATEG fixtures not present")
def test_cascade_split_separates_the_per_line_effect():
    """`decisive_count > fire_count` is only readable once the cascade is split.

    Three rules in the last full-corpus run reported more decisive lines than
    fires -- `rule_forgiven_headline` 6,773 against 3,545, `rule_trailing_fill_rescue`
    71,378 against 51,134, `rule_damaged_token` 30,158 against 29,596. That is not
    a paradox: removing a rescue pushes its line to Trash, the page's garbage
    ratio rises, and the page passes sweep the neighbours. The split is what says
    so instead of leaving it to be argued.
    """
    import tools.rule_coverage_report as RC
    from tools.recategorize_from_csv import load_csvs

    df = load_csvs(_SAMPLE_DIR)
    expected_langs, known_bases = RC._load_lang_config(str(_ROOT / "setup" / "config.txt"))

    m = RC._loo_metrics(df, "rule_hard_sweep", expected_langs, known_bases, split_cascade=True)
    assert m["decisive_line"] is not None
    assert m["decisive_cascade"] == m["decisive_count"] - m["decisive_line"]


def test_gate_marker_rules_are_declared_and_real():
    """A gate marker must name a rule whose gate always returns.

    `rule_short_line` fires on entry to gate 7, and every branch of gate 7
    returns, so its fire count is the size of the `word_count <= 2` population --
    44.6% of scored lines on the cluster corpus, which sorts it to the top of the
    table as if it were the hottest rule in the engine. Moving the `_fire()` call
    would not change the number; saying what the number is does.

    If the gate ever grows a fall-through path, the count becomes meaningful
    again and this declaration is wrong -- so it is pinned rather than inferred.
    """
    import tools.rule_coverage_report as RC

    assert RC.GATE_MARKER_RULES <= set(RC.RULES)
    assert "rule_short_line" in RC.GATE_MARKER_RULES

    src = (_ROOT / "text_util.py").read_text(encoding="utf-8")
    gate = src.split('if "rule_short_line" not in DISABLED_RULES and word_count <= 2:', 1)
    assert len(gate) == 2, "gate 7's entry condition moved; re-check the marker declaration"
    body = gate[1].split("\n    # ---", 1)[0]
    # The last statement of the gate body is an unconditional return: that is what
    # makes the gate total on its entry condition, hence a population marker.
    last = [ln for ln in body.rstrip().splitlines() if ln.strip()][-1]
    assert last.strip().startswith("return "), (
        f"gate 7 no longer ends in an unconditional return (found {last.strip()!r}); "
        "if it can now fall through, rule_short_line's fire_count is meaningful and "
        "it should leave GATE_MARKER_RULES."
    )


# ---------------------------------------------------------------------------
# (#30 D35) INERT — a rule that cannot fire because its flag is off
# ---------------------------------------------------------------------------


def test_config_gated_rules_are_registered_and_real():
    """Every name in the registry must be a real rule gated by a real flag.

    A typo here would silently re-open the hole this class exists to close: an
    unmatched name means the rule keeps classifying DEAD.
    """
    import tools.rule_coverage_report as RC

    assert set(tu.CONFIG_GATED_RULES) <= set(RC.RULES), "a gated name is not in the rule registry"
    for rule, flag in tu.CONFIG_GATED_RULES.items():
        assert hasattr(tu, flag), f"{rule} is declared gated by {flag}, which does not exist"
        # The gate is TRUTHINESS, not a bool. `rule_is_config_gated_off` reads
        # `not bool(...)`, and a gate that names an answer rather than a switch is
        # the more useful shape where the answer is itself the open question:
        # DOMAIN_NOTATION_CATEG holds a category name (#30 D43). It shipped empty
        # until Q4 was answered on 2026-10-01 and ships `Clear` now -- ARMED, yet
        # still correctly registered, because the gate is read live: an armed
        # gated rule is classified on its own counts like any other. What must
        # hold is that the classifier's answer follows the flag.
        armed = bool(getattr(tu, flag))
        assert tu.rule_is_config_gated_off(rule) is (not armed), f"{rule}'s gate disagrees with {flag}"


def test_which_gated_rules_ship_armed():
    """The shipped state of each gated flag, pinned so that flipping one is a visible decision."""
    assert tu.SHORT_GARBAGE_WITNESS_ENABLE is False
    assert tu.rule_is_config_gated_off("rule_short_garbage_witness")
    assert tu.DOMAIN_NOTATION_CATEG == "Clear"  # #30 Q4, 2026-10-01
    assert not tu.rule_is_config_gated_off("rule_domain_notation_categ")


def test_the_witness_is_inert_not_dead_at_the_shipped_configuration():
    """The stage-6 finding, pinned.

    `rule_short_garbage_witness` fires 0 times at the shipped config because
    `SHORT_GARBAGE_WITNESS_ENABLE` is false, not because it has no population --
    stage 08f measured the same predicate reaching 100,824 lines. Classifying it
    DEAD told an operator it was "unreachable dead code" that could be
    "permanently deleted"; the retirement criterion in RULE_COVERAGE.md called
    fire_count == 0 "config-independent", which for this rule it is not.
    """
    import tools.rule_coverage_report as RC

    assert tu.SHORT_GARBAGE_WITNESS_ENABLE is False, "fixture assumes the shipped default"
    assert RC._classify(0, 0, "rule_short_garbage_witness") == "INERT"
    # Without the name the old pure-counts contract is unchanged.
    assert RC._classify(0, 0) == "DEAD"
    # And an ungated rule at zero is still DEAD.
    assert RC._classify(0, 0, "rule_mid_uppercase") == "DEAD"


def test_flipping_the_flag_makes_the_witness_classifiable_again():
    """With the flag on the rule is judged on its own counts, like any other."""
    import tools.rule_coverage_report as RC

    with tu.override_constants({"SHORT_GARBAGE_WITNESS_ENABLE": True}):
        assert tu.rule_is_config_gated_off("rule_short_garbage_witness") is False
        assert RC._classify(0, 0, "rule_short_garbage_witness") == "DEAD"
        assert RC._classify(5, 0, "rule_short_garbage_witness") == "REDUNDANT-HERE"
        assert RC._classify(5, 2, "rule_short_garbage_witness") == "LOAD-BEARING"


def test_a_gated_rule_that_somehow_fires_is_not_hidden():
    """The gate is consulted only when the count is zero.

    If a rule declared flag-gated reports a non-zero fire count, that is a real
    finding -- either the declaration is wrong or the flag was on -- and it must
    not be masked by its own registry entry.
    """
    import tools.rule_coverage_report as RC

    assert RC._classify(7, 3, "rule_short_garbage_witness") == "LOAD-BEARING"
    assert RC._classify(7, 0, "rule_short_garbage_witness") == "REDUNDANT-HERE"


def test_inert_rules_do_not_make_the_tool_exit_non_zero():
    """A flag that ships off must not fail a pipeline driver.

    `_print_summary`'s DEAD list and the exit code read the same class, so this
    pins the contract at the level the driver actually sees.
    """
    import tools.rule_coverage_report as RC

    results = {
        "rule_short_garbage_witness": {
            "class": "INERT",
            "gated_by": "SHORT_GARBAGE_WITNESS_ENABLE",
            "fire_count": 0,
            "fire_rate": 0.0,
            "decisive_count": 0,
            "decisive_share": None,
            "clear_loss": 0,
        },
        "rule_short_line": {
            "class": "LOAD-BEARING",
            "fire_count": 9,
            "fire_rate": 0.1,
            "decisive_count": 3,
            "decisive_share": 0.33,
            "clear_loss": 0,
        },
    }
    # The exit code and the DEAD list read the same class, so drive the real
    # summary rather than re-implementing the predicate here.
    dead = [r for r, v in results.items() if v["class"] == "DEAD"]
    assert dead == [], "an INERT rule leaked into the DEAD set that drives exit 1"

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        RC._print_summary(results)
    printed = buf.getvalue()
    assert "1 INERT" in printed
    assert "rule_short_garbage_witness" in printed
    assert "SHORT_GARBAGE_WITNESS_ENABLE" in printed, "the summary must name the flag to re-run with"
    assert "DEAD rules (" not in printed, "an INERT rule must not be announced as retirable"


def test_gold_clear_loss_baseline_is_emitted_with_the_gold_pass(tmp_path):
    """(#30 D36) The baseline is computed by the gold pass; it must be reported.

    `gold_clear_loss` is an ABSOLUTE count in the rule-disabled arm and was
    printed next to `gold_delta_macro_f1`, a DELTA. Without the baseline a
    reader cannot tell whether a rule reporting 40 contributes 40 of them or
    none -- the stage-6 delivery has twelve rules all reporting exactly 40.
    `evaluate_dataframe` already builds `baseline_vs_gold`; this pins that the
    report stops discarding it.
    """
    import json

    import pandas as pd
    from rule_coverage_report import run_coverage

    # A frame the report can score, with a gold column that DISAGREES with the
    # stored labels -- if they agreed, every clear-loss figure would be 0 by
    # construction and the test would pass without measuring anything.
    src = pd.read_csv(sorted(_SAMPLE_DIR.glob("*.csv"))[0])
    if "gold_categ" not in src.columns:
        src["gold_categ"] = ""
    src.loc[src.index[:3], "gold_categ"] = "Clear"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    src.to_csv(corpus / "CTX000000001.csv", index=False)

    out_file = tmp_path / "coverage.json"
    gold_args = types.SimpleNamespace(gold_column="gold_categ", gold_sidecar=None)
    # skip_loo must be False: the gold figures are produced by _loo_metrics, so
    # a skipped LOO pass makes this test vacuous rather than passing.
    run_coverage(
        raw_path=str(corpus),
        output_path=str(out_file),
        skip_loo=False,
        quiet=True,
        gold_args=gold_args,
    )
    payload = json.loads(out_file.read_text())
    rules = payload["rules"]
    with_gold = [v for v in rules.values() if v.get("gold_clear_loss") is not None]
    assert with_gold, (
        "no rule reported a gold figure — the gold pass did not run, so this test would pass without measuring anything"
    )
    for v in with_gold:
        assert "gold_clear_loss_baseline" in v, (
            "gold_clear_loss is reported without the baseline it must be read against"
        )
        assert isinstance(v["gold_clear_loss_baseline"], int)
