"""Golden tests for the intent engine.

These lock in the behaviours the NLP guide asks for. Because the engine is
rule-based, any lexicon change shows up here as a diff rather than as drift.
"""

from __future__ import annotations

import pytest

from lead_triage.core.text import find_negation_spans, split_clauses
from lead_triage.nlp import (
    BudgetStatus,
    BuyingStage,
    DecisionAuthority,
    IntentExtractor,
    IntentType,
    PainSeverity,
    Urgency,
)


@pytest.fixture(scope="module")
def ex() -> IntentExtractor:
    return IntentExtractor()


# ------------------------------------------------- clause & negation
def test_contrast_marker_starts_a_new_operative_clause():
    clauses = split_clauses("This is a priority, but we have no budget.")
    assert len(clauses) == 2
    assert clauses[1].is_operative and clauses[1].contrast_marker == "but"
    assert not clauses[0].is_operative


def test_negation_scope_stops_at_until():
    # "no budget until next quarter" negates the budget, not the quarter.
    text = "we don't have budget until next quarter"
    spans = find_negation_spans(text)
    quarter = text.index("next quarter")
    assert not any(a <= quarter < b for a, b in spans)


def test_negation_scope_stops_at_a_contrast_marker():
    text = "no real budget yet but sharp and might grow"
    spans = find_negation_spans(text)
    sharp = text.index("sharp")
    assert not any(a <= sharp < b for a, b in spans)


# ------------------------------------------------------ core readings
def test_strong_buyer_extracts_every_dimension(ex):
    p = ex.extract(
        "We're a full-service marketing agency, 23 people. Summarizing call "
        "recordings into crm notes is eating our week. Want it automated end to "
        "end. Budget approved, ready to pilot in the next 2 weeks. I make the "
        "call here."
    )
    assert p.intent_type is IntentType.PURCHASE
    assert p.buying_stage is BuyingStage.PURCHASE
    assert p.budget_status is BudgetStatus.APPROVED
    assert p.urgency is Urgency.HIGH
    assert p.decision_authority is DecisionAuthority.EXPLICIT
    assert p.pain_severity is PainSeverity.HIGH
    assert p.timeline_days == 14
    assert "Call summarisation" in p.problems
    assert not p.negative_signals
    assert p.confidence > 0.8


def test_the_guides_contradiction_example(ex):
    """Guide §8: do not average contradictory signals."""
    p = ex.extract("This is a priority, but we don't have budget until next quarter.")
    assert p.pain_severity is PainSeverity.HIGH          # priority survives
    assert p.budget_status is BudgetStatus.NOT_LOCKED    # deferred, not absent
    assert p.urgency is Urgency.MODERATE                 # the quarter survives
    assert p.timeline_days == 90
    assert p.contradictions, "the budget conflict must be recorded, not averaged"


def test_negation_flips_rather_than_suppresses(ex):
    """Guide §9's inverse: negated is not the same as missing."""
    p = ex.extract("very early startup, 3 people, no real budget yet but sharp.")
    assert p.budget_status is BudgetStatus.NO_BUDGET
    assert "NO_BUDGET" in {n.code for n in p.negative_signals}


def test_post_contrast_clause_carries_the_operative_meaning(ex):
    p = ex.extract("Car dealership wanting a lead chatbot. "
                   "Not your usual client but we have money to spend.")
    assert p.budget_status is BudgetStatus.ALLOCATED
    assert "OUT_OF_ICP" in {n.code for n in p.negative_signals}


# ------------------------------------------------- intent type routing
@pytest.mark.parametrize(
    "note,expected",
    [
        ("Not looking to buy — I'm a developer looking for a role. Attaching my CV.",
         IntentType.EMPLOYMENT),
        ("hi! CS student, i love what you do. could you send a free template?",
         IntentType.EDUCATIONAL),
        ("You have WON $1,000,000!!! Click here to claim.", IntentType.SPAM),
        ("Cheap SMM panel, buy followers and likes, DM for rates.", IntentType.SPAM),
        ("I actually run a competing automation agency, just seeing how you "
         "package your offer.", IntentType.COMPETITIVE),
        ("VC here — wanting to intro you to a few portfolio companies.",
         IntentType.PARTNERSHIP),
        ("Journalist writing about the AI automation space, looking for a quote.",
         IntentType.RESEARCH),
        ("We have automation devs on our bench, would love to place candidates.",
         IntentType.VENDOR_PITCH),
        ("Exploring automating qualifying inbound leads. Comparing a few options.",
         IntentType.PURCHASE),
    ],
)
def test_intent_type_classification(ex, note, expected):
    assert ex.extract(note).intent_type is expected


def test_enthusiastic_student_does_not_read_as_a_buyer(ex):
    """Guide §4: intent type routes; it is not just a low score."""
    p = ex.extract(
        "I'm researching AI automation for my final-year project, need it by "
        "Friday, my supervisor is very keen."
    )
    assert p.intent_type is IntentType.EDUCATIONAL
    assert "STUDENT_OR_RESEARCH" in {n.code for n in p.negative_signals}


# ------------------------------------------------------------ quality
def test_missing_information_is_reported_not_penalised(ex):
    p = ex.extract("Interested in automating lead routing.")
    assert p.budget_status is BudgetStatus.UNKNOWN
    assert p.decision_authority is DecisionAuthority.UNKNOWN
    assert any("budget" in m for m in p.missing_information)
    assert not p.negative_signals


def test_empty_note_yields_unknowns_not_negatives(ex):
    p = ex.extract("")
    assert p.confidence == 0.0
    assert p.buying_stage is BuyingStage.UNKNOWN
    assert not p.negative_signals


def test_confidence_is_certainty_not_value(ex):
    """Guide §7: a clearly-worded dead lead is HIGH confidence, LOW intent."""
    spam = ex.extract("You have WON $1,000,000!!! Click here to claim.")
    assert spam.confidence > 0.7
    assert spam.buying_stage is BuyingStage.UNKNOWN


def test_job_title_never_overrides_an_explicit_statement(ex):
    p = ex.extract("Not sure who signs off internally.", title_authority="LIKELY")
    assert p.decision_authority is DecisionAuthority.NONE


def test_job_title_fills_an_unstated_authority(ex):
    p = ex.extract("Want to automate lead routing.", title_authority="LIKELY")
    assert p.decision_authority is DecisionAuthority.LIKELY


def test_every_resolution_carries_its_evidence(ex):
    p = ex.extract("Budget approved, ready to pilot in the next 2 weeks. "
                   "I make the call here.")
    for dim, res in p.resolutions.items():
        if res.is_unknown:
            continue
        assert res.evidence, f"{dim} resolved without evidence"
        assert res.rule_id
        if res.span:
            assert 0 <= res.span[0] < res.span[1] <= len(p.raw_notes)


def test_evidence_spans_point_at_the_original_text(ex):
    note = "Budget approved, wants to start ASAP."
    p = ex.extract(note)
    res = p.resolutions["budget_status"]
    start, end = res.span
    assert "budget approved" in note[start:end].lower()
