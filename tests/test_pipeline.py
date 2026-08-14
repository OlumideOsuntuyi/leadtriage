"""End-to-end tests against the real 520-row messy CSV."""

from __future__ import annotations

import pytest

from lead_triage import ScoringEngine, load_leads
from lead_triage.ingest.loader import Settings, default_dataset_path
from lead_triage.nlp import IntentType


@pytest.fixture(scope="module")
def pipeline():
    path = default_dataset_path()
    if not path.exists():
        pytest.skip("default dataset not present")
    records, report = load_leads(path)
    ScoringEngine().score_all(records)
    return records, report


def test_loads_and_drops_instrumentation_rows(pipeline):
    records, report = pipeline
    assert report["rows_in_file"] == 520
    assert report["rows_dropped"] == 3          # header / asdf / TESTROW
    assert len(records) == 517
    assert not report["unmapped_fields"]


def test_date_order_policy_is_learned_from_the_file(pipeline):
    _, report = pipeline
    assert report["date_order_policy"] == {"/": "month_first", "-": "day_first"}


def test_every_record_gets_a_band_and_a_route(pipeline):
    records, _ = pipeline
    valid = {"CONTACT_NOW", "CONSIDER", "NURTURE", "DISQUALIFY"}
    assert all(r.band in valid for r in records)
    assert all(r.route for r in records)
    assert all(0 <= r.display_score <= 100 for r in records)


def test_every_score_is_fully_explained(pipeline):
    records, _ = pipeline
    for r in records:
        if not r.explanation:
            continue
        total = sum(float(l["points"]) for l in r.explanation)
        assert abs(total - r.score) < 0.15, f"{r.id_display} unexplained points"


def test_disqualifying_enquiry_types_never_reach_sales(pipeline):
    records, _ = pipeline
    dead = {IntentType.SPAM, IntentType.VENDOR_PITCH, IntentType.COMPETITIVE,
            IntentType.EMPLOYMENT}
    for r in records:
        if r.intent and r.intent.intent_type in dead:
            assert r.band == "DISQUALIFY", f"{r.id_display} {r.intent.intent_type}"


def test_non_purchase_is_capped_not_deleted(pipeline):
    """The user's rule: a non-purchase enquiry can still be nurtured."""
    records, _ = pipeline
    soft = [r for r in records if r.intent
            and r.intent.intent_type in (IntentType.EDUCATIONAL, IntentType.RESEARCH,
                                         IntentType.PARTNERSHIP)]
    assert soft, "expected some educational/research/partnership enquiries"
    assert all(r.band != "CONTACT_NOW" for r in soft)
    assert any(r.band in ("NURTURE", "CONSIDER") for r in soft)
    assert all(r.route in ("Community", "Marketing", "Partnerships") for r in soft)


def test_top_leads_all_have_budget_and_a_timeline(pipeline):
    records, _ = pipeline
    top = sorted(records, key=lambda r: -r.score)[:25]
    for r in top:
        assert r.intent.budget_status.name in ("APPROVED", "ALLOCATED", "BUDGETED")
        assert r.intent.urgency.name != "UNKNOWN"
        assert r.band == "CONTACT_NOW"


def test_duplicate_ids_are_detected(pipeline):
    records, report = pipeline
    assert report["duplicate_ids"] > 0
    assert any(r.is_duplicate for r in records)


def test_column_mapping_is_configurable():
    """Rename every column in the file; the system should still load it."""
    csv = (
        "ref,submitted,contact,e-mail,account,headcount,site,role,channel,spend,message\n"
        "ACME-12,2024-06-01,Ada,ada@acme.io,Acme,30,acme.io,Founder,referral,$8k/mo,"
        "\"Budget approved, ready to pilot in 2 weeks. I make the call here.\"\n"
    )
    settings = Settings.load()
    records, report = load_leads(csv, settings)
    assert len(records) == 1
    assert not report["unmapped_fields"]
    rec = records[0]
    assert rec.company == "Acme" and rec.monthly_budget == 8000
    assert rec.intent.budget_status.name == "APPROVED"


def test_pipeline_is_fast_enough_to_be_interactive(pipeline):
    """The whole point of a rule-based engine: no model load, no GPU."""
    import time
    path = default_dataset_path()
    start = time.perf_counter()
    records, _ = load_leads(path)
    ScoringEngine().score_all(records)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"{elapsed:.2f}s for {len(records)} leads"
