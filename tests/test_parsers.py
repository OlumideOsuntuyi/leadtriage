"""Parser tests. Every case here is a real value from the source CSV or a
close variant, so the suite doubles as documentation of what the messy data
actually contains.
"""

from __future__ import annotations

from datetime import date

import pytest

from lead_triage.core import (
    AmountConfig,
    DateConfig,
    DateOrder,
    LeadIdConfig,
    ParseCode,
    RangePolicy,
    infer_order_policy,
    parse_amount,
    parse_date,
    parse_email,
    parse_lead_id,
    parse_website,
    sort_lead_ids,
)


# ---------------------------------------------------------------- amount
@pytest.mark.parametrize(
    "text,expected",
    [
        ("6k", 6000), ("$12k", 12000), ("$8,000", 8000), ("5000/month", 5000),
        ("8000", 8000), ("15k/mo", 15000), ("$11k/mo", 11000), ("5,000/mo", 5000),
        ("$8,500", 8500), ("500", 500), ("₦2.5m", 2_500_000), ("18k", 18000),
    ],
)
def test_amount_single_values(text, expected):
    r = parse_amount(text)
    assert r.ok and r.value.value == expected


@pytest.mark.parametrize(
    "text,lo,hi",
    [("$6k-$8k", 6000, 8000), ("5k - 8k", 5000, 8000), ("8k-12k", 8000, 12000),
     ("5k-7k", 5000, 7000), ("$6-8k", 6000, 8000), ("between 5 and 8k", 5000, 8000),
     ("35-55", 35, 55)],
)
def test_amount_ranges_expose_every_reading(text, lo, hi):
    r = parse_amount(text, AmountConfig(min_plausible=1))
    assert r.ok and r.value.is_range
    assert r.value.options == {"min": lo, "max": hi, "avg": (lo + hi) / 2}
    assert ParseCode.AMOUNT_RANGE in r.warnings


@pytest.mark.parametrize(
    "policy,expected", [("min", 6000), ("max", 8000), ("avg", 7000)]
)
def test_amount_range_policy(policy, expected):
    r = parse_amount("$6k-$8k", range_policy=policy)
    assert r.value.value == expected


@pytest.mark.parametrize(
    "text,code",
    [
        ("", ParseCode.EMPTY),
        ("TBD", ParseCode.PLACEHOLDER),
        ("depends", ParseCode.PLACEHOLDER),
        ("depends what you can do", ParseCode.PLACEHOLDER),
        ("budget way below range", ParseCode.AMOUNT_NO_DIGITS),
        ("not a number", ParseCode.AMOUNT_NO_DIGITS),
        ("asdf", ParseCode.JUNK),
    ],
)
def test_amount_failure_codes_are_specific(text, code):
    r = parse_amount(text)
    assert not r.ok and r.code is code


def test_amount_zero_parses_but_is_flagged():
    r = parse_amount("0")
    assert r.ok and r.value.value == 0 and ParseCode.AMOUNT_ZERO in r.warnings


def test_amount_hourly_rate_is_not_a_monthly_budget():
    r = parse_amount("$5/hr")
    assert r.ok and r.value.period == "hour" and r.meta["monthly"] is None


def test_amount_annual_converts_to_monthly():
    r = parse_amount("$120,000/yr")
    assert r.ok and round(r.meta["monthly"]) == 10000


# ------------------------------------------------------------------ date
def test_date_order_is_inferred_from_the_whole_column():
    column = ["06/28/2024", "19-06-2024", "04-06-2024", "6/13/24"]
    policy = infer_order_policy(column)
    assert policy["/"] is DateOrder.MONTH_FIRST   # 06/28 proves it
    assert policy["-"] is DateOrder.DAY_FIRST     # 19-06 proves it
    cfg = DateConfig().with_policy(policy)
    assert parse_date("04-06-2024", cfg).value == date(2024, 6, 4)
    assert parse_date("06/07/2024", cfg).value == date(2024, 6, 7)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2024-06-08", date(2024, 6, 8)), ("2024-6-7", date(2024, 6, 7)),
        ("Jun 7 2024", date(2024, 6, 7)), ("June 7, 2024", date(2024, 6, 7)),
        ("7th June 2024", date(2024, 6, 7)), ("20240607", date(2024, 6, 7)),
        ("6/1/24", date(2024, 6, 1)), ("19-06-2024", date(2024, 6, 19)),
        ("2024-06-08T09:30:00", date(2024, 6, 8)),
    ],
)
def test_date_formats(text, expected):
    assert parse_date(text).value == expected


@pytest.mark.parametrize(
    "text,code",
    [("", ParseCode.EMPTY), ("asdf", ParseCode.JUNK),
     ("2024-13-45", ParseCode.DATE_IMPOSSIBLE),
     ("hello world", ParseCode.DATE_UNKNOWN_FORMAT),
     ("45-46-2024", ParseCode.DATE_IMPOSSIBLE)],
)
def test_date_failure_codes(text, code):
    r = parse_date(text)
    assert not r.ok and r.code is code


def test_ambiguous_date_is_flagged_not_silently_guessed():
    r = parse_date("04-06-2024", DateConfig())
    assert r.ok and ParseCode.DATE_AMBIGUOUS_ORDER in r.warnings


# ----------------------------------------------------------------- email
@pytest.mark.parametrize(
    "text,code",
    [
        ("", ParseCode.EMPTY),
        ("weird-email-no-domain", ParseCode.EMAIL_NO_AT),
        ("ivan@", ParseCode.EMAIL_MISSING_DOMAIN),
        ("@acme.com", ParseCode.EMAIL_MISSING_LOCAL),
        ("obi@leverageside", ParseCode.EMAIL_MISSING_TLD),
        ("deji m.@scaleforge", ParseCode.EMAIL_INVALID_CHARS),
        ("a@b@c.com", ParseCode.EMAIL_MULTIPLE_AT),
        ("test@test.com", ParseCode.JUNK),
        ("obi@leverageside.1", ParseCode.EMAIL_INVALID_TLD),
    ],
)
def test_email_failures_say_what_is_wrong(text, code):
    r = parse_email(text)
    assert not r.ok and r.code is code


def test_email_obfuscation_is_repaired():
    r = parse_email("kunle[at]meridianbound.com")
    assert r.ok and r.value.address == "kunle@meridianbound.com"
    assert r.code is ParseCode.OK_REPAIRED


def test_email_soft_flags_do_not_fail_the_parse():
    free = parse_email("blessing@proton.me")
    assert free.ok and ParseCode.EMAIL_FREEMAIL in free.warnings
    role = parse_email("info@acme.com")
    assert role.ok and ParseCode.EMAIL_ROLE_ADDRESS in role.warnings


def test_email_display_name_is_stripped():
    r = parse_email("Jane Doe <jane@acme.com>")
    assert r.ok and r.value.address == "jane@acme.com"


# --------------------------------------------------------------- lead id
@pytest.mark.parametrize(
    "text,canonical,number",
    [("L-1369", "L-1369", 1369), ("1341", "L-1341", 1341),
     ("L-1205-dup", "L-1205-dup", 1205), ("L-0007", "L-7", 7)],
)
def test_lead_id_parsing(text, canonical, number):
    r = parse_lead_id(text)
    assert r.ok and str(r.value) == canonical and r.value.number == number


@pytest.mark.parametrize(
    "text,code",
    [("", ParseCode.ID_MISSING), ("TESTROW", ParseCode.JUNK),
     ("header", ParseCode.JUNK), ("no-digits-here", ParseCode.ID_NO_NUMBER)],
)
def test_lead_id_failure_codes(text, code):
    r = parse_lead_id(text)
    assert not r.ok and r.code is code


def test_lead_ids_sort_numerically_across_formats_invalid_last():
    out = sort_lead_ids(["L-1369", "1341", "L-99", "7", "", "TESTROW", "L-1205-dup"])
    assert [str(r.value) if r.ok else None for r in out] == [
        "L-7", "L-99", "L-1205-dup", "L-1341", "L-1369", None, None
    ]


def test_lead_id_pattern_is_configurable():
    cfg = LeadIdConfig(
        pattern=r"^(?P<prefix>[A-Z]{3})(?P<number>\d+)$",
        canonical_format="{prefix}/{number}", pad_to=5,
    )
    r = parse_lead_id("ACM42", cfg)
    assert r.ok and str(r.value) == "ACM/00042"


# --------------------------------------------------------------- website
@pytest.mark.parametrize(
    "text,url,registrable",
    [
        ("luxauto.io", "https://luxauto.io", "luxauto.io"),
        ("www.apexsend.co", "https://apexsend.co", "apexsend.co"),
        ("http://upshiftloop.agency", "http://upshiftloop.agency", "upshiftloop.agency"),
        ("https://x.co.uk/path", "https://x.co.uk", "x.co.uk"),
        ("sub.foo.com.ng", "https://sub.foo.com.ng", "foo.com.ng"),
    ],
)
def test_website_normalisation(text, url, registrable):
    r = parse_website(text)
    assert r.ok and r.value.url == url and r.value.registrable_domain == registrable


@pytest.mark.parametrize(
    "text,code",
    [("", ParseCode.EMPTY), ("asdf", ParseCode.JUNK), ("website", ParseCode.JUNK),
     ("localhost", ParseCode.WEB_NO_DOT), ("ivan@x.com", ParseCode.WEB_IS_EMAIL),
     ("foo.123", ParseCode.WEB_INVALID_TLD)],
)
def test_website_failure_codes(text, code):
    r = parse_website(text)
    assert not r.ok and r.code is code


# ------------------------------------------------------- shared contract
def test_every_parser_returns_the_same_shape():
    for r in (parse_amount("6k"), parse_date("2024-06-01"), parse_email("a@b.com"),
              parse_lead_id("L-1"), parse_website("a.com")):
        assert r.ok is True
        assert isinstance(r.code, ParseCode)
        assert isinstance(r.raw, str)
        assert isinstance(r.to_dict(), dict)
        assert bool(r) is True
