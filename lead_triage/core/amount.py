"""Task 1 — decipher a monetary amount out of free text.

Handles: '6k', '$12k', '$8,000', '5000/month', '8000', '$6k-$8k', '5k - 8k',
'15k/mo', '₦2.5m', 'up to 10k', 'between 5 and 8k', '$8,500', '0', 'TBD'.

Ranges return every reading (min / max / avg) plus a single `value` chosen by
the configured RangePolicy, so the caller never has to guess which one it got.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .result import ParseCode, ParseResult

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD", "usd": "USD", "us$": "USD",
    "£": "GBP", "gbp": "GBP",
    "€": "EUR", "eur": "EUR",
    "₦": "NGN", "ngn": "NGN", "n": "NGN",
    "₹": "INR", "inr": "INR",
    "r": "ZAR", "zar": "ZAR",
    "ksh": "KES", "kes": "KES",
}

MULTIPLIERS: dict[str, float] = {
    "k": 1_000.0, "K": 1_000.0,
    "m": 1_000_000.0, "mm": 1_000_000.0, "M": 1_000_000.0,
    "b": 1_000_000_000.0,
}

#: How many of `period` fit in a month. Used to normalise to monthly.
PERIOD_TO_MONTHLY: dict[str, float] = {
    "month": 1.0,
    "week": 4.345,
    "day": 30.44,
    "hour": 0.0,     # hourly rates are not budgets; flagged, not converted
    "quarter": 1 / 3,
    "year": 1 / 12,
    "one_off": 0.0,
}

PERIOD_PATTERNS: list[tuple[str, str]] = [
    (r"/\s*mo\b|/\s*month\b|per\s+month\b|p/?m\b|a\s+month\b|monthly\b|/\s*mth\b", "month"),
    (r"/\s*wk\b|/\s*week\b|per\s+week\b|weekly\b|a\s+week\b", "week"),
    (r"/\s*day\b|per\s+day\b|daily\b|a\s+day\b", "day"),
    (r"/\s*hr\b|/\s*hour\b|per\s+hour\b|hourly\b|an\s+hour\b", "hour"),
    (r"/\s*q\b|per\s+quarter\b|quarterly\b|a\s+quarter\b", "quarter"),
    (r"/\s*yr\b|/\s*year\b|per\s+year\b|annually\b|per\s+annum\b|p\.?a\.?\b|a\s+year\b", "year"),
    (r"\bone[-\s]?off\b|\bone[-\s]?time\b|\btotal\b|\bproject\b", "one_off"),
]

#: Strings that mean "no answer given", not "not a number".
PLACEHOLDERS = frozenset(
    {
        "tbd", "t.b.d", "tba", "n/a", "na", "none", "nil", "null", "-", "--",
        "unknown", "unsure", "not sure", "depends", "it depends", "depends what you can do",
        "ask", "tbc", "?", "??", "open", "flexible", "negotiable", "confidential",
        "prefer not to say", "wont share", "won't share",
    }
)

JUNK = frozenset({"asdf", "test", "qwerty", "xxx", "budget", "monthly_budget"})

RANGE_SEPARATORS = r"(?:\s*(?:-|–|—|to|and|until|thru|through|\.\.\.?)\s*)"


class RangePolicy(str, Enum):
    """Which reading of '$6k-$8k' becomes `.value`."""

    MIN = "min"
    MAX = "max"
    AVG = "avg"
    MIDPOINT = "avg"  # alias


@dataclass(frozen=True)
class AmountConfig:
    default_currency: str = "USD"
    range_policy: RangePolicy = RangePolicy.AVG
    normalise_to_monthly: bool = True
    #: Assume a bare number with no period marker is already monthly.
    assume_period: str = "month"
    min_plausible: float = 1.0
    max_plausible: float = 10_000_000.0
    #: Treat a bare '5' or '8' as thousands (agency budgets are never $5).
    bare_small_number_is_thousands: bool = False


@dataclass(frozen=True)
class Amount:
    """A parsed amount. `value` is the policy-selected figure."""

    value: float
    minimum: float
    maximum: float
    currency: str
    period: str
    is_range: bool
    monthly: float | None = None
    policy: str = "avg"
    options: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        sym = {"USD": "$", "GBP": "£", "EUR": "€", "NGN": "₦"}.get(self.currency, self.currency + " ")
        if self.is_range:
            return f"{sym}{self.minimum:,.0f}–{sym}{self.maximum:,.0f}/{self.period}"
        return f"{sym}{self.value:,.0f}/{self.period}"


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_CUR = r"(?:us\$|[$£€₦₹]|\b(?:usd|gbp|eur|ngn|inr|kes|zar)\b)"
_MULT = r"(?:k|m|mm|b)\b"

_TOKEN_RE = re.compile(
    rf"(?P<cur>{_CUR})?\s*(?P<num>{_NUMBER})\s*(?P<mult>{_MULT})?",
    re.IGNORECASE,
)

_OPEN_UPPER = re.compile(r"\b(?:up\s+to|under|below|less\s+than|max(?:imum)?(?:\s+of)?|<=?)\b", re.I)
_OPEN_LOWER = re.compile(r"\b(?:from|at\s+least|over|above|more\s+than|min(?:imum)?(?:\s+of)?|starting(?:\s+at)?|>=?|\+)\b", re.I)
_APPROX = re.compile(r"\b(?:approx(?:imately)?|around|about|circa|roughly|~|ish)\b", re.I)


def _detect_currency(text: str, default: str) -> tuple[str, bool]:
    m = re.search(_CUR, text, re.IGNORECASE)
    if not m:
        return default, False
    return CURRENCY_SYMBOLS.get(m.group(0).lower(), default), True


def _detect_period(text: str) -> str | None:
    for pattern, name in PERIOD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


def _scale(num_text: str, mult: str | None, cfg: AmountConfig) -> float:
    value = float(num_text.replace(",", ""))
    if mult:
        value *= MULTIPLIERS[mult.lower()]
    elif cfg.bare_small_number_is_thousands and value < 100:
        value *= 1_000.0
    return value


def _propagate_multiplier(tokens: list[tuple[float, str | None, str]], cfg: AmountConfig) -> list[float]:
    """'5k - 8k' is easy. '6-8k' means 6000-8000, not 6-8000.

    If a later token carries a multiplier and an earlier one does not, and the
    earlier raw number is small enough that it must be shorthand, borrow it.
    """
    values: list[float] = []
    trailing = next((m for _, m, _ in reversed(tokens) if m), None)
    for raw_num, mult, _raw in tokens:
        if mult is None and trailing is not None and raw_num < 1000:
            values.append(raw_num * MULTIPLIERS[trailing.lower()])
        else:
            values.append(raw_num * (MULTIPLIERS[mult.lower()] if mult else 1.0))
    return values


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def parse_amount(
    text: Any,
    config: AmountConfig | None = None,
    *,
    range_policy: RangePolicy | str | None = None,
) -> ParseResult[Amount]:
    """Parse a monetary amount from arbitrary text.

    Returns a ParseResult. On failure `.code` explains why:
    EMPTY, PLACEHOLDER, JUNK, AMOUNT_NO_DIGITS, NOT_A_VALUE, OUT_OF_RANGE.
    """
    cfg = config or AmountConfig()
    if range_policy is not None:
        policy = RangePolicy(range_policy) if isinstance(range_policy, str) else range_policy
        cfg = AmountConfig(**{**cfg.__dict__, "range_policy": policy})

    raw = "" if text is None else str(text)
    s = raw.strip()

    if not s:
        return ParseResult.failure(ParseCode.EMPTY, raw, "No value supplied.")

    low = s.lower().strip(" .,;:")
    if low in JUNK:
        return ParseResult.failure(ParseCode.JUNK, raw, f"'{s}' is a placeholder/test value, not an amount.")
    if low in PLACEHOLDERS or _looks_like_placeholder(low):
        return ParseResult.failure(
            ParseCode.PLACEHOLDER, raw,
            f"'{s}' states that no figure was given — this is unknown budget, not zero budget.",
        )
    if not re.search(r"\d", s):
        return ParseResult.failure(ParseCode.AMOUNT_NO_DIGITS, raw, f"No digits found in '{s}'.")

    currency, explicit_currency = _detect_currency(s, cfg.default_currency)
    period = _detect_period(s)

    matches = list(_TOKEN_RE.finditer(s))
    tokens = [(float(m.group("num").replace(",", "")), m.group("mult"), m.group(0)) for m in matches]
    if not tokens:
        return ParseResult.failure(ParseCode.NOT_A_VALUE, raw, f"Could not read a number from '{s}'.")

    # A range needs two numbers joined by a separator, not just two numbers.
    is_range = False
    if len(tokens) >= 2:
        between = s[matches[0].end():matches[1].start()]
        is_range = bool(re.fullmatch(RANGE_SEPARATORS, between)) or bool(
            re.search(r"^\s*(?:-|–|—|to|and|through|thru)\s*$", between, re.I)
        )

    warnings: list[ParseCode] = []
    values = _propagate_multiplier(tokens, cfg)

    if is_range:
        lo, hi = values[0], values[1]
        if lo > hi:
            lo, hi = hi, lo
            warnings.append(ParseCode.AMOUNT_INVERTED_RANGE)
        options = {"min": lo, "max": hi, "avg": (lo + hi) / 2.0}
        chosen = options[cfg.range_policy.value]
        warnings.append(ParseCode.AMOUNT_RANGE)
    else:
        val = values[0]
        lo = hi = val
        if _OPEN_UPPER.search(s):          # "up to 10k"
            lo, chosen = 0.0, val
            warnings.append(ParseCode.AMOUNT_OPEN_ENDED)
        elif _OPEN_LOWER.search(s) or s.rstrip().endswith("+"):  # "10k+"
            hi, chosen = float("inf"), val
            warnings.append(ParseCode.AMOUNT_OPEN_ENDED)
        else:
            chosen = val
        options = {"min": lo, "max": val if hi == float("inf") else hi, "avg": chosen}

    if chosen == 0:
        warnings.append(ParseCode.AMOUNT_ZERO)
    elif not (cfg.min_plausible <= chosen <= cfg.max_plausible):
        return ParseResult.failure(
            ParseCode.OUT_OF_RANGE, raw,
            f"{chosen:,.0f} is outside the plausible range "
            f"[{cfg.min_plausible:,.0f}, {cfg.max_plausible:,.0f}].",
            currency=currency,
        )

    if period is None:
        period = cfg.assume_period
        warnings.append(ParseCode.AMOUNT_NO_PERIOD)

    monthly: float | None = None
    if cfg.normalise_to_monthly:
        factor = PERIOD_TO_MONTHLY.get(period, 1.0)
        if factor:
            monthly = chosen * factor
            if period != "month":
                warnings.append(ParseCode.AMOUNT_PERIOD_CONVERTED)
        else:
            monthly = None  # hourly / one-off: not a monthly budget

    code = ParseCode.OK
    if _APPROX.search(s) or ParseCode.AMOUNT_RANGE in warnings:
        code = ParseCode.OK_ASSUMED

    amount = Amount(
        value=chosen,
        minimum=lo,
        maximum=hi,
        currency=currency,
        period=period,
        is_range=is_range,
        monthly=monthly,
        policy=cfg.range_policy.value,
        options=options,
    )
    detail = (
        f"Range read as {options['min']:,.0f}–{options['max']:,.0f}; "
        f"'{cfg.range_policy.value}' policy selected {chosen:,.0f}."
        if is_range else f"Read {chosen:,.0f} {currency} per {period}."
    )
    return ParseResult.success(
        amount, raw, code, detail, tuple(warnings),
        currency=currency, period=period, monthly=monthly,
        explicit_currency=explicit_currency, options=options,
    )


def _looks_like_placeholder(low: str) -> bool:
    return bool(
        re.fullmatch(r"(?:wont|won't|will\s+not)\s+share.*", low)
        or re.fullmatch(r"depends.*", low)
        or re.fullmatch(r"(?:not\s+)?(?:sure|decided|set|fixed)", low)
    )


def parse_amount_options(text: Any, config: AmountConfig | None = None) -> dict[str, float] | None:
    """Convenience: every reading of a range, or None if unparseable."""
    res = parse_amount(text, config)
    return res.value.options if res.ok and res.value else None
