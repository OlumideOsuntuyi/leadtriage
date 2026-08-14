"""Task 3 — parse a date out of any format, with an explicit return code.

The dataset mixes MM/DD/YYYY ('06/28/2024'), DD-MM-YYYY ('19-06-2024'),
ISO ('2024-6-7'), textual ('Jun 7 2024') and 2-digit years ('6/1/24') in the
same column. A single parser cannot know whether '04-06-2024' is 4 June or
6 April, so the module does two things:

1. `infer_order_policy()` scans the whole column first. If any value with a
   given separator has an unambiguous day (>12) in a given position, that
   settles the order for every value using that separator.
2. Values that remain ambiguous are parsed under the configured fallback and
   flagged DATE_AMBIGUOUS_ORDER, so nothing silently guesses.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable

from .result import ParseCode, ParseResult

MONTHS: dict[str, int] = {}
for _i, _names in enumerate(
    [
        ("jan", "january"), ("feb", "february"), ("mar", "march"),
        ("apr", "april"), ("may",), ("jun", "june"),
        ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
        ("oct", "october"), ("nov", "november"), ("dec", "december"),
    ],
    start=1,
):
    for _n in _names:
        MONTHS[_n] = _i

JUNK = frozenset({"asdf", "test", "lead_id", "created", "date", "n/a", "na", "tbd", "-", "none", "null"})


class DateOrder(str, Enum):
    DAY_FIRST = "day_first"
    MONTH_FIRST = "month_first"
    AUTO = "auto"


@dataclass(frozen=True)
class DateConfig:
    #: Fallback when the value itself is ambiguous and no policy was inferred.
    fallback_order: DateOrder = DateOrder.MONTH_FIRST
    #: Per-separator overrides, normally produced by infer_order_policy().
    order_by_separator: dict[str, DateOrder] = None  # type: ignore[assignment]
    two_digit_year_pivot: int = 70  # <=70 -> 20xx, else 19xx
    min_year: int = 1990
    max_year: int = 2100
    #: Optional sanity window for lead data; None disables the check.
    plausible_from: date | None = None
    plausible_to: date | None = None
    reject_future: bool = False
    dayfirst_default: bool = False

    def with_policy(self, policy: dict[str, DateOrder]) -> "DateConfig":
        return replace(self, order_by_separator=policy)


_TEXT_MONTH = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ORD = r"(?:st|nd|rd|th)?"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ISO-ish: 2024-06-08, 2024/6/7, 2024.06.08
    (re.compile(r"^(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})$"), "iso"),
    # compact 20240607
    (re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$"), "compact"),
    # numeric with separator: 06/28/2024, 19-06-2024, 6/1/24
    (re.compile(r"^(?P<a>\d{1,2})(?P<sep>[-/.])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{2}|\d{4})$"), "numeric"),
    # Jun 7 2024 / June 7, 2024 / Jun 7th 2024
    (re.compile(rf"^{_TEXT_MONTH}\.?\s+(?P<d>\d{{1,2}}){_ORD}[,\s]+(?P<y>\d{{2}}|\d{{4}})$", re.I), "mdy_text"),
    # 7 Jun 2024 / 7th June, 2024
    (re.compile(rf"^(?P<d>\d{{1,2}}){_ORD}\s+{_TEXT_MONTH}\.?[,\s]+(?P<y>\d{{2}}|\d{{4}})$", re.I), "dmy_text"),
    # Jun 2024 (no day)
    (re.compile(rf"^{_TEXT_MONTH}\.?[,\s]+(?P<y>\d{{4}})$", re.I), "my_text"),
    # 2024-06 (no day)
    (re.compile(r"^(?P<y>\d{4})[-/](?P<m>\d{1,2})$"), "ym"),
    # ISO datetime
    (re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[T ]\d{2}:\d{2}.*$"), "iso"),
]


def _expand_year(y: int, cfg: DateConfig) -> tuple[int, bool]:
    if y >= 100:
        return y, False
    return (2000 + y if y <= cfg.two_digit_year_pivot else 1900 + y), True


def infer_order_policy(values: Iterable[Any]) -> dict[str, DateOrder]:
    """Learn day-first vs month-first per separator from a whole column.

    '19-06-2024' proves dash-separated values are day-first.
    '06/28/2024' proves slash-separated values are month-first.
    Returns {separator: DateOrder} for separators where the evidence is clear.
    """
    evidence: dict[str, Counter] = {}
    numeric = _PATTERNS[2][0]
    for v in values:
        s = "" if v is None else str(v).strip()
        m = numeric.match(s)
        if not m:
            continue
        a, b, sep = int(m.group("a")), int(m.group("b")), m.group("sep")
        c = evidence.setdefault(sep, Counter())
        if a > 12 and b <= 12:
            c["day_first"] += 1
        elif b > 12 and a <= 12:
            c["month_first"] += 1
    policy: dict[str, DateOrder] = {}
    for sep, c in evidence.items():
        if not c:
            continue
        winner, n = c.most_common(1)[0]
        loser = c.get("month_first" if winner == "day_first" else "day_first", 0)
        # Require a clear majority; mixed columns stay ambiguous on purpose.
        if n > loser:
            policy[sep] = DateOrder(winner)
    return policy


def parse_date(
    text: Any,
    config: DateConfig | None = None,
    *,
    order: DateOrder | str | None = None,
) -> ParseResult[date]:
    """Parse a date from arbitrary text.

    Failure codes: EMPTY, JUNK, DATE_UNKNOWN_FORMAT, DATE_IMPOSSIBLE,
    OUT_OF_RANGE. Soft flags: DATE_AMBIGUOUS_ORDER, DATE_TWO_DIGIT_YEAR,
    DATE_OUT_OF_WINDOW, DATE_FUTURE.
    """
    cfg = config or DateConfig()
    if order is not None:
        cfg = replace(cfg, fallback_order=DateOrder(order) if isinstance(order, str) else order)

    raw = "" if text is None else str(text)
    s = raw.strip().strip(",")
    if not s:
        return ParseResult.failure(ParseCode.EMPTY, raw, "No date supplied.")
    if s.lower() in JUNK:
        return ParseResult.failure(ParseCode.JUNK, raw, f"'{s}' is a placeholder, not a date.")
    if not re.search(r"\d", s):
        return ParseResult.failure(ParseCode.DATE_UNKNOWN_FORMAT, raw, f"No digits in '{s}'.")

    s = re.sub(r"\s+", " ", s)
    warnings: list[ParseCode] = []
    code = ParseCode.OK

    for pattern, kind in _PATTERNS:
        m = pattern.match(s)
        if not m:
            continue
        g = m.groupdict()

        if kind in ("iso", "compact", "ym"):
            y, mo = int(g["y"]), int(g["m"])
            d = int(g.get("d") or 1)
            if not g.get("d"):
                code = ParseCode.OK_ASSUMED
        elif kind in ("mdy_text", "dmy_text", "my_text"):
            mo = MONTHS[g["mon"].lower().rstrip(".")]
            d = int(g.get("d") or 1)
            if not g.get("d"):
                code = ParseCode.OK_ASSUMED
            y, two = _expand_year(int(g["y"]), cfg)
            if two:
                warnings.append(ParseCode.DATE_TWO_DIGIT_YEAR)
        else:  # numeric
            a, b, sep = int(g["a"]), int(g["b"]), g["sep"]
            y, two = _expand_year(int(g["y"]), cfg)
            if two:
                warnings.append(ParseCode.DATE_TWO_DIGIT_YEAR)

            if a > 12 and b <= 12:
                d, mo = a, b
            elif b > 12 and a <= 12:
                mo, d = a, b
            elif a > 12 and b > 12:
                return ParseResult.failure(
                    ParseCode.DATE_IMPOSSIBLE, raw, f"Neither component of '{s}' can be a month."
                )
            else:
                resolved = (cfg.order_by_separator or {}).get(sep) or cfg.fallback_order
                if resolved is DateOrder.DAY_FIRST:
                    d, mo = a, b
                else:
                    mo, d = a, b
                warnings.append(ParseCode.DATE_AMBIGUOUS_ORDER)
                code = ParseCode.OK_ASSUMED

        if kind in ("iso", "compact", "ym") and "y" in g:
            two = False

        try:
            parsed = date(y, mo, d)
        except ValueError as exc:
            return ParseResult.failure(ParseCode.DATE_IMPOSSIBLE, raw, f"'{s}' is not a real date ({exc}).")

        if not (cfg.min_year <= parsed.year <= cfg.max_year):
            return ParseResult.failure(
                ParseCode.OUT_OF_RANGE, raw,
                f"Year {parsed.year} outside [{cfg.min_year}, {cfg.max_year}].",
            )
        if cfg.plausible_from and parsed < cfg.plausible_from:
            warnings.append(ParseCode.DATE_OUT_OF_WINDOW)
        if cfg.plausible_to and parsed > cfg.plausible_to:
            warnings.append(ParseCode.DATE_OUT_OF_WINDOW)
        if parsed > date.today():
            if cfg.reject_future:
                return ParseResult.failure(ParseCode.DATE_FUTURE, raw, f"'{s}' is in the future.")
            warnings.append(ParseCode.DATE_FUTURE)

        detail = f"Read as {parsed.isoformat()}"
        if ParseCode.DATE_AMBIGUOUS_ORDER in warnings:
            src = "column-inferred" if (cfg.order_by_separator or {}).get(g.get("sep", "")) else "fallback"
            detail += f" using the {src} {(cfg.order_by_separator or {}).get(g.get('sep',''), cfg.fallback_order).value} rule"
        return ParseResult.success(
            parsed, raw, code, detail + ".", tuple(dict.fromkeys(warnings)),
            iso=parsed.isoformat(), pattern=kind,
        )

    return ParseResult.failure(
        ParseCode.DATE_UNKNOWN_FORMAT, raw, f"'{s}' does not match any known date format."
    )


def parse_date_column(values: Iterable[Any], config: DateConfig | None = None) -> list[ParseResult[date]]:
    """Parse a whole column, inferring the day/month order from it first."""
    values = list(values)
    cfg = (config or DateConfig()).with_policy(infer_order_policy(values))
    return [parse_date(v, cfg) for v in values]
