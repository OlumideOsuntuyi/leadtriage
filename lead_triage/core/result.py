"""Uniform parse result contract shared by every parser in the system.

Every parser in `lead_triage.core` returns a `ParseResult`. That is the whole
point: one contract, one enum of return codes, one place to check `.ok`.
Hard failures set `ok=False` and carry a single `code`. Soft observations that
do not invalidate the parse (a repaired separator, a freemail domain, a zero
budget) go into `warnings` and leave `ok=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ParseCode(str, Enum):
    """Return codes. Values are stable strings so they can live in configs,
    CSV exports and API payloads without a mapping table."""

    # --- success ---------------------------------------------------------
    OK = "OK"
    OK_NORMALISED = "OK_NORMALISED"  # parsed after cleaning the input
    OK_REPAIRED = "OK_REPAIRED"      # parsed after fixing a clear typo
    OK_ASSUMED = "OK_ASSUMED"        # ambiguous input resolved by policy

    # --- generic failures ------------------------------------------------
    EMPTY = "EMPTY"
    NOT_A_VALUE = "NOT_A_VALUE"      # nothing resembling the target type
    PLACEHOLDER = "PLACEHOLDER"      # "TBD", "n/a", "depends", "unknown"
    JUNK = "JUNK"                    # test/QA rows: "asdf", "test", "header"
    AMBIGUOUS = "AMBIGUOUS"          # several valid readings, policy declined
    OUT_OF_RANGE = "OUT_OF_RANGE"

    # --- amount ----------------------------------------------------------
    AMOUNT_NO_DIGITS = "AMOUNT_NO_DIGITS"
    AMOUNT_INVERTED_RANGE = "AMOUNT_INVERTED_RANGE"
    AMOUNT_ZERO = "AMOUNT_ZERO"
    AMOUNT_RANGE = "AMOUNT_RANGE"
    AMOUNT_OPEN_ENDED = "AMOUNT_OPEN_ENDED"
    AMOUNT_PERIOD_CONVERTED = "AMOUNT_PERIOD_CONVERTED"
    AMOUNT_NO_PERIOD = "AMOUNT_NO_PERIOD"

    # --- date ------------------------------------------------------------
    DATE_UNKNOWN_FORMAT = "DATE_UNKNOWN_FORMAT"
    DATE_IMPOSSIBLE = "DATE_IMPOSSIBLE"          # 2024-13-45
    DATE_AMBIGUOUS_ORDER = "DATE_AMBIGUOUS_ORDER"  # 04-06-2024
    DATE_TWO_DIGIT_YEAR = "DATE_TWO_DIGIT_YEAR"
    DATE_OUT_OF_WINDOW = "DATE_OUT_OF_WINDOW"
    DATE_FUTURE = "DATE_FUTURE"

    # --- email -----------------------------------------------------------
    EMAIL_NO_AT = "EMAIL_NO_AT"
    EMAIL_MULTIPLE_AT = "EMAIL_MULTIPLE_AT"
    EMAIL_MISSING_LOCAL = "EMAIL_MISSING_LOCAL"
    EMAIL_MISSING_DOMAIN = "EMAIL_MISSING_DOMAIN"
    EMAIL_MISSING_TLD = "EMAIL_MISSING_TLD"
    EMAIL_INVALID_TLD = "EMAIL_INVALID_TLD"
    EMAIL_INVALID_CHARS = "EMAIL_INVALID_CHARS"
    EMAIL_FREEMAIL = "EMAIL_FREEMAIL"
    EMAIL_DISPOSABLE = "EMAIL_DISPOSABLE"
    EMAIL_ROLE_ADDRESS = "EMAIL_ROLE_ADDRESS"

    # --- lead id ---------------------------------------------------------
    ID_MISSING = "ID_MISSING"
    ID_MALFORMED = "ID_MALFORMED"
    ID_NO_NUMBER = "ID_NO_NUMBER"
    ID_HAS_SUFFIX = "ID_HAS_SUFFIX"
    ID_DUPLICATE = "ID_DUPLICATE"

    # --- website ---------------------------------------------------------
    WEB_NO_DOT = "WEB_NO_DOT"
    WEB_INVALID_TLD = "WEB_INVALID_TLD"
    WEB_INVALID_HOST = "WEB_INVALID_HOST"
    WEB_SCHEME_ADDED = "WEB_SCHEME_ADDED"
    WEB_IS_EMAIL = "WEB_IS_EMAIL"
    WEB_DOMAIN_MISMATCH = "WEB_DOMAIN_MISMATCH"


#: Codes that describe an observation rather than a failure.
SOFT_CODES = frozenset(
    {
        ParseCode.OK,
        ParseCode.OK_NORMALISED,
        ParseCode.OK_REPAIRED,
        ParseCode.OK_ASSUMED,
        ParseCode.AMOUNT_ZERO,
        ParseCode.AMOUNT_RANGE,
        ParseCode.AMOUNT_OPEN_ENDED,
        ParseCode.AMOUNT_PERIOD_CONVERTED,
        ParseCode.AMOUNT_NO_PERIOD,
        ParseCode.DATE_AMBIGUOUS_ORDER,
        ParseCode.DATE_TWO_DIGIT_YEAR,
        ParseCode.DATE_OUT_OF_WINDOW,
        ParseCode.DATE_FUTURE,
        ParseCode.EMAIL_FREEMAIL,
        ParseCode.EMAIL_DISPOSABLE,
        ParseCode.EMAIL_ROLE_ADDRESS,
        ParseCode.ID_HAS_SUFFIX,
        ParseCode.ID_DUPLICATE,
        ParseCode.WEB_SCHEME_ADDED,
        ParseCode.WEB_DOMAIN_MISMATCH,
    }
)


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    """The single return type of every parser.

    Attributes
    ----------
    ok:       True when `value` is usable.
    value:    the parsed value, or None on failure.
    code:     the primary return code.
    raw:      the untouched input, always preserved for audit.
    detail:   human-readable explanation, safe to show in a UI.
    warnings: non-fatal observations (see SOFT_CODES).
    meta:     parser-specific extras (currency, range bounds, domain, ...).
    """

    ok: bool
    value: T | None
    code: ParseCode
    raw: str
    detail: str = ""
    warnings: tuple[ParseCode, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    # -- constructors -----------------------------------------------------
    @classmethod
    def success(
        cls,
        value: T,
        raw: str,
        code: ParseCode = ParseCode.OK,
        detail: str = "",
        warnings: tuple[ParseCode, ...] = (),
        **meta: Any,
    ) -> "ParseResult[T]":
        return cls(True, value, code, raw, detail, warnings, meta)

    @classmethod
    def failure(
        cls,
        code: ParseCode,
        raw: str,
        detail: str = "",
        warnings: tuple[ParseCode, ...] = (),
        **meta: Any,
    ) -> "ParseResult[T]":
        return cls(False, None, code, raw, detail, warnings, meta)

    # -- helpers ----------------------------------------------------------
    def has(self, code: ParseCode) -> bool:
        return self.code is code or code in self.warnings

    @property
    def all_codes(self) -> tuple[ParseCode, ...]:
        return (self.code, *self.warnings)

    def unwrap_or(self, default: T) -> T:
        return self.value if self.ok and self.value is not None else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "code": self.code.value,
            "raw": self.raw,
            "detail": self.detail,
            "warnings": [w.value for w in self.warnings],
            **self.meta,
        }

    def __bool__(self) -> bool:  # `if parse_amount(x):`
        return self.ok
