"""Task 5 — parse and sort lead IDs across mixed formats.

Real column contains 'L-1234', '4321', 'L-1205-dup', '', 'TESTROW', 'header'.
The parser decomposes an ID into (prefix, number, suffix) so that a numeric
sort works across formats, and so '4321' and 'L-4321' can be recognised as the
same lead if the caller wants that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .result import ParseCode, ParseResult

JUNK = frozenset({"asdf", "test", "testrow", "header", "lead_id", "id", "n/a", "na", "none", "null", "-"})


@dataclass(frozen=True)
class LeadIdConfig:
    #: Named groups prefix / number / suffix. Configure for another scheme.
    pattern: str = r"^(?P<prefix>[A-Za-z]{1,5})?[\s_-]*(?P<number>\d+)(?:[\s_-]+(?P<suffix>[A-Za-z0-9_-]+))?$"
    canonical_format: str = "{prefix}-{number}"
    default_prefix: str = "L"
    pad_to: int = 0            # 0 = no zero padding
    uppercase_prefix: bool = True
    allow_missing_prefix: bool = True
    #: Values matching these are junk, not malformed IDs.
    junk: frozenset[str] = field(default_factory=lambda: JUNK)
    #: Where unparseable IDs go when sorting.
    invalid_sort_last: bool = True


@dataclass(frozen=True)
class LeadId:
    canonical: str
    prefix: str
    number: int
    suffix: str = ""
    had_prefix: bool = True

    @property
    def sort_key(self) -> tuple:
        return (0, self.number, self.suffix, self.prefix)

    def __str__(self) -> str:
        return self.canonical


def parse_lead_id(text: Any, config: LeadIdConfig | None = None) -> ParseResult[LeadId]:
    """Parse a lead ID.

    Failure codes: ID_MISSING (empty), JUNK, ID_MALFORMED, ID_NO_NUMBER.
    Soft flag: ID_HAS_SUFFIX (e.g. the '-dup' markers in the source file).
    """
    cfg = config or LeadIdConfig()
    raw = "" if text is None else str(text)
    s = raw.strip()

    if not s:
        return ParseResult.failure(ParseCode.ID_MISSING, raw, "Lead ID is missing.")
    if s.lower() in cfg.junk:
        return ParseResult.failure(ParseCode.JUNK, raw, f"'{s}' is a test/header row, not a lead ID.")

    m = re.match(cfg.pattern, s)
    if not m:
        if not re.search(r"\d", s):
            return ParseResult.failure(
                ParseCode.ID_NO_NUMBER, raw, f"'{s}' contains no number — cannot be ordered."
            )
        return ParseResult.failure(
            ParseCode.ID_MALFORMED, raw, f"'{s}' does not match the configured ID pattern."
        )

    g = m.groupdict()
    prefix = (g.get("prefix") or "").strip("-_ ")
    had_prefix = bool(prefix)
    if not had_prefix:
        if not cfg.allow_missing_prefix:
            return ParseResult.failure(ParseCode.ID_MALFORMED, raw, f"'{s}' is missing the required prefix.")
        prefix = cfg.default_prefix
    if cfg.uppercase_prefix:
        prefix = prefix.upper()

    number = int(g["number"])
    suffix = (g.get("suffix") or "").strip()

    num_text = str(number).zfill(cfg.pad_to) if cfg.pad_to else str(number)
    canonical = cfg.canonical_format.format(prefix=prefix, number=num_text)
    if suffix:
        canonical = f"{canonical}-{suffix}"

    warnings = (ParseCode.ID_HAS_SUFFIX,) if suffix else ()
    code = ParseCode.OK if (had_prefix and not suffix) else ParseCode.OK_NORMALISED
    detail = f"Parsed as {canonical}."
    if not had_prefix:
        detail += f" Prefix '{cfg.default_prefix}' assumed."
    if suffix:
        detail += f" Suffix '{suffix}' retained — likely a duplicate marker."

    return ParseResult.success(
        LeadId(canonical, prefix, number, suffix, had_prefix), raw, code, detail, warnings,
        number=number, prefix=prefix, suffix=suffix,
    )


def sort_lead_ids(
    values: Iterable[Any],
    config: LeadIdConfig | None = None,
    *,
    reverse: bool = False,
) -> list[ParseResult[LeadId]]:
    """Sort raw ID strings numerically, invalid ones grouped at the end."""
    cfg = config or LeadIdConfig()
    results = [parse_lead_id(v, cfg) for v in values]
    return _sorted(results, cfg, reverse)


def sort_records(
    records: Sequence[dict],
    id_field: str = "lead_id",
    config: LeadIdConfig | None = None,
    *,
    reverse: bool = False,
    mark_duplicates: bool = True,
) -> list[dict]:
    """Sort dict records by lead ID, optionally flagging duplicate numbers.

    Each record gains `_id_result` (the ParseResult) so the caller keeps the
    return code alongside the row.
    """
    cfg = config or LeadIdConfig()
    out: list[dict] = []
    seen: dict[int, int] = {}
    for rec in records:
        res = parse_lead_id(rec.get(id_field), cfg)
        if mark_duplicates and res.ok and res.value is not None:
            n = res.value.number
            seen[n] = seen.get(n, 0) + 1
            if seen[n] > 1:
                res = ParseResult(
                    ok=res.ok, value=res.value, code=res.code, raw=res.raw,
                    detail=res.detail + f" Duplicate of an earlier lead with number {n}.",
                    warnings=res.warnings + (ParseCode.ID_DUPLICATE,), meta=res.meta,
                )
        out.append({**rec, "_id_result": res})

    def key(rec: dict) -> tuple:
        r: ParseResult[LeadId] = rec["_id_result"]
        if r.ok and r.value is not None:
            return r.value.sort_key
        rank = 1 if cfg.invalid_sort_last else -1
        return (rank, 0, str(rec.get(id_field) or ""), "")

    return sorted(out, key=key, reverse=reverse)


def _sorted(
    results: list[ParseResult[LeadId]], cfg: LeadIdConfig, reverse: bool
) -> list[ParseResult[LeadId]]:
    def key(r: ParseResult[LeadId]) -> tuple:
        if r.ok and r.value is not None:
            return r.value.sort_key
        return (1 if cfg.invalid_sort_last else -1, 0, r.raw, "")

    return sorted(results, key=key, reverse=reverse)


def make_sort_key(config: LeadIdConfig | None = None) -> Callable[[Any], tuple]:
    """A reusable key function for sorting anything by a raw lead-ID string."""
    cfg = config or LeadIdConfig()

    def _key(value: Any) -> tuple:
        r = parse_lead_id(value, cfg)
        if r.ok and r.value is not None:
            return r.value.sort_key
        return (1 if cfg.invalid_sort_last else -1, 0, str(value or ""), "")

    return _key
