"""CSV → fully parsed, NLP-enriched lead records.

The loader owns the messy-data problem: column aliasing, instrumentation rows,
duplicate IDs, and the fact that a date column can only be disambiguated by
looking at the whole column at once.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..core import (
    AmountConfig,
    DateConfig,
    DateOrder,
    EmailConfig,
    LeadIdConfig,
    ParseCode,
    ParseResult,
    RangePolicy,
    WebsiteConfig,
    domains_match,
    infer_order_policy,
    parse_amount,
    parse_date,
    parse_email,
    parse_lead_id,
    parse_website,
)
from ..nlp import IntentExtractor, IntentProfile

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Settings:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        p = Path(path or CONFIG_DIR / "columns.yaml")
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- derived parser configs ------------------------------------------
    @property
    def amount_config(self) -> AmountConfig:
        a = self.raw.get("parsing", {}).get("amount", {})
        return AmountConfig(
            default_currency=a.get("default_currency", "USD"),
            range_policy=RangePolicy(a.get("range_policy", "avg")),
            normalise_to_monthly=a.get("normalise_to_monthly", True),
            assume_period=a.get("assume_period", "month"),
            min_plausible=float(a.get("min_plausible", 1)),
            max_plausible=float(a.get("max_plausible", 10_000_000)),
        )

    @property
    def date_config(self) -> DateConfig:
        d = self.raw.get("parsing", {}).get("date", {})
        return DateConfig(
            fallback_order=DateOrder(d.get("fallback_order", "month_first")),
            two_digit_year_pivot=int(d.get("two_digit_year_pivot", 70)),
            reject_future=bool(d.get("reject_future", False)),
        )

    @property
    def email_config(self) -> EmailConfig:
        e = self.raw.get("parsing", {}).get("email", {})
        return EmailConfig(
            repair_obfuscation=e.get("repair_obfuscation", True),
            reject_freemail=e.get("reject_freemail", False),
            flag_role=e.get("flag_role", True),
        )

    @property
    def lead_id_config(self) -> LeadIdConfig:
        i = self.raw.get("parsing", {}).get("lead_id", {})
        base = LeadIdConfig()
        return LeadIdConfig(
            pattern=i.get("pattern", base.pattern),
            canonical_format=i.get("canonical_format", base.canonical_format),
            default_prefix=i.get("default_prefix", base.default_prefix),
            invalid_sort_last=i.get("invalid_sort_last", True),
        )

    @property
    def website_config(self) -> WebsiteConfig:
        w = self.raw.get("parsing", {}).get("website", {})
        return WebsiteConfig(
            default_scheme=w.get("default_scheme", "https"),
            strip_www=w.get("strip_www", True),
        )


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------

@dataclass
class LeadRecord:
    """One lead: raw row, parse results, NLP profile, derived fit facts."""

    row_index: int
    raw: dict[str, str] = field(default_factory=dict)

    lead_id: ParseResult | None = None
    created: ParseResult | None = None
    email: ParseResult | None = None
    website: ParseResult | None = None
    budget: ParseResult | None = None

    name: str = ""
    company: str = ""
    title: str = ""
    source: str = ""
    notes: str = ""

    employees: int | None = None
    employee_band: str = "unknown"
    employees_raw: str = ""

    title_authority: str | None = None
    is_non_buyer_title: bool = False
    is_icp_agency: bool | None = None
    source_quality: float = 0.4
    domain_match: bool | None = None
    is_duplicate: bool = False

    intent: IntentProfile | None = None

    # scoring fills these
    score: float = 0.0
    display_score: float = 0.0
    band: str = ""
    band_label: str = ""
    route: str = ""
    explanation: list[dict[str, Any]] = field(default_factory=list)
    caps_applied: list[str] = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def id_display(self) -> str:
        if self.lead_id and self.lead_id.ok and self.lead_id.value:
            return str(self.lead_id.value)
        return self.raw.get("lead_id", "") or f"(row {self.row_index})"

    @property
    def monthly_budget(self) -> float | None:
        if self.budget and self.budget.ok and self.budget.value:
            return self.budget.value.monthly if self.budget.value.monthly is not None else self.budget.value.value
        return None

    @property
    def data_issues(self) -> list[str]:
        out = []
        for label, res in (
            ("Lead ID", self.lead_id), ("Created", self.created), ("Email", self.email),
            ("Website", self.website), ("Budget", self.budget),
        ):
            if res is None:
                continue
            if not res.ok:
                out.append(f"{label}: {res.code.value}")
            else:
                for w in res.warnings:
                    if w in (ParseCode.EMAIL_FREEMAIL, ParseCode.EMAIL_ROLE_ADDRESS,
                             ParseCode.ID_DUPLICATE, ParseCode.DATE_AMBIGUOUS_ORDER,
                             ParseCode.AMOUNT_RANGE):
                        out.append(f"{label}: {w.value}")
        if not self.notes.strip():
            out.append("Notes: EMPTY")
        return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _normalise_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")


def resolve_columns(headers: Iterable[str], mapping: dict[str, list[str]]) -> dict[str, str]:
    """canonical field -> actual header present in the file."""
    norm = {_normalise_header(h): h for h in headers}
    out: dict[str, str] = {}
    for field_name, aliases in mapping.items():
        for alias in aliases:
            key = _normalise_header(alias)
            if key in norm:
                out[field_name] = norm[key]
                break
    return out


def _parse_employees(raw: str, bands: list[dict]) -> tuple[int | None, str]:
    """'35-55' -> 45, '19+' -> 19, '1' -> 1, '' -> None."""
    s = (raw or "").strip()
    if not s:
        return None, "unknown"
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return None, "unknown"
    value = round(sum(nums) / len(nums)) if len(nums) > 1 else nums[0]
    for b in bands:
        if b["min"] <= value <= b["max"]:
            return value, b["name"]
    return value, "unknown"


def _title_authority(title: str, cfg: dict) -> tuple[str | None, bool]:
    t = (title or "").strip().lower()
    if not t:
        return None, False
    non_buyer = any(k in t for k in cfg.get("non_buyer", []))
    for level in ("EXPLICIT", "LIKELY", "INFLUENCER", "NONE"):
        for key in cfg.get("authority", {}).get(level, []) or []:
            if key in t:
                return level, non_buyer
    return None, non_buyer


def _is_icp(company: str, notes: str, icp: dict) -> bool | None:
    text = f"{company} {notes}".lower()
    for pat in icp.get("disqualifying_note_patterns", []):
        if re.search(pat, text):
            return False
    for pat in icp.get("agency_note_patterns", []):
        if re.search(pat, text):
            return True
    for pat in icp.get("positive_company_patterns", []):
        if re.search(pat, (company or "").lower()):
            return True
    return None


def _is_path_like(source: Any) -> bool:
    """True only for something that could actually be a filename.

    `load_leads` accepts either a path or raw CSV text. Calling Path().exists()
    on a 100 KB CSV string raises OSError (name too long), so screen first.
    """
    if isinstance(source, Path):
        return source.exists()
    if not isinstance(source, str):
        return False
    if len(source) > 512 or "\n" in source or "\r" in source:
        return False
    try:
        return Path(source).exists()
    except OSError:
        return False


def load_leads(
    source: str | Path | bytes | io.StringIO,
    settings: Settings | None = None,
    extractor: IntentExtractor | None = None,
    *,
    run_nlp: bool = True,
) -> tuple[list[LeadRecord], dict[str, Any]]:
    """Load a CSV into LeadRecords. Returns (records, report)."""
    st = settings or Settings.load()
    ex = extractor or IntentExtractor()

    if _is_path_like(source):
        text = Path(source).read_text(  # type: ignore[arg-type]
            encoding=st.get("dataset", {}).get("encoding", "utf-8"), errors="replace"
        )
    elif isinstance(source, bytes):
        text = source.decode("utf-8", errors="replace")
    elif isinstance(source, io.StringIO):
        text = source.getvalue()
    else:
        text = str(source)

    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    cols = resolve_columns(headers, st["mapping"])
    rows = list(reader)

    drop = st.get("dataset", {}).get("drop_rows_matching", {}) or {}
    dropped: list[dict[str, str]] = []
    kept: list[dict[str, str]] = []
    for r in rows:
        skip = False
        for fname, bad in drop.items():
            col = cols.get(fname)
            if col and str(r.get(col, "")).strip().lower() in {b.lower() for b in bad}:
                skip = True
                break
        (dropped if skip else kept).append(r)

    # Date order must be inferred across the whole column, not per value.
    date_cfg = st.date_config
    if st.get("parsing", {}).get("date", {}).get("infer_order_from_column", True):
        col = cols.get("created")
        if col:
            date_cfg = date_cfg.with_policy(infer_order_policy(r.get(col, "") for r in kept))

    amount_cfg, email_cfg = st.amount_config, st.email_config
    id_cfg, web_cfg = st.lead_id_config, st.website_config
    bands = st.get("employees", {}).get("bands", [])
    titles_cfg = st.get("titles", {})
    icp_cfg = st.get("icp", {})
    src_quality = st.get("sources", {}).get("quality", {})

    records: list[LeadRecord] = []
    seen_numbers: dict[int, int] = {}

    for i, row in enumerate(kept):
        def g(fname: str) -> str:
            col = cols.get(fname)
            return (row.get(col) or "").strip() if col else ""

        rec = LeadRecord(row_index=i, raw={k: (v or "") for k, v in row.items()})
        rec.name, rec.company = g("name"), g("company")
        rec.title, rec.source, rec.notes = g("title"), g("source"), g("notes")
        rec.employees_raw = g("employees")

        rec.lead_id = parse_lead_id(g("lead_id"), id_cfg)
        rec.created = parse_date(g("created"), date_cfg)
        rec.email = parse_email(g("email"), email_cfg)
        rec.website = parse_website(g("website"), web_cfg)
        rec.budget = parse_amount(g("monthly_budget"), amount_cfg)

        if rec.lead_id.ok and rec.lead_id.value:
            n = rec.lead_id.value.number
            seen_numbers[n] = seen_numbers.get(n, 0) + 1
            if seen_numbers[n] > 1:
                rec.is_duplicate = True
                rec.lead_id = ParseResult(
                    ok=rec.lead_id.ok, value=rec.lead_id.value, code=rec.lead_id.code,
                    raw=rec.lead_id.raw, detail=rec.lead_id.detail + " Duplicate ID number.",
                    warnings=rec.lead_id.warnings + (ParseCode.ID_DUPLICATE,), meta=rec.lead_id.meta,
                )

        rec.employees, rec.employee_band = _parse_employees(rec.employees_raw, bands)
        rec.title_authority, rec.is_non_buyer_title = _title_authority(rec.title, titles_cfg)
        rec.is_icp_agency = _is_icp(rec.company, rec.notes, icp_cfg)
        rec.source_quality = float(src_quality.get(rec.source.lower(), src_quality.get("unknown", 0.4)))
        rec.domain_match = domains_match(
            rec.website.value if rec.website.ok else None,
            rec.email.value.domain if (rec.email.ok and rec.email.value) else None,
        )

        if run_nlp:
            rec.intent = ex.extract(
                rec.notes, lead_id=rec.id_display, company=rec.company,
                contact=rec.name, title_authority=rec.title_authority,
            )
        records.append(rec)

    report = {
        "rows_in_file": len(rows),
        "rows_loaded": len(records),
        "rows_dropped": len(dropped),
        "dropped_examples": [d for d in dropped[:5]],
        "headers": headers,
        "column_map": cols,
        "unmapped_fields": [f for f in st["mapping"] if f not in cols],
        "date_order_policy": {k: v.value for k, v in (date_cfg.order_by_separator or {}).items()},
        "duplicate_ids": sum(1 for r in records if r.is_duplicate),
    }
    return records, report


def default_dataset_path(settings: Settings | None = None) -> Path:
    st = settings or Settings.load()
    p = Path(st.get("dataset", {}).get("default_path", "data/default_leads.csv"))
    return p if p.is_absolute() else (PACKAGE_ROOT / p)
