"""Reusable, dependency-free parsers. Each returns a ParseResult."""
from .result import ParseResult, ParseCode, SOFT_CODES
from .amount import parse_amount, parse_amount_options, Amount, AmountConfig, RangePolicy
from .dates import parse_date, parse_date_column, infer_order_policy, DateConfig, DateOrder
from .emails import parse_email, Email, EmailConfig
from .leadid import parse_lead_id, sort_lead_ids, sort_records, make_sort_key, LeadId, LeadIdConfig
from .website import parse_website, domains_match, Website, WebsiteConfig
from . import text

__all__ = [
    "ParseResult", "ParseCode", "SOFT_CODES",
    "parse_amount", "parse_amount_options", "Amount", "AmountConfig", "RangePolicy",
    "parse_date", "parse_date_column", "infer_order_policy", "DateConfig", "DateOrder",
    "parse_email", "Email", "EmailConfig",
    "parse_lead_id", "sort_lead_ids", "sort_records", "make_sort_key", "LeadId", "LeadIdConfig",
    "parse_website", "domains_match", "Website", "WebsiteConfig",
    "text",
]
