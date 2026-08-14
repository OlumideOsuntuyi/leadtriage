"""Task 4 — validate and normalise an email address.

Distinguishes *why* an address is wrong (no @, no domain, no TLD, bad chars,
empty) rather than returning a single boolean, and separates hard failures
from commercially relevant observations (freemail, role address, disposable).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .result import ParseCode, ParseResult

FREEMAIL = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "hotmail.com",
    "outlook.com", "live.com", "msn.com", "aol.com", "icloud.com", "me.com",
    "proton.me", "protonmail.com", "pm.me", "gmx.com", "mail.com", "zoho.com",
    "yandex.com", "inbox.com", "rocketmail.com", "ymail.com",
})

DISPOSABLE = frozenset({
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com",
    "throwawaymail.com", "trashmail.com", "yopmail.com", "sharklasers.com",
    "getnada.com", "dispostable.com", "maildrop.cc", "fakeinbox.com",
})

ROLE_LOCALS = frozenset({
    "info", "admin", "sales", "support", "hello", "contact", "team", "office",
    "enquiries", "inquiries", "marketing", "billing", "accounts", "hr", "jobs",
    "careers", "noreply", "no-reply", "webmaster", "postmaster", "help", "lead",
    "leads", "newsletter",
})

JUNK_VALUES = frozenset({"email", "asdf", "test", "n/a", "na", "none", "-", "null", "test@test.com", "asdf@asdf.com"})

#: Common ways an @ gets mangled by forms and scrapers.
AT_SUBSTITUTES = [
    (re.compile(r"\s*\[\s*at\s*\]\s*", re.I), "@"),
    (re.compile(r"\s*\(\s*at\s*\)\s*", re.I), "@"),
    (re.compile(r"\s+at\s+(?=[A-Za-z0-9-]+\.[A-Za-z]{2,})", re.I), "@"),
    (re.compile(r"\s*\{\s*at\s*\}\s*", re.I), "@"),
]
DOT_SUBSTITUTES = [
    (re.compile(r"\s*\[\s*dot\s*\]\s*", re.I), "."),
    (re.compile(r"\s*\(\s*dot\s*\)\s*", re.I), "."),
]

_LOCAL_OK = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")
_LABEL_OK = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_TLD_OK = re.compile(r"^[A-Za-z]{2,24}$")


@dataclass(frozen=True)
class EmailConfig:
    repair_obfuscation: bool = True
    lowercase: bool = True
    strip_display_name: bool = True
    flag_freemail: bool = True
    flag_role: bool = True
    reject_freemail: bool = False
    extra_freemail: frozenset[str] = field(default_factory=frozenset)
    extra_disposable: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Email:
    address: str
    local: str
    domain: str
    tld: str
    is_freemail: bool = False
    is_role: bool = False
    is_disposable: bool = False

    def __str__(self) -> str:
        return self.address


def parse_email(text: Any, config: EmailConfig | None = None) -> ParseResult[Email]:
    """Validate an email address.

    Failure codes: EMPTY, JUNK, EMAIL_NO_AT, EMAIL_MULTIPLE_AT,
    EMAIL_MISSING_LOCAL, EMAIL_MISSING_DOMAIN, EMAIL_MISSING_TLD,
    EMAIL_INVALID_TLD, EMAIL_INVALID_CHARS.
    """
    cfg = config or EmailConfig()
    raw = "" if text is None else str(text)
    s = raw.strip()

    if not s:
        return ParseResult.failure(ParseCode.EMPTY, raw, "No email supplied.")
    if s.lower() in JUNK_VALUES:
        return ParseResult.failure(ParseCode.JUNK, raw, f"'{s}' is a test/placeholder value.")

    warnings: list[ParseCode] = []
    code = ParseCode.OK

    # "Jane Doe <jane@x.com>"
    if cfg.strip_display_name:
        m = re.search(r"<([^>]+)>", s)
        if m:
            s = m.group(1).strip()
            code = ParseCode.OK_NORMALISED

    if cfg.repair_obfuscation:
        before = s
        for pat, sub in AT_SUBSTITUTES + DOT_SUBSTITUTES:
            s = pat.sub(sub, s)
        if s != before:
            code = ParseCode.OK_REPAIRED

    s = s.strip().strip(".,;:")
    s = re.sub(r"^(?:mailto:)", "", s, flags=re.I)

    at_count = s.count("@")
    if at_count == 0:
        return ParseResult.failure(
            ParseCode.EMAIL_NO_AT, raw, f"'{raw.strip()}' has no @ — it is not an address at all."
        )
    if at_count > 1:
        return ParseResult.failure(
            ParseCode.EMAIL_MULTIPLE_AT, raw, f"'{raw.strip()}' contains {at_count} @ symbols."
        )

    local, _, domain = s.partition("@")
    local, domain = local.strip(), domain.strip()

    if not local:
        return ParseResult.failure(ParseCode.EMAIL_MISSING_LOCAL, raw, "Missing the name before the @.")
    if not domain:
        return ParseResult.failure(ParseCode.EMAIL_MISSING_DOMAIN, raw, "Missing the domain after the @.")

    if " " in local or not _LOCAL_OK.match(local):
        bad = sorted({c for c in local if not _LOCAL_OK.match(c)})
        return ParseResult.failure(
            ParseCode.EMAIL_INVALID_CHARS, raw,
            f"Invalid character(s) in the name part: {' '.join(repr(c) for c in bad) or 'whitespace'}.",
            local=local, domain=domain,
        )
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return ParseResult.failure(ParseCode.EMAIL_INVALID_CHARS, raw, "Misplaced dot in the name part.")

    if " " in domain:
        return ParseResult.failure(ParseCode.EMAIL_INVALID_CHARS, raw, "Whitespace in the domain.")
    if "." not in domain:
        return ParseResult.failure(
            ParseCode.EMAIL_MISSING_TLD, raw,
            f"Domain '{domain}' has no top-level domain (expected something like '{domain}.com').",
            local=local, domain=domain,
        )

    labels = domain.split(".")
    if any(not _LABEL_OK.match(lbl) for lbl in labels):
        return ParseResult.failure(
            ParseCode.EMAIL_INVALID_CHARS, raw, f"Malformed domain '{domain}'.", local=local, domain=domain
        )
    tld = labels[-1]
    if not _TLD_OK.match(tld):
        return ParseResult.failure(
            ParseCode.EMAIL_INVALID_TLD, raw, f"'{tld}' is not a valid top-level domain.",
            local=local, domain=domain,
        )

    if cfg.lowercase:
        local_out, domain_out = local, domain.lower()
        if domain_out != domain and code is ParseCode.OK:
            code = ParseCode.OK_NORMALISED
        local_out = local_out.lower()
    else:
        local_out, domain_out = local, domain

    is_free = domain_out in FREEMAIL or domain_out in cfg.extra_freemail
    is_disp = domain_out in DISPOSABLE or domain_out in cfg.extra_disposable
    is_role = local_out.split("+")[0] in ROLE_LOCALS

    if is_free and cfg.reject_freemail:
        return ParseResult.failure(
            ParseCode.EMAIL_FREEMAIL, raw, f"'{domain_out}' is a personal mailbox, not a company domain."
        )
    if cfg.flag_freemail and is_free:
        warnings.append(ParseCode.EMAIL_FREEMAIL)
    if is_disp:
        warnings.append(ParseCode.EMAIL_DISPOSABLE)
    if cfg.flag_role and is_role:
        warnings.append(ParseCode.EMAIL_ROLE_ADDRESS)

    email = Email(
        address=f"{local_out}@{domain_out}",
        local=local_out,
        domain=domain_out,
        tld=tld.lower(),
        is_freemail=is_free,
        is_role=is_role,
        is_disposable=is_disp,
    )
    detail = "Valid address."
    if code is ParseCode.OK_REPAIRED:
        detail = f"Repaired obfuscation: '{raw.strip()}' → '{email.address}'."
    elif code is ParseCode.OK_NORMALISED:
        detail = f"Normalised to '{email.address}'."
    return ParseResult.success(
        email, raw, code, detail, tuple(warnings),
        domain=email.domain, local=email.local, freemail=is_free, role=is_role,
    )
