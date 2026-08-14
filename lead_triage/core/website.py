"""Task 6 — parse and normalise a website string.

Accepts 'luxauto.io', 'www.apexsend.co', 'http://upshiftloop.agency',
'https://x.com/path?q=1'. Rejects empty strings, junk ('asdf'), bare hosts
with no dot, and invalid TLDs — each with its own return code.

Also exposes `registrable_domain`, which is what you actually compare against
an email domain to check that a lead's address matches their company.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .result import ParseCode, ParseResult

JUNK = frozenset({"asdf", "test", "website", "n/a", "na", "none", "null", "-", "tbd", "no website", "none yet"})

#: Second-level suffixes that are not the registrable part on their own.
MULTI_PART_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.za", "com.au", "net.au", "org.au",
    "co.nz", "com.ng", "org.ng", "co.ke", "com.br", "co.in", "com.sg", "co.jp",
    "com.mx", "com.tr", "co.il", "com.gh",
})

_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_TLD = re.compile(r"^[A-Za-z]{2,24}$")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


@dataclass(frozen=True)
class WebsiteConfig:
    default_scheme: str = "https"
    strip_www: bool = True
    lowercase_host: bool = True
    keep_path: bool = False
    keep_query: bool = False
    allow_ip: bool = False
    require_tld: bool = True
    extra_junk: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Website:
    url: str
    scheme: str
    host: str
    registrable_domain: str
    tld: str
    path: str = ""
    had_scheme: bool = True
    had_www: bool = False

    def __str__(self) -> str:
        return self.url


def _registrable(host: str) -> tuple[str, str]:
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_SUFFIXES:
        return ".".join(parts[-3:]), ".".join(parts[-2:])
    if len(parts) >= 2:
        return ".".join(parts[-2:]), parts[-1]
    return host, ""


def parse_website(text: Any, config: WebsiteConfig | None = None) -> ParseResult[Website]:
    """Parse a website string.

    Failure codes: EMPTY, JUNK, WEB_NO_DOT, WEB_INVALID_HOST,
    WEB_INVALID_TLD, WEB_IS_EMAIL.
    Soft flag: WEB_SCHEME_ADDED.
    """
    cfg = config or WebsiteConfig()
    raw = "" if text is None else str(text)
    s = raw.strip().strip("<>\"'")

    if not s:
        return ParseResult.failure(ParseCode.EMPTY, raw, "No website supplied.")
    low = s.lower()
    if low in JUNK or low in cfg.extra_junk:
        return ParseResult.failure(ParseCode.JUNK, raw, f"'{s}' is a placeholder, not a website.")
    if re.search(r"\s", s) and not re.match(r"^https?://", s, re.I):
        s = s.split()[0]
    if "@" in s and not re.match(r"^https?://", s, re.I):
        return ParseResult.failure(
            ParseCode.WEB_IS_EMAIL, raw, f"'{s}' looks like an email address, not a website."
        )

    warnings: list[ParseCode] = []
    code = ParseCode.OK

    had_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s))
    if not had_scheme:
        s = f"{cfg.default_scheme}://{s}"
        warnings.append(ParseCode.WEB_SCHEME_ADDED)
        code = ParseCode.OK_NORMALISED

    try:
        parts = urlsplit(s)
    except ValueError as exc:
        return ParseResult.failure(ParseCode.WEB_INVALID_HOST, raw, f"Unparseable URL ({exc}).")

    host = (parts.hostname or "").strip(".")
    if not host:
        return ParseResult.failure(ParseCode.WEB_INVALID_HOST, raw, f"No host found in '{raw.strip()}'.")
    if cfg.lowercase_host:
        host = host.lower()

    if _IPV4.match(host):
        if not cfg.allow_ip:
            return ParseResult.failure(ParseCode.WEB_INVALID_HOST, raw, "IP addresses are not accepted.")
    else:
        if "." not in host:
            return ParseResult.failure(
                ParseCode.WEB_NO_DOT, raw,
                f"'{host}' has no dot — a website needs a domain and a TLD (e.g. '{host}.com').",
            )
        labels = host.split(".")
        if any(not _LABEL.match(lbl) for lbl in labels):
            bad = [lbl for lbl in labels if not _LABEL.match(lbl)]
            return ParseResult.failure(
                ParseCode.WEB_INVALID_HOST, raw, f"Invalid part(s) in host: {', '.join(bad)}."
            )
        if cfg.require_tld and not _TLD.match(labels[-1]):
            return ParseResult.failure(
                ParseCode.WEB_INVALID_TLD, raw, f"'{labels[-1]}' is not a valid top-level domain."
            )

    had_www = host.startswith("www.")
    if cfg.strip_www and had_www:
        host = host[4:]
        if code is ParseCode.OK:
            code = ParseCode.OK_NORMALISED

    registrable, tld = _registrable(host)
    path = parts.path if cfg.keep_path else ""
    if path in ("/",):
        path = ""
    url = f"{parts.scheme}://{host}{path}"
    if cfg.keep_query and parts.query:
        url += f"?{parts.query}"

    detail = f"Normalised to {url}."
    if not had_scheme:
        detail += f" Scheme '{cfg.default_scheme}' assumed."
    return ParseResult.success(
        Website(url, parts.scheme, host, registrable, tld, path, had_scheme, had_www),
        raw, code, detail, tuple(warnings),
        host=host, registrable_domain=registrable, tld=tld,
    )


def domains_match(website: Website | None, email_domain: str | None) -> bool | None:
    """True/False if both are present, None if either is unknown."""
    if website is None or not email_domain:
        return None
    return _registrable(email_domain.lower())[0] == website.registrable_domain
