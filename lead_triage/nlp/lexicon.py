"""Loads and compiles lexicon.yaml into executable rule objects.

Kept separate from the extractor so a caller can swap in a completely
different vocabulary (another market, another language) without touching the
engine, and so a bad regex fails loudly at load time rather than silently at
match time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_LEXICON = Path(__file__).resolve().parent.parent / "config" / "lexicon.yaml"


@dataclass(frozen=True)
class CompiledRule:
    id: str
    dimension: str
    value: str
    negated_value: str | None
    strength: float
    specificity: float
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class IntentTypeRule:
    id: str
    type: str
    priority: int
    confidence: float
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class NegativeRule:
    code: str
    reason: str
    severity: float
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class TimelineRule:
    pattern: re.Pattern[str]
    label: str
    days: int | None
    days_from_group: int | None
    multiplier: int


@dataclass(frozen=True)
class ProblemRule:
    id: str
    label: str
    category: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass
class Lexicon:
    settings: dict[str, Any] = field(default_factory=dict)
    rules: tuple[CompiledRule, ...] = ()
    intent_types: tuple[IntentTypeRule, ...] = ()
    negatives: tuple[NegativeRule, ...] = ()
    timelines: tuple[TimelineRule, ...] = ()
    problems: tuple[ProblemRule, ...] = ()
    source: str = ""

    @property
    def dimensions(self) -> set[str]:
        return {r.dimension for r in self.rules}

    def rules_for(self, dimension: str) -> list[CompiledRule]:
        return [r for r in self.rules if r.dimension == dimension]


def _compile(patterns: Iterable[str], where: str) -> tuple[re.Pattern[str], ...]:
    out = []
    for p in patterns or ():
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid regex in {where}: {p!r} — {exc}") from exc
    return tuple(out)


def load_lexicon(path: str | Path | None = None, data: dict | None = None) -> Lexicon:
    """Load, validate and compile a lexicon.

    Raises ValueError with the offending rule id on any malformed entry, so a
    typo in the config surfaces immediately instead of quietly matching nothing.
    """
    if data is None:
        p = Path(path or DEFAULT_LEXICON)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        source = str(p)
    else:
        source = "<inline>"

    settings = data.get("settings", {}) or {}
    d_strength = float(settings.get("default_strength", 0.6))
    d_spec = float(settings.get("default_specificity", 0.6))

    rules = []
    seen: set[str] = set()
    for raw in data.get("rules", []) or []:
        rid = raw.get("id")
        if not rid:
            raise ValueError(f"Rule without an id: {raw}")
        if rid in seen:
            raise ValueError(f"Duplicate rule id: {rid}")
        seen.add(rid)
        if not raw.get("dimension") or not raw.get("value"):
            raise ValueError(f"Rule {rid} needs both 'dimension' and 'value'.")
        rules.append(
            CompiledRule(
                id=rid,
                dimension=str(raw["dimension"]),
                value=str(raw["value"]),
                negated_value=(str(raw["negated_value"]) if raw.get("negated_value") else None),
                strength=float(raw.get("strength", d_strength)),
                specificity=float(raw.get("specificity", d_spec)),
                patterns=_compile(raw.get("patterns", []), f"rule '{rid}'"),
            )
        )

    intent_types = tuple(
        sorted(
            (
                IntentTypeRule(
                    id=str(r.get("id", r.get("type", "?"))),
                    type=str(r["type"]),
                    priority=int(r.get("priority", 0)),
                    confidence=float(r.get("confidence", 0.7)),
                    patterns=_compile(r.get("patterns", []), f"intent_type '{r.get('id')}'"),
                )
                for r in data.get("intent_types", []) or []
            ),
            key=lambda r: -r.priority,
        )
    )

    negatives = tuple(
        NegativeRule(
            code=str(r["code"]),
            reason=str(r.get("reason", "")),
            severity=float(r.get("severity", 0.5)),
            patterns=_compile(r.get("patterns", []), f"negative '{r.get('code')}'"),
        )
        for r in data.get("negative_signals", []) or []
    )

    timelines = tuple(
        TimelineRule(
            pattern=_compile([r["pattern"]], "timeline")[0],
            label=str(r.get("label", "")),
            days=(int(r["days"]) if r.get("days") is not None else None),
            days_from_group=(int(r["days_from_group"]) if r.get("days_from_group") else None),
            multiplier=int(r.get("multiplier", 1)),
        )
        for r in data.get("timeline", []) or []
    )

    problems = tuple(
        ProblemRule(
            id=str(r["id"]),
            label=str(r.get("label", r["id"])),
            category=str(r.get("category", "Other")),
            patterns=_compile(r.get("patterns", []), f"problem '{r.get('id')}'"),
        )
        for r in data.get("problems", []) or []
    )

    return Lexicon(settings, tuple(rules), intent_types, negatives, timelines, problems, source)
