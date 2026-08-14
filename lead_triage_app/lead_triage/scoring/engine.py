"""Scoring & triage.

This module contains no vocabulary and no thresholds — everything comes from
scoring.yaml. It reads the NLP's structured signals plus the parsed structured
fields, and answers the one question the NLP deliberately refuses to answer:
how much does all of that matter?

Three deliberate properties:

* UNKNOWN is worth 0, never a penalty. Incomplete records are not punished
  for being incomplete (guide §9).
* intent_type ROUTES rather than scores. A student with a 2-week "deadline"
  cannot out-add a real buyer, because their ceiling is NURTURE regardless
  of how many points the note accumulates.
* Every point is logged with its reason, so any band is fully explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..ingest.loader import LeadRecord
from ..nlp.schema import DIMENSIONS, IntentType

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class ScoringConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ScoringConfig":
        p = Path(path or CONFIG_DIR / "scoring.yaml")
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def band_order(self) -> list[str]:
        return [b["name"] for b in self.raw.get("bands", [])]


class ScoringEngine:
    def __init__(self, config: ScoringConfig | str | Path | None = None):
        self.cfg = config if isinstance(config, ScoringConfig) else ScoringConfig.load(config)
        self.bands = self.cfg.get("bands", [])
        self.order = self.cfg.band_order  # best -> worst

    # -- public -----------------------------------------------------------
    def score(self, rec: LeadRecord) -> LeadRecord:
        lines: list[dict[str, Any]] = []
        total = 0.0

        total += self._intent_points(rec, lines)
        total += self._fit_points(rec, lines)
        total += self._quality_points(rec, lines)
        total += self._negative_points(rec, lines)

        routing = self._routing(rec)
        mult = float(routing.get("multiplier", 1.0))
        if mult != 1.0:
            before = total
            total *= mult
            lines.append({
                "group": "Routing",
                "label": f"{rec.intent.intent_type.value if rec.intent else 'UNKNOWN'} multiplier ×{mult}",
                "points": round(total - before, 1),
                "evidence": routing.get("note", ""),
            })

        rec.score = round(total, 1)
        rec.display_score = self._display(total)
        rec.route = routing.get("route", "Sales")

        band = self._band_for(total)
        caps: list[str] = []

        ceiling = routing.get("max_band")
        if ceiling and self._worse(ceiling, band):
            caps.append(
                f"Capped at {self._label(ceiling)} — {rec.intent.intent_type.value if rec.intent else 'UNKNOWN'} enquiry"
                + (f": {routing['note']}" if routing.get("note") else "")
            )
            band = ceiling

        # hard_cap_codes are PLACEMENTS, not just ceilings: "NO_BUDGET →
        # NURTURE" means put this lead in nurture, in both directions. A
        # cash-poor startup that scores below the nurture threshold still
        # belongs in nurture, and a strong lead that admits it has no budget
        # still drops to it. DISQUALIFY placements are applied last so they
        # always win.
        hard_caps = self._hard_caps(rec)
        placements = [(b, why) for b, why in hard_caps if b != "DISQUALIFY"]
        if placements:
            target, why = max(placements, key=lambda c: self.order.index(c[0]))
            if target != band:
                verb = "Raised to" if self._worse(band, target) else "Held at"
                caps.append(f"{verb} {self._label(target)} — {why.split('— ', 1)[-1]}")
                band = target
        for cap_band, reason in hard_caps:
            if cap_band == "DISQUALIFY" and band != "DISQUALIFY":
                caps.append(reason)
                band = cap_band

        # A floor, not just a ceiling. A student or an analyst scores badly on
        # a pipeline scale and would otherwise fall to DISQUALIFY — but a
        # non-purchase enquiry is not dead, it belongs to a different team.
        #
        # The floor deliberately outranks the negative-signal caps: a student
        # writing "not looking to buy" is stating the very thing that makes
        # them EDUCATIONAL, so letting NOT_A_BUYER also disqualify them counts
        # the same sentence twice. Types that really are dead (spam, job
        # applicants, competitors) simply declare no min_band and are never
        # lifted.
        floor = routing.get("min_band")
        if floor and self._worse(band, floor):
            caps.append(
                f"Raised to {self._label(floor)} — "
                + (routing.get("note") or "non-purchase enquiry, routed rather than dropped")
            )
            band = floor

        rec.band = band
        rec.band_label = self._label(band)
        rec.explanation = lines
        rec.caps_applied = caps
        return rec

    def score_all(self, records: Iterable[LeadRecord]) -> list[LeadRecord]:
        return [self.score(r) for r in records]

    def band_meta(self, name: str) -> dict[str, Any]:
        return next((b for b in self.bands if b["name"] == name), {})

    # -- components -------------------------------------------------------
    def _intent_points(self, rec: LeadRecord, lines: list[dict]) -> float:
        w = self.cfg.get("intent_weights", {})
        if rec.intent is None:
            return 0.0
        subtotal = 0.0
        for dim in DIMENSIONS:
            table = w.get(dim)
            if not table:
                continue
            res = rec.intent.resolutions.get(dim)
            value = res.value.name if res else "UNKNOWN"
            pts = float(table.get(value, table.get(str(value), 0)) or 0)
            if pts:
                subtotal += pts
                lines.append({
                    "group": "Intent",
                    "label": f"{dim.replace('_', ' ').capitalize()}: {value}",
                    "points": pts,
                    "evidence": res.evidence if res else "",
                    "confidence": round(res.confidence, 2) if res else 0.0,
                })

        bonus = float(self.cfg.get("problem_identified_bonus", 0) or 0)
        if rec.intent.problems and bonus:
            subtotal += bonus
            lines.append({
                "group": "Intent", "label": f"Named problem: {', '.join(rec.intent.problems)}",
                "points": bonus, "evidence": "; ".join(rec.intent.problem_evidence.values()),
            })

        floor = float(self.cfg.get("confidence_floor", 0.55))
        factor = max(floor, rec.intent.confidence)
        if factor < 1.0 and subtotal:
            adj = subtotal * factor - subtotal
            lines.append({
                "group": "Intent",
                "label": f"NLP confidence ×{factor:.2f}",
                "points": round(adj, 1),
                "evidence": f"{len(rec.intent.missing_information)} field(s) not stated in the note",
            })
            subtotal *= factor
        return subtotal

    def _fit_points(self, rec: LeadRecord, lines: list[dict]) -> float:
        f = self.cfg.get("fit_weights", {})
        total = 0.0

        pts = float(f.get("employee_band", {}).get(rec.employee_band, 0) or 0)
        if pts:
            total += pts
            lines.append({"group": "Fit", "label": f"Company size: {rec.employee_band}",
                          "points": pts, "evidence": rec.employees_raw or "not stated"})

        icp_table = f.get("is_icp_agency", {})
        key = "unknown" if rec.is_icp_agency is None else str(bool(rec.is_icp_agency)).lower()
        pts = float(icp_table.get(key, icp_table.get(rec.is_icp_agency, 0)) or 0)
        if pts:
            total += pts
            label = {"true": "In target profile (agency)", "false": "Outside target profile",
                     "unknown": "Target profile unknown"}[key]
            total_line = {"group": "Fit", "label": label, "points": pts,
                          "evidence": rec.company or ""}
            lines.append(total_line)

        sq = float(f.get("source_quality_points", 0) or 0) * rec.source_quality
        if sq:
            total += sq
            lines.append({"group": "Fit", "label": f"Source: {rec.source or 'unknown'}",
                          "points": round(sq, 1), "evidence": f"channel quality {rec.source_quality:.2f}"})

        b = f.get("budget_vs_floor", {})
        floor = float(b.get("floor", 0) or 0)
        monthly = rec.monthly_budget
        if monthly is None:
            pts = float(b.get("unknown", 0) or 0)
            ev = rec.raw.get("monthly_budget", "") or "not stated"
            lbl = "Stated budget: unknown"
        elif monthly > floor:
            pts, lbl, ev = float(b.get("above_floor", 0) or 0), "Stated budget above floor", f"${monthly:,.0f}/mo"
        elif monthly == floor:
            pts, lbl, ev = float(b.get("at_floor", 0) or 0), "Stated budget at floor", f"${monthly:,.0f}/mo"
        else:
            pts, lbl, ev = float(b.get("below_floor", 0) or 0), "Stated budget below floor", f"${monthly:,.0f}/mo"
        if pts:
            total += pts
            lines.append({"group": "Fit", "label": lbl, "points": pts, "evidence": ev})

        if rec.is_non_buyer_title:
            pts = float(f.get("non_buyer_title", 0) or 0)
            total += pts
            lines.append({"group": "Fit", "label": f"Non-buying role: {rec.title}",
                          "points": pts, "evidence": rec.title})
        return total

    def _quality_points(self, rec: LeadRecord, lines: list[dict]) -> float:
        q = self.cfg.get("quality_weights", {})
        total = 0.0

        def add(key: str, label: str, evidence: str = "") -> None:
            nonlocal total
            pts = float(q.get(key, 0) or 0)
            if pts:
                total += pts
                lines.append({"group": "Data quality", "label": label, "points": pts, "evidence": evidence})

        if rec.email and not rec.email.ok:
            add("invalid_email", f"Invalid email ({rec.email.code.value})", rec.email.raw)
        elif rec.email and rec.email.ok and rec.email.value:
            if rec.email.value.is_freemail:
                add("freemail_email", "Personal email domain", rec.email.value.domain)
            if rec.email.value.is_role:
                add("role_email", "Role mailbox", rec.email.value.address)

        if rec.website and not rec.website.ok:
            key = "missing_website" if rec.website.code.value == "EMPTY" else "invalid_website"
            add(key, f"Website {rec.website.code.value.lower().replace('_', ' ')}", rec.website.raw)

        if rec.domain_match:
            add("email_domain_matches_website", "Email domain matches website")
        if not rec.notes.strip():
            add("missing_notes", "No notes — nothing to assess intent from")
        if rec.is_duplicate:
            add("duplicate_lead", "Duplicate lead ID", rec.id_display)
        if rec.lead_id and not rec.lead_id.ok:
            add("missing_lead_id", f"Lead ID {rec.lead_id.code.value}", rec.lead_id.raw)
        return total

    def _negative_points(self, rec: LeadRecord, lines: list[dict]) -> float:
        if rec.intent is None or not rec.intent.negative_signals:
            return 0.0
        factor = float(self.cfg.get("negative_signal_penalty", 0) or 0)
        # Some negative signals are already represented as a dimension value —
        # NO_BUDGET is scored once through budget_status. Charging the penalty
        # too would deduct for the same sentence twice, which is what drags
        # "no budget yet but sharp and might grow" below a nurture lead.
        already = set(self.cfg.get("negative_signals_already_scored", []) or [])
        total = 0.0
        for n in rec.intent.negative_signals:
            if n.code in already:
                lines.append({"group": "Negative signal",
                              "label": f"{n.code}: {n.reason}", "points": 0.0,
                              "evidence": f"{n.evidence} — already counted in the intent block"})
                continue
            pts = -round(n.severity * factor, 1)
            total += pts
            lines.append({"group": "Negative signal", "label": f"{n.code}: {n.reason}",
                          "points": pts, "evidence": n.evidence})
        return total

    # -- routing / caps ---------------------------------------------------
    def _routing(self, rec: LeadRecord) -> dict[str, Any]:
        table = self.cfg.get("intent_routing", {})
        itype = rec.intent.intent_type.value if rec.intent else IntentType.UNKNOWN.value
        return table.get(itype, table.get("UNKNOWN", {})) or {}

    def _hard_caps(self, rec: LeadRecord) -> list[tuple[str, str]]:
        caps = self.cfg.get("hard_cap_codes", {}) or {}
        out: list[tuple[str, str]] = []
        if rec.intent is None:
            return out
        for n in rec.intent.negative_signals:
            band = caps.get(n.code)
            if band:
                out.append((band, f"Capped at {self._label(band)} — {n.reason.lower()} (“{n.evidence}”)"))
        return out

    # -- helpers -----------------------------------------------------------
    def _band_for(self, score: float) -> str:
        for b in self.bands:
            if score >= float(b.get("min_score", -1e9)):
                return b["name"]
        return self.bands[-1]["name"] if self.bands else "DISQUALIFY"

    def _worse(self, candidate: str, current: str) -> bool:
        try:
            return self.order.index(candidate) > self.order.index(current)
        except ValueError:
            return False

    def _label(self, band: str) -> str:
        return self.band_meta(band).get("label", band.replace("_", " ").title())

    def _display(self, raw: float) -> float:
        d = self.cfg.get("display", {})
        lo, hi = float(d.get("raw_min", -60)), float(d.get("raw_max", 110))
        return round(max(0.0, min(100.0, (raw - lo) / (hi - lo) * 100)), 1)
