"""The intent extraction engine.

Pipeline, in order:

  raw note
    → normalise (length-preserving, so offsets stay valid)
    → split into clauses, marking which sit after a contrast connective
    → find negation scopes inside each clause
    → fire lexicon rules, emitting one Signal per match with its evidence span
    → resolve each dimension (max strength wins; never a sum)
    → classify intent type (priority-ordered; routes, does not score)
    → extract problems, timeline, explicit negative signals
    → compute per-dimension and overall confidence, contradictions, gaps

Everything the engine knows about the market lives in lexicon.yaml.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from ..core.text import Clause, modality_at, normalise, snippet, split_clauses
from .lexicon import CompiledRule, Lexicon, load_lexicon
from .schema import (
    DERIVED_DIMENSIONS,
    DIMENSIONS,
    BuyingStage,
    Commitment,
    Contradiction,
    IntentProfile,
    IntentType,
    Level,
    NegativeSignal,
    Polarity,
    Resolution,
    Signal,
    SolutionIntent,
    Urgency,
)

#: Which dimensions we expect a real buyer's note to mention. Anything absent
#: is reported as missing information rather than treated as a negative (§9).
EXPECTED = ("budget_status", "urgency", "decision_authority", "buying_stage", "pain_severity")

_HUMAN = {
    "budget_status": "budget",
    "urgency": "timeline / urgency",
    "decision_authority": "decision authority",
    "buying_stage": "buying stage",
    "pain_severity": "pain severity",
    "problem_recognition": "problem recognition",
    "competitive_evaluation": "competitive evaluation",
}


class IntentExtractor:
    """Reusable, stateless-per-call extractor. Build once, run over many notes."""

    def __init__(self, lexicon: Lexicon | str | None = None):
        if isinstance(lexicon, Lexicon):
            self.lexicon = lexicon
        else:
            self.lexicon = load_lexicon(lexicon)
        s = self.lexicon.settings
        self.min_conf = float(s.get("min_signal_confidence", 0.15))

    # -- public -----------------------------------------------------------
    def extract(
        self,
        notes: Any,
        *,
        lead_id: str = "",
        company: str = "",
        contact: str = "",
        title_authority: str | None = None,
    ) -> IntentProfile:
        """Extract an IntentProfile from a note.

        `title_authority` lets the caller feed in a job-title-derived authority
        level (e.g. LIKELY for a Founder). It is kept strictly weaker than any
        explicit statement in the note, so "I make the call here" always wins
        and a title alone never masquerades as explicit authority.
        """
        raw = "" if notes is None else str(notes)
        profile = IntentProfile(
            lead_id=lead_id, company=company, contact=contact,
            raw_notes=raw, note_length=len(raw.split()),
        )
        if not raw.strip():
            profile.missing_information = ("notes are empty",) + tuple(
                f"no {_HUMAN[d]} stated" for d in EXPECTED
            )
            profile.confidence = 0.0
            profile = self._apply_title_authority(profile, title_authority, raw)
            return profile

        clauses = split_clauses(raw)
        signals = self._fire_rules(raw, clauses)

        profile.all_signals = tuple(signals)
        profile.resolutions, contradictions = self._resolve_all(signals)
        profile.contradictions = tuple(contradictions)

        profile = self._apply_title_authority(profile, title_authority, raw)

        itype, iconf, ievid = self._classify_intent_type(raw, clauses, profile)
        profile.intent_type = itype
        profile.intent_type_confidence = iconf
        profile.intent_type_evidence = ievid

        profile.problems, profile.problem_categories, profile.problem_evidence = self._problems(raw)
        profile.timeline_text, profile.timeline_days = self._timeline(raw)
        profile.negative_signals = self._negatives(raw)

        profile.commitment = self._derive_commitment(profile)
        profile.solution_intent = self._derive_solution_intent(profile)

        profile.missing_information = self._missing(profile)
        profile.confidence = self._overall_confidence(profile)
        return profile

    def extract_many(self, notes: Iterable[Any], **kw) -> list[IntentProfile]:
        return [self.extract(n, **kw) for n in notes]

    # -- stage 1: fire rules ---------------------------------------------
    def _fire_rules(self, raw: str, clauses: list[Clause]) -> list[Signal]:
        signals: list[Signal] = []
        for clause in clauses:
            for rule in self.lexicon.rules:
                for pattern in rule.patterns:
                    for m in pattern.finditer(clause.text):
                        sig = self._make_signal(rule, m, clause, raw)
                        if sig is not None:
                            signals.append(sig)
        return signals

    def _make_signal(self, rule: CompiledRule, m, clause: Clause, raw: str) -> Signal | None:
        local = m.start()
        negated = clause.is_negated_at(local)

        value = rule.value
        polarity = Polarity.POSITIVE
        strength = rule.strength
        if negated:
            if rule.negated_value is None:
                return None  # negation suppresses this reading entirely
            value = rule.negated_value
            polarity = Polarity.NEGATIVE
            strength = rule.strength * 0.95

        abs_start = clause.start + local
        abs_end = clause.start + m.end()
        conf = min(1.0, rule.specificity * modality_at(clause.text, local))
        if negated:
            conf *= 0.9

        return Signal(
            dimension=rule.dimension,
            value=value,
            strength=strength,
            specificity=rule.specificity,
            polarity=polarity,
            rule_id=rule.id,
            evidence=snippet(raw, abs_start, abs_end),
            start=abs_start,
            end=abs_end,
            clause_index=clause.index,
            is_operative=clause.is_operative,
            negated=negated,
            confidence=conf,
        )

    # -- stage 2: resolve --------------------------------------------------
    def _resolve_all(
        self, signals: list[Signal]
    ) -> tuple[dict[str, Resolution], list[Contradiction]]:
        resolutions: dict[str, Resolution] = {}
        contradictions: list[Contradiction] = []

        for dim, enum_cls in DIMENSIONS.items():
            group = [s for s in signals if s.dimension == dim and s.confidence >= self.min_conf]
            if not group:
                resolutions[dim] = Resolution(
                    dimension=dim, value=list(enum_cls)[0], confidence=0.0,
                    evidence="", rule_id="", span=None, is_unknown=True,
                )
                continue

            winner = max(group, key=self._priority)
            try:
                value = enum_cls[winner.value]
            except KeyError as exc:
                raise ValueError(
                    f"Rule '{winner.rule_id}' produced value '{winner.value}' "
                    f"which is not a member of {enum_cls.__name__}."
                ) from exc

            supporting = tuple(s for s in group if s.value == winner.value)
            conflicting = tuple(s for s in group if s.value != winner.value)

            # A contradiction is a materially different reading, not a
            # neighbouring one. Two ladder steps apart, or opposite polarity.
            real_conflicts = []
            for s in conflicting:
                try:
                    other = enum_cls[s.value]
                except KeyError:
                    continue
                far = abs(other.ordinal - value.ordinal) >= 2
                opposed = s.polarity is not winner.polarity
                if far or opposed:
                    real_conflicts.append(s)

            if real_conflicts:
                worst = max(real_conflicts, key=lambda s: s.strength)
                rule_used = (
                    "post-contrast clause takes precedence"
                    if winner.is_operative and not worst.is_operative
                    else "stronger, more specific evidence retained"
                )
                contradictions.append(
                    Contradiction(
                        dimension=dim, kept=value.name, rejected=worst.value, rule=rule_used,
                        evidence_kept=winner.evidence, evidence_rejected=worst.evidence,
                    )
                )

            # Confidence: winner's own confidence, lifted slightly by
            # corroboration, cut by genuine disagreement.
            conf = winner.confidence
            corroboration = min(0.15, 0.05 * (len(supporting) - 1))
            penalty = 0.2 if real_conflicts else 0.0
            conf = max(0.05, min(1.0, conf + corroboration - penalty))

            resolutions[dim] = Resolution(
                dimension=dim, value=value, confidence=conf,
                evidence=winner.evidence, rule_id=winner.rule_id,
                span=(winner.start, winner.end),
                supporting=supporting, conflicting=tuple(real_conflicts),
                is_unknown=False,
            )
        return resolutions, contradictions

    @staticmethod
    def _priority(s: Signal) -> tuple:
        """Winner selection. Never a sum — the strongest, most specific,
        most operative reading wins outright."""
        return (s.is_operative, s.strength, s.specificity, s.confidence, -s.start)

    # -- stage 3: intent type ---------------------------------------------
    def _classify_intent_type(
        self, raw: str, clauses: list[Clause], profile: IntentProfile
    ) -> tuple[IntentType, float, str]:
        norm = normalise(raw)
        for rule in self.lexicon.intent_types:  # already priority-sorted
            for pattern in rule.patterns:
                m = pattern.search(norm)
                if not m:
                    continue
                # A hit inside a negation scope is not a classification.
                clause = next((c for c in clauses if c.start <= m.start() < c.end), None)
                if clause is not None and clause.is_negated_at(m.start() - clause.start):
                    continue
                try:
                    itype = IntentType(rule.type)
                except ValueError:
                    continue
                return itype, rule.confidence, snippet(raw, m.start(), m.end())
        return IntentType.UNKNOWN, 0.3, ""

    # -- stage 4: problems / timeline / negatives -------------------------
    def _problems(self, raw: str) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
        norm = normalise(raw)
        labels, cats, evid = [], [], {}
        for p in self.lexicon.problems:
            for pattern in p.patterns:
                m = pattern.search(norm)
                if m:
                    if p.label not in labels:
                        labels.append(p.label)
                        evid[p.label] = snippet(raw, m.start(), m.end())
                    if p.category not in cats:
                        cats.append(p.category)
                    break
        return tuple(labels), tuple(cats), evid

    def _timeline(self, raw: str) -> tuple[str, int | None]:
        norm = normalise(raw)
        best: tuple[str, int] | None = None
        for t in self.lexicon.timelines:
            m = t.pattern.search(norm)
            if not m:
                continue
            if t.days_from_group is not None:
                try:
                    days = int(m.group(t.days_from_group)) * t.multiplier
                except (IndexError, ValueError, TypeError):
                    continue
                label = t.label.format(m.group(t.days_from_group))
            else:
                days = t.days or 0
                label = t.label
            if best is None or days < best[1]:  # soonest explicit horizon wins
                best = (label, days)
        return best if best else ("", None)

    def _negatives(self, raw: str) -> tuple[NegativeSignal, ...]:
        norm = normalise(raw)
        out: list[NegativeSignal] = []
        seen: set[str] = set()
        for rule in self.lexicon.negatives:
            if rule.code in seen:
                continue
            for pattern in rule.patterns:
                m = pattern.search(norm)
                if m:
                    out.append(
                        NegativeSignal(
                            code=rule.code, reason=rule.reason,
                            evidence=snippet(raw, m.start(), m.end()),
                            span=(m.start(), m.end()), severity=rule.severity,
                        )
                    )
                    seen.add(rule.code)
                    break
        return tuple(sorted(out, key=lambda n: -n.severity))

    # -- stage 5: derived views (non-scoring) -----------------------------
    @staticmethod
    def _derive_commitment(p: IntentProfile) -> Commitment:
        stage, urg = p.buying_stage, p.urgency
        if stage is BuyingStage.PURCHASE:
            return Commitment.READY_TO_BUY
        if urg.ordinal >= Urgency.HIGH.ordinal and stage.ordinal >= BuyingStage.EVALUATION.ordinal:
            return Commitment.CONCRETE_TIMELINE
        if stage is BuyingStage.DECISION:
            return Commitment.CONCRETE_TIMELINE
        if stage is BuyingStage.EVALUATION:
            return Commitment.ACTIVE_EVALUATION
        if stage is BuyingStage.EXPLORATION:
            return Commitment.EXPLORATION
        if stage is BuyingStage.AWARENESS:
            return Commitment.INTEREST
        if any(n.code in ("NOT_A_BUYER", "JOB_SEEKER", "SPAM") for n in p.negative_signals):
            return Commitment.NONE
        return Commitment.UNKNOWN

    @staticmethod
    def _derive_solution_intent(p: IntentProfile) -> SolutionIntent:
        return {
            BuyingStage.UNKNOWN: SolutionIntent.UNKNOWN,
            BuyingStage.AWARENESS: SolutionIntent.GENERAL_INTEREST,
            BuyingStage.EXPLORATION: SolutionIntent.EXPLORATION,
            BuyingStage.EVALUATION: SolutionIntent.EVALUATION,
            BuyingStage.DECISION: SolutionIntent.EVALUATION,
            BuyingStage.PURCHASE: SolutionIntent.IMPLEMENTATION,
        }[p.buying_stage]

    # -- stage 6: authority from title ------------------------------------
    def _apply_title_authority(
        self, p: IntentProfile, title_level: str | None, raw: str
    ) -> IntentProfile:
        if not title_level:
            return p
        from .schema import DecisionAuthority

        try:
            level = DecisionAuthority[title_level]
        except KeyError:
            return p
        current = p.resolutions.get("decision_authority")
        # Never let a job title override something the lead said explicitly.
        if current is not None and not current.is_unknown:
            return p
        p.resolutions["decision_authority"] = Resolution(
            dimension="decision_authority", value=level, confidence=0.55,
            evidence="(inferred from job title)", rule_id="authority.from_title",
            span=None, is_unknown=False,
        )
        return p

    # -- stage 7: quality --------------------------------------------------
    @staticmethod
    def _missing(p: IntentProfile) -> tuple[str, ...]:
        out = [f"no {_HUMAN[d]} stated" for d in EXPECTED if p.resolutions.get(d, None) is None or p.resolutions[d].is_unknown]
        if p.note_length < 6:
            out.append("note is very short")
        if not p.problems:
            out.append("no specific problem identified")
        return tuple(out)

    def _overall_confidence(self, p: IntentProfile) -> float:
        known = [r for r in p.resolutions.values() if not r.is_unknown]
        if not known:
            base = 0.1
        else:
            base = sum(r.confidence for r in known) / len(known)
            coverage = len(known) / max(1, len(DIMENSIONS))
            base = 0.65 * base + 0.35 * coverage
        if p.note_length < 6:
            base *= 0.7
        elif p.note_length > 25:
            base = min(1.0, base * 1.05)
        base -= 0.05 * len(p.contradictions)
        if p.intent_type is IntentType.UNKNOWN:
            base *= 0.85
        else:
            base = 0.75 * base + 0.25 * p.intent_type_confidence

        # Confidence measures certainty of interpretation, not lead value (§7).
        # "You have WON $1,000,000" fills in almost no dimensions, yet we are
        # extremely sure what it is — so a decisive classification sets a floor.
        if p.intent_type not in (IntentType.UNKNOWN, IntentType.PURCHASE):
            base = max(base, p.intent_type_confidence * 0.9)
        elif p.negative_signals:
            base = max(base, max(n.severity for n in p.negative_signals) * 0.7)
        return round(max(0.0, min(1.0, base)), 3)


# module-level convenience -------------------------------------------------
_DEFAULT: IntentExtractor | None = None


def extract_intent(notes: Any, **kw) -> IntentProfile:
    """One-shot extraction using the default lexicon."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = IntentExtractor()
    return _DEFAULT.extract(notes, **kw)
