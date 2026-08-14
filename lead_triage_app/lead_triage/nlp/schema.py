"""Typed output contract for the intent NLP layer.

Design notes that matter:

* Every dimension has an explicit UNKNOWN member at ordinal 0. Missing
  information is never silently the same as a negative reading (guide §9).
* `buying_stage` is the single progression ladder. `solution_intent` and
  `commitment` are derived VIEWS of it, marked non-scoring, because they are
  measured from the same evidence and would otherwise be triple-counted.
* Nothing here is a score. This layer answers "what does the note say", and
  the scoring engine separately answers "how much does that matter" (§10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Level(Enum):
    """Base for ordered categorical dimensions."""

    @property
    def ordinal(self) -> int:
        return list(type(self)).index(self)

    @classmethod
    def max(cls, values):
        vals = [v for v in values if v is not None]
        return max(vals, key=lambda v: v.ordinal) if vals else list(cls)[0]

    def __str__(self) -> str:  # "PRICE_SENSITIVE" -> "Price sensitive"
        return self.name.replace("_", " ").capitalize()


class IntentType(str, Enum):
    """What kind of contact this is. Classified FIRST — it routes the lead
    rather than contributing points, because an enthusiastic student would
    otherwise out-add a quiet buyer."""

    PURCHASE = "PURCHASE"
    RESEARCH = "RESEARCH"
    PARTNERSHIP = "PARTNERSHIP"
    EMPLOYMENT = "EMPLOYMENT"
    EDUCATIONAL = "EDUCATIONAL"
    COMPETITIVE = "COMPETITIVE"
    VENDOR_PITCH = "VENDOR_PITCH"   # someone selling TO us
    SPAM = "SPAM"
    UNKNOWN = "UNKNOWN"


class BuyingStage(Level):
    UNKNOWN = "UNKNOWN"
    AWARENESS = "AWARENESS"
    EXPLORATION = "EXPLORATION"
    EVALUATION = "EVALUATION"
    DECISION = "DECISION"
    PURCHASE = "PURCHASE"


class ProblemRecognition(Level):
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class PainSeverity(Level):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Urgency(Level):
    UNKNOWN = "UNKNOWN"
    NEGATIVE = "NEGATIVE"      # explicitly no plans
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class BudgetStatus(Level):
    UNKNOWN = "UNKNOWN"
    NO_BUDGET = "NO_BUDGET"
    PRICE_SENSITIVE = "PRICE_SENSITIVE"
    NOT_LOCKED = "NOT_LOCKED"
    BUDGETED = "BUDGETED"
    ALLOCATED = "ALLOCATED"
    APPROVED = "APPROVED"


class DecisionAuthority(Level):
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"
    INFLUENCER = "INFLUENCER"
    LIKELY = "LIKELY"          # inferred from job title
    EXPLICIT = "EXPLICIT"      # "I make the call here"


class CompetitiveEvaluation(Level):
    UNKNOWN = "UNKNOWN"
    NO = "NO"
    YES = "YES"


class Commitment(Level):
    """Derived view of buying_stage + urgency. Non-scoring by default."""

    UNKNOWN = "UNKNOWN"
    NONE = "NONE"
    INTEREST = "INTEREST"
    EXPLORATION = "EXPLORATION"
    ACTIVE_EVALUATION = "ACTIVE_EVALUATION"
    CONCRETE_TIMELINE = "CONCRETE_TIMELINE"
    READY_TO_BUY = "READY_TO_BUY"


class SolutionIntent(Level):
    """Derived view of buying_stage. Non-scoring by default."""

    UNKNOWN = "UNKNOWN"
    NONE = "NONE"
    GENERAL_INTEREST = "GENERAL_INTEREST"
    EXPLORATION = "EXPLORATION"
    EVALUATION = "EVALUATION"
    IMPLEMENTATION = "IMPLEMENTATION"


#: Dimension name -> enum class. Drives the generic resolver and the UI.
DIMENSIONS: dict[str, type[Level]] = {
    "buying_stage": BuyingStage,
    "problem_recognition": ProblemRecognition,
    "pain_severity": PainSeverity,
    "urgency": Urgency,
    "budget_status": BudgetStatus,
    "decision_authority": DecisionAuthority,
    "competitive_evaluation": CompetitiveEvaluation,
}

#: Dimensions computed from others; excluded from scoring to avoid
#: double-counting the same evidence.
DERIVED_DIMENSIONS: dict[str, type[Level]] = {
    "commitment": Commitment,
    "solution_intent": SolutionIntent,
}


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class Signal:
    """One rule firing on one span of text. The atomic unit of the engine.

    Signals are never summed. A resolver picks a winner per dimension, so
    saying "we have budget" three times does not out-rank "budget approved".
    """

    dimension: str
    value: str
    strength: float               # 0..1, how strong this reading is
    specificity: float            # 0..1, phrase match > bare keyword
    polarity: Polarity
    rule_id: str
    evidence: str                 # quote from the ORIGINAL note
    start: int
    end: int
    clause_index: int
    is_operative: bool            # clause sits after a contrast marker
    negated: bool
    confidence: float             # per-signal, after modality damping

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension, "value": self.value,
            "strength": round(self.strength, 3), "polarity": self.polarity.value,
            "rule_id": self.rule_id, "evidence": self.evidence,
            "span": [self.start, self.end], "clause": self.clause_index,
            "operative": self.is_operative, "negated": self.negated,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class Resolution:
    """The winning reading of one dimension, plus the trail behind it."""

    dimension: str
    value: Level
    confidence: float
    evidence: str
    rule_id: str
    span: tuple[int, int] | None
    supporting: tuple[Signal, ...] = ()
    conflicting: tuple[Signal, ...] = ()
    is_unknown: bool = False

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.name,
            "label": str(self.value),
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "rule_id": self.rule_id,
            "span": list(self.span) if self.span else None,
            "supporting": [s.to_dict() for s in self.supporting],
            "conflicting": [s.to_dict() for s in self.conflicting],
            "unknown": self.is_unknown,
        }


@dataclass(frozen=True)
class NegativeSignal:
    """An explicit disqualifier, with the reason — not just a low score (§3)."""

    code: str
    reason: str
    evidence: str
    span: tuple[int, int]
    severity: float  # 0..1


@dataclass(frozen=True)
class Contradiction:
    dimension: str
    kept: str
    rejected: str
    rule: str
    evidence_kept: str
    evidence_rejected: str


@dataclass
class IntentProfile:
    """Everything the NLP layer knows about one lead's notes."""

    # context
    lead_id: str = ""
    company: str = ""
    contact: str = ""
    raw_notes: str = ""

    # intent
    intent_type: IntentType = IntentType.UNKNOWN
    intent_type_confidence: float = 0.0
    intent_type_evidence: str = ""

    # resolved dimensions
    resolutions: dict[str, Resolution] = field(default_factory=dict)

    # derived views
    commitment: Commitment = Commitment.UNKNOWN
    solution_intent: SolutionIntent = SolutionIntent.UNKNOWN

    # problem extraction
    problems: tuple[str, ...] = ()
    problem_categories: tuple[str, ...] = ()
    problem_evidence: dict[str, str] = field(default_factory=dict)

    # timeline
    timeline_text: str = ""
    timeline_days: int | None = None

    # negatives & quality
    negative_signals: tuple[NegativeSignal, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    missing_information: tuple[str, ...] = ()
    all_signals: tuple[Signal, ...] = ()
    confidence: float = 0.0
    note_length: int = 0

    # ---- accessors ------------------------------------------------------
    def level(self, dimension: str) -> Level:
        r = self.resolutions.get(dimension)
        if r is not None:
            return r.value
        return list(DIMENSIONS[dimension])[0]

    def name(self, dimension: str) -> str:
        return self.level(dimension).name

    def evidence(self, dimension: str) -> str:
        r = self.resolutions.get(dimension)
        return r.evidence if r else ""

    def confidence_of(self, dimension: str) -> float:
        r = self.resolutions.get(dimension)
        return r.confidence if r else 0.0

    @property
    def buying_stage(self) -> BuyingStage: return self.level("buying_stage")          # type: ignore[return-value]
    @property
    def budget_status(self) -> BudgetStatus: return self.level("budget_status")        # type: ignore[return-value]
    @property
    def urgency(self) -> Urgency: return self.level("urgency")                         # type: ignore[return-value]
    @property
    def pain_severity(self) -> PainSeverity: return self.level("pain_severity")        # type: ignore[return-value]
    @property
    def problem_recognition(self) -> ProblemRecognition: return self.level("problem_recognition")  # type: ignore[return-value]
    @property
    def decision_authority(self) -> DecisionAuthority: return self.level("decision_authority")     # type: ignore[return-value]
    @property
    def competitive_evaluation(self) -> CompetitiveEvaluation: return self.level("competitive_evaluation")  # type: ignore[return-value]

    @property
    def is_purchase_intent(self) -> bool:
        return self.intent_type is IntentType.PURCHASE

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": {
                "lead_id": self.lead_id, "company": self.company,
                "contact": self.contact, "raw_notes": self.raw_notes,
            },
            "intent": {
                "intent_type": self.intent_type.value,
                "intent_type_confidence": round(self.intent_type_confidence, 3),
                "intent_type_evidence": self.intent_type_evidence,
                "buying_stage": self.buying_stage.name,
                "solution_intent": self.solution_intent.name,
                "commitment": self.commitment.name,
            },
            "pain": {
                "problems": list(self.problems),
                "problem_categories": list(self.problem_categories),
                "problem_recognition": self.problem_recognition.name,
                "pain_severity": self.pain_severity.name,
            },
            "buying_signals": {
                "budget_status": self.budget_status.name,
                "urgency": self.urgency.name,
                "timeline": self.timeline_text,
                "timeline_days": self.timeline_days,
                "competitive_evaluation": self.competitive_evaluation.name,
            },
            "authority": {"decision_authority": self.decision_authority.name},
            "negative_signals": [
                {"code": n.code, "reason": n.reason, "evidence": n.evidence, "severity": n.severity}
                for n in self.negative_signals
            ],
            "quality": {
                "confidence": round(self.confidence, 3),
                "missing_information": list(self.missing_information),
                "contradictions": [c.__dict__ for c in self.contradictions],
                "per_dimension": {k: v.to_dict() for k, v in self.resolutions.items()},
            },
        }
