"""Shared text handling: normalisation, clause splitting, negation scope,
modality detection. Offsets into the ORIGINAL string are preserved throughout
so the UI can highlight the exact evidence span in the raw note.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

#: Character-for-character replacements only, so offsets never shift.
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "\t": " ", "\n": " ", "\r": " ",
    "…": ".",
}
_TRANS = str.maketrans(_CHAR_MAP)


def normalise(text: str) -> str:
    """Lowercase + fold punctuation without changing string length.

    Length preservation is the whole point: every offset computed on the
    normalised text points at the same character in the original.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    if len(folded) != len(text):  # NFKC changed width; fall back to raw
        folded = text
    return folded.translate(_TRANS).lower()


# --------------------------------------------------------------------------
# clauses
# --------------------------------------------------------------------------

#: Connectives that mark a contrast. In English the clause AFTER one of these
#: carries the operative meaning: "this is a priority, BUT we have no budget".
CONTRAST_MARKERS = [
    "but", "however", "although", "though", "that said", "even so",
    "unfortunately", "sadly", "except", "unless", "whereas", "still",
    "on the other hand", "having said that", "then again", "albeit",
]

#: Connectives that merely continue the thought.
CONTINUATION_MARKERS = ["and", "also", "plus", "as well as", "additionally", "moreover", "furthermore"]

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\s+[-–—]\s+")
_CONTRAST_RE = re.compile(
    r"(?:(?<=[\s,;])|^)(" + "|".join(re.escape(m) for m in CONTRAST_MARKERS) + r")(?=[\s,]|$)"
)
_SUBCLAUSE = re.compile(r",\s+(?=(?:so|because|since|while|which|as)\b)")


@dataclass
class Clause:
    """A span of the note with its position and role."""

    text: str            # normalised text of this clause
    start: int           # offset into the original note
    end: int
    index: int
    is_operative: bool = False   # sits after a contrast marker
    contrast_marker: str = ""
    negation_spans: list[tuple[int, int]] = field(default_factory=list)  # clause-local

    @property
    def raw_slice(self) -> slice:
        return slice(self.start, self.end)

    def is_negated_at(self, local_pos: int) -> bool:
        return any(a <= local_pos < b for a, b in self.negation_spans)


def split_clauses(text: str) -> list[Clause]:
    """Split a note into clauses, tracking contrast structure.

    Sentence boundaries and contrast connectives both start a new clause.
    A clause introduced by a contrast marker is flagged `is_operative`,
    which the resolver uses to break ties between conflicting signals.
    """
    norm = normalise(text)
    if not norm.strip():
        return []

    # First pass: sentences.
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_END.finditer(norm):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(norm):
        spans.append((pos, len(norm)))

    # Second pass: split each sentence on contrast and subordination markers.
    clauses: list[Clause] = []
    for s_start, s_end in spans:
        segment = norm[s_start:s_end]
        cuts: list[tuple[int, str]] = [(0, "")]
        for m in _CONTRAST_RE.finditer(segment):
            cuts.append((m.start(), m.group(1)))
        for m in _SUBCLAUSE.finditer(segment):
            cuts.append((m.end(), ""))
        cuts.sort()

        for i, (cut, marker) in enumerate(cuts):
            end = cuts[i + 1][0] if i + 1 < len(cuts) else len(segment)
            body = segment[cut:end]
            if not body.strip():
                continue
            lead = len(body) - len(body.lstrip(" ,"))
            abs_start = s_start + cut + lead
            abs_end = s_start + end
            ctext = norm[abs_start:abs_end].rstrip()
            if not ctext:
                continue
            clauses.append(
                Clause(
                    text=ctext,
                    start=abs_start,
                    end=abs_start + len(ctext),
                    index=len(clauses),
                    is_operative=bool(marker),
                    contrast_marker=marker,
                )
            )

    for c in clauses:
        c.negation_spans = find_negation_spans(c.text)
    return clauses


# --------------------------------------------------------------------------
# negation
# --------------------------------------------------------------------------

NEGATION_CUES = [
    r"\bnot\b", r"\bno\b", r"\bnone\b", r"\bnever\b", r"\bnothing\b",
    r"\bn't\b", r"\bdon'?t\b", r"\bdoesn'?t\b", r"\bdidn'?t\b", r"\bisn'?t\b",
    r"\baren'?t\b", r"\bwasn'?t\b", r"\bweren'?t\b", r"\bwon'?t\b", r"\bwont\b",
    r"\bcan'?t\b", r"\bcannot\b", r"\bcouldn'?t\b", r"\bshouldn'?t\b",
    r"\bwouldn'?t\b", r"\bhaven'?t\b", r"\bhasn'?t\b", r"\bhadn'?t\b",
    r"\bwithout\b", r"\black(?:ing|s)?\s+(?:of\s+)?\b", r"\bhardly\b",
    r"\bbarely\b", r"\bscarcely\b", r"\brather than\b", r"\binstead of\b",
    r"\bfar from\b", r"\bnowhere near\b",
]
_NEG_RE = re.compile("|".join(NEGATION_CUES))

#: A negation stops at these — "no budget, but we want it automated".
#: `until`/`before` matter especially: in "no budget until next quarter" the
#: negation applies to the budget, NOT to the quarter, so the timeline signal
#: after it must survive.
_SCOPE_STOP = re.compile(
    r"[,;:]|\b(?:but|however|although|though|so|because|since|yet|until|till|"
    r"unless|before|after|once|when|while|except)\b"
)

#: Phrases where a negation cue is part of a fixed expression and must NOT
#: negate what follows.
NEGATION_EXCEPTIONS = [
    r"\bno\s+(?:brainer|doubt|question|problem)\b",
    r"\bnot\s+only\b",
    r"\bnot\s+sure\s+who\b",     # authority signal, handled by its own rule
    r"\bno\s+longer\b",
]
_NEG_EXC_RE = re.compile("|".join(NEGATION_EXCEPTIONS))

_SCOPE_WINDOW = 60  # characters


def find_negation_spans(clause_text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans within a clause that fall under negation."""
    spans: list[tuple[int, int]] = []
    blocked = [(m.start(), m.end()) for m in _NEG_EXC_RE.finditer(clause_text)]
    for m in _NEG_RE.finditer(clause_text):
        if any(a <= m.start() < b for a, b in blocked):
            continue
        start = m.end()
        stop = _SCOPE_STOP.search(clause_text, start)
        end = min(stop.start() if stop else len(clause_text), start + _SCOPE_WINDOW)
        if end > start:
            spans.append((start, end))
    return _merge(spans)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for a, b in spans[1:]:
        if a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# --------------------------------------------------------------------------
# modality / hedging
# --------------------------------------------------------------------------

HEDGE_TERMS = [
    "might", "maybe", "may ", "perhaps", "possibly", "could", "would",
    "considering", "thinking about", "not sure", "unsure", "somewhat",
    "kind of", "sort of", "eventually", "someday", "at some point",
    "curious", "vague", "loosely", "in future", "down the line", "ish",
]
_HEDGE_RE = re.compile("|".join(re.escape(t) for t in HEDGE_TERMS))

BOOSTER_TERMS = [
    "definitely", "absolutely", "certainly", "urgently", "immediately",
    "asap", "right away", "must", "need to", "have to", "ready to",
    "committed", "approved", "signed off", "priority", "keen", "already",
]
_BOOST_RE = re.compile("|".join(re.escape(t) for t in BOOSTER_TERMS))

_HEDGE_WINDOW = 45


def modality_at(clause_text: str, pos: int) -> float:
    """Confidence multiplier for a match at `pos`: <1 hedged, >1 boosted."""
    lo, hi = max(0, pos - _HEDGE_WINDOW), min(len(clause_text), pos + _HEDGE_WINDOW)
    window = clause_text[lo:hi]
    factor = 1.0
    if _HEDGE_RE.search(window):
        factor *= 0.75
    if _BOOST_RE.search(window):
        factor *= 1.15
    return max(0.4, min(1.2, factor))


def snippet(raw: str, start: int, end: int, pad: int = 24) -> str:
    """A readable evidence quote from the ORIGINAL note."""
    lo, hi = max(0, start - pad), min(len(raw), end + pad)
    out = raw[lo:hi].strip().replace("\n", " ")
    return ("…" if lo > 0 else "") + out + ("…" if hi < len(raw) else "")
