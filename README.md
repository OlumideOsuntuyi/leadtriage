# Lead Triage

Turns a messy leads CSV into an explainable **Contact now / Consider / Nurture / Disqualify** decision, with the evidence for every call.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens with the bundled dataset already loaded. Upload any other CSV from the sidebar.

---

## What's lightweight about it

No models, no embeddings, no GPU, no network calls. The intent engine is regex + a weighted lexicon, so **517 leads parse, classify and score in ~0.6 s** on one CPU core.

| Layer | Dependencies |
|---|---|
| `lead_triage/` — parsers, NLP, scoring | **PyYAML only** (Python 3.10+ stdlib otherwise) |
| `app/` — the UI | Streamlit, Altair, pandas |

The core package is importable on its own with no UI stack:

```python
from lead_triage import load_leads, ScoringEngine

records, report = load_leads("data/default_leads.csv")
ScoringEngine().score_all(records)

for r in sorted(records, key=lambda r: -r.score)[:5]:
    print(r.band, r.display_score, r.company, r.intent.budget_status.name)
```

---

## The six parsers (tasks 1, 3–6)

All five field parsers share **one return contract** — `ParseResult(ok, value, code, raw, detail, warnings, meta)` — so checking a result is the same everywhere and return codes are a single enum.

```python
from lead_triage.core import parse_amount, parse_date, parse_email, parse_lead_id, parse_website

parse_amount("$6k-$8k").value.options   # {'min': 6000, 'max': 8000, 'avg': 7000}
parse_amount("TBD").code                # ParseCode.PLACEHOLDER  (unknown ≠ zero)
parse_email("obi@leverageside").code    # ParseCode.EMAIL_MISSING_TLD
parse_website("asdf").code              # ParseCode.JUNK
parse_lead_id("1341").value             # L-1341
```

**Amount** — `6k`, `$12k`, `$8,000`, `5000/month`, `$6k-$8k`, `5k - 8k`, `15k/mo`, `up to 10k`, `10k+`, `between 5 and 8k`, `₦2.5m`. Ranges return **every** reading (`min`/`max`/`avg`) plus one selected by the configured policy, so the caller never has to guess which it got. `6-8k` correctly reads as 6000–8000, not 6–8000. Periods normalise to monthly; `$5/hr` parses but is explicitly *not* a monthly budget.

**Date** — the hard part isn't the formats, it's that `04-06-2024` is genuinely ambiguous. `parse_date_column()` scans the whole column first: `19-06-2024` proves dash-separated values are day-first, `06/28/2024` proves slash-separated ones are month-first, and every ambiguous value in that file is then settled the same way. Anything still ambiguous is parsed *and* flagged `DATE_AMBIGUOUS_ORDER` — never silently guessed.

**Email** — distinguishes *why* it's wrong (`EMAIL_NO_AT`, `EMAIL_MISSING_DOMAIN`, `EMAIL_MISSING_TLD`, `EMAIL_INVALID_CHARS`, `EMAIL_MULTIPLE_AT`) instead of returning one boolean. Repairs `kunle[at]x.com`. Flags freemail and role mailboxes as *observations*, not failures.

**Lead ID** — decomposes into `(prefix, number, suffix)` so `L-1234` and `4321` sort together numerically, `L-1205-dup` keeps its duplicate marker, and invalid IDs group at the end. The pattern is a config field, not a hard-coded regex.

**Website** — normalises scheme and `www`, exposes `registrable_domain` (handling `co.uk`, `com.ng`) so you can check a lead's email domain against their own site.

**Sorting (task 5)** — `sort_lead_ids()`, `sort_records()`, or `make_sort_key()` to sort anything else by ID.

---

## The intent NLP (task 2)

The engine answers *"what does this note say"*. It never answers *"is this a good lead"* — that's the scoring layer's job.

```
note → normalise → clauses → negation scope → rules fire → Signals
     → per-dimension resolution → intent-type routing → problems, timeline,
       negatives, confidence, contradictions
```

**Three design decisions worth knowing about:**

1. **One ladder, not three.** The brief specified Solution Intent, Buying Stage and Commitment as separate dimensions. They're measured from the same evidence, so scoring all three would count one phrase three times. Only `buying_stage` is extracted; the other two are exposed as derived, non-scoring views.

2. **Intent type routes, it doesn't score.** *"Researching AI automation for my final-year project, need it by Friday"* fires high urgency and real commitment. Every signal is correct and the lead is worth nothing. So the enquiry type is classified first and routes the lead. Points can't fix this — a keen student will always out-add a quiet buyer.

3. **Negation flips, it doesn't cancel.** *"No real budget"* resolves to `NO_BUDGET`, not `UNKNOWN`. Missing ≠ negative, and negated ≠ missing.

**Signals are never summed.** Each rule match emits a `Signal` carrying its strength, specificity, clause, and character span into the original note. A resolver picks a winner per dimension — strongest, most specific, most operative reading wins outright. Saying "we have budget" three times does not outrank "budget approved".

**Contrast handling.** Clauses are split on `but / however / although / unfortunately / …`, and the clause *after* a contrast marker carries the operative meaning:

```python
extract_intent("This is a priority, but we don't have budget until next quarter.")
# pain_severity   HIGH        "This is a priority"
# budget_status   NOT_LOCKED  "no budget until"     ← deferred, not absent
# urgency         MODERATE    "next quarter"        ← survives the negation
# contradictions  1 recorded, not averaged away
```

Negation scope stops at `until / unless / before / but / …` — in *"no budget until next quarter"* the negation applies to the budget, not to the quarter.

**Confidence is separate from intent.** It's computed from mechanical facts (how many rules fired, phrase vs bare keyword, hedging nearby, note length, contradictions present), and a decisive classification sets a floor: spam scores 0 on every buying dimension but reads at 85% confidence, because we're certain what it is.

**Evidence is free.** Every signal already carries its character span, so the UI highlights the exact words behind each classification with no second pass.

---

## Scoring & triage

The NLP says *what the note says*; `scoring.yaml` says *how much that matters*. Score = intent + fit + data quality, then routed and banded.

* **`UNKNOWN` is worth 0**, never a penalty. Incomplete records aren't punished for being incomplete.
* **Non-purchase enquiries are routed, not deleted.** Each enquiry type declares a `max_band` (ceiling) *and* a `min_band` (floor) plus a destination team. A student lands in **Nurture → Community**, a VC in **Consider → Partnerships**, a job applicant in **Disqualify → Recruiting**. The floor is what stops a non-buyer from silently collapsing to Disqualify just because they score badly on a scale built for buyers.
* **No double-charging.** Negative signals already represented as a dimension value (`NO_BUDGET`) are displayed but cost 0 points, because `budget_status` already scored them.
* **Every point is logged with its reason**, so any band opens into a full breakdown.

---

## Configuration

There is no business vocabulary in the Python. Three YAML files, all editable live in the app's Configuration page:

| File | Controls |
|---|---|
| `config/columns.yaml` | column aliases, per-field parse policy, employee bands, title→authority map, source quality, ICP definition |
| `config/lexicon.yaml` | every pattern, weight, enquiry type, negative signal, timeline horizon and problem category |
| `config/scoring.yaml` | point weights, band thresholds, routing table, hard caps |

Point the column mapping at different headers and the same system reads a different CSV — see `test_column_mapping_is_configurable`.

---

## The app

Five pages: **Dashboard** (triage outcome, score distribution, stage × budget matrix, enquiry types, problems, channel quality), **Leads** (filter, search, sort, click through, export), **Lead detail** (the note with evidence highlighted, every extracted signal with its quote, the full score breakdown, contradictions, what wasn't stated), **Data quality** (parse rate and return codes per field, with example inputs), **Configuration** (live YAML editing).

Band colours were validated for colour-vision deficiency (worst adjacent ΔE 11.2 deutan, normal-vision ΔE 15.2); every band mark also carries a text label, so identity is never colour alone.

---

## Tests

```bash
pytest -q     # 116 tests, ~1.4s
```

`test_parsers.py` uses real values from the source CSV. `test_nlp.py` locks in the guide's contradiction, negation and routing behaviours as golden cases. `test_pipeline.py` runs the full 520-row file end to end and asserts, among other things, that every score is fully explained by its own breakdown.

---

## What the bundled dataset exercises

520 rows, deliberately messy: two lead-ID formats plus duplicates and a blank; five date formats with real day/month ambiguity; emails missing the `@`, the domain, or the TLD, plus `[at]` obfuscation; websites with and without scheme and `www`; budgets as ranges, placeholders (`TBD`, `depends`), zeros and hourly rates; employee counts like `35-55` and `19+`; three instrumentation rows (`header`, `asdf`, `TESTROW`) dropped on load; and notes covering buyers, students, job applicants, competitors, VCs, journalists, vendor pitches and outright spam.
