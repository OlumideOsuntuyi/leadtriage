"""Lead Triage — Streamlit frontend."""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from lead_triage import ScoringEngine, load_leads
from lead_triage.ingest.loader import LeadRecord, Settings, default_dataset_path
from lead_triage.nlp import DIMENSIONS, IntentExtractor
from lead_triage.scoring.engine import ScoringConfig

from . import components as C
from . import theme as T

CONFIG_DIR = Path(__file__).resolve().parent.parent / "lead_triage" / "config"
PAGES = ["Dashboard", "Leads", "Lead detail", "Data quality", "Configuration"]


# ==========================================================================
# data
# ==========================================================================
@st.cache_resource(show_spinner=False)
def _extractor(lexicon_text: str | None) -> IntentExtractor:
    if lexicon_text:
        from lead_triage.nlp.lexicon import load_lexicon
        return IntentExtractor(load_lexicon(data=yaml.safe_load(lexicon_text)))
    return IntentExtractor()


@st.cache_data(show_spinner="Parsing and analysing leads…")
def build(csv_text: str, scoring_yaml: str, columns_yaml: str, lexicon_yaml: str | None):
    settings = Settings(yaml.safe_load(columns_yaml))
    records, report = load_leads(csv_text, settings, _extractor(lexicon_yaml))
    engine = ScoringEngine(ScoringConfig(yaml.safe_load(scoring_yaml)))
    engine.score_all(records)
    return records, report, _frame(records)


def _frame(records: list[LeadRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        p = r.intent
        rows.append({
            "row": r.row_index,
            "Lead ID": r.id_display,
            "Name": r.name,
            "Company": r.company,
            "Title": r.title,
            "Band": r.band,
            "Band label": r.band_label,
            "Score": r.display_score,
            "Raw score": r.score,
            "Route": r.route,
            "Intent type": p.intent_type.value if p else "UNKNOWN",
            "Buying stage": p.buying_stage.name if p else "UNKNOWN",
            "Budget status": p.budget_status.name if p else "UNKNOWN",
            "Urgency": p.urgency.name if p else "UNKNOWN",
            "Authority": p.decision_authority.name if p else "UNKNOWN",
            "Pain": p.pain_severity.name if p else "UNKNOWN",
            "Competitive": p.competitive_evaluation.name if p else "UNKNOWN",
            "Confidence": round(p.confidence, 2) if p else 0.0,
            "Timeline": p.timeline_text if p else "",
            "Timeline days": p.timeline_days if p else None,
            "Problems": ", ".join(p.problems) if p else "",
            "Categories": ", ".join(p.problem_categories) if p else "",
            "Negatives": ", ".join(n.code for n in p.negative_signals) if p else "",
            "Contradictions": len(p.contradictions) if p else 0,
            "Source": r.source,
            "Employees": r.employees,
            "Size band": r.employee_band,
            "ICP agency": {True: "Yes", False: "No", None: "Unknown"}[r.is_icp_agency],
            "Email": (r.email.value.address if r.email and r.email.ok and r.email.value else r.raw.get("email", "")),
            "Email status": r.email.code.value if r.email else "",
            "Website": (r.website.value.url if r.website and r.website.ok and r.website.value else ""),
            "Website status": r.website.code.value if r.website else "",
            "Created": (r.created.value.isoformat() if r.created and r.created.ok else ""),
            "Created status": r.created.code.value if r.created else "",
            "Budget (raw)": r.raw.get("monthly_budget", ""),
            "Budget (monthly)": r.monthly_budget,
            "Budget status code": r.budget.code.value if r.budget else "",
            "Data issues": len(r.data_issues),
            "Notes": r.notes,
        })
    return pd.DataFrame(rows)


def _read_config(name: str) -> str:
    return (CONFIG_DIR / name).read_text(encoding="utf-8")


# ==========================================================================
# charts
# ==========================================================================
def chart_bands(df: pd.DataFrame) -> alt.Chart:
    counts = (
        df.groupby(["Band", "Band label"], as_index=False)
        .size().rename(columns={"size": "Leads"})
    )
    counts["order"] = counts["Band"].map({b: i for i, b in enumerate(T.BAND_ORDER)})
    counts = counts.sort_values("order")
    headroom = float(counts["Leads"].max()) * 1.12
    base = alt.Chart(counts).encode(
        y=alt.Y("Band label:N", sort=list(counts["Band label"]), title=None,
                axis=alt.Axis(labelFontSize=12, labelColor=T.INK)),
        x=alt.X("Leads:Q", title=None, axis=alt.Axis(grid=True),
                scale=alt.Scale(domain=[0, headroom], nice=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("Band:N", scale=T.band_scale(), legend=None),
        tooltip=["Band label", "Leads"],
    )
    labels = base.mark_text(align="left", dx=8, fontSize=12, fontWeight=600, color=T.INK).encode(
        text="Leads:Q"
    )
    return (bars + labels).properties(height=150)


def chart_score_distribution(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, color=T.LIME_DEEP)
        .encode(
            x=alt.X("Score:Q", bin=alt.Bin(maxbins=28), title="Triage score (0–100)"),
            y=alt.Y("count():Q", title="Leads"),
            tooltip=[alt.Tooltip("count():Q", title="Leads"),
                     alt.Tooltip("Score:Q", bin=alt.Bin(maxbins=28), title="Score")],
        )
        .properties(height=190)
    )


def chart_stage_budget(df: pd.DataFrame) -> alt.Chart:
    stages = ["UNKNOWN", "AWARENESS", "EXPLORATION", "EVALUATION", "DECISION", "PURCHASE"]
    budgets = ["NO_BUDGET", "PRICE_SENSITIVE", "NOT_LOCKED", "BUDGETED", "ALLOCATED", "APPROVED", "UNKNOWN"]
    m = (df.groupby(["Buying stage", "Budget status"], as_index=False)
           .size().rename(columns={"size": "Leads"}))
    base = alt.Chart(m).encode(
        x=alt.X("Buying stage:N", sort=stages, title=None,
                axis=alt.Axis(labelAngle=-35, labelFontSize=10)),
        y=alt.Y("Budget status:N", sort=budgets, title=None, axis=alt.Axis(labelFontSize=10)),
    )
    cells = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Leads:Q", scale=alt.Scale(range=T.SEQUENTIAL),
                        legend=alt.Legend(title="Leads", gradientLength=110)),
        tooltip=["Buying stage", "Budget status", "Leads"],
    )
    text = base.mark_text(fontSize=10, fontWeight=600).encode(
        text="Leads:Q",
        color=alt.condition(alt.datum.Leads > m["Leads"].max() * 0.55,
                            alt.value("#FFFFFF"), alt.value(T.INK_2)),
    )
    return (cells + text).properties(height=300)


def chart_bar_count(df: pd.DataFrame, column: str, title: str, height: int = 230,
                    limit: int = 12) -> alt.Chart:
    counts = df[column].replace("", "—").value_counts().head(limit).reset_index()
    counts.columns = [column, "Leads"]
    headroom = float(counts["Leads"].max()) * 1.12
    base = alt.Chart(counts).encode(
        y=alt.Y(f"{column}:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=11, labelColor=T.INK_2)),
        x=alt.X("Leads:Q", title=None, scale=alt.Scale(domain=[0, headroom], nice=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=T.LIME_DEEP).encode(
        tooltip=[column, "Leads"]
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11, color=T.INK_2).encode(text="Leads:Q")
    return (bars + labels).properties(height=height, title=title)


def chart_source_quality(df: pd.DataFrame) -> alt.Chart:
    d = (df.groupby(["Source", "Band"], as_index=False).size()
           .rename(columns={"size": "Leads"}))
    d["Source"] = d["Source"].replace("", "—")
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=4, height=20, stroke=T.SURFACE, strokeWidth=2)
        .encode(
            y=alt.Y("Source:N", sort="-x", title=None),
            x=alt.X("Leads:Q", title=None, stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("Band:N", scale=T.band_scale(), sort=T.BAND_ORDER,
                            legend=alt.Legend(title=None)),
            order=alt.Order("Band:N"),
            tooltip=["Source", "Band", "Leads"],
        )
        .properties(height=210)
    )


def chart_problems(df: pd.DataFrame) -> alt.Chart:
    rows: list[str] = []
    for cell in df["Problems"]:
        rows.extend([p.strip() for p in str(cell).split(",") if p.strip()])
    if not rows:
        return alt.Chart(pd.DataFrame({"Problem": [], "Leads": []})).mark_bar()
    counts = pd.Series(rows).value_counts().reset_index()
    counts.columns = ["Problem", "Leads"]
    headroom = float(counts["Leads"].max()) * 1.12
    base = alt.Chart(counts).encode(
        y=alt.Y("Problem:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=11, labelColor=T.INK_2)),
        x=alt.X("Leads:Q", title=None, scale=alt.Scale(domain=[0, headroom], nice=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=15, color=T.GREEN).encode(
        tooltip=["Problem", "Leads"]
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11, color=T.INK_2).encode(text="Leads:Q")
    return (bars + labels).properties(height=max(160, 22 * len(counts)))


# ==========================================================================
# pages
# ==========================================================================
def page_dashboard(records: list[LeadRecord], report: dict, df: pd.DataFrame) -> None:
    total = len(df)
    hot = int((df["Band"] == "CONTACT_NOW").sum())
    consider = int((df["Band"] == "CONSIDER").sum())
    nurture = int((df["Band"] == "NURTURE").sum())
    disq = int((df["Band"] == "DISQUALIFY").sum())
    pipeline = df.loc[df["Band"].isin(["CONTACT_NOW", "CONSIDER"]), "Budget (monthly)"].sum()
    issues = int((df["Data issues"] > 0).sum())
    avg_conf = df["Confidence"].mean()

    cols = st.columns([1.25, 1, 1, 1, 1])
    with cols[0]:
        C.card(C.stat_tile(
            "Pipeline value", f"${pipeline:,.0f}",
            f"stated monthly budget across {hot + consider} actionable leads",
            accent=T.INK, meter=100 * (hot + consider) / max(total, 1),
        ))
    for col, (label, n, colour, sub) in zip(cols[1:], [
        ("Contact now", hot, T.GREEN, "budget, timeline and authority"),
        ("Consider", consider, T.LIME_DEEP, "real intent, one blocker"),
        ("Nurture", nurture, T.AMBER, "genuine but not ready"),
        ("Disqualify", disq, T.GREY, "routed away from sales"),
    ]):
        with col:
            C.card(C.stat_tile(label, f"{n}", f"{n / max(total,1):.0%} · {sub}",
                               accent=colour, meter=100 * n / max(total, 1)))

    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True):
            st.markdown(
                C.section("Triage outcome",
                          f"{total} leads loaded · {report['rows_dropped']} instrumentation rows dropped"),
                unsafe_allow_html=True)
            st.altair_chart(chart_bands(df), width="stretch")
            st.markdown(C.section("Score distribution", "Where the population sits on the 0–100 scale"),
                        unsafe_allow_html=True)
            st.altair_chart(chart_score_distribution(df), width="stretch")

    with right:
        with st.container(border=True):
            st.markdown(C.section("Stage × budget", "The matrix that decides contact-now vs nurture"),
                        unsafe_allow_html=True)
            st.altair_chart(chart_stage_budget(df), width="stretch")
        st.write("")
        with st.container(border=True):
            st.markdown(C.section("Enquiry type", "Non-purchase enquiries are routed, not deleted"),
                        unsafe_allow_html=True)
            st.altair_chart(chart_bar_count(df, "Intent type", "", height=250), width="stretch")

    st.write("")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown(C.section("Problems named in notes",
                                  "What is actually creating the buying intent"), unsafe_allow_html=True)
            st.altair_chart(chart_problems(df), width="stretch")
    with b:
        with st.container(border=True):
            st.markdown(C.section("Channel quality", "Band mix by acquisition source"),
                        unsafe_allow_html=True)
            st.altair_chart(chart_source_quality(df), width="stretch")
        st.write("")
        routes = df["Route"].value_counts()
        chips = "".join(
            C.tag(f"{r} · {n}", "good" if r == "Sales" else "") for r, n in routes.items()
        )
        C.card(C.section("Routing", "Where each enquiry should actually go") + chips)

    st.write("")
    with st.container(border=True):
        st.markdown(C.section("Top 10 by triage score", "Sorted on the same score the bands use"),
                    unsafe_allow_html=True)
        top = df.nlargest(10, "Raw score")[
            ["Lead ID", "Company", "Title", "Score", "Band label", "Buying stage",
             "Budget status", "Urgency", "Timeline", "Problems"]
        ]
        st.dataframe(top, width="stretch", hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn(
                         "Score", min_value=0, max_value=100, format="%.0f")})


def page_leads(records: list[LeadRecord], df: pd.DataFrame) -> None:
    filter_box = st.container(border=True)
    f = filter_box.columns([1.3, 1.2, 1.1, 1, 1, 1.4])
    bands = f[0].multiselect("Band", T.BAND_ORDER, default=T.BAND_ORDER,
                             format_func=lambda b: b.replace("_", " ").title())
    types = f[1].multiselect("Enquiry type", sorted(df["Intent type"].unique()))
    sources = f[2].multiselect("Source", sorted(x for x in df["Source"].unique() if x))
    sizes = f[3].multiselect("Size", ["solo", "micro", "small", "mid", "large", "unknown"])
    min_conf = f[4].slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    query = f[5].text_input("Search", placeholder="company, name, email, note…")
    st.write("")

    view = df[df["Band"].isin(bands)]
    if types:
        view = view[view["Intent type"].isin(types)]
    if sources:
        view = view[view["Source"].isin(sources)]
    if sizes:
        view = view[view["Size band"].isin(sizes)]
    view = view[view["Confidence"] >= min_conf]
    if query:
        q = query.lower()
        mask = (
            view["Company"].str.lower().str.contains(q, na=False)
            | view["Name"].str.lower().str.contains(q, na=False)
            | view["Email"].str.lower().str.contains(q, na=False)
            | view["Notes"].str.lower().str.contains(q, na=False)
            | view["Lead ID"].str.lower().str.contains(q, na=False)
        )
        view = view[mask]

    table_box = st.container(border=True)
    table_box.markdown(
        C.section(f"{len(view)} of {len(df)} leads",
                  "Click a row to open the full evidence trail on the Lead detail page"),
        unsafe_allow_html=True,
    )
    cols = ["Lead ID", "Company", "Name", "Title", "Score", "Band label", "Intent type",
            "Buying stage", "Budget status", "Urgency", "Authority", "Confidence",
            "Timeline", "Source", "Employees", "Negatives", "Data issues"]
    event = table_box.dataframe(
        view[cols].sort_values("Score", ascending=False),
        width="stretch", hide_index=True, height=520,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Confidence": st.column_config.ProgressColumn("Conf.", min_value=0, max_value=1, format="%.2f"),
        },
    )

    sel = event.selection.rows if hasattr(event, "selection") else []
    if sel:
        ordered = view.sort_values("Score", ascending=False)
        st.session_state["selected_row"] = int(ordered.iloc[sel[0]]["row"])
        st.session_state["page"] = "Lead detail"
        st.rerun()

    csv = view.drop(columns=["row"]).to_csv(index=False).encode()
    st.download_button("Download this selection as CSV", csv, "triaged_leads.csv", "text/csv")


def page_detail(records: list[LeadRecord], df: pd.DataFrame) -> None:
    options = sorted(records, key=lambda r: -r.score)
    labels = [f"{r.display_score:5.1f}  {r.id_display}  ·  {r.company or r.name or '—'}" for r in options]
    current = st.session_state.get("selected_row")
    index = next((i for i, r in enumerate(options) if r.row_index == current), 0)
    choice = st.selectbox("Lead", range(len(options)), index=index, format_func=lambda i: labels[i])
    rec = options[choice]
    st.session_state["selected_row"] = rec.row_index
    p = rec.intent

    head = st.columns([2.2, 1, 1, 1])
    with head[0]:
        C.card(
            f'<div class="lt-label">{C.band_chip(rec.band, rec.band_label)} '
            f'&nbsp;<span class="lt-tag">Route: {rec.route}</span></div>'
            f'<div class="lt-figure lt-figure-sm" style="margin-top:.5rem">'
            f"{rec.company or rec.name or rec.id_display}</div>"
            f'<div class="lt-sub">{rec.name or "—"}'
            f'{" · " + rec.title if rec.title else ""} · {rec.id_display}</div>'
        )
    with head[1]:
        C.card(C.stat_tile("Triage score", f"{rec.display_score:.0f}",
                           f"raw {rec.score:+.1f}", accent=T.BAND_COLOURS.get(rec.band),
                           meter=rec.display_score))
    with head[2]:
        C.card(C.stat_tile("NLP confidence", f"{p.confidence:.0%}" if p else "—",
                           "certainty of reading, not lead value",
                           accent=T.INK, meter=(p.confidence * 100) if p else 0))
    with head[3]:
        C.card(C.stat_tile("Stated budget",
                           f"${rec.monthly_budget:,.0f}" if rec.monthly_budget else "—",
                           (rec.budget.detail[:60] if rec.budget else ""), accent=T.INK))

    left, right = st.columns([1.25, 1])

    with left:
        spans: list[tuple[int, int, str]] = []
        if p:
            for res in p.resolutions.values():
                if res.span:
                    spans.append((res.span[0], res.span[1], "pos"))
            for n in p.negative_signals:
                spans.append((n.span[0], n.span[1], "neg"))
        C.card(
            C.section("The note, with extracted evidence",
                      "Green = signal used · orange = explicit negative signal")
            + C.highlight_note(rec.notes, spans)
        )

        if p:
            grid = st.container(border=True)
            grid.markdown(C.section("Extracted signals",
                                    "Every classification with its supporting quote"),
                          unsafe_allow_html=True)
            keys = list(DIMENSIONS.keys())
            for i in range(0, len(keys), 2):
                cc = grid.columns(2)
                for col, dim in zip(cc, keys[i:i + 2]):
                    res = p.resolutions.get(dim)
                    with col:
                        st.markdown(
                            C.dimension_cell(
                                dim.replace("_", " ").title(),
                                res.value.name if res else "UNKNOWN",
                                (res.evidence if res and not res.is_unknown else ""),
                                bool(res.is_unknown) if res else True,
                                res.confidence if res else 0.0,
                            ),
                            unsafe_allow_html=True,
                        )
                grid.write("")

            meta = (
                C.section("Derived & contextual", "Views of the same evidence — not scored separately")
                + C.tag(f"Enquiry type: {p.intent_type.value}", "good" if p.is_purchase_intent else "warn")
                + C.tag(f"Commitment: {p.commitment.name}")
                + C.tag(f"Solution intent: {p.solution_intent.name}")
                + (C.tag(f"Timeline: {p.timeline_text} (~{p.timeline_days}d)") if p.timeline_text else "")
                + "".join(C.tag(f"Problem: {x}") for x in p.problems)
                + "".join(C.tag(f"Category: {x}") for x in p.problem_categories)
            )
            C.card(meta)

    with right:
        caps_html = ""
        if rec.caps_applied:
            caps_html = (
                f'<div style="margin-top:.7rem;padding-top:.7rem;border-top:1px solid {T.LINE};">'
                + "".join(f'<div class="lt-sub" style="color:{T.AMBER};">&#9650; {c}</div>'
                          for c in rec.caps_applied)
                + "</div>"
            )
        C.card(
            C.section("Why this score", "Every point, with its reason")
            + C.explanation_table(rec.explanation) + caps_html
        )

        if p and p.negative_signals:
            body = C.section("Negative signals", "Extracted with a reason, not a silent deduction")
            for n in p.negative_signals:
                body += (
                    f'<div style="margin-bottom:.55rem;"><div style="font-size:.8rem;'
                    f'font-weight:600;color:{T.AMBER};">{n.code}</div>'
                    f'<div class="lt-sub">{n.reason}</div>'
                    f'<div class="lt-sub" style="font-style:italic;">“{n.evidence}”</div></div>'
                )
            C.card(body)

        if p and p.contradictions:
            body = C.section("Contradictions", "Kept apart rather than averaged together")
            for c in p.contradictions:
                body += (
                    f'<div style="margin-bottom:.55rem;"><div style="font-size:.8rem;font-weight:600;">'
                    f'{c.dimension.replace("_"," ").title()}: kept <b>{c.kept}</b> over {c.rejected}</div>'
                    f'<div class="lt-sub">{c.rule}</div>'
                    f'<div class="lt-sub" style="font-style:italic;">kept: “{c.evidence_kept}”</div>'
                    f'<div class="lt-sub" style="font-style:italic;">rejected: “{c.evidence_rejected}”</div></div>'
                )
            C.card(body)

        if p and p.missing_information:
            C.card(
                C.section("Not stated in the note", "Missing ≠ negative — these score zero, not minus")
                + "".join(C.tag(m) for m in p.missing_information)
            )

        issues = rec.data_issues
        C.card(
            C.section("Record quality", "Parser return codes for this row")
            + ("".join(C.tag(i, "warn") for i in issues) if issues else C.tag("No parse issues", "good"))
        )

        with st.expander("Raw NLP output (JSON)"):
            st.json(p.to_dict() if p else {})


def page_quality(records: list[LeadRecord], report: dict, df: pd.DataFrame) -> None:
    fields = [("Lead ID", "lead_id"), ("Created", "created"), ("Email", "email"),
              ("Website", "website"), ("Budget", "budget")]

    tiles = st.columns(5)
    for col, (label, attr) in zip(tiles, fields):
        results = [getattr(r, attr) for r in records if getattr(r, attr) is not None]
        # A blank optional field is missing data, not a parse failure — judge
        # the parser only on values that were actually supplied.
        supplied = [x for x in results if x.raw.strip()]
        blank = len(results) - len(supplied)
        ok = sum(1 for x in supplied if x.ok)
        pct = 100 * ok / max(len(supplied), 1)
        colour = T.GREEN if pct >= 90 else (T.LIME_DEEP if pct >= 75 else T.AMBER)
        with col:
            C.card(C.stat_tile(
                f"{label} parsed", f"{pct:.0f}%",
                f"{ok} of {len(supplied)} supplied · {blank} blank",
                accent=colour, meter=pct,
            ))

    codes_box = st.container(border=True)
    codes_box.markdown(C.section("Return codes by field",
                                 "Every value that failed, and exactly why it failed"),
                       unsafe_allow_html=True)
    rows = []
    for label, attr in fields:
        counts = Counter()
        examples: dict[str, str] = {}
        for r in records:
            res = getattr(r, attr)
            if res is None:
                continue
            for code in res.all_codes:
                counts[code.value] += 1
                examples.setdefault(code.value, res.raw or "(empty)")
        for code, n in counts.most_common():
            rows.append({"Field": label, "Return code": code, "Rows": n,
                         "OK": not code.startswith(("EMPTY", "NOT_", "JUNK", "AMOUNT_NO_DIGITS",
                                                    "DATE_UNKNOWN", "DATE_IMPOSSIBLE", "EMAIL_NO",
                                                    "EMAIL_MISSING", "EMAIL_INVALID", "EMAIL_MULTIPLE",
                                                    "ID_MISSING", "ID_MALFORMED", "ID_NO", "WEB_NO",
                                                    "WEB_INVALID", "WEB_IS", "PLACEHOLDER", "OUT_OF")),
                         "Example input": examples[code][:60]})
    qdf = pd.DataFrame(rows)
    codes_box.dataframe(qdf, width="stretch", hide_index=True, height=430)

    st.write("")
    a, b = st.columns(2)
    with a:
        load_box = st.container(border=True)
        load_box.markdown(C.section("Load report", "What the loader did to your file"),
                          unsafe_allow_html=True)
        load_box.markdown(
            "".join(
                C.tag(f"{k.replace('_', ' ')}: {v}")
                for k, v in report.items()
                if k in ("rows_in_file", "rows_loaded", "rows_dropped", "duplicate_ids")
            )
            + "<br>"
            + C.tag(f"Date order inferred: {json.dumps(report['date_order_policy'])}")
            + (C.tag(f"Unmapped fields: {report['unmapped_fields']}", "warn")
               if report["unmapped_fields"] else C.tag("All columns mapped", "good")),
            unsafe_allow_html=True,
        )
        load_box.caption(
            "Day-first vs month-first is learned from the whole column: a value like "
            "'19-06-2024' proves dash-separated dates are day-first, which then settles "
            "every ambiguous '04-06-2024' in the same file."
        )
    with b:
        issue_box = st.container(border=True)
        issue_box.markdown(C.section("Rows with issues", "Ranked by number of parse problems"),
                           unsafe_allow_html=True)
        bad = df[df["Data issues"] > 0].nlargest(25, "Data issues")[
            ["Lead ID", "Company", "Data issues", "Email status", "Website status",
             "Created status", "Budget status code"]
        ]
        issue_box.dataframe(bad, width="stretch", hide_index=True, height=330)


def page_config(state: dict) -> None:
    C.card(
        C.section("Live configuration",
                  "Everything the system knows about your market lives in these files. "
                  "Edit here, press Apply, and the whole pipeline re-runs."),
        tight=True,
    )

    tabs = st.tabs(["Scoring & bands", "Columns & parsing", "Intent lexicon"])
    keys = ["scoring_yaml", "columns_yaml", "lexicon_yaml"]
    names = ["scoring.yaml", "columns.yaml", "lexicon.yaml"]

    for tab, key, name in zip(tabs, keys, names):
        with tab:
            text = st.text_area(name, value=state[key], height=520, key=f"edit_{key}",
                                label_visibility="collapsed")
            c1, c2, c3 = st.columns([1, 1, 5])
            if c1.button("Apply", key=f"apply_{key}", type="primary"):
                try:
                    yaml.safe_load(text)
                except yaml.YAMLError as exc:
                    st.error(f"Invalid YAML — nothing applied.\n\n{exc}")
                else:
                    state[key] = text
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Applied. Re-scoring…")
                    st.rerun()
            if c2.button("Reset", key=f"reset_{key}"):
                state[key] = _read_config(name)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


# ==========================================================================
# shell
# ==========================================================================
def run() -> None:
    st.set_page_config(page_title="Lead Triage", page_icon="◆", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(T.CSS, unsafe_allow_html=True)

    for key, name in (("scoring_yaml", "scoring.yaml"), ("columns_yaml", "columns.yaml"),
                      ("lexicon_yaml", "lexicon.yaml")):
        st.session_state.setdefault(key, _read_config(name))
    st.session_state.setdefault("page", "Dashboard")
    st.session_state.setdefault("csv_text", None)
    st.session_state.setdefault("csv_name", None)

    with st.sidebar:
        st.markdown('<div class="lt-brand">Lead<span>Triage</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="lt-brand-sub">Intent · Fit · Routing</div>', unsafe_allow_html=True)

        # key="page" makes the widget itself the source of truth, so a
        # programmatic jump (row click -> Lead detail) and a manual click
        # cannot disagree about which page is active.
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

        st.divider()
        st.markdown('<div class="lt-label">Dataset</div>', unsafe_allow_html=True)
        upload = st.file_uploader("Load a CSV", type=["csv"], label_visibility="collapsed")
        if upload is not None:
            st.session_state["csv_text"] = upload.getvalue().decode("utf-8", errors="replace")
            st.session_state["csv_name"] = upload.name
        if st.session_state["csv_text"] is not None:
            st.caption(f"Loaded: **{st.session_state['csv_name']}**")
            if st.button("Revert to default dataset"):
                st.session_state["csv_text"] = None
                st.session_state["csv_name"] = None
                st.rerun()
        else:
            st.caption(f"Default: `{default_dataset_path().name}`")

    csv_text = st.session_state["csv_text"]
    if csv_text is None:
        path = default_dataset_path()
        if not path.exists():
            st.error(f"Default dataset not found at {path}. Upload a CSV to continue.")
            return
        csv_text = path.read_text(encoding="utf-8", errors="replace")

    try:
        records, report, df = build(
            csv_text, st.session_state["scoring_yaml"],
            st.session_state["columns_yaml"], st.session_state["lexicon_yaml"],
        )
    except Exception as exc:  # configuration errors must be legible, not a stack trace
        st.error(f"Could not build the pipeline: {exc}")
        st.stop()
        return

    if page == "Dashboard":
        page_dashboard(records, report, df)
    elif page == "Leads":
        page_leads(records, df)
    elif page == "Lead detail":
        page_detail(records, df)
    elif page == "Data quality":
        page_quality(records, report, df)
    else:
        page_config(st.session_state)
