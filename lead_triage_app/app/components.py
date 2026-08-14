"""Reusable presentation pieces: stat tiles, band chips, evidence highlighting."""

from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st

from . import theme as T


def card(body: str, tight: bool = False) -> None:
    cls = "lt-card lt-card-tight" if tight else "lt-card"
    st.markdown(f'<div class="{cls}">{body}</div>', unsafe_allow_html=True)


def stat_tile(label: str, figure: str, sub: str = "", accent: str | None = None,
              meter: float | None = None) -> str:
    """A big-number tile. Returns HTML so it can be composed inside a card."""
    colour = accent or T.INK
    bar = ""
    if meter is not None:
        pct = max(0.0, min(100.0, meter))
        bar = (
            f'<div class="lt-meter"><div style="width:{pct:.1f}%;'
            f'background:{colour};"></div></div>'
        )
    return (
        f'<div class="lt-label">{html.escape(label)}</div>'
        f'<div class="lt-figure" style="color:{colour}">{figure}</div>'
        f'<div class="lt-sub">{sub}</div>{bar}'
    )


def band_chip(band: str, label: str | None = None) -> str:
    colour = T.BAND_COLOURS.get(band, T.GREY)
    text = label or band.replace("_", " ").title()
    return (
        f'<span class="lt-chip lt-chip-{band}">'
        f'<span class="dot" style="background:{colour}"></span>{html.escape(text)}</span>'
    )


def tag(text: str, kind: str = "") -> str:
    cls = {"warn": " lt-tag-warn", "good": " lt-tag-good"}.get(kind, "")
    return f'<span class="lt-tag{cls}">{html.escape(text)}</span>'


def section(title: str, subtitle: str = "") -> str:
    sub = f'<div class="lt-hsub">{html.escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="lt-h">{html.escape(title)}</div>{sub}'


def dimension_cell(name: str, value: str, evidence: str, unknown: bool,
                   confidence: float = 0.0) -> str:
    cls = "lt-dim unknown" if unknown else "lt-dim"
    ev = html.escape(evidence) if evidence else "not stated in the note"
    conf = f" · {confidence:.0%} conf" if confidence and not unknown else ""
    return (
        f'<div class="{cls}"><div class="k">{html.escape(name)}{conf}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="e">{ev}</div></div>'
    )


def highlight_note(raw: str, spans: Iterable[tuple[int, int, str]]) -> str:
    """Render a note with evidence spans marked.

    `spans` are (start, end, kind) offsets into the ORIGINAL string — which is
    exactly what the extractor produces, because normalisation preserves length.
    """
    if not raw:
        return '<div class="lt-note"><em>No notes on this lead.</em></div>'

    merged: list[tuple[int, int, str]] = []
    for start, end, kind in sorted(spans, key=lambda s: (s[0], -s[1])):
        if start < 0 or end > len(raw) or end <= start:
            continue
        if merged and start < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), prev[2] or kind)
        else:
            merged.append((start, end, kind))

    out, cursor = [], 0
    for start, end, kind in merged:
        out.append(html.escape(raw[cursor:start]))
        cls = ' class="neg"' if kind == "neg" else ""
        out.append(f"<mark{cls}>{html.escape(raw[start:end])}</mark>")
        cursor = end
    out.append(html.escape(raw[cursor:]))
    return f'<div class="lt-note">{"".join(out)}</div>'


def explanation_table(lines: list[dict[str, Any]]) -> str:
    """Score breakdown as a compact HTML table, grouped and signed."""
    if not lines:
        return '<div class="lt-sub">No scoring signals fired.</div>'
    rows = []
    for group in ("Intent", "Fit", "Data quality", "Negative signal", "Routing"):
        items = [l for l in lines if l.get("group") == group]
        if not items:
            continue
        rows.append(
            f'<tr><td colspan="3" style="padding-top:.55rem;font-size:.66rem;'
            f'letter-spacing:.07em;text-transform:uppercase;color:{T.INK_3};">'
            f"{html.escape(group)}</td></tr>"
        )
        for l in items:
            pts = float(l.get("points", 0))
            colour = T.GREEN if pts > 0 else (T.AMBER if pts < 0 else T.INK_3)
            ev = html.escape(str(l.get("evidence", "") or ""))[:110]
            rows.append(
                f'<tr><td style="padding:.2rem 0;font-size:.79rem;color:{T.INK};">'
                f'{html.escape(str(l.get("label","")))}</td>'
                f'<td style="font-size:.7rem;color:{T.INK_3};font-style:italic;'
                f'padding-left:.8rem;max-width:280px;">{ev}</td>'
                f'<td style="text-align:right;font-weight:650;font-size:.79rem;'
                f'color:{colour};white-space:nowrap;padding-left:.8rem;">'
                f'{pts:+.1f}</td></tr>'
            )
    return f'<table style="width:100%;border-collapse:collapse;">{"".join(rows)}</table>'
