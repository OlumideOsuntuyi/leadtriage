"""Visual system for the app: design tokens, injected CSS, and the Altair theme.

Palette provenance
------------------
Band colours were re-stepped from the brand greens and validated with the
dataviz palette validator (light, surface #FFFFFF, adjacent pairs):

    #0E6B45, #93C51D, #C25A0F, #7A8076
    PASS lightness band · PASS CVD separation (worst ΔE 11.2 deutan)
    PASS normal-vision floor (worst ΔE 15.2)
    FAIL chroma floor — #7A8076 is deliberately achromatic: "Disqualify"
         is an inactive state, and greying it out is the point.
    WARN contrast — #93C51D sits at 2.05:1, which obliges visible labels.
         Every band mark in this app carries its text label, and every chart
         has a table view, so identity is never colour-alone.

The sequential green ramp is single-hue (12° spread), monotone in lightness,
and used only for continuous magnitude (the stage × budget matrix), where the
lightest step is allowed to recede toward the surface.
"""

from __future__ import annotations

import altair as alt

# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------
CANVAS = "#F1F1ED"
SURFACE = "#FFFFFF"
SURFACE_SUNK = "#F7F7F3"
INK = "#14161A"
INK_2 = "#61655F"
INK_3 = "#9BA096"
LINE = "#E7E7DF"
LINE_SOFT = "#F0F0E9"

LIME = "#C8E85B"
LIME_DEEP = "#93C51D"
GREEN = "#0E6B45"
GREEN_SOFT = "#E7F3EA"
AMBER = "#C25A0F"
AMBER_SOFT = "#FBEFE2"
GREY = "#7A8076"

BAND_COLOURS = {
    "CONTACT_NOW": GREEN,
    "CONSIDER": LIME_DEEP,
    "NURTURE": AMBER,
    "DISQUALIFY": GREY,
}
BAND_ORDER = ["CONTACT_NOW", "CONSIDER", "NURTURE", "DISQUALIFY"]

SEQUENTIAL = ["#D2E8A8", "#B4D97A", "#93C51D", "#6FA218", "#4E7C14", "#2F5A10"]

RADIUS = 20
RADIUS_SM = 12

FONT = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


# --------------------------------------------------------------------------
# Altair
# --------------------------------------------------------------------------
@alt.theme.register("acru", enable=True)
def _acru_theme() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": SURFACE,
            "font": FONT,
            "view": {"stroke": "transparent", "continuousHeight": 220},
            "arc": {"stroke": SURFACE, "strokeWidth": 2},
            "axis": {
                "domain": False,
                "grid": True,
                "gridColor": LINE_SOFT,
                "gridWidth": 1,
                "labelColor": INK_3,
                "labelFont": FONT,
                "labelFontSize": 11,
                "labelPadding": 8,
                "tickColor": LINE,
                "tickSize": 0,
                "titleColor": INK_2,
                "titleFont": FONT,
                "titleFontSize": 11,
                "titleFontWeight": 500,
                "titlePadding": 12,
            },
            "axisX": {"grid": False},
            "legend": {
                "labelColor": INK_2,
                "labelFontSize": 11,
                "titleColor": INK_3,
                "titleFontSize": 11,
                "titleFontWeight": 500,
                "symbolType": "circle",
                "symbolSize": 90,
                "orient": "top",
                "direction": "horizontal",
                "offset": 6,
            },
            "title": {
                "color": INK,
                "fontSize": 13,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 12,
                "subtitleColor": INK_3,
                "subtitleFontSize": 11,
            },
            "bar": {"cornerRadiusEnd": 4, "color": LIME_DEEP},
            "rect": {"stroke": SURFACE, "strokeWidth": 2},
            "line": {"strokeWidth": 2, "color": GREEN},
            "point": {"size": 70, "filled": True, "color": GREEN},
            "text": {"font": FONT, "fontSize": 11, "color": INK_2},
            "range": {"category": [GREEN, LIME_DEEP, AMBER, GREY], "heatmap": SEQUENTIAL},
        }
    }


def band_scale() -> alt.Scale:
    return alt.Scale(domain=BAND_ORDER, range=[BAND_COLOURS[b] for b in BAND_ORDER])


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: {FONT}; }}

.stApp {{ background: {CANVAS}; }}
.block-container {{ padding: 1.6rem 2.2rem 3rem 2.2rem; max-width: 1500px; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.2rem; }}
.lt-brand {{
    font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em;
    color: {INK}; padding: 0 0.35rem 0.15rem 0.35rem;
}}
.lt-brand span {{ color: {GREEN}; }}
.lt-brand-sub {{
    font-size: 0.7rem; color: {INK_3}; padding: 0 0.35rem 1.1rem 0.35rem;
    letter-spacing: 0.03em; text-transform: uppercase;
}}

/* ---------- cards ---------- */
.lt-card {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS}px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 1px 2px rgba(20,22,26,.03), 0 8px 24px -18px rgba(20,22,26,.18);
    margin-bottom: 0.9rem;
}}
.lt-card-tight {{ padding: 0.9rem 1.1rem; }}

/* st.container(border=True) is the card wrapper for anything containing
   real Streamlit widgets — a hand-written <div> cannot wrap them. */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS}px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(20,22,26,.03), 0 8px 24px -18px rgba(20,22,26,.18);
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {{
    box-shadow: none; padding: 0.65rem 0.8rem; border-radius: {RADIUS_SM}px;
}}

.lt-label {{
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.07em;
    text-transform: uppercase; color: {INK_3}; margin-bottom: 0.35rem;
}}
.lt-figure {{
    font-size: 2.1rem; font-weight: 700; letter-spacing: -0.035em;
    color: {INK}; line-height: 1.05;
}}
.lt-figure-sm {{ font-size: 1.45rem; }}
.lt-sub {{ font-size: 0.75rem; color: {INK_3}; margin-top: 0.3rem; }}
.lt-delta-up {{ color: {GREEN}; font-weight: 600; }}
.lt-delta-down {{ color: {AMBER}; font-weight: 600; }}

.lt-h {{
    font-size: 0.95rem; font-weight: 600; color: {INK};
    letter-spacing: -0.01em; margin: 0 0 0.15rem 0;
}}
.lt-hsub {{ font-size: 0.75rem; color: {INK_3}; margin-bottom: 0.7rem; }}

/* ---------- band chips ---------- */
.lt-chip {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.2rem 0.62rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: -0.005em;
    border: 1px solid transparent; white-space: nowrap;
}}
.lt-chip .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.lt-chip-CONTACT_NOW {{ background: {GREEN_SOFT}; color: {GREEN}; border-color: #CFE6D8; }}
.lt-chip-CONSIDER    {{ background: #F2F8DE; color: #4E7C14; border-color: #E0EDBC; }}
.lt-chip-NURTURE     {{ background: {AMBER_SOFT}; color: {AMBER}; border-color: #F2DFC9; }}
.lt-chip-DISQUALIFY  {{ background: #F2F2EE; color: {GREY}; border-color: {LINE}; }}

.lt-tag {{
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 7px;
    background: {SURFACE_SUNK}; border: 1px solid {LINE};
    font-size: 0.68rem; color: {INK_2}; margin: 0 0.25rem 0.3rem 0;
    font-weight: 500;
}}
.lt-tag-warn {{ background: {AMBER_SOFT}; border-color: #F2DFC9; color: {AMBER}; }}
.lt-tag-good {{ background: {GREEN_SOFT}; border-color: #CFE6D8; color: {GREEN}; }}

/* ---------- evidence ---------- */
.lt-note {{
    background: {SURFACE_SUNK}; border: 1px solid {LINE};
    border-radius: {RADIUS_SM}px; padding: 0.85rem 1rem;
    font-size: 0.86rem; line-height: 1.65; color: {INK};
}}
.lt-note mark {{
    background: #E4F2C6; border-radius: 4px; padding: 0.05em 0.18em;
    box-shadow: inset 0 -2px 0 {LIME_DEEP}; color: {INK};
}}
.lt-note mark.neg {{ background: #FBE6D6; box-shadow: inset 0 -2px 0 {AMBER}; }}

/* ---------- score bar ---------- */
.lt-meter {{
    height: 7px; border-radius: 999px; background: {LINE_SOFT};
    overflow: hidden; margin-top: 0.55rem;
}}
.lt-meter > div {{ height: 100%; border-radius: 999px; }}

/* ---------- dimension grid ---------- */
.lt-dim {{
    border: 1px solid {LINE}; border-radius: {RADIUS_SM}px;
    padding: 0.65rem 0.8rem; background: {SURFACE}; height: 100%;
}}
.lt-dim .k {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; color: {INK_3}; }}
.lt-dim .v {{ font-size: 0.98rem; font-weight: 650; color: {INK}; margin: 0.15rem 0 0.2rem 0; }}
.lt-dim .e {{ font-size: 0.7rem; color: {INK_3}; font-style: italic; line-height: 1.4; }}
.lt-dim.unknown .v {{ color: {INK_3}; font-weight: 500; }}

/* ---------- streamlit widget polish ---------- */
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: {RADIUS_SM}px; }}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input {{
    border-radius: 10px !important; border-color: {LINE} !important;
    background: {SURFACE} !important;
}}
.stButton > button, .stDownloadButton > button {{
    border-radius: 999px; border: 1px solid {LINE}; background: {SURFACE};
    color: {INK}; font-weight: 550; font-size: 0.8rem; padding: 0.35rem 1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {GREEN}; color: {GREEN};
}}
button[kind="primary"] {{ background: {INK} !important; color: #fff !important; border: none !important; }}
div[data-testid="stExpander"] {{
    border: 1px solid {LINE}; border-radius: {RADIUS_SM}px; background: {SURFACE};
}}
div[data-testid="stExpander"] summary {{ font-size: 0.82rem; font-weight: 550; }}
div[data-testid="stVegaLiteChart"] {{ overflow: visible; }}
div[data-testid="stVegaLiteChart"] > div,
div[data-testid="stVegaLiteChart"] canvas,
div[data-testid="stVegaLiteChart"] svg {{ background: transparent !important; }}
.stElementContainer:has(> div[data-testid="stVegaLiteChart"]) {{ background: transparent; }}
hr {{ border-color: {LINE}; }}
.stRadio [role="radiogroup"] {{ gap: 0.15rem; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {GREEN}; }}
</style>
"""
