# -*- coding: utf-8 -*-
"""
Gemstones & Pearls — Trade Dashboards (Streamlit)

Modules:
- Pearls (HS 7101)
- Gemstones (HS 7103)

Data source format: ITC Trade Map style (USD thousand in source tables).
This dashboard follows the same layout language as your Diamonds/Gold dashboards:
premium theme, hero header, auto-fit KPI cards, Top‑N + Others snapshot, country trends, and downloads.
"""

from __future__ import annotations

import re


def _sync_shared(shared_key: str, widget_key: str, last_key_name: str) -> None:
    # Sync a widget value into a shared session key and remember the last widget that changed.
    try:
        st.session_state[shared_key] = int(st.session_state.get(widget_key))
    except Exception:
        st.session_state[shared_key] = st.session_state.get(widget_key)
    st.session_state[last_key_name] = widget_key

import html
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st



# =========================
# Theme (Gold-dashboard style)
# =========================
THEMES: dict[str, dict[str, str]] = {
    "Pearls": {
        "bg1": "#eef7ff",
        "bg2": "#f7fbf8",
        "accent": "#2a9d8f",
        "accent2": "#264653",
        "accent_rgb": "42,157,143",
        "accent2_rgb": "38,70,83",
        "emoji": "🦪",
    },
    "Gemstones": {
        "bg1": "#f7f2ff",
        "bg2": "#f1fff8",
        "accent": "#7c3aed",
        "accent2": "#2d6a4f",
        "accent_rgb": "124,58,237",
        "accent2_rgb": "45,106,79",
        "emoji": "💎",
    },
}

COMMON_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap");

:root{
  --bg1: {{BG1}};
  --bg2: {{BG2}};
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --accent: {{ACCENT}};
  --accent2: {{ACCENT2}};
  --accent_rgb: {{ACCENT_RGB}};     /* e.g., 184,134,11 */
  --accent2_rgb: {{ACCENT2_RGB}};   /* e.g., 139,90,43 */
  --border:rgba(15,23,42,.08);
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
  --shadow2: 0 10px 28px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"]{
  font-family: "Manrope", sans-serif;
  color: var(--text);
}

.stApp{
  background:
    radial-gradient(1200px 600px at 15% 15%, var(--bg1) 0%, rgba(255,255,255,0) 60%),
    radial-gradient(900px 500px at 95% 10%, var(--bg2) 0%, rgba(255,255,255,0) 55%),
    linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.block-container{
  max-width: 1400px;
  padding-top: 1.8rem;
  padding-bottom: 2.2rem;
}

h1,h2,h3,h4{
  font-family: "Playfair Display", serif;
  letter-spacing: .2px;
}

/* Sidebar (dark, like Gold dashboard) */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b1220 0%, #0b1220 55%, #070b14 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] *{
  color: rgba(255,255,255,.92) !important;
}
.sidebar-title{
  font-weight: 850;
  letter-spacing: .3px;
  font-size: 22px;
  margin: 2px 0 6px 0;
}
.sidebar-sub{
  color: rgba(255,255,255,.72) !important;
  font-size: 12.5px;
  margin-bottom: 10px;
}
.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(var(--accent_rgb), .35);
  background: rgba(var(--accent_rgb), .14);
  color: rgba(255,255,255,.92) !important;
  font-size: 12px;
  font-weight: 750;
}

/* Header card (hero) */
.header-wrap{
  background: rgba(255,255,255,.78);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 18px 20px;
  margin: 4px 0 14px 0;
  backdrop-filter: blur(10px);
}
.header-wrap.hero{ position: relative; overflow: hidden; }
.header-wrap.hero::before{
  content:"";
  position:absolute;
  top:-80px;
  right:-90px;
  width:260px;
  height:260px;
  background: radial-gradient(circle at 30% 30%, rgba(var(--accent_rgb), .35) 0%, rgba(var(--accent_rgb),0) 70%);
  transform: rotate(10deg);
  pointer-events:none;
}
.header-wrap.hero::after{
  content:"";
  position:absolute;
  left:0; right:0; bottom:0;
  height:4px;
  background: linear-gradient(90deg, rgba(var(--accent_rgb),0), rgba(var(--accent_rgb),.55), rgba(var(--accent2_rgb),.45), rgba(var(--accent_rgb),0));
  opacity:.95;
  pointer-events:none;
}
.hero-row{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; position:relative; z-index:2; }
.hero-left{ display:flex; gap:14px; align-items:flex-start; }
.hero-right{ display:flex; flex-direction:column; align-items:flex-end; gap:8px; }
.hero-icon{
  width:54px;
  height:54px;
  border-radius:18px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: linear-gradient(135deg, rgba(var(--accent_rgb),.18), rgba(var(--accent2_rgb),.10));
  border: 1px solid rgba(var(--accent_rgb), .26);
  box-shadow: 0 10px 26px rgba(15,23,42,0.08);
  font-size: 26px;
}
.hero-art{
  position:absolute;
  right: 16px;
  bottom: -8px;
  width: 190px;
  opacity: .16;
  pointer-events:none;
  z-index:1;
}
.h1{
  font-size: 38px;
  line-height: 1.08;
  font-weight: 900;
  color: var(--text);
  margin: 0;
}
.hsub{
  margin-top: 6px;
  font-size: 14px;
  color: var(--muted);
  font-weight: 650;
}

/* KPI cards (auto-fit values via inline style) */
.kpi{
  background: rgba(255,255,255,.86);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow2);
  padding: 14px 14px;
  display:flex;
  gap: 12px;
  align-items:center;
  min-height: 84px;
}
.kpi .ico{
  width:44px;
  height:44px;
  border-radius: 14px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: rgba(var(--accent_rgb), .14);
  border: 1px solid rgba(var(--accent_rgb), .25);
  font-size: 22px;
}
.kpi .lbl{
  font-size: 12.5px;
  color: var(--muted);
  font-weight: 750;
  margin-bottom: 2px;
}
.kpi .val{
  font-size: 22px;
  font-weight: 900;
  color: var(--text);
  line-height: 1.05;
}
.kpi .sub{
  font-size: 12.5px;
  color: var(--muted);
  font-weight: 650;
  margin-top: 3px;
}
.kpi-grid{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px;
  margin: 10px 0 16px;
}

/* Tabs (pill style) */
.stTabs [data-baseweb="tab-list"]{
  gap: .35rem;
  padding: .22rem;
  background: rgba(255,255,255,.62);
  border-radius: 999px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  border-radius: 999px;
  padding: .45rem 1rem;
  color: var(--muted);
  font-weight: 700;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  background: rgba(var(--accent_rgb), .14);
  color: rgba(15,23,42,.95);
  border: 1px solid rgba(var(--accent_rgb), .18);
}

/* Plot & table containers */
div.stPlotlyChart, div[data-testid="stDataFrame"]{
  background: rgba(255,255,255,.86);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 6px;
  box-shadow: var(--shadow2);
}
div[data-baseweb="select"] > div{
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.92);
}

input[type="radio"], input[type="checkbox"]{ accent-color: var(--accent) !important; }

/* Buttons */
.stButton > button{
  border-radius: 12px;
  border: 1px solid rgba(var(--accent_rgb), .55);
  background: linear-gradient(140deg, rgba(var(--accent_rgb),1) 0%, rgba(var(--accent2_rgb),1) 100%);
  color: #ffffff;
  font-weight: 750;
  padding: 0.45rem 1rem;
  box-shadow: 0 12px 22px rgba(var(--accent_rgb), 0.20);
}
.stButton > button:hover{ filter: brightness(0.98); }

hr.soft{
  border:none;
  border-top:1px solid rgba(15,23,42,.08);
  margin: 10px 0 14px 0;
}
</style>
"""


def apply_theme(theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["Pearls"])
    css = (
        COMMON_CSS.replace("{{BG1}}", theme["bg1"])
        .replace("{{BG2}}", theme["bg2"])
        .replace("{{ACCENT}}", theme["accent"])
        .replace("{{ACCENT2}}", theme["accent2"])
        .replace("{{ACCENT_RGB}}", theme["accent_rgb"])
        .replace("{{ACCENT2_RGB}}", theme["accent2_rgb"])
    )
    st.markdown(css, unsafe_allow_html=True)


def style_plotly(fig, *, theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["Pearls"])
    try:
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Manrope, sans-serif", size=13, color="#142032"),
            title_font=dict(family="Playfair Display, serif", size=20, color="#142032"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(
                bgcolor="rgba(255,255,255,0.65)",
                bordercolor="rgba(20, 32, 50, 0.10)",
                borderwidth=1,
            ),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Manrope, sans-serif"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False)
    except Exception:
        pass



def kpi_card(icon: str, label: str, value: str, sub: str = "") -> None:
    """Gold-style KPI card with conservative auto-shrink for long values."""
    v = "" if value is None else str(value)
    digits = len(re.sub(r"[^0-9]", "", v))
    if digits >= 13 or len(v) >= 16:
        val_style = "font-size:18px; line-height:1.05; word-break:break-word;"
    elif digits >= 10 or len(v) >= 13:
        val_style = "font-size:20px; line-height:1.05; word-break:break-word;"
    else:
        val_style = "font-size:22px; line-height:1.05; word-break:break-word;"

    html_card = dedent(
        f"""
        <div class="kpi">
          <div class="ico">{html.escape(str(icon))}</div>
          <div>
            <div class="lbl">{html.escape(str(label))}</div>
            <div class="val" style="{val_style}">{html.escape(v)}</div>
            <div class="sub">{html.escape(str(sub))}</div>
          </div>
        </div>
        """
    ).strip()

    st.markdown(html_card, unsafe_allow_html=True)

def metric_card(label: str, value: str, *, tooltip: str | None = None) -> None:
    label_txt = html.escape(str(label))
    raw_val = "" if value is None else str(value)
    value_txt = html.escape(raw_val)
    tip_txt = html.escape(str(tooltip if tooltip is not None else raw_val))

    n = len(raw_val.strip())
    fs = "1.15rem"
    if n >= 18:
        fs = "1.05rem"
    if n >= 28:
        fs = "0.98rem"
    if n >= 40:
        fs = "0.92rem"

    st.markdown(
        f"""
        <div class="metric-card" title="{tip_txt}">
          <div class="label">{label_txt}</div>
          <div class="value" style="font-size:{fs};">{value_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header(title: str, subtitle: str, *, badge: str | None = None, icon: str = "💠") -> None:
    title_txt = html.escape(str(title))
    subtitle_txt = html.escape(str(subtitle))
    badge_html = f'<span class="badge">{html.escape(str(badge))}</span>' if badge else ""

    # Minimal accent art (uses CSS variables for color)
    art_svg = """
    <svg viewBox="0 0 220 140" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="g1" cx="35%" cy="30%" r="70%">
          <stop offset="0%" stop-color="rgba(var(--accent_rgb),0.60)"/>
          <stop offset="100%" stop-color="rgba(var(--accent_rgb),0)"/>
        </radialGradient>
        <radialGradient id="g2" cx="65%" cy="55%" r="75%">
          <stop offset="0%" stop-color="rgba(var(--accent2_rgb),0.45)"/>
          <stop offset="100%" stop-color="rgba(var(--accent2_rgb),0)"/>
        </radialGradient>
      </defs>
      <circle cx="70" cy="55" r="60" fill="url(#g1)"/>
      <circle cx="150" cy="80" r="70" fill="url(#g2)"/>
      <path d="M140 22 L170 52 L140 82 L110 52 Z" fill="rgba(var(--accent_rgb),0.22)"/>
      <path d="M175 45 L195 65 L175 85 L155 65 Z" fill="rgba(var(--accent2_rgb),0.18)"/>
    </svg>
    """

    st.markdown(
        f"""
        <div class="header-wrap hero">
          <div class="hero-row">
            <div class="hero-left">
              <div class="hero-icon">{html.escape(icon)}</div>
              <div>
                <div class="h1">{title_txt}</div>
                <div class="hsub">{subtitle_txt}</div>
              </div>
            </div>
            <div class="hero-right">{badge_html}</div>
          </div>
          <div class="hero-art">{art_svg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pick_first_existing(candidates: list[str]) -> str | None:
    base_dirs: list[Path] = []
    try:
        base_dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    base_dirs.append(Path.cwd())

    for p in candidates:
        cand0 = Path(str(p)).expanduser()
        if cand0.is_absolute() and cand0.exists():
            return str(cand0)
        for base in base_dirs:
            cand = (base / str(p)).expanduser()
            if cand.exists():
                return str(cand)
        if cand0.exists():
            return str(cand0)
    return None


# =========================
# Parsing helpers (TradeMap style)
# =========================
def _extract_year_cols(header_row: list) -> tuple[list[int], list[int]]:
    years: list[int] = []
    col_idxs: list[int] = []
    for j, h in enumerate(header_row):
        if j == 0 or pd.isna(h):
            continue
        if isinstance(h, (int, np.integer)) and 1900 <= int(h) <= 2100:
            years.append(int(h))
            col_idxs.append(j)
            continue
        if isinstance(h, (float, np.floating)) and np.isfinite(h) and float(h).is_integer():
            y = int(h)
            if 1900 <= y <= 2100:
                years.append(y)
                col_idxs.append(j)
                continue
        s = str(h)
        m = re.search(r"(19\d{2}|20\d{2})", s)
        if m:
            years.append(int(m.group(1)))
            col_idxs.append(j)
    return years, col_idxs


def _parse_simple_flow_sheet(path: str, sheet_name: str, flow: str) -> pd.DataFrame:
    """
    Parse a single-block TradeMap sheet (e.g., 'Imports(7101)' or 'Exports(7101)').
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    col0 = df.iloc[:, 0].astype(str).str.strip().str.lower()
    hdr_idx = col0[col0.isin(["importers", "exporters"])].index.tolist()
    hdr = int(hdr_idx[0]) if hdr_idx else 0

    header = df.iloc[hdr].tolist()
    years, col_idxs = _extract_year_cols(header)
    if not years:
        raise ValueError(f"Could not detect year columns in sheet: {sheet_name}")

    block = df.iloc[hdr + 1 :, [0] + col_idxs].copy()
    block.columns = ["country"] + [str(y) for y in years]
    block = block.dropna(subset=["country"])
    block["country"] = block["country"].astype(str).str.strip()
    block = block[~block["country"].str.lower().isin(["nan", "none", ""])]

    long_df = block.melt(id_vars=["country"], var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["flow"] = flow
    return long_df.dropna(subset=["year", "value"])


def _parse_trade_block(df: pd.DataFrame, header_row_idx: int, end_row_idx_exclusive: int, flow: str) -> pd.DataFrame:
    header = df.iloc[header_row_idx].tolist()
    years, col_idxs = _extract_year_cols(header)

    # Fallback: sometimes the real header row is the next row
    if not years and header_row_idx + 1 < len(df):
        header2 = df.iloc[header_row_idx + 1].tolist()
        years2, col2 = _extract_year_cols(header2)
        if years2:
            header_row_idx = header_row_idx + 1
            years, col_idxs = years2, col2

    if not years:
        raise ValueError(f"Could not detect year columns for {flow}.")

    block = df.iloc[header_row_idx + 1 : end_row_idx_exclusive, [0] + col_idxs].copy()
    block.columns = ["country"] + [str(y) for y in years]
    block = block.dropna(subset=["country"])
    block["country"] = block["country"].astype(str).str.strip()
    block = block[~block["country"].str.lower().isin(["nan", "none", ""])]

    long_df = block.melt(id_vars=["country"], var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["flow"] = flow
    return long_df.dropna(subset=["year", "value"])


def _find_sheet_name(xls: pd.ExcelFile, token: str) -> str | None:
    tok = str(token).strip()
    for sn in xls.sheet_names:
        if str(sn).strip() == tok:
            return sn
    return None


def _hs6_desc_from_sheet(path: str, sheet_name: str) -> str:
    """
    Extracts a clean HS6 description from the sheet's 'Product:' line (top area).
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=6, usecols=[0, 1])
    except Exception:
        return str(sheet_name)

    candidates: list[str] = []
    for c in [1, 0]:
        if c in df.columns:
            candidates += df[c].dropna().astype(str).tolist()

    for v in candidates:
        if "Product:" in v:
            s = v.split("Product:", 1)[1].strip()
            s = re.sub(r"^\s*\d+\s*", "", s)
            return s.strip().strip(".")
    return str(sheet_name)


def _parse_hs6_sheet(path: str, sheet_name: str, hs6: str) -> pd.DataFrame:
    """
    Parse a HS6 sheet that contains BOTH imports and exports blocks ('Importers' and 'Exporters').
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    low = col0.str.lower()

    imp_idx = low[low == "importers"].index.tolist()
    exp_idx = low[low == "exporters"].index.tolist()
    if not imp_idx or not exp_idx:
        raise ValueError(f"Sheet {sheet_name}: could not locate Importers/Exporters blocks.")

    imp_row = int(imp_idx[0])
    exp_row = int(exp_idx[0])

    imp_long = _parse_trade_block(df, imp_row, exp_row - 1, flow="Imports")
    exp_end = df.iloc[exp_row + 1 :, 0].last_valid_index()
    exp_long = _parse_trade_block(df, exp_row, int(exp_end) + 1, flow="Exports")

    out = pd.concat([imp_long, exp_long], ignore_index=True)
    out["hs6"] = hs6
    out["hs_desc"] = _hs6_desc_from_sheet(path, sheet_name)
    return out


@st.cache_data(show_spinner=False)
def load_total_trade(path: str, imports_sheet: str, exports_sheet: str) -> pd.DataFrame:
    imp = _parse_simple_flow_sheet(path, imports_sheet, "Imports")
    exp = _parse_simple_flow_sheet(path, exports_sheet, "Exports")
    out = pd.concat([imp, exp], ignore_index=True)
    out["hs6"] = "TOTAL"
    out["hs_desc"] = "Total"
    return out


@st.cache_data(show_spinner=False)
def load_all_hs6_trade(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    hs6_sheets = [s for s in xls.sheet_names if str(s).strip().isdigit() and len(str(s).strip()) == 6]
    frames: list[pd.DataFrame] = []
    for hs6 in hs6_sheets:
        sn = _find_sheet_name(xls, hs6) or str(hs6)
        try:
            frames.append(_parse_hs6_sheet(path, sn, str(hs6).strip()))
        except Exception:
            # Skip any malformed sheets without breaking the app
            continue
    if not frames:
        return pd.DataFrame(columns=["country", "year", "value", "flow", "hs6", "hs_desc"])
    return pd.concat(frames, ignore_index=True)



@st.cache_data(show_spinner=False)
def load_total_trade_from_hs6(path: str) -> pd.DataFrame:
    """Build TOTAL trade (partner countries) by summing all HS6 sheets in the workbook."""
    hs6_df = load_all_hs6_trade(path)
    if hs6_df.empty:
        return pd.DataFrame(columns=["country", "year", "value", "flow", "hs6", "hs_desc"])
    tot = hs6_df.groupby(["country", "year", "flow"], as_index=False)["value"].sum()
    tot["hs6"] = "TOTAL"
    tot["hs_desc"] = "Total"
    return tot


def _world_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a year-wise 'World' series for Exports/Imports and Balance.
    If 'World' row is missing, uses sum of countries (excluding World).
    """
    d = df.copy()
    d["country_l"] = d["country"].astype(str).str.strip().str.lower()

    w = d[d["country_l"] == "world"].copy()
    if w.empty:
        w = d[d["country_l"] != "world"].groupby(["year", "flow"], as_index=False)["value"].sum()
    else:
        w = w.groupby(["year", "flow"], as_index=False)["value"].sum()

    p = w.pivot(index="year", columns="flow", values="value").fillna(0.0).reset_index().sort_values("year")
    for col in ["Exports", "Imports"]:
        if col not in p.columns:
            p[col] = 0.0
    p["Trade Balance"] = p["Exports"] - p["Imports"]
    return p


def _usd_scale_choice(key_prefix: str) -> tuple[str, float]:
    """
    TradeMap values are in USD thousand in the source file.
    Display scaling only:
      - USD Mn: divide by 1e3
      - USD Bn: divide by 1e6
    """
    scale = st.radio(
        "Value display unit (scaling only)",
        ["USD thousand (as in file)", "USD Mn", "USD Bn"],
        index=1,
        horizontal=True,
        key=f"{key_prefix}_usd_unit",
    )
    if scale == "USD Mn":
        return "USD Mn", 1e3
    if scale == "USD Bn":
        return "USD Bn", 1e6
    return "USD thousand", 1.0


def _top_n_plus_others(
    base: pd.DataFrame,
    *,
    flow: str,
    snap_year: int,
    top_n: int,
    metric: str,
    world: pd.DataFrame,
    usd_div: float,
    usd_label: str,
) -> tuple[pd.DataFrame, str, float]:
    """
    Returns:
      - snapshot table (Top N + Others)
      - metric title
      - world_total (raw units, USD thousand)
    """
    snap = base[(base["flow"] == flow) & (base["year"] == snap_year)].copy()
    if snap.empty:
        return pd.DataFrame(columns=["rank", "country", "metric_val", "metric_disp"]), "", float("nan")

    snap = snap.groupby(["country"], as_index=False)["value"].sum()
    snap["country_l"] = snap["country"].astype(str).str.strip().str.lower()
    snap = snap[snap["country_l"] != "world"].copy()
    snap = snap.drop(columns=["country_l"])

    if snap.empty:
        return pd.DataFrame(columns=["rank", "country", "metric_val", "metric_disp"]), "", float("nan")

    world_row = world[world["year"] == snap_year]
    world_total = float(world_row[flow].iloc[0]) if not world_row.empty else float(snap["value"].sum())

    if metric == "Value":
        snap["metric_val"] = snap["value"] / usd_div
        metric_title = f"Value ({usd_label})"
        fmt = lambda v: f"{v:,.2f}"
    else:
        snap["metric_val"] = np.where(world_total > 0, (snap["value"] / world_total) * 100.0, np.nan)
        metric_title = "Share of world (%)"
        fmt = lambda v: f"{v:,.2f}%"

    snap = snap.sort_values("metric_val", ascending=False).reset_index(drop=True)

    top = snap.head(int(top_n)).copy()
    rest = snap.iloc[int(top_n):].copy()

    others_value = float(rest["value"].sum()) if not rest.empty else 0.0
    if others_value > 0:
        if metric == "Value":
            others_metric = others_value / usd_div
        else:
            others_metric = (others_value / world_total) * 100.0 if world_total > 0 else np.nan

        top = pd.concat(
            [
                top,
                pd.DataFrame({"country": ["Others"], "value": [others_value], "metric_val": [others_metric]}),
            ],
            ignore_index=True,
        )

    top["rank"] = np.arange(1, len(top) + 1)
    top["metric_disp"] = top["metric_val"].apply(lambda x: fmt(x) if np.isfinite(x) else "NA")
    return top[["rank", "country", "metric_val", "metric_disp"]].copy(), metric_title, world_total


def _safe_year_range(df: pd.DataFrame) -> tuple[int, int]:
    y_min = int(pd.to_numeric(df["year"], errors="coerce").dropna().min())
    y_max = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    return y_min, y_max


# =========================
# UI renderers
# =========================
def render_trade_module(
    *,
    theme_name: str,
    title: str,
    hs_code: str,
    file_candidates: list[str],
    imports_sheet: str,
    exports_sheet: str,
    total_from_hs6: bool = False,
) -> None:
    apply_theme(theme_name)
    theme = THEMES.get(theme_name, THEMES["Pearls"])

    path = pick_first_existing(file_candidates)
    if path is None:
        st.error(
            "Default Excel file not found.\n\n"
            f"Place one of these in the app folder: {', '.join(file_candidates)}"
        )
        return

    total_df = load_total_trade_from_hs6(path) if total_from_hs6 else load_total_trade(path, imports_sheet=imports_sheet, exports_sheet=exports_sheet)
    hs6_df = load_all_hs6_trade(path)

    # Sidebar
    st.sidebar.markdown(
        f'<div class="sidebar-title">{html.escape(title)} DASHBOARD</div>'
        f'<div class="sidebar-sub">{theme["emoji"]} HS {hs_code} • ITC Trade Map style tables</div>'
        f'<div style="margin-top:6px;"><span class="badge">HS {hs_code}</span></div>',
        unsafe_allow_html=True,
    )
    scope = st.sidebar.radio(
        "Dataset",
        ["Total trade", "HS6 breakdown"],
        index=0,
        key=f"{hs_code}_scope",
    )
    st.sidebar.checkbox("Show data labels", value=False, key=f"{hs_code}_show_labels")

    # Gemstone segmentation (only for HS 7103)
    segment_codes = None
    segment_label = None
    if hs_code == "7103":
        seg_options = [
            ("All gemstones (HS 7103 total)", None),
            ("710310 — Precious & Semi‑Precious Stones (Unworked) • Raw materials", ["710310"]),
            ("710391 — Ruby, Sapphire & Emerald (Worked) • Finished", ["710391"]),
            ("710399 — Other Semi‑Precious Stones (Worked) • Finished", ["710399"]),
            ("Worked gemstones (710391 + 710399)", ["710391", "710399"]),
        ]
        seg_labels = [s for s, _ in seg_options]
        seg_sel = st.sidebar.selectbox(
            "Gemstone segment",
            options=seg_labels,
            index=0,
            key=f"{hs_code}_segment",
        )
        segment_label = seg_sel
        for lab, codes in seg_options:
            if lab == seg_sel:
                segment_codes = codes
                break


    # Base df
    if scope == "Total trade":
        if segment_codes:
            base = hs6_df[hs6_df["hs6"].astype(str).isin(segment_codes)].copy()
            base = base.groupby(["country", "year", "flow"], as_index=False)["value"].sum()
            base["hs6"] = "TOTAL"
            base["hs_desc"] = "Total"
            selected_hs6_label = segment_label or f"Total HS {hs_code}"
        else:
            base = total_df.copy()
            selected_hs6_label = f"Total HS {hs_code}"
    else:
        # HS6 selection
        if hs6_df.empty:
            st.warning("No HS6 sheets parsed from this workbook.")
            return
        hs6_options = (
            hs6_df[["hs6", "hs_desc"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["hs6"])
            .assign(label=lambda d: d["hs6"].astype(str) + " — " + d["hs_desc"].astype(str))
        )
        if segment_codes:
            hs6_options = hs6_options[hs6_options["hs6"].astype(str).isin(segment_codes)].copy()
            if hs6_options.empty:
                st.warning("No HS6 sheets found for the selected gemstone segment.")
                return
        
        labels = hs6_options["label"].tolist()
        default_sel = labels if len(labels) <= 3 else labels[:3]
        hs6_sel = st.sidebar.multiselect(
            "Select HS6 products",
            options=labels,
            default=default_sel,
            key=f"{hs_code}_hs6_sel",
        )
        hs6_pick = hs6_options.loc[hs6_options["label"].isin(hs6_sel), "hs6"].astype(str).tolist()
        if not hs6_pick:
            hs6_pick = hs6_options["hs6"].astype(str).tolist()
        base = hs6_df[hs6_df["hs6"].astype(str).isin(hs6_pick)].copy()
        base = base.groupby(["country", "year", "flow"], as_index=False)["value"].sum()
        base["hs6"] = "HS6"
        base["hs_desc"] = "Selected HS6"
        selected_hs6_label = "HS6 selection"

    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    base = base.dropna(subset=["year", "country"])

    if base.empty:
        st.warning("No data available for the selected scope.")
        return

    y_min, y_max = _safe_year_range(base)
    st.sidebar.subheader("Filters")
    yr_lo, yr_hi = st.sidebar.slider(
        "Year range",
        min_value=y_min,
        max_value=y_max,
        value=(y_min, y_max),
        step=1,
        key=f"{hs_code}_year_range",
    )
    flow = st.sidebar.selectbox(
        "Partner ranking based on",
        ["Exports", "Imports"],
        index=0,
        key=f"{hs_code}_flow",
    )
    metric = st.sidebar.radio(
        "Metric",
        ["Value", "Share of world (%)"],
        index=0,
        key=f"{hs_code}_metric",
    )

    # Filter by year range
    base = base[(base["year"] >= int(yr_lo)) & (base["year"] <= int(yr_hi))].copy()
    if base.empty:
        st.warning("No records in the selected year range.")
        return

    st.markdown(
        f"""
        <div style="margin-top:0.25rem;"></div>
        """,
        unsafe_allow_html=True,
    )

    render_hero_header(
        title=f"{title} Dashboard",
        subtitle=f"{selected_hs6_label} • ITC Trade Map style tables • values stored as USD thousand (display scaling only).",
        badge=f"{int(yr_lo)}–{int(yr_hi)}",
        icon=theme["emoji"],
    )
    st.caption(f"File loaded: `{Path(path).name}`")

    # Display unit (main area)
    st.markdown("#### Display unit")
    st.caption("Trade values are stored as USD thousand in the source file. Scaling affects display only.")
    usd_label, usd_div = _usd_scale_choice(f"{hs_code}_unit")

    show_labels = bool(st.session_state.get(f"{hs_code}_show_labels", False))

    # World series for KPIs and shares
    world = _world_series(base)
    years = world["year"].astype(int).tolist()
    if not years:
        st.warning("World series could not be computed.")
        return
    snap_default = years[-1]
    shared_snap_key = f"{hs_code}_snap_year"
    last_snap_key = f"{shared_snap_key}_last"
    snap_overview_key = f"{hs_code}_snap_year_overview"
    snap_topn_key = f"{hs_code}_snap_year_topn"

    if shared_snap_key not in st.session_state or st.session_state.get(shared_snap_key) not in years:
        st.session_state[shared_snap_key] = snap_default

    # Keep both widgets in sync without duplicate keys
    last_changed = st.session_state.get(last_snap_key)
    if last_changed == snap_overview_key:
        st.session_state[snap_topn_key] = st.session_state[shared_snap_key]
    elif last_changed == snap_topn_key:
        st.session_state[snap_overview_key] = st.session_state[shared_snap_key]
    else:
        st.session_state[snap_overview_key] = st.session_state[shared_snap_key]
        st.session_state[snap_topn_key] = st.session_state[shared_snap_key]
    if f"{hs_code}_topn" not in st.session_state:
        st.session_state[f"{hs_code}_topn"] = 10

    tab_overview, tab_topn, tab_country, tab_download = st.tabs(
        ["📌 Overview", "🌍 Top N countries", "📈 Country trend", "⬇️ Download"]
    )

    # -----------------------------
    # Overview
    # -----------------------------
    with tab_overview:
        csel1, csel2 = st.columns([1.3, 2.7])
        with csel1:
            snap_year = st.selectbox(
                "KPI year",
                options=years,
                index=years.index(int(st.session_state.get(shared_snap_key, snap_default))),
                key=snap_overview_key,
                on_change=_sync_shared,
                args=(shared_snap_key, snap_overview_key, last_snap_key),
            )
        with csel2:
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

        snap_year = int(snap_year)

        row = world[world["year"] == snap_year]
        prev = world[world["year"] == (snap_year - 1)]

        exp = float(row["Exports"].iloc[0]) if not row.empty else 0.0
        imp = float(row["Imports"].iloc[0]) if not row.empty else 0.0
        bal = exp - imp

        exp_prev = float(prev["Exports"].iloc[0]) if not prev.empty else np.nan
        imp_prev = float(prev["Imports"].iloc[0]) if not prev.empty else np.nan

        exp_yoy = (exp - exp_prev) / exp_prev * 100 if exp_prev and exp_prev > 0 else np.nan
        imp_yoy = (imp - imp_prev) / imp_prev * 100 if imp_prev and imp_prev > 0 else np.nan

        first_year = int(world["year"].min())
        last_year = int(world["year"].max())
        n_years = max(last_year - first_year, 0)

        exp_start = float(world.loc[world["year"] == first_year, "Exports"].iloc[0]) if not world.empty else np.nan
        exp_end = float(world.loc[world["year"] == last_year, "Exports"].iloc[0]) if not world.empty else np.nan
        imp_start = float(world.loc[world["year"] == first_year, "Imports"].iloc[0]) if not world.empty else np.nan
        imp_end = float(world.loc[world["year"] == last_year, "Imports"].iloc[0]) if not world.empty else np.nan

        exp_cagr = ((exp_end / exp_start) ** (1 / n_years) - 1) * 100 if n_years and exp_start and exp_start > 0 else np.nan
        imp_cagr = ((imp_end / imp_start) ** (1 / n_years) - 1) * 100 if n_years and imp_start and imp_start > 0 else np.nan

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi_card(
                "📤",
                f"Exports ({snap_year})",
                f"{exp / usd_div:,.2f} {usd_label}",
                sub=(f"{exp_yoy:+.1f}% YoY" if np.isfinite(exp_yoy) else ""),
            )
        with k2:
            kpi_card(
                "📥",
                f"Imports ({snap_year})",
                f"{imp / usd_div:,.2f} {usd_label}",
                sub=(f"{imp_yoy:+.1f}% YoY" if np.isfinite(imp_yoy) else ""),
            )
        with k3:
            kpi_card(
                "⚖️",
                f"Trade balance ({snap_year})",
                f"{bal / usd_div:,.2f} {usd_label}",
                sub=("Surplus" if bal >= 0 else "Deficit"),
            )
        with k4:
            kpi_card(
                "📈",
                f"CAGR Exports ({first_year}–{last_year})",
                (f"{exp_cagr:+.1f}%" if np.isfinite(exp_cagr) else "NA"),
                sub="",
            )
        with k5:
            kpi_card(
                "📉",
                f"CAGR Imports ({first_year}–{last_year})",
                (f"{imp_cagr:+.1f}%" if np.isfinite(imp_cagr) else "NA"),
                sub="",
            )

# Global trend chart
        trend = world.copy()
        trend["Exports_disp"] = trend["Exports"] / usd_div
        trend["Imports_disp"] = trend["Imports"] / usd_div
        trend["Balance_disp"] = trend["Trade Balance"] / usd_div

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["Exports_disp"],
                name="Exports",
                mode="lines+markers" + ("+text" if show_labels else ""),
                text=[f"{v:,.2f}" for v in trend["Exports_disp"]] if show_labels else None,
                textposition="top center",
                line=dict(color=THEMES[theme_name]["accent"], width=3),
                marker=dict(size=7, color=THEMES[theme_name]["accent"]),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["Imports_disp"],
                name="Imports",
                mode="lines+markers" + ("+text" if show_labels else ""),
                text=[f"{v:,.2f}" for v in trend["Imports_disp"]] if show_labels else None,
                textposition="top center",
                line=dict(color=THEMES[theme_name]["accent2"], width=3),
                marker=dict(size=7, color=THEMES[theme_name]["accent2"]),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["Balance_disp"],
                name="Trade balance",
                mode="lines",
                line=dict(color="rgba(90,107,130,0.75)", width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title=f"Global trade trend ({usd_label})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text=usd_label, secondary_y=False)
        fig.update_yaxes(title_text="Balance", secondary_y=True)
        style_plotly(fig, theme_name=theme_name)
        st.plotly_chart(fig, use_container_width=True)

        # HS6 composition (optional, only when HS6 data exists)
        if not hs6_df.empty:
            st.markdown("### HS6 composition (World)")
            c1, c2 = st.columns([1.4, 2.6])
            with c1:
                comp_flow = st.radio(
                    "Flow for composition",
                    ["Exports", "Imports"],
                    index=0,
                    horizontal=True,
                    key=f"{hs_code}_comp_flow",
                )
                years_opts = sorted(hs6_df["year"].dropna().astype(int).unique().tolist())
                if not years_opts:
                    comp_year = None
                    st.info("No HS6-year data available for composition.")
                else:
                    comp_year = st.selectbox(
                        "Year",
                        options=years_opts,
                        index=len(years_opts) - 1,
                        key=f"{hs_code}_comp_year",
                    )
            with c2:
                if comp_year is None:
                    st.info("HS6 composition not available.")
                    comp = pd.DataFrame(columns=["country","hs6","hs_desc","value","flow","year"])
                else:
                    comp = hs6_df[(hs6_df["year"].astype(int) == int(comp_year)) & (hs6_df["flow"] == comp_flow)].copy()
                comp["country_l"] = comp["country"].astype(str).str.strip().str.lower()
                comp = comp[comp["country_l"] == "world"].copy()
                if comp.empty:
                    st.info("World row missing in HS6 sheets for this year — composition not available.")
                else:
                    comp = comp.groupby(["hs6", "hs_desc"], as_index=False)["value"].sum()
                    comp["share"] = np.where(comp["value"].sum() > 0, comp["value"] / comp["value"].sum() * 100.0, np.nan)
                    comp = comp.sort_values("value", ascending=False)

                    fig_pie = px.pie(
                        comp,
                        names="hs6",
                        values="value",
                        hover_data={"hs_desc": True, "share": ":.2f"},
                        title=f"HS6 composition — {comp_flow} ({comp_year})",
                        hole=0.45,
                    )
                    fig_pie.update_traces(textinfo="percent+label")
                    style_plotly(fig_pie, theme_name=theme_name)
                    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # Top N partners
    # -----------------------------
    with tab_topn:
        c1, c2 = st.columns([2, 3])
        with c1:
            snap_year = st.selectbox(
                "Snapshot year",
                options=years,
                index=years.index(int(st.session_state.get(shared_snap_key, snap_default))),
                key=snap_topn_key,
                on_change=_sync_shared,
                args=(shared_snap_key, snap_topn_key, last_snap_key),
            )
        with c2:
            top_n = st.slider(
                "Top N countries (snapshot)",
                min_value=3,
                max_value=30,
                value=int(st.session_state.get(f"{hs_code}_topn", 10)),
                step=1,
                key=f"{hs_code}_topn",
            )

        snap_df, metric_title, world_total = _top_n_plus_others(
            base,
            flow=flow,
            snap_year=int(snap_year),
            top_n=int(top_n),
            metric=metric,
            world=world,
            usd_div=usd_div,
            usd_label=usd_label,
        )

        if snap_df.empty:
            st.info("No snapshot data available for the selected year.")
        else:
            plot_df = snap_df.sort_values("metric_val", ascending=True)
            fig_bar = px.bar(
                plot_df,
                x="metric_val",
                y="country",
                orientation="h",
                text=("metric_disp" if show_labels else None),
                title=f"Top {int(top_n)} partners + Others — {flow} ({int(snap_year)})",
                labels={"metric_val": metric_title, "country": "Country"},
            )
            fig_bar.update_traces(marker_color=THEMES[theme_name]["accent"], textposition="outside")
            fig_bar.update_layout(xaxis_title=metric_title, yaxis_title="Country")
            fig_bar.update_yaxes(categoryorder="array", categoryarray=plot_df["country"].tolist())
            style_plotly(fig_bar, theme_name=theme_name)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Snapshot table (Top N + Others + Total)")
            view = snap_df[["rank", "country", "metric_val"]].rename(
                columns={"country": "Partner", "metric_val": metric_title}
            )

            # Total row: show the overall picture explicitly
            if metric == "Value":
                total_val = float(world_total / usd_div) if np.isfinite(world_total) else float(view[metric_title].sum())
            else:
                total_val = 100.0
            total_row = pd.DataFrame({"rank": [""], "Partner": ["Total"], metric_title: [total_val]})
            view_total = pd.concat([view, total_row], ignore_index=True)

            def _bold_total_row(row):
                return ["font-weight: 800;"] * len(row) if str(row.get("Partner", "")) == "Total" else [""] * len(row)

            # Reduce extra zeros but keep readability
            fmt_map = {metric_title: ("{:,.2f}%" if metric != "Value" else "{:,.2f}")}
            st.dataframe(
                view_total.style.apply(_bold_total_row, axis=1).format(fmt_map),
                use_container_width=True,
            )

            csv = view_total.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download snapshot (CSV)",
                data=csv,
                file_name=f"{hs_code}_{flow}_{int(snap_year)}_top{int(top_n)}_plus_others.csv",
                mime="text/csv",
                key=f"{hs_code}_dl_snapshot",
            )

    # -----------------------------
    # Country trend
    # -----------------------------
    with tab_country:
        partners = sorted(
            base[base["country"].astype(str).str.strip().str.lower() != "world"]["country"].unique().tolist()
        )
        default_sel = ["India"] if "India" in partners else (partners[:5] if len(partners) >= 5 else partners)

        sel = st.multiselect(
            "Select countries",
            options=partners,
            default=default_sel,
            key=f"{hs_code}_countries",
        )
        include_world = st.checkbox("Include World", value=False, key=f"{hs_code}_include_world")
        if include_world:
            sel = ["World"] + [c for c in sel if c != "World"]

        if not sel:
            st.info("Select at least one country to see the trend.")
        else:
            d = base[(base["flow"] == flow) & (base["country"].isin(sel))].copy()
            d = d.groupby(["year", "country"], as_index=False)["value"].sum()

            if metric == "Value":
                d["y"] = d["value"] / usd_div
                y_title = f"Value ({usd_label})"
            else:
                totals = world.set_index("year")[flow].to_dict()
                d["y"] = d.apply(
                    lambda r: (r["value"] / totals.get(int(r["year"]), np.nan) * 100.0) if totals.get(int(r["year"]), 0) else np.nan,
                    axis=1,
                )
                d.loc[d["country"].astype(str).str.strip().str.lower() == "world", "y"] = 100.0
                y_title = "Share of world (%)"

            fig_line = px.line(
                d.sort_values("year"),
                x="year",
                y="y",
                color="country",
                markers=True,
                text=("y" if show_labels else None),
                title=f"Country trend — {flow}",
                labels={"year": "Year", "y": y_title, "country": "Country"},
            )
            if show_labels:
                fig_line.update_traces(textposition="top center")
            fig_line.update_layout(xaxis_title="Year", yaxis_title=y_title)
            style_plotly(fig_line, theme_name=theme_name)
            st.plotly_chart(fig_line, use_container_width=True)

            wide = d.pivot(index="country", columns="year", values="y").reset_index()
            st.dataframe(wide, use_container_width=True)

    # -----------------------------
    # Download
    # -----------------------------
    with tab_download:
        st.markdown("### Downloadable table (years as columns)")
        top_n_dl = st.slider(
            "Top N countries (by total over selected years)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            key=f"{hs_code}_topn_dl",
        )

        dl = base[base["flow"] == flow].copy()
        dl["country_l"] = dl["country"].astype(str).str.strip().str.lower()
        dl = dl[dl["country_l"] != "world"].copy()
        dl = dl.drop(columns=["country_l"])
        dl = dl.groupby(["country", "year"], as_index=False)["value"].sum()

        totals = dl.groupby("country", as_index=False)["value"].sum().sort_values("value", ascending=False).head(int(top_n_dl))
        dl = dl[dl["country"].isin(totals["country"])].copy()

        if metric == "Value":
            dl["val"] = dl["value"] / usd_div
            wide = dl.pivot(index="country", columns="year", values="val").fillna(0.0)
            wide.index.name = "Country"
            wide_reset = wide.reset_index()
            st.dataframe(wide_reset, use_container_width=True)

            csv = wide_reset.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download values (CSV)",
                data=csv,
                file_name=f"{hs_code}_{flow}_values_{int(yr_lo)}-{int(yr_hi)}_top{int(top_n_dl)}.csv",
                mime="text/csv",
                key=f"{hs_code}_dl_values",
            )

            yoy = wide.pct_change(axis=1) * 100.0
            yoy_reset = yoy.reset_index()
            st.markdown("### YoY growth (%)")
            st.dataframe(yoy_reset, use_container_width=True)

            csv2 = yoy_reset.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download YoY growth (CSV)",
                data=csv2,
                file_name=f"{hs_code}_{flow}_yoy_{int(yr_lo)}-{int(yr_hi)}_top{int(top_n_dl)}.csv",
                mime="text/csv",
                key=f"{hs_code}_dl_yoy",
            )
        else:
            # Compute share-of-world wide table
            totals_world = world.set_index("year")[flow].to_dict()
            dl["share"] = dl.apply(
                lambda r: (r["value"] / totals_world.get(int(r["year"]), np.nan) * 100.0)
                if totals_world.get(int(r["year"]), 0) else np.nan,
                axis=1,
            )
            wide = dl.pivot(index="country", columns="year", values="share")
            wide.index.name = "Country"
            wide_reset = wide.reset_index()
            st.dataframe(wide_reset, use_container_width=True)

            csv = wide_reset.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download shares (CSV)",
                data=csv,
                file_name=f"{hs_code}_{flow}_share_{int(yr_lo)}-{int(yr_hi)}_top{int(top_n_dl)}.csv",
                mime="text/csv",
                key=f"{hs_code}_dl_share",
            )


def main() -> None:
    st.set_page_config(page_title="Gemstones & Pearls Dashboards (HS 7101 & 7103)", page_icon="💠", layout="wide")

    st.sidebar.markdown('<div class="sidebar-title">GEMSTONES • PEARLS</div>' 
                    '<div class="sidebar-sub">Interactive trade dashboards • ITC Trade Map tables</div>',
                    unsafe_allow_html=True)
    module = st.sidebar.radio(
        "Choose module",
        ["Pearls Dashboard (HS 7101)", "Gemstones Dashboard (HS 7103)"],
        index=0,
        key="main_module",
    )

    if module == "Pearls Dashboard (HS 7101)":
        render_trade_module(
            theme_name="Pearls",
            title="Pearls",
            hs_code="7101",
            file_candidates=["Pearls(7101).xlsx", "Pearls (7101).xlsx", "Pearls_7101.xlsx"],
            imports_sheet="Imports(7101)",
            exports_sheet="Exports(7101)",
        )
    else:
        render_trade_module(
            theme_name="Gemstones",
            title="Gemstones",
            hs_code="7103",
            file_candidates=["Gemstones(7103).xlsx", "Gemstones (7103).xlsx", "Gemstones_7103.xlsx"],
            imports_sheet="Imports(7103)",
            exports_sheet="Exports(7103)",
            total_from_hs6=True,
        )


if __name__ == "__main__":
    main()
