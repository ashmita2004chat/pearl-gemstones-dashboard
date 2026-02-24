# -*- coding: utf-8 -*-
"""
Selected HS-6 Trade Dashboard (Gold-layout style)
Reference layout: gold_wgc_fixed_colors_layout_v5.py

Data:
- Global_Import_Export_filtered_hs_from_screenshots.xlsx
  Sheets: Import, Export
  Base unit in file: US$ Mn (values)

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================
# App config
# =========================
st.set_page_config(
    page_title="Selected HS-6 Trade Dashboard",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling (unique theme, gold-layout inspired)
# =========================
APP_CSS = """
<style>
:root{
  --bg1:#e9fff9;
  --bg2:#f3edff;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --accent:#14b8a6; /* teal */
  --accent2:#7c3aed; /* violet */
  --border:rgba(15,23,42,.08);
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
}
.stApp{
  background: radial-gradient(1200px 600px at 15% 15%, var(--bg1) 0%, rgba(255,255,255,0) 60%),
              radial-gradient(900px 500px at 95% 10%, var(--bg2) 0%, rgba(255,255,255,0) 55%),
              linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b1220 0%, #0b1220 55%, #070b14 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] *{
  color: rgba(255,255,255,.92) !important;
}
.sidebar-title{
  font-weight:900;
  letter-spacing: .3px;
  font-size: 21.5px;
  margin: 2px 0 6px 0;
}
.sidebar-sub{
  color: rgba(255,255,255,.70) !important;
  font-size: 12.5px;
  margin-bottom: 10px;
}
.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(20,184,166,.35);
  background: rgba(20,184,166,.14);
  color: rgba(255,255,255,.92) !important;
  font-size: 12px;
  font-weight: 800;
}
.badge.purple{
  border: 1px solid rgba(124,58,237,.35);
  background: rgba(124,58,237,.14);
}
.header-wrap{
  background: rgba(255,255,255,.78);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 18px 20px;
  margin: 4px 0 14px 0;
  backdrop-filter: blur(10px);
}

/* --- Hero header enhancements --- */
.header-wrap.hero{ position: relative; overflow: hidden; }
.header-wrap.hero::before{
  content: "";
  position: absolute;
  top: -80px;
  right: -90px;
  width: 260px;
  height: 260px;
  background: radial-gradient(circle at 30% 30%, rgba(20,184,166,.35) 0%, rgba(20,184,166,0) 70%);
  transform: rotate(10deg);
  pointer-events: none;
}
.header-wrap.hero::after{
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 4px;
  background: linear-gradient(90deg, rgba(20,184,166,0), rgba(20,184,166,.55), rgba(124,58,237,.45), rgba(20,184,166,0));
  opacity: .95;
  pointer-events: none;
}
.hero-row{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; position:relative; z-index:2; }
.hero-left{ display:flex; gap:14px; align-items:flex-start; }
.hero-right{ display:flex; flex-direction:column; align-items:flex-end; gap:8px; }
.hero-icon{
  width: 54px;
  height: 54px;
  border-radius: 18px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: linear-gradient(135deg, rgba(20,184,166,.18), rgba(124,58,237,.10));
  border: 1px solid rgba(20,184,166,.26);
  box-shadow: 0 10px 26px rgba(15,23,42,0.08);
}
.hero-icon svg{ width: 34px; height: 34px; }
.hero-art{
  position:absolute;
  right: 16px;
  bottom: -8px;
  width: 190px;
  opacity: .18;
  pointer-events:none;
  z-index:1;
}
.hero-art svg{ width: 100%; height: auto; }
.h1{
  font-size: 38px;
  line-height: 1.08;
  font-weight: 950;
  color: var(--text);
  margin: 0;
}
.hsub{
  margin-top: 6px;
  font-size: 14px;
  color: var(--muted);
  font-weight: 650;
}
.kpi{
  background: rgba(255,255,255,.86);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
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
  background: rgba(20,184,166,.14);
  border: 1px solid rgba(20,184,166,.25);
  font-size: 22px;
}
.kpi .ico.purple{
  background: rgba(124,58,237,.14);
  border: 1px solid rgba(124,58,237,.25);
}
.kpi .lbl{
  font-size: 12.5px;
  color: var(--muted);
  font-weight: 800;
  margin-bottom: 2px;
}
.kpi .val{
  font-size: 22px;
  font-weight: 950;
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
.kpi-grid .kpi{ margin: 0; }
.small-note{
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 650;
}
hr.soft{
  border: none;
  border-top: 1px solid rgba(15,23,42,.08);
  margin: 10px 0 14px 0;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# =========================
# File
# =========================
DATA_FILE = "Global_Import_Export_filtered_hs_from_screenshots.xlsx"


# =========================
# Exclusions (handled in Gold/Silver dashboards)
# =========================
EXCLUDED_HS = {711411, 711419, 711420, 711510, 711590, 711810, 711890}
# =========================
# Hero visuals (inline SVG)
# =========================
GEM_ICON_SVG = """
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" fill="none">
  <defs>
    <linearGradient id="g1" x1="14" y1="14" x2="50" y2="50" gradientUnits="userSpaceOnUse">
      <stop stop-color="#9ff9e8"/>
      <stop offset="0.55" stop-color="#14b8a6"/>
      <stop offset="1" stop-color="#7c3aed"/>
    </linearGradient>
  </defs>
  <path d="M32 6 L52 22 L44 54 H20 L12 22 Z" fill="url(#g1)" opacity="0.95"/>
  <path d="M12 22 H52" stroke="rgba(255,255,255,0.55)" stroke-width="2"/>
  <path d="M32 6 L20 22 L32 54 L44 22 Z" stroke="rgba(255,255,255,0.55)" stroke-width="2" opacity="0.7"/>
  <circle cx="32" cy="33" r="8" fill="rgba(15,23,42,0.10)"/>
</svg>
"""

GEM_ART_SVG = """
<svg viewBox="0 0 240 120" xmlns="http://www.w3.org/2000/svg" fill="none">
  <defs>
    <linearGradient id="a1" x1="30" y1="10" x2="220" y2="110">
      <stop stop-color="#b7fff1"/>
      <stop offset="0.6" stop-color="#14b8a6"/>
      <stop offset="1" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="shine" x1="0" y1="0" x2="1" y2="0">
      <stop stop-color="rgba(255,255,255,0)"/>
      <stop offset="0.5" stop-color="rgba(255,255,255,0.55)"/>
      <stop offset="1" stop-color="rgba(255,255,255,0)"/>
    </linearGradient>
  </defs>
  <g opacity="0.95">
    <path d="M36 38 L86 16 H210 L196 84 H52 Z" fill="url(#a1)"/>
    <path d="M86 16 L196 84" stroke="rgba(255,255,255,0.35)" stroke-width="3"/>
    <path d="M60 60 L190 60" stroke="url(#shine)" stroke-width="8" opacity="0.55"/>
    <path d="M52 84 L196 84" stroke="rgba(15,23,42,0.20)" stroke-width="3"/>
    <path d="M86 16 H210" stroke="rgba(15,23,42,0.16)" stroke-width="3"/>
  </g>
</svg>
"""

# =========================
# Helpers
# =========================
def resolve_file(filename: str) -> Path:
    candidates = [
        Path(filename),
        Path("/mnt/data") / filename,
        Path("/content") / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def _fmt_num(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{x:,.{decimals}f}"
    except Exception:
        return "—"

def _fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{decimals}f}%"

def _yoy(series: pd.Series, year: int) -> float:
    prev = year - 1
    if year not in series.index or prev not in series.index:
        return np.nan
    a = series.loc[prev]
    b = series.loc[year]
    if a is None or b is None or pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return (b / a - 1.0) * 100.0

def _cagr(series: pd.Series, start_year: int, end_year: int) -> float:
    if start_year not in series.index or end_year not in series.index:
        return np.nan
    start = series.loc[start_year]
    end = series.loc[end_year]
    n = end_year - start_year
    if n <= 0 or pd.isna(start) or pd.isna(end) or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1.0 / n) - 1.0

def _kpi_card(icon: str, label: str, value: str, sub: str = "", icon_cls: str = "") -> None:
    v = "" if value is None else str(value)
    digits = len(re.sub(r"[^0-9]", "", v))
    if digits >= 13 or len(v) >= 16:
        val_style = "font-size:18px; line-height:1.05; word-break:break-word;"
    elif digits >= 10 or len(v) >= 13:
        val_style = "font-size:20px; line-height:1.05; word-break:break-word;"
    else:
        val_style = "font-size:22px; line-height:1.05; word-break:break-word;"

    html = dedent(
        f"""
        <div class="kpi">
          <div class="ico {icon_cls}">{icon}</div>
          <div>
            <div class="lbl">{label}</div>
            <div class="val" style="{val_style}">{v}</div>
            <div class="sub">{sub}</div>
          </div>
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)

def _add_line_point_labels(fig, fmt: str = "{:,.2f}"):
    for tr in getattr(fig, "data", []) or []:
        if not hasattr(tr, "y") or tr.y is None:
            continue
        try:
            yvals = list(tr.y)
        except Exception:
            yvals = [tr.y]
        if len(yvals) == 0:
            continue

        txt = []
        for v in yvals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                txt.append("")
            else:
                try:
                    txt.append(fmt.format(float(v)))
                except Exception:
                    txt.append(str(v))
        tr.text = txt
        tr.textposition = "top center"
        mode = getattr(tr, "mode", "") or ""
        if "text" not in mode:
            tr.mode = (mode + "+text") if mode else "lines+text"

def _add_bar_labels(fig, orientation: str = "h", fmt: str = "{:,.2f}"):
    for tr in getattr(fig, "data", []) or []:
        arr = getattr(tr, "x", None) if orientation == "h" else getattr(tr, "y", None)
        if arr is None:
            continue
        try:
            vals = list(arr)
        except Exception:
            vals = [arr]

        txt = []
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                txt.append("")
            else:
                try:
                    txt.append(fmt.format(float(v)))
                except Exception:
                    txt.append(str(v))

        tr.text = txt
        tr.textposition = "outside"
        tr.cliponaxis = False

def _trade_unit_label(unit: str) -> str:
    return {"US$ Mn": "US$ Mn", "US$ Bn": "US$ Bn", "US$ (Absolute)": "US$"}.get(unit, unit)

def _scale_trade_from_mn(val_mn: float, unit: str) -> tuple[float, str]:
    """
    Base is US$ Mn from the file.
    """
    if val_mn is None or (isinstance(val_mn, float) and np.isnan(val_mn)):
        return (np.nan, "")
    if unit == "US$ Mn":
        return (val_mn * 1.0, "US$ Mn")
    if unit == "US$ Bn":
        return (val_mn / 1000.0, "US$ Bn")
    if unit == "US$ (Absolute)":
        return (val_mn * 1_000_000.0, "US$")
    return (val_mn * 1.0, "US$ Mn")

def _latest_year_with_data(series: pd.Series) -> int | None:
    s = series.copy()
    s = s.dropna()
    # treat zeros as missing for "latest"
    s = s[s > 0]
    if s.empty:
        return None
    return int(s.index.max())


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_trade_workbook(path: str) -> dict:
    imp = pd.read_excel(path, sheet_name="Import", header=1)
    exp = pd.read_excel(path, sheet_name="Export", header=1)

    # Clean colnames
    imp.columns = [str(c).strip() for c in imp.columns]
    exp.columns = [str(c).strip() for c in exp.columns]

    # Rename core columns
    imp = imp.rename(columns={
        "HS Code": "hs6", "HS Code ": "hs6",
        "Commodity Description": "description",
        "Commodity Category": "category",
        "Type of Commodity": "type",
        "Importers": "partner",
        "Unit": "unit", "Unit ": "unit",
    })
    exp = exp.rename(columns={
        "HS Code": "hs6", "HS Code ": "hs6",
        "Commodity Description": "description",
        "Commodity Category": "category",
        "Exporters": "partner",
        "Unit": "unit", "Unit ": "unit",
    })

    imp["hs6"] = pd.to_numeric(imp["hs6"], errors="coerce").astype("Int64")
    exp["hs6"] = pd.to_numeric(exp["hs6"], errors="coerce").astype("Int64")


    # Remove HS codes that belong to Gold/Silver (handled in separate dashboards)
    imp = imp[~imp["hs6"].isin(EXCLUDED_HS)].copy()
    exp = exp[~exp["hs6"].isin(EXCLUDED_HS)].copy()
    # Map missing type into exports from imports
    type_map = (
        imp[["hs6", "type"]]
        .dropna()
        .assign(type=lambda d: d["type"].astype(str).str.strip())
        .drop_duplicates("hs6")
        .set_index("hs6")["type"]
        .to_dict()
    )
    exp["type"] = exp["hs6"].map(type_map)

    # Identify year columns
    def _extract_year_cols(df: pd.DataFrame) -> dict:
        year_cols = {}
        for c in df.columns:
            m = re.search(r"value in\s*(\d{4})", str(c))
            if m:
                year_cols[c] = int(m.group(1))
        return year_cols

    imp_year = _extract_year_cols(imp)
    exp_year = _extract_year_cols(exp)

    # Keep only useful columns + years
    keep_imp = [c for c in ["hs6", "description", "category", "type", "partner", "unit"] if c in imp.columns] + list(imp_year.keys())
    keep_exp = [c for c in ["hs6", "description", "category", "type", "partner", "unit"] if c in exp.columns] + list(exp_year.keys())

    imp = imp[keep_imp].copy()
    exp = exp[keep_exp].copy()

    # Rename year cols to int year
    imp = imp.rename(columns=imp_year)
    exp = exp.rename(columns=exp_year)

    # Numeric conversion
    for y in imp_year.values():
        imp[y] = pd.to_numeric(imp[y], errors="coerce")
    for y in exp_year.values():
        exp[y] = pd.to_numeric(exp[y], errors="coerce")

    imp["partner"] = imp["partner"].astype(str).str.strip()
    exp["partner"] = exp["partner"].astype(str).str.strip()

    # HS meta
    hs_meta = (
        imp[["hs6", "description", "category", "type"]]
        .dropna(subset=["hs6"])
        .drop_duplicates("hs6")
        .sort_values("hs6")
        .reset_index(drop=True)
    )

    years = sorted(set(imp_year.values()) | set(exp_year.values()))

    def _wide_for(df: pd.DataFrame, hs: int, years: list[int]) -> pd.DataFrame:
        d = df[df["hs6"] == hs].copy()
        cols = ["partner"] + [y for y in years if y in d.columns]
        d = d[cols].copy()
        # Aggregate in case duplicates exist
        d = d.groupby("partner", as_index=False).sum(numeric_only=True)
        # Ensure all years exist
        for y in years:
            if y not in d.columns:
                d[y] = np.nan
        return d[["partner"] + years]

    hs_list = [int(x) for x in hs_meta["hs6"].dropna().unique().tolist()]
    hs_data = {}
    for hs in hs_list:
        hs_data[hs] = {
            "imp": _wide_for(imp, hs, years),
            "exp": _wide_for(exp, hs, years),
        }

    # Total across HS: sum by partner over all HS
    total_imp = imp.groupby("partner", as_index=False)[years].sum(numeric_only=True)
    total_exp = exp.groupby("partner", as_index=False)[years].sum(numeric_only=True)

    return {
        "years": years,
        "hs_meta": hs_meta,
        "hs_data": hs_data,
        "total_imp": total_imp,
        "total_exp": total_exp,
    }


# =========================
# Sidebar
# =========================
data_path = resolve_file(DATA_FILE)
if not data_path.exists():
    st.error(
        f"Missing file: {DATA_FILE}. Place it in the same folder as app.py (or /mnt/data or /content)."
    )
    st.stop()

data = load_trade_workbook(str(data_path))
years = data["years"]
hs_meta = data["hs_meta"]
hs_data = data["hs_data"]

miny, maxy = (min(years), max(years)) if years else (None, None)

st.sidebar.markdown('<div class="sidebar-title">SELECTED HS-6 • TRADE</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Imports & Exports (Trade Map) • 2005–2024</div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge">Gold-style layout</span> <span class="badge purple">Unique theme</span>', unsafe_allow_html=True)
st.sidebar.markdown("<br/>", unsafe_allow_html=True)

# Navigation items
nav_items = ["Trade — Total (All selected HS)"]
for r in hs_meta.itertuples(index=False):
    hs = int(r.hs6)
    desc = str(r.description) if pd.notna(r.description) else ""
    desc = re.sub(r"\s+", " ", desc).strip()
    if len(desc) > 48:
        desc = desc[:48].rstrip() + "…"
    nav_items.append(f"{hs} — {desc}")

page = st.sidebar.radio("Navigation", nav_items, index=0, key="nav_main")

st.sidebar.markdown("<hr class='soft'/>", unsafe_allow_html=True)
trade_unit = st.sidebar.radio(
    "Trade value display unit (base: US$ Mn)",
    ["US$ Mn", "US$ Bn", "US$ (Absolute)"],
    index=0,
    key="trade_unit",
)

st.sidebar.markdown("<hr class='soft'/>", unsafe_allow_html=True)
if "show_labels_v2" not in st.session_state:
    st.session_state["show_labels_v2"] = True
show_labels = st.sidebar.checkbox(
    "Show data labels on charts",
    value=st.session_state.get("show_labels_v2", True),
    key="show_labels_v2",
)

st.sidebar.markdown("<div class='small-note'>Tip: Keep the Excel file next to app.py.</div>", unsafe_allow_html=True)


# =========================
# Page renderers
# =========================
def hero(title: str, subtitle: str, badge_text: str = ""):
    badge_html = f"<div class='badge'>{badge_text}</div>" if badge_text else ""
    st.markdown(
        f"""
        <div class="header-wrap hero">
          <div class="hero-row">
            <div class="hero-left">
              <div class="hero-icon">{GEM_ICON_SVG}</div>
              <div>
                <div class="h1">{title}</div>
                <div class="hsub">{subtitle}</div>
              </div>
            </div>
            <div class="hero-right">
              {badge_html}
            </div>
          </div>
          <div class="hero-art">{GEM_ART_SVG}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _get_years_from_trade(df_wide: pd.DataFrame) -> list[int]:
    ys = [c for c in df_wide.columns if isinstance(c, int)]
    return sorted(ys)

def _world_series(df_wide: pd.DataFrame) -> pd.Series:
    """Return a 'World' series.

    TradeMap sheets sometimes contain World row = 0 for recent years (even when partner rows have values).
    To avoid KPIs showing 0 incorrectly, we fall back to summing all non-World partners for any year where:
      - World is missing/NaN, or
      - World == 0 but the sum of non-World partners > 0
    """
    ys = _get_years_from_trade(df_wide)

    # Base series from World row (if present)
    s_world_df = df_wide.loc[df_wide["partner"].astype(str).str.lower() == "world", ys]
    if s_world_df.empty:
        s_world = pd.Series(index=ys, dtype=float)
    else:
        s_world = s_world_df.iloc[0].copy()

    # Fallback series = sum of non-World partners
    non_world = df_wide.loc[df_wide["partner"].astype(str).str.lower() != "world", ys].copy()
    s_sum = non_world.sum(numeric_only=True)

    # Ensure aligned index
    s_world = pd.to_numeric(s_world, errors="coerce").reindex(ys)
    s_sum = pd.to_numeric(s_sum, errors="coerce").reindex(ys).fillna(0.0)

    # Replace problematic years
    out = s_world.copy()
    for y in ys:
        wv = out.get(y, np.nan)
        sv = s_sum.get(y, 0.0)
        if pd.isna(wv) and sv > 0:
            out[y] = sv
        elif (not pd.isna(wv)) and float(wv) == 0.0 and sv > 0:
            out[y] = sv

    return out

def show_trade(title: str, imp_wide: pd.DataFrame, exp_wide: pd.DataFrame, page_key: str, subtitle: str):
    ys = _get_years_from_trade(exp_wide if exp_wide is not None else imp_wide)
    if not ys:
        st.error("No year columns found.")
        return

    miny, maxy = min(ys), max(ys)

    world_exp = _world_series(exp_wide)
    world_imp = _world_series(imp_wide)

    # default snapshot year: latest nonzero (either flow)
    yexp = _latest_year_with_data(world_exp)
    yimp = _latest_year_with_data(world_imp)
    snap_default = max([y for y in [yexp, yimp] if y is not None], default=maxy)

    hero(title, subtitle, badge_text=f"{miny}–{maxy}")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1.0])
    with c1:
        snap_year = st.selectbox("Snapshot year", ys, index=ys.index(snap_default), key=f"{page_key}_year")
    with c2:
        partner_mode = st.selectbox("Partner ranking based on", ["Exports", "Imports"], index=0, key=f"{page_key}_mode")
    with c3:
        top_n = st.slider("Top N countries", 5, 30, 10, key=f"{page_key}_topn")
    with c4:
        metric = st.radio("Metric", ["Value", "Share of World (%)"], index=0, key=f"{page_key}_metric")

    unit_lbl = _trade_unit_label(trade_unit)

    ctx = (
        f"<div class='small-note'>"
        f"<b>Snapshot year:</b> {snap_year} &nbsp; • &nbsp; "
        f"<b>Partner ranking:</b> {partner_mode} &nbsp; • &nbsp; "
        f"<b>Top N:</b> {top_n} &nbsp; • &nbsp; "
        f"<b>Unit:</b> {unit_lbl} (base: US$ Mn)"
        f"</div>"
    )

    # World totals for snapshot year
    exp_val_mn = float(world_exp.get(snap_year, np.nan))
    imp_val_mn = float(world_imp.get(snap_year, np.nan))
    bal_mn = exp_val_mn - imp_val_mn if not (np.isnan(exp_val_mn) or np.isnan(imp_val_mn)) else np.nan

    exp_disp, exp_unit_lbl = _scale_trade_from_mn(exp_val_mn, trade_unit)
    imp_disp, _ = _scale_trade_from_mn(imp_val_mn, trade_unit)
    bal_disp, _ = _scale_trade_from_mn(bal_mn, trade_unit)

    exp_yoy = _yoy(world_exp, snap_year)
    imp_yoy = _yoy(world_imp, snap_year)
    exp_cagr = _cagr(world_exp, ys[0], snap_year)
    imp_cagr = _cagr(world_imp, ys[0], snap_year)

    # Rank source
    df_rank_src = exp_wide if partner_mode == "Exports" else imp_wide
    rank = df_rank_src[["partner", snap_year]].copy().rename(columns={snap_year: "value_mn"})
    rank["value_mn"] = pd.to_numeric(rank["value_mn"], errors="coerce")
    rank = rank[rank["partner"].str.lower() != "world"].dropna(subset=["value_mn"]).sort_values("value_mn", ascending=False)

    top_partner = rank["partner"].iloc[0] if len(rank) else "—"
    top_partner_val_mn = float(rank["value_mn"].iloc[0]) if len(rank) else np.nan
    top_partner_disp, _ = _scale_trade_from_mn(top_partner_val_mn, trade_unit)

    tabs = st.tabs(["Overview", "Countries", "Country trend", "Download"])

    # ----- Overview -----
    with tabs[0]:
        st.markdown(f"## Overview • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns([1, 1, 1, 1, 1, 1])
        with k1:
            _kpi_card("📤", f"Exports ({snap_year})", f"{_fmt_num(exp_disp, 2)} {exp_unit_lbl}", sub=f"YoY: {_fmt_pct(exp_yoy)}", icon_cls="")
        with k2:
            _kpi_card("📥", f"Imports ({snap_year})", f"{_fmt_num(imp_disp, 2)} {exp_unit_lbl}", sub=f"YoY: {_fmt_pct(imp_yoy)}", icon_cls="purple")
        with k3:
            _kpi_card(
                "⚖️",
                f"Trade Balance ({snap_year})",
                f"{_fmt_num(bal_disp, 2)} {exp_unit_lbl}",
                sub=("Surplus" if bal_mn > 0 else "Deficit" if bal_mn < 0 else ""),
                icon_cls=""
            )
        with k4:
            _kpi_card("📈", f"CAGR Exports ({ys[0]}–{snap_year})", f"{_fmt_pct(exp_cagr * 100)}", sub="", icon_cls="")
        with k5:
            _kpi_card("📉", f"CAGR Imports ({ys[0]}–{snap_year})", f"{_fmt_pct(imp_cagr * 100)}", sub="", icon_cls="purple")
        with k6:
            _kpi_card("🏆", "Top Partner", f"{top_partner}", sub=f"{_fmt_num(top_partner_disp, 2)} {exp_unit_lbl} ({partner_mode})", icon_cls="")

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

        st.markdown(f"### Global trade trend • {ys[0]}–{maxy}")
        df_world = pd.DataFrame({
            "year": ys,
            "Exports_mn": [world_exp.get(y, np.nan) for y in ys],
            "Imports_mn": [world_imp.get(y, np.nan) for y in ys],
        })
        df_world["Exports"], _ = zip(*df_world["Exports_mn"].apply(lambda v: _scale_trade_from_mn(v, trade_unit)))
        df_world["Imports"], _ = zip(*df_world["Imports_mn"].apply(lambda v: _scale_trade_from_mn(v, trade_unit)))
        df_world["Balance"] = df_world["Exports"] - df_world["Imports"]

        from plotly.subplots import make_subplots

        # Use subplots to avoid confusing overlaps:
        # Row 1: Exports (left) + Imports (right)
        # Row 2: Trade Balance (bar) on its own scale
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.68, 0.32],
            specs=[[{"secondary_y": True}], [{}]],
        )

        # Row 1 — Exports (left)
        fig.add_trace(
            go.Scatter(
                x=df_world["year"],
                y=df_world["Exports"],
                mode="lines+markers",
                name="Exports",
                line=dict(color="#14b8a6", width=3),
            ),
            row=1, col=1, secondary_y=False
        )

        # Row 1 — Imports (right)
        fig.add_trace(
            go.Scatter(
                x=df_world["year"],
                y=df_world["Imports"],
                mode="lines+markers",
                name="Imports",
                line=dict(color="#7c3aed", width=3),
            ),
            row=1, col=1, secondary_y=True
        )

        # Row 2 — Trade Balance (separate, clearer)
        fig.add_trace(
            go.Bar(
                x=df_world["year"],
                y=df_world["Balance"],
                name="Trade Balance",
                marker_color="rgba(15,23,42,.35)",
            ),
            row=2, col=1
        )

        fig.add_vline(x=snap_year, line_dash="dot", line_width=2, line_color="rgba(15,23,42,.35)")

        fig.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            title=f"Global trade trend ({exp_unit_lbl})",
            legend_title_text="",
            barmode="relative",
        )

        fig.update_yaxes(title_text=f"Exports ({exp_unit_lbl})", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text=f"Imports ({exp_unit_lbl})", row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text=f"Trade Balance ({exp_unit_lbl})", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)

        if show_labels:
            _add_line_point_labels(fig, fmt="{:,.2f}")
        st.plotly_chart(fig, use_container_width=True)
    # ----- Countries -----
    with tabs[1]:
        st.markdown(f"## Countries • {partner_mode} • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)

        world_total_mn = exp_val_mn if partner_mode == "Exports" else imp_val_mn
        if world_total_mn is None or (isinstance(world_total_mn, float) and np.isnan(world_total_mn)) or world_total_mn == 0:
            world_total_mn = float(rank["value_mn"].sum())

        rank_top = rank.head(top_n).copy()
        top_sum_mn = float(rank_top["value_mn"].sum())
        others_mn = float(max(0.0, float(world_total_mn) - top_sum_mn))

        plot_df = rank_top[["partner", "value_mn"]].copy()
        if others_mn > 0:
            plot_df = pd.concat([plot_df, pd.DataFrame([{"partner": "Others", "value_mn": others_mn}])], ignore_index=True)

        plot_df["value"], _ = zip(*plot_df["value_mn"].apply(lambda v: _scale_trade_from_mn(v, trade_unit)))
        plot_df["share_pct"] = (plot_df["value_mn"] / float(world_total_mn)) * 100.0 if world_total_mn else np.nan

        x_col = "value" if metric == "Value" else "share_pct"
        x_title = exp_unit_lbl if metric == "Value" else "Share of World (%)"
        lbl_fmt = "{:,.2f}" if metric == "Value" else "{:,.2f}%"

        plot_df_plot = plot_df.sort_values(x_col, ascending=True)

        fig2 = px.bar(plot_df_plot, x=x_col, y="partner", orientation="h", title=f"Top {top_n} countries + Others")
        fig2.update_traces(marker_color="#14b8a6" if partner_mode == "Exports" else "#7c3aed")
        fig2.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig2.update_xaxes(title=x_title)
        fig2.update_yaxes(title="")
        if show_labels:
            _add_bar_labels(fig2, orientation="h", fmt=lbl_fmt)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Data (Top N + Others)")
        value_col = f"value ({exp_unit_lbl})"
        tbl = plot_df.copy()
        tbl = tbl[["partner", "value", "share_pct"]].rename(columns={"value": value_col, "share_pct": "share_of_world_pct"})

        total_val, _ = _scale_trade_from_mn(world_total_mn, trade_unit)
        total_row = pd.DataFrame([{
            "partner": "Total",
            value_col: total_val,
            "share_of_world_pct": 100.0 if world_total_mn else np.nan
        }])
        tbl = pd.concat([tbl, total_row], ignore_index=True)

        tbl_disp = tbl.copy()
        for _c in [value_col, "share_of_world_pct"]:
            if _c in tbl_disp.columns:
                tbl_disp[_c] = pd.to_numeric(tbl_disp[_c], errors="coerce").round(2)

        def _bold_total_row(row):
            is_total = str(row.get("partner", "")).strip().lower() == "total"
            return ["font-weight: 850" if is_total else ""] * len(row)

        try:
            sty = (
                tbl_disp.style
                .apply(_bold_total_row, axis=1)
                .format({value_col: "{:,.2f}", "share_of_world_pct": "{:,.2f}"})
            )
            if hasattr(sty, "hide"):
                sty = sty.hide(axis="index")
            st.dataframe(sty, use_container_width=True)
        except Exception:
            st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

        st.download_button(
            "Download (Top N + Others + Total) CSV",
            data=tbl_disp.to_csv(index=False).encode("utf-8"),
            file_name=f"{page_key}_countries_{partner_mode.lower()}_{snap_year}_top{top_n}.csv",
            mime="text/csv",
        )

    # ----- Country trend -----
    with tabs[2]:
        st.markdown(f"## Country trend • {partner_mode} • {snap_year}")
        st.markdown(ctx, unsafe_allow_html=True)

        default_partner = top_partner if top_partner != "—" else (rank["partner"].iloc[0] if len(rank) else "World")
        partner_opts = rank["partner"].head(60).tolist()
        partner_idx = partner_opts.index(default_partner) if default_partner in partner_opts else 0
        partner = st.selectbox("Select country", partner_opts, index=partner_idx, key=f"{page_key}_partner")

        df_src = exp_wide if partner_mode == "Exports" else imp_wide
        partner_row = df_src[df_src["partner"].str.lower() == str(partner).lower()]
        if partner_row.empty:
            st.info("No data available for the selected country.")
        else:
            s = partner_row.iloc[0][ys]
            df_tr = pd.DataFrame({"year": ys, "value_mn": pd.to_numeric(s.values, errors="coerce")})
            df_tr["value"], _ = zip(*df_tr["value_mn"].apply(lambda v: _scale_trade_from_mn(v, trade_unit)))

            world_series = world_exp if partner_mode == "Exports" else world_imp
            df_tr["world_mn"] = [world_series.get(y, np.nan) for y in ys]
            df_tr["share_pct"] = np.where(
                (df_tr["world_mn"].notna()) & (df_tr["world_mn"] != 0),
                (df_tr["value_mn"] / df_tr["world_mn"]) * 100.0,
                np.nan,
            )

            y_col = "value" if metric == "Value" else "share_pct"
            y_title = exp_unit_lbl if metric == "Value" else "Share of World (%)"
            lbl_fmt = "{:,.2f}" if metric == "Value" else "{:,.2f}%"

            fig3 = px.line(df_tr, x="year", y=y_col, markers=True, title=f"Country trend • {partner_mode}: {partner}")
            fig3.update_traces(line=dict(color="#14b8a6" if partner_mode == "Exports" else "#7c3aed", width=3))
            fig3.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=60, b=10))
            fig3.update_yaxes(title=y_title)
            if show_labels:
                _add_line_point_labels(fig3, fmt=lbl_fmt)
            st.plotly_chart(fig3, use_container_width=True)

    # ----- Download -----
    with tabs[3]:
        st.markdown("## Download")
        st.markdown(ctx, unsafe_allow_html=True)
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown("### Download (year range)")

        d1, d2 = st.columns(2)
        with d1:
            dl_start_year = st.selectbox("Start year", ys, index=0, key=f"{page_key}_dl_start_year")
        with d2:
            dl_end_year = st.selectbox("End year", ys, index=len(ys) - 1, key=f"{page_key}_dl_end_year")

        if dl_start_year > dl_end_year:
            dl_start_year, dl_end_year = dl_end_year, dl_start_year

        sel_years = [y for y in ys if dl_start_year <= y <= dl_end_year]

        exp_long = exp_wide[["partner"] + sel_years].melt(id_vars="partner", var_name="year", value_name="exports_mn")
        imp_long = imp_wide[["partner"] + sel_years].melt(id_vars="partner", var_name="year", value_name="imports_mn")

        dl = exp_long.merge(imp_long, on=["partner", "year"], how="outer")
        dl["year"] = dl["year"].astype(int)

        dl["exports_mn"] = pd.to_numeric(dl["exports_mn"], errors="coerce")
        dl["imports_mn"] = pd.to_numeric(dl["imports_mn"], errors="coerce")
        dl["balance_mn"] = dl["exports_mn"] - dl["imports_mn"]

        dl = dl.sort_values(["partner", "year"])
        dl["yoy_exports_pct"] = dl.groupby("partner")["exports_mn"].pct_change() * 100.0
        dl["yoy_imports_pct"] = dl.groupby("partner")["imports_mn"].pct_change() * 100.0

        def _scaled(v):
            return _scale_trade_from_mn(v, trade_unit)[0]

        dl["exports"] = dl["exports_mn"].apply(_scaled)
        dl["imports"] = dl["imports_mn"].apply(_scaled)
        dl["balance"] = dl["balance_mn"].apply(_scaled)

        out_cols = ["partner", "year", "exports", "imports", "balance", "yoy_exports_pct", "yoy_imports_pct"]
        dl_out = dl[out_cols].copy()

        st.caption(
            f"Download range: {dl_start_year}–{dl_end_year} • Rows: {len(dl_out):,} • Unit: {unit_lbl}"
        )
        st.dataframe(dl_out.head(500), use_container_width=True, height=420)

        st.download_button(
            "Download trade (Exports+Imports+Balance + YoY) CSV",
            data=dl_out.to_csv(index=False).encode("utf-8"),
            file_name=f"{page_key}_trade_{dl_start_year}_{dl_end_year}.csv",
            mime="text/csv",
        )


# =========================
# Routing
# =========================
if page == "Trade — Total (All selected HS)":
    show_trade(
        "Trade — Total (Selected HS)",
        data["total_imp"],
        data["total_exp"],
        page_key="trade_total",
        subtitle="Global trade totals + partner analysis • Top-N + Others + Total • Downloads"
    )
else:
    # parse hs code from page
    m = re.match(r"^(\d{6})\s+—", page)
    hs = int(m.group(1)) if m else None
    meta = hs_meta[hs_meta["hs6"] == hs].iloc[0] if hs is not None and (hs_meta["hs6"] == hs).any() else None

    desc = (str(meta["description"]) if meta is not None and pd.notna(meta["description"]) else "").strip()
    cat = (str(meta["category"]) if meta is not None and pd.notna(meta["category"]) else "").strip()
    typ = (str(meta["type"]) if meta is not None and pd.notna(meta["type"]) else "").strip()

    subtitle = f"{desc} • Category: {cat or '—'} • Type: {typ or '—'} • Values in US$ Mn"
    show_trade(
        f"Trade — HS {hs}",
        hs_data[hs]["imp"],
        hs_data[hs]["exp"],
        page_key=f"trade_{hs}",
        subtitle=subtitle
    )
