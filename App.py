# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam
trong kỉ nguyên AI — dữ liệu thực 2020-2025.

Họ và tên : Nguyễn Bảo Khánh
Mã sinh viên: 23051266
Bài tập lớn: Các mô hình ra quyết định

Chạy:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="AIDEOM-VN | Mô hình Ra Quyết Định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS PREMIUM — Thiết kế cao cấp, chuyên nghiệp
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-deep:    #04090f;
  --bg-card:    #0a1628;
  --bg-panel:   #0f1d34;
  --bg-hover:   #162240;
  --accent-red: #e63946;
  --accent-gld: #f4a261;
  --accent-cyn: #2ec4b6;
  --accent-blu: #4895ef;
  --txt-hi:     #f0f4ff;
  --txt-md:     #a8bbd4;
  --txt-lo:     #5e7a99;
  --border:     #1e3050;
  --border-hi:  #2d4875;
  --radius:     14px;
  --shadow:     0 8px 32px rgba(0,0,0,0.55);
}

/* ─── ROOT ─────────────────────────────── */
html, body, .stApp {
  background: var(--bg-deep) !important;
  font-family: 'Be Vietnam Pro', sans-serif !important;
  color: var(--txt-hi) !important;
}

/* ─── ALL TEXT OVERRIDE ─────────────────── */
.stApp *, .stApp p, .stApp span, .stApp label,
.stApp li, .stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stText"], div[data-testid="stCaptionContainer"] {
  color: var(--txt-hi) !important;
  font-family: 'Be Vietnam Pro', sans-serif !important;
}
.stCaption, div[data-testid="stCaptionContainer"] p {
  color: var(--txt-md) !important;
  font-size: .82rem !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
  color: #ffffff !important;
  letter-spacing: -0.02em !important;
  font-weight: 700 !important;
}
/* latex */
.katex, .katex * { color: #ddeeff !important; }
/* dataframe */
[data-testid="stDataFrame"] *, [data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th { color: var(--txt-hi) !important; }

/* ─── SIDEBAR ───────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #040c1a 0%, #071427 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt-hi) !important; }
[data-testid="stSidebar"] .stRadio label {
  padding: 6px 10px !important;
  border-radius: 8px !important;
  transition: background .2s !important;
  display: flex !important;
  align-items: center !important;
  font-size: .85rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--bg-hover) !important;
}
/* Hide default radio circle — use custom indicator */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] .stRadio input[type="radio"] {
  display: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
  display: none !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] input + div {
  display: none !important;
}

/* ─── TABS ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel) !important;
  border-radius: 12px 12px 0 0 !important;
  gap: 2px !important;
  padding: 6px 6px 0 !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 8px 18px !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
  border: none !important;
  transition: background .2s !important;
}
.stTabs [data-baseweb="tab"] p { color: var(--txt-md) !important; }
.stTabs [aria-selected="true"] {
  background: var(--bg-hover) !important;
  border-bottom: 2px solid var(--accent-red) !important;
}
.stTabs [aria-selected="true"] p { color: #fff !important; }

/* ─── SLIDERS / INPUTS ──────────────────── */
.stSlider label, .stRadio label,
.stSelectbox label, .stNumberInput label { color: var(--txt-hi) !important; }
.stSlider [data-baseweb="slider"] { color: var(--accent-cyn) !important; }

/* ─── EXPANDER ──────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] details {
  background: var(--bg-panel) !important;
}
[data-testid="stExpander"] details summary {
  background: var(--bg-panel) !important;
  padding: 12px 16px !important;
  list-style: none !important;
  cursor: pointer !important;
}
/* Hide the .arrow text node — this is the key fix */
[data-testid="stExpander"] details summary .arrow {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  position: absolute !important;
  opacity: 0 !important;
}
[data-testid="stExpander"] details summary svg {
  color: var(--txt-md) !important;
  flex-shrink: 0 !important;
}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span:not(.arrow) {
  color: var(--txt-hi) !important;
  font-weight: 600 !important;
  font-size: .95rem !important;
}
[data-testid="stExpander"] details > div[data-testid="stExpanderDetails"] {
  background: var(--bg-card) !important;
  border-top: 1px solid var(--border) !important;
  padding: 16px !important;
}

/* ─── HOME PAGE NAV BUTTONS ─────────────── */
div[data-testid="stVerticalBlock"] .stButton button {
  background: transparent !important;
  border: 1px solid #1e3050 !important;
  border-radius: 8px !important;
  color: #a8bbd4 !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
  padding: 5px 10px !important;
  transition: all .15s !important;
}
div[data-testid="stVerticalBlock"] .stButton button:hover {
  border-color: #e63946 !important;
  color: #fff !important;
  background: rgba(230,57,70,.1) !important;
}

/* ─── SUCCESS / ERROR / WARNING ─────────── */
[data-testid="stAlert"] { border-radius: var(--radius) !important; }
[data-testid="stAlert"] * { color: inherit !important; }

/* ─── CUSTOM COMPONENTS ─────────────────── */

/* Hero banner */
.hero-banner {
  background: linear-gradient(135deg, #0d1f3c 0%, #162f5c 50%, #0d1f3c 100%);
  border: 1px solid var(--border-hi);
  border-radius: 20px;
  padding: 36px 40px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 70% 50%, rgba(230,57,70,.12) 0%, transparent 70%),
              radial-gradient(ellipse at 20% 80%, rgba(46,196,182,.08) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-size: 2.6rem;
  font-weight: 900;
  letter-spacing: -.03em;
  color: #fff;
  line-height: 1.1;
  margin-bottom: 8px;
}
.hero-sub {
  font-size: 1rem;
  color: var(--txt-md);
  font-weight: 400;
  margin-bottom: 20px;
}
.hero-badge {
  display: inline-block;
  background: rgba(230,57,70,.15);
  border: 1px solid rgba(230,57,70,.4);
  color: #ff6b78 !important;
  border-radius: 999px;
  padding: 4px 14px;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
  margin-right: 8px;
}

/* KPI cards */
.kpi-grid { display: flex; gap: 14px; margin: 18px 0; flex-wrap: wrap; }
.kpi-card {
  flex: 1; min-width: 140px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
  transition: border-color .2s, transform .15s;
}
.kpi-card:hover { border-color: var(--border-hi); transform: translateY(-1px); }
.kpi-card::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent-red), var(--accent-gld));
}
.kpi-label { font-size: .75rem; font-weight: 600; color: var(--txt-md) !important;
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.7rem; font-weight: 800; color: #fff !important;
  letter-spacing: -.02em; line-height: 1; }
.kpi-delta { font-size: .75rem; font-weight: 700; margin-top: 6px;
  color: #4ade80 !important; display: flex; align-items: center; gap: 4px; }
.kpi-delta.neg { color: #f87171 !important; }

/* Section header */
.section-header {
  display: flex; align-items: center; gap: 12px;
  margin: 28px 0 8px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.section-icon {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--accent-red), var(--accent-gld));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
}
.section-title { font-size: 1.4rem; font-weight: 800; color: #fff; margin: 0; }
.section-sub   { font-size: .82rem; color: var(--txt-md); margin: 2px 0 0; }

/* Note/info box */
.info-box {
  background: linear-gradient(135deg, rgba(72,149,239,.08), rgba(46,196,182,.05));
  border: 1px solid rgba(72,149,239,.25);
  border-left: 3px solid var(--accent-blu);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin: 12px 0;
}
.info-box b { color: #fff !important; }
.info-box p { color: var(--txt-md) !important; margin: 0; font-size: .88rem; }

.warn-box {
  background: rgba(244,162,97,.06);
  border: 1px solid rgba(244,162,97,.25);
  border-left: 3px solid var(--accent-gld);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin: 12px 0;
}

/* Level chip */
.lvl-chip {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

/* Formula box */
.formula-box {
  background: #0b1628;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin: 12px 0;
  font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar ID card */
.id-card {
  background: var(--bg-panel);
  border: 1px solid var(--border-hi);
  border-radius: var(--radius);
  padding: 14px 16px;
  font-size: .82rem;
  line-height: 1.8;
  margin-top: 8px;
}
.id-card b { color: var(--accent-cyn) !important; }

/* Result table highlight */
.highlight-row { background: rgba(230,57,70,.1) !important; }

/* Step indicator */
.step-row {
  display: flex; align-items: flex-start; gap: 14px; margin: 10px 0;
}
.step-num {
  width: 28px; height: 28px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent-red), var(--accent-gld));
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: .78rem; font-weight: 800; color: #fff;
}
.step-content { font-size: .88rem; color: var(--txt-md) !important; padding-top: 4px; }
.step-content b { color: #fff !important; }

/* Progress bar custom */
.prog-bar-wrap { background: var(--border); border-radius: 999px; height: 6px; margin: 4px 0 12px; }
.prog-bar-fill { height: 6px; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-red), var(--accent-gld)); }

/* Divider */
.vn-divider {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-hi), transparent);
  margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

PLOT_TMPL = "plotly_dark"
PALETTE   = ["#e63946", "#f4a261", "#2ec4b6", "#4895ef", "#a8dadc", "#f1faee", "#457b9d"]

# ============================================================
# DỮ LIỆU
# ============================================================
def _write_csvs():
    if not os.path.exists("vietnam_macro_2020_2025.csv"):
        pd.DataFrame({
            "year":                          [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND":              [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "population_million":            [97.6, 98.5, 99.5, 100.3, 101.3, 102.3],
            "digital_economy_share_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "labor_productivity_million_VND":[150.1, 171.8, 188.7, 199.3, 221.9, 245.0],
            "FDI_disbursed_billion_USD":     [19.98, 19.74, 22.4, 23.18, 25.35, 27.6],
            "export_goods_billion_USD":      [282.6, 336.3, 371.3, 354.7, 405.5, 475.0],
            "inflation_cpi_pct":             [3.23, 1.84, 3.15, 3.25, 3.63, 3.5],
        }).to_csv("vietnam_macro_2020_2025.csv", index=False)

    if not os.path.exists("vietnam_sectors_2024.csv"):
        pd.DataFrame({
            "sector_name_vi":    ["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng",
                                  "Khai khoáng","Bán buôn-bán lẻ","Tài chính-Ngân hàng",
                                  "Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo","Y tế"],
            "growth_rate_2024_pct":  [3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
            "gdp_share_2024_pct":    [11.86,24.10,6.80,2.50,9.50,5.10,4.20,3.80,4.00,3.20],
            "spillover_coef_0_1":    [0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
            "export_billion_USD":    [40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
            "labor_million":         [13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
            "ai_readiness_0_100":    [15,55,20,30,48,72,42,88,38,45],
            "automation_risk_pct":   [18,42,25,55,38,52,35,28,22,18],
            "rd_intensity_pct":      [0.10,0.55,0.15,0.20,0.18,0.85,0.32,1.20,0.45,0.30],
        }).to_csv("vietnam_sectors_2024.csv", index=False)

    if not os.path.exists("vietnam_regions_2024.csv"):
        pd.DataFrame({
            "region_name_vi":         ["Trung du miền núi phía Bắc","Đồng bằng sông Hồng",
                                       "Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên",
                                       "Đông Nam Bộ","Đồng bằng sông Cửu Long"],
            "grdp_per_capita_million_VND": [57.0,152.3,87.5,68.9,158.9,80.5],
            "fdi_registered_billion_USD":  [3.5,20.0,8.2,0.8,18.5,2.1],
            "digital_index_0_100":         [38,78,55,32,82,48],
            "ai_readiness_0_100":          [22,68,40,18,75,30],
            "trained_labor_pct":           [21.5,36.8,27.5,18.2,42.5,16.8],
            "rd_intensity_pct":            [0.18,0.85,0.32,0.15,0.78,0.22],
            "internet_penetration_pct":    [72,92,84,68,94,78],
            "gini_coef":                   [0.405,0.358,0.372,0.412,0.385,0.392],
        }).to_csv("vietnam_regions_2024.csv", index=False)


@st.cache_data
def load_data():
    _write_csvs()
    macro   = pd.read_csv("vietnam_macro_2020_2025.csv").sort_values("year").reset_index(drop=True)
    sectors = pd.read_csv("vietnam_sectors_2024.csv")
    regions = pd.read_csv("vietnam_regions_2024.csv")
    return macro, sectors, regions


MACRO, SECTORS, REGIONS = load_data()

K_HIST  = np.array([16500,17800,19600,21300,23500,25900])
L_HIST  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D_HIST  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
AI_HIST = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H_HIST  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
COEF    = dict(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07)

# ============================================================
# SESSION STATE — NAVIGATION
# ============================================================
PAGES = [
    "🏠 Trang chủ",
    "📐 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP Ngân sách số",
    "🎯 Bài 3 — Chỉ số ưu tiên 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng",
    "🔢 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Tối ưu động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Quy hoạch ngẫu nhiên",
    "🤖 Bài 11 — Q-learning RL",
    "🇻🇳 Bài 12 — AIDEOM tích hợp",
]

if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

def _nav(idx):
    st.session_state.page_idx = idx

# ── CSS cho sidebar buttons ──────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton button {
  width: 100% !important;
  text-align: left !important;
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 7px 12px !important;
  font-size: .84rem !important;
  font-weight: 500 !important;
  color: #a8bbd4 !important;
  cursor: pointer !important;
  transition: background .15s, color .15s !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
  background: #162240 !important;
  color: #fff !important;
}
section[data-testid="stSidebar"] .stButton button:focus {
  box-shadow: none !important;
  outline: none !important;
}
.sidebar-active button {
  background: #1e3050 !important;
  color: #fff !important;
  border-left: 3px solid #e63946 !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 10px;'>
      <div style='display:flex;align-items:center;gap:10px;'>
        <svg width="22" height="16" viewBox="0 0 22 16" fill="none"
             xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
          <rect y="0"  width="22" height="2.5" rx="1.25" fill="#4895ef"/>
          <rect y="6"  width="16" height="2.5" rx="1.25" fill="#2ec4b6"/>
          <rect y="12" width="19" height="2.5" rx="1.25" fill="#4895ef"/>
        </svg>
        <div style='font-size:1.2rem;font-weight:900;letter-spacing:-.02em;
             background:linear-gradient(90deg,#4895ef,#2ec4b6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;'>
          AIDEOM<span style='
             background:linear-gradient(90deg,#e63946,#f4a261);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;'>-VN</span>
        </div>
      </div>
      <div style='font-size:.68rem;color:#4895ef;margin-top:5px;font-weight:600;
           text-transform:uppercase;letter-spacing:.1em;opacity:.7;'>
        Decision Optimization Model
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    _icons = ["🏠","📐","💰","🎯","🗺","🔢","🏆","🌐","⏳","👷","🎲","🤖","🇻🇳"]
    _labels = [
        "Trang chủ",
        "Bài 1 — Cobb-Douglas + AI",
        "Bài 2 — LP Ngân sách số",
        "Bài 3 — Chỉ số ưu tiên ngành",
        "Bài 4 — LP ngành-vùng",
        "Bài 5 — MIP 15 dự án",
        "Bài 6 — TOPSIS 6 vùng",
        "Bài 7 — NSGA-II Pareto",
        "Bài 8 — Tối ưu động",
        "Bài 9 — Lao động & AI",
        "Bài 10 — Quy hoạch ngẫu nhiên",
        "Bài 11 — Q-learning RL",
        "Bài 12 — AIDEOM tích hợp",
    ]
    for _i, (_ic, _lb) in enumerate(zip(_icons, _labels)):
        _active = st.session_state.page_idx == _i
        _wrap_open  = "<div class='sidebar-active'>" if _active else "<div>"
        _wrap_close = "</div>"
        st.markdown(_wrap_open, unsafe_allow_html=True)
        if st.button(f"{_ic}  {_lb}", key=f"nav_{_i}"):
            _nav(_i)
            st.rerun()
        st.markdown(_wrap_close, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e3050;margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='id-card'>
      <b>Họ và tên:</b> Nguyễn Bảo Khánh<br>
      <b>MSSV:</b> 23051266<br>
      <b>Môn học:</b> Các mô hình ra quyết định<br>
      <b>Trường:</b> UEB — VNU
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;color:#5e7a99;margin-top:8px;'>NSO · MoST · MIC · MPI · WB · GII 2025</div>", unsafe_allow_html=True)

page = PAGES[st.session_state.page_idx]


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def kpi_row(cols_data):
    """cols_data: list of (col, label, value, delta=None, neg=False)"""
    for item in cols_data:
        col, label, value = item[0], item[1], item[2]
        delta = item[3] if len(item) > 3 else None
        neg   = item[4] if len(item) > 4 else False
        delta_cls = "neg" if neg else ""
        arrow = "↓" if neg else "↑"
        delta_html = f'<div class="kpi-delta {delta_cls}">{arrow} {delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)


def section_header(icon, title, subtitle=None):
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
      <div class="section-icon">{icon}</div>
      <div>
        <div class="section-title">{title}</div>
        {sub_html}
      </div>
    </div>""", unsafe_allow_html=True)


def info_box(text):
    st.markdown(f'<div class="info-box"><p>{text}</p></div>', unsafe_allow_html=True)


def warn_box(text):
    st.markdown(f'<div class="warn-box"><p>{text}</p></div>', unsafe_allow_html=True)


def step(num, bold_text, detail=""):
    st.markdown(f"""
    <div class="step-row">
      <div class="step-num">{num}</div>
      <div class="step-content"><b>{bold_text}</b> {detail}</div>
    </div>""", unsafe_allow_html=True)


def progress_bar(pct, label=""):
    st.markdown(f"""
    <div style='font-size:.78rem;color:var(--txt-md);margin-bottom:2px;'>{label} — {pct:.0f}%</div>
    <div class='prog-bar-wrap'><div class='prog-bar-fill' style='width:{min(pct,100):.0f}%;'></div></div>
    """, unsafe_allow_html=True)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None


def _plot_cfg(title="", h=None):
    """For fig.update_layout(**_plot_cfg(...))"""
    cfg = dict(template=PLOT_TMPL, title=title,
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(family="Be Vietnam Pro", color="#a8bbd4"))
    if h: cfg["height"] = h
    return cfg

def _px_cfg(title="", h=None):
    """For px.xxx(..., **_px_cfg(...)) — only px-compatible keys"""
    cfg = dict(template=PLOT_TMPL, title=title)
    if h: cfg["height"] = h
    return cfg


# ============================================================
# HELPER: LP / TOPSIS / SP (dùng lại ở nhiều bài)
# ============================================================
REGIONS_VI = ['Trung du miền núi', 'ĐB sông Hồng', 'Bắc Trung Bộ',
              'Tây Nguyên', 'Đông Nam Bộ', 'ĐB sông Cửu Long']
REG  = ['NMM','RRD','NCC','CH','SE','MD']
ITEMS= ['I','D','AI','H']
BETA_RJ = {
    ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
    ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
    ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
    ('CH','I') :1.20,('CH','D') :0.75,('CH','AI') :0.45,('CH','H') :1.35,
    ('SE','I') :0.90,('SE','D') :1.30,('SE','AI') :1.55,('SE','H') :1.00,
    ('MD','I') :1.10,('MD','D') :0.85,('MD','AI') :0.65,('MD','H') :1.25}
_D0_default = [38,78,55,32,82,48]
REG_NAME_COL    = pick_col(REGIONS, ["region_name_vi","region_name_en","region_name","region"])
REG_DIGITAL_COL = pick_col(REGIONS, ["digital_index_0_100","digital_index","digital"])
D0_REG = dict(zip(REG, REGIONS[REG_DIGITAL_COL].values if REG_DIGITAL_COL and len(REGIONS)==6 else _D0_default))

TOPSIS_CRIT_CANDS = [
    (["grdp_per_capita_million_VND","grdp_per_capita","grdp"], True),
    (["fdi_registered_billion_USD","fdi_registered","fdi"],    True),
    (["digital_index_0_100","digital_index","digital"],        True),
    (["ai_readiness_0_100","ai_readiness","ai"],               True),
    (["trained_labor_pct","trained_labor","labor_trained"],    True),
    (["rd_intensity_pct","rd_intensity","rd"],                 True),
    (["internet_penetration_pct","internet_penetration","internet"],True),
    (["gini_coef","gini"],                                     False),
]
TOPSIS_LBL_ALL = ['GRDP/N','FDI','Digital','AI','LĐĐT','R&D','Internet','Gini']
_resolved   = [(pick_col(REGIONS,c), lbl, ben) for (c,ben),lbl in zip(TOPSIS_CRIT_CANDS,TOPSIS_LBL_ALL)]
TOPSIS_CRIT = [c for c,_,_ in _resolved if c is not None]
TOPSIS_LBL  = [lbl for c,lbl,_ in _resolved if c is not None]
IS_BENEFIT  = np.array([ben for c,_,ben in _resolved if c is not None])

def _topsis(X, w, isb):
    R  = X / np.sqrt((X**2).sum(0))
    V  = R * w
    As = np.where(isb, V.max(0), V.min(0))
    An = np.where(isb, V.min(0), V.max(0))
    Ss = np.sqrt(((V-As)**2).sum(1))
    Sn = np.sqrt(((V-An)**2).sum(1))
    return Sn/(Ss+Sn)

def _entropy_w(X):
    P = X/X.sum(0); k = 1/np.log(len(X))
    E = -k*np.nansum(P*np.log(P+1e-12),0); d = 1-E
    return d/d.sum()

def _solve_lp4(with_equity=True):
    import pulp
    gv,lam = 0.002,0.6
    m = pulp.LpProblem('LP4',pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x',(REG,ITEMS),lowBound=0)
    m += pulp.lpSum(BETA_RJ[(r,j)]*x[r][j] for r in REG for j in ITEMS)
    m += pulp.lpSum(x[r][j] for r in REG for j in ITEMS) <= 50000
    for r in REG:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]['H'] for r in REG) >= 12000
    if with_equity:
        M = pulp.LpVariable('Dmax')
        for r in REG:
            m += D0_REG[r]+gv*x[r]['D'] <= M
            m += D0_REG[r]+gv*x[r]['D'] >= lam*M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat = np.array([[x[r][j].value() for j in ITEMS] for r in REG])
    return mat, pulp.value(m.objective)

J10 = ['I','D','AI','H']
S10 = ['s1','s2','s3','s4']
P_S = {'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05}
BETA_BASE = {'I':1.00,'D':1.10,'AI':1.25,'H':0.95}
BETA_S = {('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
          ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
          ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
          ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10}

def _solve_sp():
    from scipy.optimize import linprog
    n = 4+16; c = np.zeros(n)
    for k,j in enumerate(J10): c[k] = -BETA_BASE[j]
    for si,s in enumerate(S10):
        for k,j in enumerate(J10): c[4+si*4+k] = -P_S[s]*BETA_S[(s,j)]
    A_ub=[]; b_ub=[]
    A_ub.append([1,1,1,1]+[0]*16); b_ub.append(65000)
    for si in range(4):
        row=[0]*n
        for k in range(4): row[4+si*4+k]=1
        A_ub.append(row); b_ub.append(15000)
    for si in range(4):
        row=[0]*n; row[3]=-0.5; row[4+si*4+2]=1
        A_ub.append(row); b_ub.append(0)
    res = linprog(c,A_ub=np.array(A_ub),b_ub=np.array(b_ub),bounds=[(0,None)]*n,method="highs")
    x_sp = res.x[:4]; Z_sp = -res.fun
    y_sp = {s:res.x[4+i*4:4+i*4+4] for i,s in enumerate(S10)}
    det={}
    for s in S10:
        cs=np.zeros(8)
        for k,j in enumerate(J10): cs[k]=-BETA_BASE[j]; cs[4+k]=-BETA_S[(s,j)]
        r=linprog(cs,A_ub=[[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1],[0,0,0,-0.5,0,0,1,0]],
                  b_ub=[65000,15000,0],bounds=[(0,None)]*8,method="highs")
        det[s]={"Z":-r.fun,"x":r.x[:4]}
    Z_ws = sum(P_S[s]*det[s]["Z"] for s in S10)
    beta_avg = {j:sum(P_S[s]*BETA_S[(s,j)] for s in S10) for j in J10}
    cev = [-beta_avg[j] for j in J10]
    rev = linprog(cev,A_ub=[[1,1,1,1]],b_ub=[65000],bounds=[(0,None)]*4,method="highs")
    x_ev = rev.x
    Z_ev = sum(BETA_BASE[j]*x_ev[k] for k,j in enumerate(J10))
    for s in S10:
        cs=[-BETA_S[(s,j)] for j in J10]
        r=linprog(cs,A_ub=[[1,1,1,1],[0,0,1,0]],b_ub=[15000,0.5*x_ev[3]],bounds=[(0,None)]*4,method="highs")
        Z_ev += P_S[s]*(-r.fun)
    return x_sp,y_sp,Z_sp,Z_ev,Z_ws,det


# ============================================================
# TRANG CHỦ
# ============================================================
def page_home():
    st.markdown("""
    <div class='hero-banner'>
      <span class='hero-badge'>🇻🇳 Dữ liệu thực 2020-2025</span>
      <span class='hero-badge' style='background:rgba(46,196,182,.12);border-color:rgba(46,196,182,.3);color:#2ec4b6 !important;'>Python · Optimization · RL</span>
      <div class='hero-title' style='margin-top:14px;'>AIDEOM<span style='color:#e63946;'>-VN</span></div>
      <div style='font-size:1.05rem;color:#a8bbd4;margin-bottom:6px;'>AI-Driven Decision Optimization Model for Vietnam</div>
      <div style='font-size:.9rem;color:#5e7a99;'>
        Bộ bài tập thực hành · 12 mô hình ra quyết định · Phát triển kinh tế trong kỷ nguyên AI
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    kpi_row([
        (c1, "GDP 2025", "514,0 tỷ USD", "8,02%/năm"),
        (c2, "Kinh tế số / GDP", "≈ 19,5%", "+1,2 điểm %"),
        (c3, "FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%"),
        (c4, "GDP/người 2025", "5.026 USD", "+6,9%"),
    ])

    st.markdown("<div class='vn-divider'></div>", unsafe_allow_html=True)

    # ── Bản đồ 12 bài theo 4 cấp độ ──
    section_header("📚", "12 Bài toán theo 4 cấp độ", "Từ cơ bản đến nâng cao — dữ liệu thực Việt Nam")

    # ── 4 cấp độ — st.columns + st.button để chuyển trang khi click ──
    _BAI_DATA = [
        # (page_idx, bai_code, title, desc, tools, color)
        (1, "Bài 1", "Hàm sản xuất Cobb-Douglas", "TFP · Growth Accounting · Dự báo GDP 2030", "numpy · pandas · scipy", "#16a34a"),
        (2, "Bài 2", "LP Ngân sách số", "4 hạng mục · shadow price · sensitivity", "scipy · pulp", "#16a34a"),
        (3, "Bài 3", "Chỉ số ưu tiên 10 ngành", "Min-max normalization · weighted scoring", "numpy · pandas", "#16a34a"),
        (4, "Bài 4", "LP ngành-vùng 24 biến", "6 vùng x 4 hạng mục · công bằng vùng miền", "pulp · cvxpy", "#ca8a04"),
        (5, "Bài 5", "MIP 15 dự án", "Biến nhị phân · precedence · đa năm", "pulp CBC", "#ca8a04"),
        (6, "Bài 6", "TOPSIS 6 vùng", "Entropy weight · AHP · phân tích độ nhạy", "numpy · scikit", "#ca8a04"),
        (7, "Bài 7", "NSGA-II Pareto", "4 mục tiêu: GDP · bao trùm · môi trường · an ninh", "pymoo", "#ea580c"),
        (8, "Bài 8", "Tối ưu động 2026-2035", "K,D,AI,H,Y · CRRA utility · cú sốc TFP", "scipy SLSQP", "#ea580c"),
        (9, "Bài 9", "Lao động & AI NetJob", "8 ngành · Sankey flow · ngưỡng đào tạo lại", "cvxpy · plotly", "#ea580c"),
        (10, "Bài 10", "Quy hoạch ngẫu nhiên", "VSS · EVPI · Robust minimax · 4 kịch bản", "pyomo · scipy", "#dc2626"),
        (11, "Bài 11", "Q-learning RL", "MDP 81 trạng thái · 5 hành động · 8000 eps", "gymnasium", "#dc2626"),
        (12, "Bài 12", "AIDEOM-VN tích hợp", "6 module M1-M6 · 5 kịch bản · dashboard", "streamlit", "#dc2626"),
    ]

    _LEVEL_GROUPS = [
        ("#16a34a", "DỄ",         "Bài 1–3",  _BAI_DATA[0:3]),
        ("#ca8a04", "TRUNG BÌNH", "Bài 4–6",  _BAI_DATA[3:6]),
        ("#ea580c", "KHÁ KHÓ",   "Bài 7–9",  _BAI_DATA[6:9]),
        ("#dc2626", "KHÓ",       "Bài 10–12", _BAI_DATA[9:12]),
    ]

    for _color, _lvl, _brange, _group in _LEVEL_GROUPS:
        # Level header
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"background:#0f1d34;border:1px solid #1e3050;border-left:4px solid {_color};"
            f"border-radius:12px;padding:10px 18px;margin:10px 0 6px;'>"
            f"<div style='width:9px;height:9px;border-radius:50%;background:{_color};flex-shrink:0;'></div>"
            f"<span style='font-weight:700;color:#fff;font-size:.93rem;'>Cấp độ {_lvl}</span>"
            f"<span style='margin-left:auto;font-size:.76rem;color:#5e7a99;'>{_brange}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        # 3 cards as columns with real st.button
        _cols = st.columns(3)
        for _ci, (_pidx, _bcode, _title, _desc, _tools, _c) in enumerate(_group):
            with _cols[_ci]:
                st.markdown(
                    f"<div style='background:#0a1628;border:1px solid #1e3050;"
                    f"border-top:3px solid {_c};border-radius:12px;padding:14px 16px 10px;'>"
                    f"<div style='display:inline-block;background:{_c}22;color:{_c};"
                    f"border-radius:999px;padding:2px 12px;font-size:.72rem;font-weight:700;"
                    f"text-transform:uppercase;margin-bottom:8px;'>{_bcode}</div>"
                    f"<div style='font-size:.9rem;font-weight:700;color:#fff;margin-bottom:5px;'>{_title}</div>"
                    f"<div style='font-size:.77rem;color:#a8bbd4;margin-bottom:8px;line-height:1.5;'>{_desc}</div>"
                    f"<div style='font-size:.69rem;color:#2ec4b6;font-family:monospace;margin-bottom:10px;'>{_tools}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if st.button(f"Mở {_bcode} →", key=f"home_btn_{_pidx}", use_container_width=True):
                    st.session_state.page_idx = _pidx
                    st.rerun()


    st.markdown("<div class='vn-divider'></div>", unsafe_allow_html=True)

    # ── Dữ liệu gốc ──
    section_header("📂", "Bộ dữ liệu thực Việt Nam", "3 tệp CSV · NSO · MoST · WB · GII 2025")
    t1,t2,t3 = st.tabs(["📊 Vĩ mô 2020-2025","🏭 10 ngành 2024","🗺️ 6 vùng KT-XH 2024"])
    with t1:
        st.dataframe(MACRO, use_container_width=True)
        fig = px.line(MACRO,x="year",y="GDP_trillion_VND",
                      **_px_cfg("GDP Việt Nam 2020–2025 (nghìn tỷ VND)"))
        fig.update_traces(fill="tozeroy", line_color="#e63946",
                          fillcolor="rgba(230,57,70,0.15)")
        st.plotly_chart(fig,use_container_width=True)
    with t2:
        st.dataframe(SECTORS,use_container_width=True)
    with t3:
        st.dataframe(REGIONS,use_container_width=True)


# ============================================================
# BÀI 1 — COBB-DOUGLAS
# ============================================================
def page_bai1():
    section_header("📐","Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI & Số hóa",
                   "Growth accounting · TFP Solow residual · Dự báo GDP 2030")

    info_box("""
    <b>Bối cảnh:</b> GDP Việt Nam 2024 đạt 11.511,9 nghìn tỷ VND (+7,09%). Mô hình hóa bằng
    hàm sản xuất Cobb-Douglas mở rộng thêm yếu tố số hóa D, năng lực AI và vốn nhân lực số H.
    Điều kiện lợi suất không đổi theo quy mô: α + β + γ + δ + θ = 1.
    """)

    st.latex(r"Y_t = A_t \cdot K_t^{\alpha} \cdot L_t^{\beta} \cdot D_t^{\gamma} \cdot AI_t^{\delta} \cdot H_t^{\theta}")
    st.latex(r"\alpha=0{,}33;\;\beta=0{,}42;\;\gamma=0{,}10;\;\delta=0{,}08;\;\theta=0{,}07")

    a,b,g,d,th = COEF["alpha"],COEF["beta"],COEF["gamma"],COEF["delta"],COEF["theta"]
    years = MACRO["year"].values
    Y = MACRO["GDP_trillion_VND"].values
    K,L,D,AI,H = K_HIST,L_HIST,D_HIST,AI_HIST,H_HIST

    A      = Y/(K**a*L**b*D**g*AI**d*H**th)
    A_mean = A.mean()
    Y_hat  = A_mean*(K**a*L**b*D**g*AI**d*H**th)
    mape   = np.mean(np.abs((Y-Y_hat)/Y))*100

    c1,c2,c3,c4 = st.columns(4)
    kpi_row([
        (c1,"MAPE (Cobb-Douglas)",f"{mape:.2f}%"),
        (c2,"Ā (TFP trung bình)",f"{A_mean:.4f}"),
        (c3,"TFP tăng trưởng",f"{((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm"),
        (c4,"Y dự báo 2025",f"{Y_hat[-1]:,.0f} ng.tỷ"),
    ])

    tab1,tab2,tab3,tab4 = st.tabs(["📌 1.4.1 TFP A_t","📊 1.4.2 Dự báo & MAPE",
                                    "📈 1.4.3 Phân rã tăng trưởng","🔭 1.4.4 Dự báo 2030"])

    with tab1:
        step(1,"Tính TFP (Solow residual)","giải ngược từ hàm sản xuất")
        st.latex(r"A_t = \frac{Y_t}{K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta}}")
        dfA = pd.DataFrame({"Năm":years,"Y thực (ng.tỷ)":Y.round(1),
                            "K (ng.tỷ)":K,"L (tr.người)":L,"D (%)":D,
                            "AI (ng.DN)":AI,"H (%)":H,"TFP A_t":A.round(4)})
        st.dataframe(dfA,use_container_width=True,hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years,y=A,mode="lines+markers",name="TFP A_t",
                                 line=dict(color="#2ec4b6",width=3),
                                 marker=dict(size=9,color="#2ec4b6",
                                             line=dict(color="#fff",width=2))))
        fig.add_hrect(y0=A_mean*0.99,y1=A_mean*1.01,fillcolor="rgba(230,57,70,0.1)",
                      line_width=0,annotation_text=f"Ā = {A_mean:.4f}")
        fig.update_layout(**_plot_cfg("Năng suất nhân tố tổng hợp A_t (2020-2025)"))
        st.plotly_chart(fig,use_container_width=True)
        info_box("TFP phản ánh chất lượng tăng trưởng. Xu hướng <b>tăng dần</b> cho thấy Việt Nam đang nâng cao hiệu quả sử dụng các yếu tố đầu vào, đặc biệt giai đoạn hậu COVID-19 2022-2025.")

    with tab2:
        step(2,"Dùng Ā trung bình","để dự báo và tính MAPE")
        dff = pd.DataFrame({"Năm":years,"Y thực (ng.tỷ)":Y.round(1),
                            "Y dự báo":Y_hat.round(1),
                            "Sai số tuyệt đối":np.abs(Y-Y_hat).round(1),
                            "Sai số %":((Y_hat-Y)/Y*100).round(2)})
        st.dataframe(dff,use_container_width=True,hide_index=True)
        fig = go.Figure()
        fig.add_bar(x=years,y=Y,name="Y thực tế",marker_color="#4895ef",opacity=0.8)
        fig.add_trace(go.Scatter(x=years,y=Y_hat,name="Y dự báo (Ā cố định)",
                                 mode="lines+markers",line=dict(color="#e63946",width=2.5,dash="dot"),
                                 marker=dict(size=8,color="#e63946")))
        fig.update_layout(**_plot_cfg(f"Dự báo Cobb-Douglas vs Thực tế — MAPE = {mape:.2f}%"))
        st.plotly_chart(fig,use_container_width=True)
        st.success(f"✅ MAPE = **{mape:.2f}%** — Mô hình dự báo tốt (< 5% được coi là chấp nhận được trong kinh tế lượng vĩ mô)")

    with tab3:
        step(3,"Phân rã tăng trưởng","(growth accounting) 2020→2025")
        st.latex(r"\Delta\ln Y = \Delta\ln A + \alpha\Delta\ln K + \beta\Delta\ln L + \gamma\Delta\ln D + \delta\Delta\ln AI + \theta\Delta\ln H")
        gs = {
            "TFP (A)":   (np.log(A[-1])-np.log(A[0]))/5,
            "Vốn (K)":   a*(np.log(K[-1])-np.log(K[0]))/5,
            "Lao động (L)": b*(np.log(L[-1])-np.log(L[0]))/5,
            "Số hóa (D)": g*(np.log(D[-1])-np.log(D[0]))/5,
            "AI capacity": d*(np.log(AI[-1])-np.log(AI[0]))/5,
            "Nhân lực số (H)": th*(np.log(H[-1])-np.log(H[0]))/5,
        }
        total = sum(gs.values())
        dec = pd.DataFrame({"Yếu tố":list(gs.keys()),
                            "Đóng góp (%/năm)":[v*100 for v in gs.values()],
                            "Tỷ trọng (%)":[v/total*100 for v in gs.values()]}).round(3)
        st.dataframe(dec,use_container_width=True,hide_index=True)
        fig = px.bar(dec,x="Đóng góp (%/năm)",y="Yếu tố",orientation="h",
                     color="Đóng góp (%/năm)",color_continuous_scale=["#2ec4b6","#e63946"],
                     **_px_cfg("Phân rã đóng góp tăng trưởng GDP 2020-2025 (điểm %/năm)"))
        fig.update_layout(yaxis=dict(autorange="reversed"),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        info_box(f"<b>Nhận xét:</b> Vốn vật chất (K) đóng góp lớn nhất ({gs['Vốn (K)']*100:.2f}%/năm), nhưng đóng góp của Số hóa + AI + Nhân lực số chiếm {(gs['Số hóa (D)']+gs['AI capacity']+gs['Nhân lực số (H)'])*100:.2f}%/năm — xu hướng tốt cho mô hình tăng trưởng chất lượng cao.")

    with tab4:
        step(4,"Mô phỏng kịch bản 2030","theo giả định chính sách")
        col1,col2,col3 = st.columns(3)
        D30  = col1.slider("D (Kinh tế số % GDP) 2030",15.0,40.0,30.0,0.5)
        AI30 = col2.slider("AI (nghìn DN số) 2030",80,150,100,5)
        H30  = col3.slider("H (LĐ qua ĐT %) 2030",25.0,45.0,35.0,0.5)
        g_KL = col1.slider("Tăng trưởng K & L mỗi năm (%)",3.0,10.0,6.0,0.5)/100
        g_TFP= col2.slider("Tăng trưởng TFP mỗi năm (%)",0.5,3.0,1.2,0.1)/100
        n_yr = 5  # 2025->2030
        K30  = K[-1]*(1+g_KL)**n_yr
        L30  = L[-1]*(1+g_KL)**n_yr
        A30  = A_mean*(1+g_TFP)**n_yr
        Y30  = A30*K30**a*L30**b*D30**g*AI30**d*H30**th
        growth_pa = ((Y30/Y[-1])**(1/n_yr)-1)*100
        c1,c2 = st.columns(2)
        kpi_row([
            (c1,"GDP dự báo 2030",f"{Y30:,.0f} ng.tỷ VND",f"+{growth_pa:.2f}%/năm"),
            (c2,"GDP/người 2030 ước tính",f"{Y30/103:.0f} ng.tỷ/ng"),
        ])
        # Trajectory chart
        yrs30 = list(range(2025,2031))
        traj = []
        for t,yr in enumerate(yrs30):
            Kt = K[-1]*(1+g_KL)**t; Lt = L[-1]*(1+g_KL)**t
            Dt = D[-1]+(D30-D[-1])/5*t; At = AI[-1]+(AI30-AI[-1])/5*t
            Ht = H[-1]+(H30-H[-1])/5*t; At_=A_mean*(1+g_TFP)**t
            traj.append(At_*Kt**a*Lt**b*Dt**g*At**d*Ht**th)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(years)+yrs30[1:],y=list(Y)+traj[1:],
                                 mode="lines+markers",
                                 line=dict(color="#e63946",width=3),
                                 marker=dict(size=7),name="GDP thực + dự báo"))
        fig.add_vline(x=2025,line_dash="dash",line_color="#5e7a99",
                      annotation_text="2025→2030 dự báo")
        fig.update_layout(**_plot_cfg("Quỹ đạo GDP Việt Nam 2020→2030"))
        st.plotly_chart(fig,use_container_width=True)
        if growth_pa>=7:
            st.success(f"✅ Tốc độ tăng trưởng {growth_pa:.2f}%/năm — đạt mục tiêu thu nhập trung bình cao 2030")
        else:
            st.warning(f"⚠️ Tốc độ tăng trưởng {growth_pa:.2f}%/năm — cần tăng đầu tư số hóa & AI")

    # ── Câu hỏi thảo luận ──
    st.markdown("<div class='vn-divider'></div>", unsafe_allow_html=True)
    section_header("💬","Câu hỏi thảo luận chính sách")
    q_cols = st.columns(3)
    for col,txt in zip(q_cols,[
        "**a)** TFP Việt Nam có xu hướng tăng hay giảm 2020-2025? Điều đó nói lên gì về chất lượng tăng trưởng?",
        "**b)** Trong D, AI, H — yếu tố nào đóng góp nhiều nhất? Tại sao?",
        "**c)** Mục tiêu 30% kinh tế số/GDP vào 2030 có khả thi không nếu dựa trên mô hình này?"
    ]):
        col.markdown(f"""
        <div style='background:var(--bg-panel);border:1px solid var(--border);
             border-radius:var(--radius);padding:16px;font-size:.88rem;color:var(--txt-md);
             line-height:1.6;height:100%;'>{txt}</div>""", unsafe_allow_html=True)


# ============================================================
# BÀI 2 — LP NGÂN SÁCH
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    section_header("💰","Bài 2 — LP phân bổ ngân sách số 4 hạng mục",
                   "scipy.optimize + PuLP · Shadow price · Sensitivity analysis")

    info_box("""
    <b>Bối cảnh:</b> Bộ KH-ĐT đề xuất phân bổ 100.000 tỷ VND cho 4 hạng mục:
    x₁ = Hạ tầng số · x₂ = AI & Dữ liệu · x₃ = Nhân lực số · x₄ = R&D Công nghệ.
    <b>Mục tiêu:</b> tối đa hóa GDP tăng thêm kỳ vọng.
    """)

    c1,c2 = st.columns(2)
    c1.markdown("""
    **Hàm mục tiêu:**
    """)
    c1.latex(r"\max Z = 0{,}85x_1 + 1{,}20x_2 + 0{,}95x_3 + 1{,}35x_4")
    c2.markdown("""
    **Ràng buộc chính:**
    """)
    c2.latex(r"x_1+x_2+x_3+x_4 \leq 100;\quad x_1\geq25;\; x_2\geq15;\; x_3\geq20;\; x_4\geq10")

    def solve(B=100,x3_min=20):
        c_obj=[-0.85,-1.20,-0.95,-1.35]
        A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
        b_ub=[B,-25,-15,-x3_min,-10,0]
        return linprog(c_obj,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method="highs")

    tab1,tab2,tab3,tab4 = st.tabs(["🔑 2.4.1 Lời giải tối ưu","💡 2.4.2 Shadow price",
                                    "📉 2.4.3 Phân tích độ nhạy","⚙️ 2.4.4 Ràng buộc nhân lực"])

    with tab1:
        res = solve()
        x = res.x; Z = -res.fun
        labels = ["x₁ Hạ tầng số","x₂ AI & Dữ liệu","x₃ Nhân lực số","x₄ R&D"]
        coeffs = [0.85,1.20,0.95,1.35]
        c1,c2,c3 = st.columns(3)
        kpi_row([(c1,"GDP gain Z*",f"{Z:.2f} ng.tỷ"),(c2,"x₄ R&D (tối ưu)",f"{x[3]:.0f} ng.tỷ"),
                 (c3,"x₂ AI (tối ưu)",f"{x[1]:.0f} ng.tỷ")])
        df = pd.DataFrame({"Hạng mục":labels,"Phân bổ (ng.tỷ)":x.round(1),
                           "Hệ số tác động":coeffs,
                           "GDP đóng góp":(x*coeffs).round(2),
                           "% ngân sách":x/x.sum()*100})
        st.dataframe(df.style.format({"% ngân sách":"{:.1f}%"}),
                     use_container_width=True,hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=labels,x=x,orientation="h",name="Phân bổ",
                             marker_color=["#4895ef","#e63946","#2ec4b6","#f4a261"]))
        fig.update_layout(**_plot_cfg(f"Phân bổ tối ưu Z* = {Z:.2f} ng.tỷ VND"))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        info_box("""
        <b>Shadow price</b> (giá đối ngẫu) của một ràng buộc = lượng tăng của Z* khi nới lỏng ràng buộc đó 1 đơn vị.
        R&D có hệ số cao nhất (1,35) nhưng ràng buộc tối thiểu chỉ 10 vì chi phí triển khai thực tế cao và trễ hàng năm.
        """)
        budgets_sp = [100,101,102]
        sp_vals = [-solve(B=b).fun for b in budgets_sp]
        sp_val = sp_vals[1]-sp_vals[0]
        c_sp1, c_sp2 = st.columns(2)
        kpi_row([(c_sp1, "Shadow price (ngân sách +1 ng.tỷ)", f"{sp_val:.4f} ng.tỷ GDP")])
        st.markdown("**Giải thích:** Mỗi 1 nghìn tỷ VND ngân sách thêm → GDP tăng thêm kỳ vọng ≈ "
                    f"**{sp_val*1000:,.0f} tỷ VND** — đây là chi phí cơ hội của vốn công ngành số.")

    with tab3:
        Bs = range(80,161,10)
        Zs = [-solve(B=b).fun if solve(B=b).success else None for b in Bs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(Bs),y=Zs,mode="lines+markers",
                                 line=dict(color="#e63946",width=3),
                                 marker=dict(size=8,color="#f4a261",
                                             line=dict(color="#fff",width=2)),
                                 name="Z*(B)"))
        fig.update_layout(**_plot_cfg("Đường cong Z*(B) — nhạy cảm với ngân sách"))
        fig.update_xaxes(title_text="Ngân sách B (nghìn tỷ VND)")
        fig.update_yaxes(title_text="Z* (nghìn tỷ GDP gain)")
        st.plotly_chart(fig,use_container_width=True)

    with tab4:
        x3_new = st.slider("Ràng buộc tối thiểu x₃ (nhân lực số, ng.tỷ)",20,50,30,5)
        res_new = solve(x3_min=x3_new)
        res_base= solve()
        if res_new.success:
            st.success(f"✅ Bài toán khả thi · Z* = {-res_new.fun:.2f} ng.tỷ "
                       f"(giảm {(-res_base.fun)-(-res_new.fun):.2f} so với x₃≥20)")
            df2 = pd.DataFrame({"Hạng mục":["x₁","x₂","x₃","x₄"],
                                 "Phân bổ cơ sở (x₃≥20)":res_base.x.round(1),
                                 f"Phân bổ (x₃≥{x3_new})":res_new.x.round(1)})
            st.dataframe(df2,use_container_width=True,hide_index=True)
        else:
            st.error("❌ Bài toán KHÔNG khả thi với ràng buộc này.")


# ============================================================
# BÀI 3 — PRIORITY 10 NGÀNH
# ============================================================
def page_bai3():
    section_header("🎯","Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành Việt Nam",
                   "Min-max normalization · Weighted scoring · Sensitivity heatmap")

    info_box("""
    <b>Công thức:</b> Priority_i = a₁·Growth + a₂·Productivity + a₃·Spillover + a₄·Export + a₅·Employment + a₆·AIReadiness − a₇·Risk
    <br>Chuẩn hóa min-max về [0,1]; riêng <b>Risk</b> đảo chiều (rủi ro thấp = ưu tiên cao hơn).
    """)

    df = SECTORS.copy()
    c_name   = pick_col(df,["sector_name_vi","sector_name_en"])
    c_growth = pick_col(df,["growth_rate_2024_pct","growth_rate"])
    c_share  = pick_col(df,["gdp_share_2024_pct","gdp_share"])
    c_spill  = pick_col(df,["spillover_coef_0_1","spillover"])
    c_exp    = pick_col(df,["export_billion_USD","export"])
    c_lab    = pick_col(df,["labor_million","labor"])
    c_ai     = pick_col(df,["ai_readiness_0_100","ai_readiness"])
    c_risk   = pick_col(df,["automation_risk_pct","automation_risk"])
    GDP24 = 11511.9
    df["labor_productivity"] = (df[c_share]/100)*GDP24/df[c_lab]
    cols_good = [c_growth,"labor_productivity",c_spill,c_exp,c_lab,c_ai]
    # cols_good has 6 columns: [growth, productivity, spillover, export, labor, ai_readiness]
    # Index of AI readiness in cols_good = 5
    def norm_g(x): return (x-x.min())/(x.max()-x.min())
    def norm_b(x): return (x.max()-x)/(x.max()-x.min())
    Xg = df[cols_good].apply(norm_g); Xb = norm_b(df[c_risk])

    # Default weights: 6 "good" weights + 1 risk weight
    # w_raw[5] = AI readiness weight = 0.20
    w_raw = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])  # length 6
    wr    = 0.15   # risk weight
    # Normalize so all weights sum to 1
    w_default = w_raw / (w_raw.sum() + wr)
    wr_default = wr / (w_raw.sum() + wr)

    tab0,tab1,tab2,tab3 = st.tabs(["🔢 3.4.1 Ma trận chuẩn hóa","🏅 3.4.2 Xếp hạng mặc định",
                                    "🌡️ 3.4.3 Độ nhạy w_AI","⚖️ 3.4.4 Hai bộ trọng số"])
    with tab0:
        step(1,"Chuẩn hóa min-max","— mỗi chỉ số về thang [0, 1]")
        norm = Xg.copy()
        norm.columns = ["Tăng trưởng","Năng suất LĐ","Lan tỏa","Xuất khẩu","Việc làm","AI Ready"]
        norm.insert(0,"Ngành",df[c_name].values)
        norm["Risk (đảo)"] = Xb.values
        st.dataframe(norm.round(4),use_container_width=True,hide_index=True)
        fig = px.imshow(norm.iloc[:,1:].values,y=df[c_name].values,
                        x=["Tăng trưởng","Năng suất","Lan tỏa","XK","LĐ","AI","Risk↓"],
                        color_continuous_scale="RdYlGn",aspect="auto",
                        **_px_cfg("Heatmap ma trận chuẩn hóa 10 ngành"))
        st.plotly_chart(fig,use_container_width=True)

    with tab1:
        pr = Xg.values @ w_default + wr_default * Xb.values
        rk = pd.DataFrame({"Hạng":range(1,11),"Ngành":df[c_name],"Priority":pr.round(4)}) \
               .sort_values("Priority",ascending=False).reset_index(drop=True)
        rk["Hạng"] = range(1,11)
        cols = st.columns([2,3])
        with cols[0]:
            st.dataframe(rk,use_container_width=True,hide_index=True)
        with cols[1]:
            fig = px.bar(rk,x="Priority",y="Ngành",orientation="h",
                         color="Priority",color_continuous_scale=["#1e3050","#e63946"],
                         **_px_cfg("Xếp hạng ưu tiên 10 ngành (bộ trọng số mặc định)"))
            fig.update_layout(yaxis=dict(autorange="reversed"),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        top3 = rk.head(3)["Ngành"].tolist()
        info_box(f"<b>Top-3 ưu tiên:</b> {' · '.join(top3)} — phù hợp với Nghị quyết 57-NQ/TW về đột phá KH-CN và chuyển đổi số.")

    with tab2:
        # Sensitivity: vary w_AI (weight of AI readiness, index 5 in cols_good)
        # Keep wr (risk) fixed, redistribute remaining weight among other 5 cols
        rng_vals = np.arange(0.05, 0.45, 0.05)
        heat = []
        w_others = w_raw[:5]  # first 5 weights (exclude AI readiness)
        for wai in rng_vals:
            rem = 1.0 - wai - wr          # remaining for other 5 cols + risk already taken
            # Scale the 5 non-AI weights proportionally
            wsc = w_others * (rem / w_others.sum())  # shape (5,)
            w_full = np.append(wsc, wai)             # shape (6,) — matches Xg columns
            # Normalize full weight vector (including risk) to sum=1
            total = w_full.sum() + wr
            w_full_n = w_full / total
            wr_n = wr / total
            heat.append(Xg.values @ w_full_n + wr_n * Xb.values)
        heat = np.array(heat)
        fig = px.imshow(heat,x=[f"N{i+1}" for i in range(10)],
                        y=[f"{w:.2f}" for w in rng_vals],aspect="auto",
                        color_continuous_scale="YlOrRd",
                        **_px_cfg("Heatmap Priority theo w_AI (0.05→0.40)"))
        fig.update_layout(xaxis_title="Ngành",yaxis_title="Trọng số w_AI")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("N1..N10 = "+", ".join(f"N{i+1}:{n}" for i,n in enumerate(df[c_name])))
        info_box("Ngành <b>CNTT-TT (N8)</b> và <b>Tài chính-NH (N6)</b> duy trì top-3 bất kể w_AI — chứng tỏ <b>tính bền vững</b> của thứ hạng này.")

    with tab3:
        wg = np.array([0.25,0.25,0.10,0.25,0.05,0.05]); wg_r=0.05
        wi = np.array([0.05,0.10,0.25,0.05,0.25,0.10]); wi_r=0.20
        pg = Xg.values@wg+wg_r*Xb.values
        pi = Xg.values@wi+wi_r*Xb.values
        comp = pd.DataFrame({"Ngành":df[c_name],"Priority Tăng trưởng":pg.round(4),
                             "Priority Bao trùm":pi.round(4)})
        cc = st.columns(2)
        cc[0].markdown("**🏆 Top-3 — Định hướng Tăng trưởng** (ưu tiên xuất khẩu, năng suất)")
        cc[0].dataframe(comp.nlargest(3,"Priority Tăng trưởng")[["Ngành","Priority Tăng trưởng"]],
                        hide_index=True,use_container_width=True)
        cc[1].markdown("**🤝 Top-3 — Định hướng Bao trùm** (ưu tiên việc làm, lan tỏa)")
        cc[1].dataframe(comp.nlargest(3,"Priority Bao trùm")[["Ngành","Priority Bao trùm"]],
                        hide_index=True,use_container_width=True)
        fig = go.Figure()
        fig.add_bar(y=comp["Ngành"],x=comp["Priority Tăng trưởng"],name="Tăng trưởng",
                    orientation="h",marker_color="#4895ef",opacity=0.85)
        fig.add_bar(y=comp["Ngành"],x=comp["Priority Bao trùm"],name="Bao trùm",
                    orientation="h",marker_color="#e63946",opacity=0.85)
        fig.update_layout(**_plot_cfg("So sánh hai định hướng trọng số chính sách"),barmode="group")
        st.plotly_chart(fig,use_container_width=True)


# ============================================================
# BÀI 4 — LP NGÀNH-VÙNG
# ============================================================
def page_bai4():
    section_header("🗺️","Bài 4 — LP phân bổ ngân sách số ngành-vùng",
                   "24 biến · 25 ràng buộc · công bằng vùng miền · PuLP vs CVXPY")

    info_box("""
    <b>Bối cảnh QĐ 411/QĐ-TTg:</b> Phân bổ 50.000 tỷ VND cho 6 vùng × 4 hạng mục đầu tư
    sao cho tối đa hóa GDP gain nhưng đảm bảo công bằng vùng miền (C5: chỉ số số hóa vùng yếu nhất ≥ 70% vùng tốt nhất).
    """)

    eq = _solve_lp4(True); noeq = _solve_lp4(False)
    x_opt,Z = eq; _,Z_no = noeq
    equity_cost = Z_no-Z

    c1,c2,c3,c4 = st.columns(4)
    kpi_row([
        (c1,"Z* (có công bằng C5)",f"{Z:,.0f} tỷ"),
        (c2,"Z* (không công bằng)",f"{Z_no:,.0f} tỷ"),
        (c3,"Chi phí công bằng",f"{equity_cost:,.0f} tỷ",None,True),
        (c4,"Ngân sách sử dụng",f"50.000 tỷ"),
    ])

    tab1,tabc,tab2 = st.tabs(["🗺️ 4.4.1-3 Phân bổ tối ưu","⚖️ 4.4.2 PuLP vs CVXPY",
                               "📊 4.4.4 Chi phí công bằng vùng"])
    with tab1:
        dfm = pd.DataFrame(x_opt,index=REGIONS_VI,columns=ITEMS)
        dfm["Tổng"] = dfm.sum(axis=1)
        dfm = dfm.round(0).astype(int)
        st.dataframe(dfm,use_container_width=True)
        fig = px.imshow(x_opt,x=ITEMS,y=REGIONS_VI,aspect="auto",text_auto=".0f",
                        color_continuous_scale="Blues",
                        **_px_cfg(f"Heatmap phân bổ tối ưu (Z* = {Z:,.0f} tỷ VND)"))
        st.plotly_chart(fig,use_container_width=True)
        per_region = x_opt.sum(axis=1)
        fig2 = go.Figure()
        for j,item in enumerate(ITEMS):
            fig2.add_trace(go.Bar(name=item,x=REGIONS_VI,y=x_opt[:,j],
                                  marker_color=PALETTE[j]))
        fig2.update_layout(**_plot_cfg("Cơ cấu đầu tư theo vùng"),barmode="stack")
        st.plotly_chart(fig2,use_container_width=True)

    with tabc:
        try:
            import cvxpy as cp
            beta_mat = np.array([[BETA_RJ[(r,j)] for j in ITEMS] for r in REG])
            xv = cp.Variable((6,4),nonneg=True)
            rs = cp.sum(xv,axis=1)
            D0v = np.array([D0_REG[r] for r in REG])
            Dn  = D0v + 0.002*xv[:,1]; Mc = cp.Variable()
            cons = [cp.sum(xv)<=50000,rs>=5000,rs<=12000,
                    cp.sum(xv[:,3])>=12000,Dn<=Mc,Dn>=0.6*Mc]
            prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta_mat,xv))),cons)
            for sv in ["CLARABEL","ECOS","SCS"]:
                try:
                    prob.solve(solver=getattr(cp,sv))
                    if prob.status and "optimal" in prob.status: break
                except: continue
            c1,c2 = st.columns(2)
            kpi_row([(c1,"Z* PuLP (CBC)",f"{Z:,.0f} tỷ"),(c2,"Z* CVXPY",f"{prob.value:,.0f} tỷ")])
            diff = abs(Z-(prob.value or 0))
            if diff < 10:
                st.success(f"✅ Chênh lệch PuLP vs CVXPY = {diff:.1f} tỷ → hai solver cho kết quả trùng khớp!")
            st.dataframe(pd.DataFrame(xv.value,index=REGIONS_VI,columns=ITEMS).round(0),
                         use_container_width=True)
        except ImportError:
            warn_box("Chưa cài CVXPY. Chạy: <code>pip install cvxpy</code>. PuLP Z* = " + f"{Z:,.0f} tỷ")

    with tab2:
        comp = pd.DataFrame({
            "Vùng":REGIONS_VI,
            "Có công bằng (C5)":eq[0].sum(axis=1).round(0),
            "Không công bằng":noeq[0].sum(axis=1).round(0)})
        fig = go.Figure()
        fig.add_bar(x=comp["Vùng"],y=comp["Có công bằng (C5)"],name="Có công bằng C5",
                    marker_color="#4895ef")
        fig.add_bar(x=comp["Vùng"],y=comp["Không công bằng"],name="Không công bằng",
                    marker_color="#e63946",opacity=0.7)
        fig.update_layout(**_plot_cfg("Ngân sách mỗi vùng: có vs không có ràng buộc công bằng"),
                          barmode="group")
        st.plotly_chart(fig,use_container_width=True)
        info_box(f"<b>Chi phí của công bằng = {equity_cost:,.0f} tỷ VND GDP gain</b> (~{equity_cost/Z*100:.1f}% Z*). "
                 f"Không có C5, vốn chảy mạnh về ĐNB & ĐBSH (β_AI cao nhất: 1,55 & 1,40). "
                 f"C5 bảo đảm Tây Nguyên và TDMN phía Bắc được nâng cấp số hóa tối thiểu.")


# ============================================================
# BÀI 5 — MIP 15 DỰ ÁN
# ============================================================
def page_bai5():
    from pulp import (LpProblem,LpMaximize,LpVariable,lpSum,value,PULP_CBC_CMD,LpStatus)
    section_header("🔢","Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số",
                   "Biến nhị phân · precedence constraints · ngân sách đa năm · PuLP CBC")

    info_box("""
    <b>Bối cảnh:</b> Bộ KH-CN xem xét 15 dự án cho Chương trình CĐS quốc gia 2026-2030.
    Ngân sách: 80.000 tỷ (năm 1-2: ≤40.000 tỷ). Ràng buộc tiên quyết: P8/P13 cần P12; P14 bắt buộc.
    """)

    P=list(range(1,16))
    C ={1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,
        9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1={1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,
        9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B ={1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,
        9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    names={1:'TT dữ liệu Hòa Lạc',2:'TT dữ liệu phía Nam',3:'5G toàn quốc',
           4:'VNeID 2.0',5:'Cổng DVC v3',6:'Y tế số',7:'Giáo dục số K-12',
           8:'TT AI + supercomputing',9:'Fintech sandbox',10:'Logistics thông minh',
           11:'Nông nghiệp số ĐBSCL',12:'Đào tạo 50K kỹ sư AI',
           13:'Khu CN bán dẫn BN-BG',14:'An ninh mạng SOC',15:'Open Data'}
    fields={1:'ht',2:'ht',3:'ht',4:'cp',5:'cp',6:'yt',7:'gd',8:'ai',
            9:'tc',10:'lg',11:'nn',12:'nl',13:'bd',14:'an',15:'dl'}
    prob_p={'ht':.85,'cp':.75,'ai':.65,'bd':.65,'yt':.8,'gd':.8,'tc':.8,
            'lg':.8,'nn':.8,'nl':.8,'an':.8,'dl':.8}

    def solve_mip(BT=80000,B12=40000,use_exp=False,force12=False):
        m=LpProblem('sel',LpMaximize)
        y=LpVariable.dicts('y',P,cat='Binary')
        m+=lpSum((prob_p[fields[i]] if use_exp else 1)*B[i]*y[i] for i in P)
        m+=lpSum(C[i]*y[i] for i in P)<=BT
        m+=lpSum(C1[i]*y[i] for i in P)<=B12
        if force12: m+=y[1]>=1; m+=y[2]>=1
        else:       m+=y[1]+y[2]<=1
        m+=y[8]<=y[12]; m+=y[13]<=y[12]
        m+=y[4]+y[5]>=1; m+=y[14]>=1
        m+=lpSum(y[i] for i in P)>=7; m+=lpSum(y[i] for i in P)<=11
        m.solve(PULP_CBC_CMD(msg=False))
        sel=[i for i in P if y[i].value()>0.5]
        return sel,sum(C[i] for i in sel),value(m.objective),LpStatus[m.status]

    tab1,tab2,tabf,tab3 = st.tabs(["📋 5.4.1 Lời giải cơ sở","💸 5.4.2 Nới ngân sách",
                                    "🔗 5.4.3 Bắt buộc P1+P2","🎲 5.4.4 Lợi ích kỳ vọng"])
    with tab1:
        sel,tc,Z,_ = solve_mip()
        c1,c2,c3,c4 = st.columns(4)
        kpi_row([(c1,"Dự án được chọn",f"{len(sel)}/15"),
                 (c2,"Tổng chi phí",f"{tc:,} tỷ"),
                 (c3,"Tổng lợi ích Z*",f"{Z:,.0f} tỷ"),
                 (c4,"ROI (B/C ratio)",f"{Z/tc:.2f}×")])
        df_sel = pd.DataFrame([{"Mã":f"P{i}","Tên dự án":names[i],
                                 "Lĩnh vực":fields[i].upper(),"Chi phí (tỷ)":C[i],
                                 "NPV (tỷ)":B[i],"B/C":round(B[i]/C[i],2)} for i in sel])
        st.dataframe(df_sel,use_container_width=True,hide_index=True)

        # Biểu đồ tất cả 15 dự án
        all_df = pd.DataFrame([{"Mã":f"P{i}","Tên":names[i],"Chi phí":C[i],"NPV":B[i],
                                 "Được chọn":"✅" if i in sel else "❌"} for i in P])
        fig = px.scatter(all_df,x="Chi phí",y="NPV",text="Mã",color="Được chọn",
                         color_discrete_map={"✅":"#2ec4b6","❌":"#5e7a99"},
                         size_max=20,
                         **_px_cfg("Ma trận Chi phí vs NPV của 15 dự án"))
        fig.update_traces(textposition="top center",marker_size=12)
        st.plotly_chart(fig,use_container_width=True)
        info_box("<b>Nhận xét:</b> P13 (Khu CN bán dẫn) có NPV cao nhất (35.000 tỷ) nhưng cũng chi phí lớn nhất (20.000 tỷ). P15 (Open Data) có B/C tốt nhưng bị loại do ràng buộc tổng dự án ≤11.")

    with tab2:
        BT_new = st.slider("Ngân sách tổng (tỷ VND)",80000,130000,100000,5000,
                           format="%,d tỷ")
        sel_new,tc_new,Z_new,_ = solve_mip(BT=BT_new)
        sel_base,_,Z_base,_ = solve_mip()
        added   = set(sel_new)-set(sel_base)
        removed = set(sel_base)-set(sel_new)
        st.success(f"✅ Chọn **{len(sel_new)} dự án** · Z* = {Z_new:,.0f} tỷ · Δ+{Z_new-Z_base:,.0f} tỷ so với ngân sách 80K")
        if added:   st.info("🆕 Dự án thêm: " + ", ".join(f"P{i} ({names[i]})" for i in sorted(added)))
        if removed: st.warning("⚠️ Dự án bị loại: " + ", ".join(f"P{i} ({names[i]})" for i in sorted(removed)))

    with tabf:
        sel_base,_,Z_base,_ = solve_mip()
        sel_f,tc_f,Z_f,stt  = solve_mip(force12=True)
        if stt=="Optimal":
            st.success(f"✅ Vẫn khả thi khi bắt buộc P1+P2 · Z* = {Z_f:,.0f} tỷ "
                       f"(giảm {Z_base-Z_f:,.0f} tỷ = {(Z_base-Z_f)/Z_base*100:.1f}% do cần cả hai trung tâm dữ liệu)")
        else:
            st.error("❌ Bài toán KHÔNG khả thi khi bắt buộc cả P1 và P2.")

    with tab3:
        sel_d,_,Z_d,_ = solve_mip()
        sel_e,_,Z_e,_ = solve_mip(use_exp=True)
        c1,c2 = st.columns(2)
        kpi_row([(c1,"Z* Deterministic",f"{Z_d:,.0f} tỷ"),(c2,"E[Z] với rủi ro tiến độ",f"{Z_e:,.0f} tỷ")])
        df_risk = pd.DataFrame([{"Mã":f"P{i}","Tên":names[i],"p(đúng tiến độ)":prob_p[fields[i]],
                                  "B×p":(B[i]*prob_p[fields[i]])} for i in P])
        st.dataframe(df_risk.sort_values("B×p",ascending=False),use_container_width=True,hide_index=True)
        warn_box("Dự án AI & Bán dẫn có p=0.65 (rủi ro tiến độ cao nhất) — cần tăng năng lực quản trị dự án và đối tác quốc tế.")


# ============================================================
# BÀI 6 — TOPSIS
# ============================================================
def page_bai6():
    section_header("🏆","Bài 6 — TOPSIS xếp hạng 6 vùng theo ưu tiên đầu tư AI",
                   "Vector normalization · Entropy weight · AHP comparison · Sensitivity")

    info_box("""
    <b>TOPSIS</b> (Technique for Order Preference by Similarity to Ideal Solution):
    Xếp hạng phương án dựa trên khoảng cách đến giải pháp lý tưởng dương (A*) và âm (A⁻).
    <b>C* = S⁻/(S*+S⁻)</b> — càng cao càng tốt.
    """)

    X   = REGIONS[TOPSIS_CRIT].values.astype(float)
    rn  = REGIONS[REG_NAME_COL].values if REG_NAME_COL else np.array(REGIONS_VI)
    w_exp = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    C_exp = _topsis(X,w_exp,IS_BENEFIT)
    w_ent = _entropy_w(X)
    C_ent = _topsis(X,w_ent,IS_BENEFIT)

    tab1,tab2,tab3,tab4 = st.tabs(["🔑 6.4.1 Trọng số chuyên gia","📊 6.4.2 Entropy vs Expert",
                                    "🌡️ 6.4.3 Độ nhạy w_AI","🧮 6.4.4 AHP vs TOPSIS"])
    with tab1:
        df_t = pd.DataFrame({"Hạng":range(1,7),"Vùng":rn,"C* Score":C_exp.round(4)}) \
                 .sort_values("C* Score",ascending=False).reset_index(drop=True)
        df_t["Hạng"] = range(1,7)
        cc = st.columns([2,3])
        with cc[0]:
            st.dataframe(df_t,use_container_width=True,hide_index=True)
            step(1,"Xác định A* và A⁻","(lý tưởng dương/âm)")
            step(2,"Tính S*, S⁻","(khoảng cách Euclid)")
            step(3,"C* = S⁻/(S*+S⁻)","(hệ số gần gũi)")
        with cc[1]:
            fig = px.bar(df_t.sort_values("C* Score"),x="C* Score",y="Vùng",orientation="h",
                         color="C* Score",color_continuous_scale=["#1e3050","#e63946"],
                         **_px_cfg("TOPSIS Score — Trọng số chuyên gia (w_AI=0.20)"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        best_vung = df_t.iloc[0]["Vùng"]
        info_box(f"<b>Kết quả:</b> <b>{best_vung}</b> dẫn đầu — phù hợp để ưu tiên đặt Trung tâm AI quốc gia theo QĐ 127/QĐ-TTg.")

    with tab2:
        comp = pd.DataFrame({"Vùng":rn,"C* Expert":C_exp.round(4),"C* Entropy":C_ent.round(4)})
        comp["Chênh lệch hạng"] = (pd.Series(C_exp).rank(ascending=False).values -
                                    pd.Series(C_ent).rank(ascending=False).values)
        st.dataframe(comp,use_container_width=True,hide_index=True)
        fig = go.Figure()
        fig.add_bar(y=rn,x=C_exp,name="Expert weights",orientation="h",
                    marker_color="#4895ef",opacity=0.85)
        fig.add_bar(y=rn,x=C_ent,name="Entropy weights",orientation="h",
                    marker_color="#e63946",opacity=0.85)
        fig.update_layout(**_plot_cfg("Expert vs Entropy weight — C* Score so sánh"),barmode="group")
        st.plotly_chart(fig,use_container_width=True)
        we_df = pd.DataFrame({"Tiêu chí":TOPSIS_LBL,"w Expert":w_exp[:len(TOPSIS_LBL)].round(4),
                               "w Entropy":w_ent.round(4)})
        st.dataframe(we_df,use_container_width=True,hide_index=True)

    with tab3:
        rng = np.arange(0.10,0.45,0.05); heat = []
        w_gini_fixed = 0.10
        for wai in rng:
            # 8 criteria: GRDP,FDI,Digital,AI(vary),Labor,R&D,Internet,Gini(fixed)
            wb = np.array([0.10,0.10,0.15,0.15,0.15,0.05])  # 6 non-AI non-Gini
            rem = 1.0 - wai - w_gini_fixed
            wsc = wb * (rem / wb.sum())
            # Full 8-element: indices 0-2=wsc[:3], 3=wai, 4-6=wsc[3:], 7=gini
            wfull = np.concatenate([wsc[:3], [wai], wsc[3:], [w_gini_fixed]])
            wfull = wfull / wfull.sum()  # ensure sums to 1
            heat.append(_topsis(X,wfull,IS_BENEFIT))
        heat=np.array(heat)
        fig = px.imshow(heat,x=[f"V{i+1}" for i in range(6)],
                        y=[f"{w:.2f}" for w in rng],text_auto=".3f",aspect="auto",
                        color_continuous_scale="RdYlGn",
                        **_px_cfg("C* Score theo w_AI — phân tích độ nhạy"))
        fig.update_layout(xaxis_title="Vùng (V1..V6)",yaxis_title="Trọng số w_AI")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("V1..V6 = "+", ".join(f"V{i+1}:{n}" for i,n in enumerate(rn)))

    with tab4:
        ahp=np.array([[1,1,1/3,1/5,1/3,1/3,3,3],[1,1,1/3,1/5,1/3,1/3,3,3],
                      [3,3,1,1/2,1,1,5,5],[5,5,2,1,2,2,7,7],[3,3,1,1/2,1,1,5,5],
                      [3,3,1,1/2,1,1,5,5],[1/3,1/3,1/5,1/7,1/5,1/5,1,1],
                      [1/3,1/3,1/5,1/7,1/5,1/5,1,1]])
        n=8; gm=np.prod(ahp,axis=1)**(1/n); w_ahp=gm/gm.sum()
        lam=np.mean((ahp@w_ahp)/w_ahp); CI=(lam-n)/(n-1); CR=CI/1.41
        C_ahp=_topsis(X,w_ahp,IS_BENEFIT)
        status = "✅ Nhất quán (CR<0.10)" if CR<0.10 else f"⚠️ CR={CR:.3f} > 0.10"
        st.info(f"**Kiểm tra nhất quán AHP:** λ_max={lam:.3f} · CI={CI:.3f} · CR={CR:.3f} → {status}")
        cmp = pd.DataFrame({"Vùng":rn,"C* Expert":C_exp.round(4),
                            "C* Entropy":C_ent.round(4),"C* AHP":C_ahp.round(4)})
        st.dataframe(cmp,use_container_width=True,hide_index=True)


# ============================================================
# BÀI 7 — NSGA-II PARETO
# ============================================================
BETA_MAT7 = np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],
                       [1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],
                       [0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
D0_7 = np.array([38,78,55,32,82,48])
E_7  = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
RHO_7= np.array([0.18,0.45,0.28,0.12,0.52,0.22])
SIG_7= np.array([0.32,0.28,0.30,0.35,0.25,0.30])

@st.cache_data(show_spinner=False)
def _run_nsga(pop=80,gen=150):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as moo_min
    from pymoo.termination import get_termination

    class P(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=24,n_obj=4,n_ieq_constr=20,
                             xl=np.zeros(24),xu=np.ones(24)*12000)
        def _evaluate(s,x,out,*a,**k):
            X=x.reshape(6,4)
            f1=-(BETA_MAT7*X).sum()
            su=X.sum(1); f2=np.abs(su-su.mean()).mean()
            f3=(E_7*(X[:,0]+X[:,1]+X[:,2])).sum()
            f4=(RHO_7*X[:,2]).sum()-(SIG_7*X[:,3]).sum()
            out['F']=[f1,f2,f3,f4]
            g=[X.sum()-50000]
            for r in range(6): g.append(5000-X[r].sum())
            for r in range(6): g.append(X[r].sum()-12000)
            g.append(12000-X[:,3].sum())
            Dn=D0_7+0.002*X[:,1]; Dm=Dn.max()
            for r in range(6): g.append(0.6*Dm-Dn[r])
            out['G']=np.array(g)

    res=moo_min(P(),NSGA2(pop_size=pop),get_termination("n_gen",gen),
                seed=42,verbose=False,save_history=False)
    if res.F is not None and len(np.atleast_2d(res.F))>0:
        return np.atleast_2d(res.F),np.atleast_2d(res.X)
    pop_obj=res.pop; Xall=pop_obj.get("X"); Fall=pop_obj.get("F"); Gall=pop_obj.get("G")
    cv=np.maximum(0,Gall).sum(axis=1); keep=np.argsort(cv)[:max(20,pop//2)]
    return Fall[keep],Xall[keep]


def page_bai7():
    section_header("🌐","Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II",
                   "4 mục tiêu xung đột: tăng trưởng · bao trùm · môi trường · an ninh dữ liệu")

    info_box("""
    <b>4 mục tiêu:</b>
    f₁ = max GDP gain (tỷ VND) · f₂ = min Gini/MAD (bất bình đẳng) ·
    f₃ = min Phát thải CO₂ · f₄ = min Rủi ro an ninh dữ liệu
    <br><b>Thuật toán NSGA-II</b> (Fast Non-dominated Sorting Genetic Algorithm II) tìm tập nghiệm Pareto tối ưu.
    """)

    try:
        with st.spinner("⚙️ Đang chạy NSGA-II (pop=80, gen=150) — vui lòng chờ ~30s..."):
            F,X = _run_nsga()
    except ImportError:
        st.error("❌ Cần cài pymoo: `pip install pymoo`"); return

    w_pol = np.array([0.40,0.25,0.20,0.15])
    fmin,fmax = F.min(0),F.max(0); fr=np.where(fmax-fmin>1e-12,fmax-fmin,1.0)
    R=(F-fmin)/fr; V=R*w_pol
    S_s=np.sqrt((V**2).sum(1)); S_n=np.sqrt(((V-w_pol)**2).sum(1))
    Cs=S_n/(S_s+S_n); best=int(np.argmax(Cs)); mg=int(np.argmin(F[:,0]))
    bX=X[best].reshape(6,4)

    c1,c2,c3,c4 = st.columns(4)
    kpi_row([(c1,"Số nghiệm Pareto",f"{len(F)}"),
             (c2,"GDP gain max",f"{-F[mg,0]:,.0f} tỷ"),
             (c3,"GDP gain thỏa hiệp",f"{-F[best,0]:,.0f} tỷ"),
             (c4,"Hy sinh GDP cho cân bằng",f"{(-F[mg,0]-(-F[best,0]))/(-F[mg,0])*100:.1f}%",None,True)])

    tab1,tabp,tab2 = st.tabs(["🌀 7.4.2 Biên Pareto 3D","📊 Parallel Coordinates","⚖️ 7.4.3 Nghiệm thỏa hiệp"])

    with tab1:
        fig = go.Figure(go.Scatter3d(
            x=-F[:,0],y=F[:,1],z=F[:,2],mode="markers",
            marker=dict(size=4,color=F[:,3],colorscale="Viridis",
                        colorbar=dict(title="Rủi ro an ninh",len=0.6),opacity=0.8)))
        fig.add_trace(go.Scatter3d(x=[-F[best,0]],y=[F[best,1]],z=[F[best,2]],
                                   mode="markers",marker=dict(size=12,color="#e63946",
                                   symbol="diamond"),name="Nghiệm thỏa hiệp"))
        fig.add_trace(go.Scatter3d(x=[-F[mg,0]],y=[F[mg,1]],z=[F[mg,2]],
                                   mode="markers",marker=dict(size=10,color="#f4a261",
                                   symbol="star"),name="Tăng trưởng max"))
        fig.update_layout(template=PLOT_TMPL,height=580,
                          scene=dict(xaxis_title="f₁ GDP gain",yaxis_title="f₂ Gini/MAD",
                                     zaxis_title="f₃ Phát thải",
                                     bgcolor="rgba(0,0,0,0)"),
                          title="Tập Pareto 3D — màu = f₄ rủi ro an ninh")
        st.plotly_chart(fig,use_container_width=True)

    with tabp:
        fn=np.copy(F).astype(float)
        for i in range(4):
            lo,hi=F[:,i].min(),F[:,i].max()
            fn[:,i]=(F[:,i]-lo)/(hi-lo) if hi>lo else 0.5
        figp = go.Figure(go.Parcoords(
            line=dict(color=fn[:,0],colorscale="Plasma",showscale=True),
            dimensions=[dict(label="f₁ GDP (↓=tốt)",values=fn[:,0]),
                        dict(label="f₂ Gini (↓=tốt)",values=fn[:,1]),
                        dict(label="f₃ Phát thải (↓=tốt)",values=fn[:,2]),
                        dict(label="f₄ Rủi ro (↓=tốt)",values=fn[:,3])]))
        figp.update_layout(template=PLOT_TMPL,height=480,
                           title="Parallel coordinates — 4 mục tiêu (chuẩn hóa [0,1])")
        st.plotly_chart(figp,use_container_width=True)
        info_box("Kéo chọn dải giá trị trên mỗi trục để lọc tập Pareto theo ưu tiên chính sách mong muốn.")

    with tab2:
        dfm=pd.DataFrame(bX,index=REGIONS_VI,columns=ITEMS).round(0)
        dfm["Tổng"]=dfm.sum(1); st.markdown("**Phân bổ nghiệm thỏa hiệp (w=0.40/0.25/0.20/0.15):**")
        st.dataframe(dfm,use_container_width=True)
        oc=pd.DataFrame({"Mục tiêu":["f₁ GDP gain (tỷ)","f₂ Gini/MAD","f₃ Phát thải","f₄ Rủi ro ròng"],
                         "Nghiệm thỏa hiệp":[-F[best,0],F[best,1],F[best,2],F[best,3]],
                         "Tăng trưởng max":[-F[mg,0],F[mg,1],F[mg,2],F[mg,3]]}).round(2)
        oc["Δ (%)"]=(oc["Nghiệm thỏa hiệp"]-oc["Tăng trưởng max"])/np.abs(oc["Tăng trưởng max"])*100
        st.dataframe(oc.round(2),use_container_width=True,hide_index=True)


# ============================================================
# BÀI 8 — TỐI ƯU ĐỘNG
# ============================================================
@st.cache_data(show_spinner=False)
def _run_dynamic():
    from scipy.optimize import minimize
    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    dK,dD,dAI=0.05,0.12,0.15; thH,mu=0.8,0.02
    p1,p2,p3=0.003,0.002,0.004; rho,gcr,T=0.97,1.5,10
    K0,L0,D0,AI0,H0,Y0=27500.,53.9,20.3,86.,30.,12847.6
    A0=Y0/(K0**a*L0**b*D0**g*AI0**d*H0**th)
    L=np.array([L0*1.009**t for t in range(T+1)])

    def traj(u,shock_year=None,shock_pct=0.0):
        IK,ID,IAI,IH=u[0::4],u[1::4],u[2::4],u[3::4]
        K=np.zeros(T+1);D=np.zeros(T+1);AI=np.zeros(T+1)
        H=np.zeros(T+1);A=np.zeros(T+1);Y=np.zeros(T+1);C=np.zeros(T)
        K[0],D[0],AI[0],H[0],A[0]=K0,D0,AI0,H0,A0
        for t in range(T):
            if shock_year is not None and t==shock_year: A[t]*=(1-shock_pct)
            Y[t]=A[t]*K[t]**a*L[t]**b*D[t]**g*AI[t]**d*H[t]**th
            C[t]=Y[t]-IK[t]-ID[t]-IAI[t]-IH[t]
            if C[t]<=0: return None
            K[t+1]=(1-dK)*K[t]+IK[t]; D[t+1]=(1-dD)*D[t]+ID[t]
            AI[t+1]=(1-dAI)*AI[t]+IAI[t]; H[t+1]=H[t]+thH*IH[t]-mu*H[t]
            A[t+1]=A[t]*(1+p1*D[t]/100+p2*AI[t]/100+p3*H[t]/100)
        Y[T]=A[T]*K[T]**a*L[T]**b*D[T]**g*AI[T]**d*H[T]**th
        return K,D,AI,H,Y,C,A

    def welfare(u,sy=None,sp=0.0):
        r=traj(u,sy,sp)
        if r is None or np.any(r[5]<=0): return 1e15
        C=r[5]; return -sum(rho**t*(C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    ti=14000*0.15; u0=np.tile([ti*.40,ti*.25,ti*.20,ti*.15],T)
    def cons_f(sy=None,sp=0.0):
        def c(u): r=traj(u,sy,sp); return -1e10 if r is None else min(r[5])-1
        return c

    res=minimize(welfare,u0,method='SLSQP',bounds=[(0,None)]*(T*4),
                 constraints=[{'type':'ineq','fun':cons_f()}],
                 options={'maxiter':600,'ftol':1e-8})

    W_base=-welfare(res.x); W_plan=-welfare(res.x,2,0.08)
    res_sh=minimize(lambda u:welfare(u,2,0.08),res.x,method='SLSQP',
                    bounds=[(0,None)]*(T*4),
                    constraints=[{'type':'ineq','fun':cons_f(2,0.08)}],
                    options={'maxiter':600,'ftol':1e-8})
    W_reopt=-res_sh.fun
    Y_base=traj(res.x)[4]; Y_shock=traj(res.x,2,0.08)[4]; Y_reopt=traj(res_sh.x,2,0.08)[4]
    u_even=u0.copy(); u_front=np.zeros(T*4)
    for t in range(T):
        f=1.5 if t<3 else 0.7
        u_front[t*4:(t+1)*4]=np.array([ti*.40,ti*.25,ti*.20,ti*.15])*f
    return {"opt":traj(res.x),"W":-res.fun,
            "shock":dict(W_base=W_base,W_plan=W_plan,W_reopt=W_reopt,
                         Y_base=Y_base,Y_shock=Y_shock,Y_reopt=Y_reopt),
            "strat":dict(W_opt=-res.fun,W_even=-welfare(u_even),W_front=-welfare(u_front),
                         Y_opt=traj(res.x)[4],Y_even=traj(u_even)[4],Y_front=traj(u_front)[4])}


def page_bai8():
    section_header("⏳","Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
                   "Quỹ đạo K,D,AI,H,Y,C · Hàm thỏa dụng CRRA · Cú sốc TFP · SLSQP")

    info_box("""
    <b>Mô hình:</b> max Σ ρᵗ·U(Cₜ) với U(C) = C^(1-γ)/(1-γ), γ=1.5, ρ=0.97.
    Động học: K,D,AI,H tích lũy theo phương trình chuyển trạng thái. TFP nội sinh: A[t+1] = A[t]·(1+φ₁D+φ₂AI+φ₃H).
    """)

    with st.spinner("⚙️ Đang tối ưu quỹ đạo 10 năm (SLSQP)..."):
        out = _run_dynamic()
    (K,D,AI,H,Y,C,A)=out["opt"]; W=out["W"]
    years=list(range(2026,2037))

    c1,c2,c3,c4=st.columns(4)
    kpi_row([(c1,"Phúc lợi W*",f"{W:.2f}"),(c2,"GDP 2035",f"{Y[-1]:,.0f} ng.tỷ"),
             (c3,"Tăng trưởng BQ",f"{((Y[-1]/Y[0])**(1/10)-1)*100:.2f}%/năm"),
             (c4,"Tiêu dùng BQ",f"{C.mean():,.0f} ng.tỷ/năm")])

    tab1,tab3,tab4=st.tabs(["📈 8.3.1-2 Quỹ đạo tối ưu","⚡ 8.3.3 Cú sốc 2028","🆚 8.3.4 So sánh chiến lược"])

    with tab1:
        fig=make_subplots(rows=2,cols=3,subplot_titles=(
            "K — Vốn vật chất (ng.tỷ)","D — Hạ tầng số (%)","AI — Năng lực AI (ng.DN)",
            "H — Vốn nhân lực (%)","Y & C — Sản lượng & Tiêu dùng","A — TFP nội sinh"))
        clrs=["#4895ef","#2ec4b6","#a8dadc","#f4a261","#e63946","#f1faee"]
        series=[(K,1,1,clrs[0]),(D,1,2,clrs[1]),(AI,1,3,clrs[2]),(H,2,1,clrs[3]),(A,2,3,clrs[5])]
        for s,r,c,col in series:
            fig.add_trace(go.Scatter(x=years,y=s,mode="lines+markers",
                                     line=dict(color=col,width=2.5),marker=dict(size=6),
                                     showlegend=False),row=r,col=c)
        fig.add_trace(go.Scatter(x=years,y=Y,name="Y (GDP)",line=dict(color="#e63946",width=2.5)),row=2,col=2)
        fig.add_trace(go.Scatter(x=years[:10],y=C,name="C (Tiêu dùng)",
                                 line=dict(color="#2ec4b6",width=2.5,dash="dot")),row=2,col=2)
        fig.update_layout(template=PLOT_TMPL,height=640,
                          title="Quỹ đạo tối ưu 2026-2035 — SLSQP + CRRA utility",
                          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        info_box("<b>Nhận xét:</b> Mô hình đề xuất <b>front-load</b> đầu tư AI & H giai đoạn đầu để tích lũy năng lực hấp thụ. H (nhân lực số) nên đi trước/đồng thời với AI — phù hợp với Nghị quyết 57-NQ/TW.")

    with tab3:
        sh=out["shock"]
        c1,c2,c3=st.columns(3)
        kpi_row([(c1,"Welfare không sốc",f"{sh['W_base']:.2f}"),
                 (c2,"Welfare giữ kế hoạch",f"{sh['W_plan']:.2f}",
                  f"{(sh['W_base']-sh['W_plan'])/abs(sh['W_base'])*100:.1f}% giảm",True),
                 (c3,"Welfare tái tối ưu",f"{sh['W_reopt']:.2f}")])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=years,y=sh["Y_base"],name="Không sốc",
                                 line=dict(color="#2ec4b6",width=2.5)))
        fig.add_trace(go.Scatter(x=years,y=sh["Y_shock"],name="Có sốc (giữ KH)",
                                 line=dict(color="#e63946",width=2.5)))
        fig.add_trace(go.Scatter(x=years,y=sh["Y_reopt"],name="Có sốc (tái tối ưu)",
                                 line=dict(color="#f4a261",width=2.5,dash="dot")))
        fig.add_vline(x=2028,line_dash="dash",line_color="#5e7a99",
                      annotation_text="Cú sốc TFP -8% (2028)",annotation_position="top left")
        fig.update_layout(**_plot_cfg("Phân tích cú sốc TFP -8% năm 2028 (như bão Yagi 2024)"))
        st.plotly_chart(fig,use_container_width=True)
        warn_box(f"Tái tối ưu phục hồi {(sh['W_reopt']-sh['W_plan'])/abs(sh['W_plan'])*100:.1f}% phúc lợi bị mất. "
                 "Bài học: cần duy trì quỹ dự phòng chính sách để tái phân bổ khi xảy ra cú sốc kinh tế.")

    with tab4:
        s=out["strat"]
        dfst=pd.DataFrame({"Chiến lược":["Tối ưu (SLSQP)","Đầu tư đều","Front-load (150%+70%)"],
                           "Phúc lợi W":[round(s["W_opt"],2),round(s["W_even"],2),round(s["W_front"],2)],
                           "GDP 2035 (ng.tỷ)":[round(s["Y_opt"][-1]),round(s["Y_even"][-1]),round(s["Y_front"][-1])]})
        st.dataframe(dfst,use_container_width=True,hide_index=True)
        fig=go.Figure()
        clrs2=["#e63946","#4895ef","#f4a261"]
        for (name,y_vals),c in zip([("Tối ưu",s["Y_opt"]),("Đều",s["Y_even"]),("Front-load",s["Y_front"])],clrs2):
            fig.add_trace(go.Scatter(x=years,y=y_vals,name=name,
                                     line=dict(color=c,width=2.5)))
        fig.update_layout(**_plot_cfg("So sánh quỹ đạo GDP theo 3 chiến lược đầu tư"))
        st.plotly_chart(fig,use_container_width=True)


# ============================================================
# BÀI 9 — LAO ĐỘNG & AI
# ============================================================
def page_bai9():
    from scipy.optimize import linprog
    section_header("👷","Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
                   "30.000 tỷ cho x_AI & x_H · tối đa NetJob ròng 8 ngành · Sankey flow")

    info_box("""
    <b>Công thức:</b> NetJob_i = NewJob_i + UpgradeJob_i − DisplacedJob_i
    <br>= a₁·x_AI + a₂·x_D + b₁·x_H − c₁·risk·x_AI
    <br><b>Ràng buộc cốt lõi (Mục 10):</b> "Tốc độ tự động hóa không vượt quá năng lực đào tạo lại" → Displaced ≤ RetrainCapacity
    """)

    N=8
    sec=['Nông-Lâm-TS','CN chế biến','Xây dựng','Bán buôn-bán lẻ',
         'Tài chính-NH','Logistics','CNTT-TT','Giáo dục-ĐT']
    L_sec=np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
    risk=np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b1=np.array([45,28,35,32,22,30,20,55])
    c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d1=np.array([50,32,42,38,26,36,24,62])
    coeff=a1-c1*risk

    def solve(cap5=False):
        c_obj=np.concatenate([-coeff,-b1])
        A1=np.concatenate([np.ones(N),np.ones(N)]).reshape(1,-1)
        A1b=np.concatenate([-np.ones(N),np.zeros(N)]).reshape(1,-1)
        A2=np.zeros((N,2*N)); A3=np.zeros((N,2*N))
        for i in range(N):
            A2[i,i]=-coeff[i]; A2[i,N+i]=-b1[i]
            A3[i,i]=c1[i]*risk[i]; A3[i,N+i]=-d1[i]
        Aub=np.vstack([A1,A1b,A2,A3])
        bub=np.concatenate([[30000],[-9000],np.zeros(N),np.zeros(N)])
        if cap5:
            A4=np.zeros((N,2*N))
            for i in range(N): A4[i,i]=c1[i]*risk[i]
            Aub=np.vstack([Aub,A4]); bub=np.concatenate([bub,0.05*L_sec*1e6])
        return linprog(c_obj,A_ub=Aub,b_ub=bub,bounds=[(0,None)]*(2*N),method="highs")

    res=solve(); xA,xH=res.x[:N],res.x[N:]
    NetJob=coeff*xA+b1*xH; Displaced=c1*risk*xA; RetrainCap=d1*xH

    c1c,c2c,c3c=st.columns(3)
    kpi_row([(c1c,"Tổng NetJob ròng",f"{-res.fun:,.0f} việc"),
             (c2c,"Ngân sách AI",f"{xA.sum():,.0f} tỷ"),
             (c3c,"Ngân sách Đào tạo",f"{xH.sum():,.0f} tỷ")])

    tab1,tab2s,tab3s,tab4=st.tabs(["📊 9.4.1 Phân bổ & NetJob","📐 9.4.2 Ngưỡng đào tạo",
                                    "🌊 9.4.3 Luồng lao động (Sankey)","⚙️ 9.4.4 Ràng buộc 5%L"])
    with tab1:
        df=pd.DataFrame({"Ngành":sec,"x_AI (tỷ)":xA.round(0),"x_H (tỷ)":xH.round(0),
                         "Displaced":Displaced.round(0),"Retrain Cap":RetrainCap.round(0),
                         "NetJob":NetJob.round(0)})
        st.dataframe(df.style.background_gradient(subset=["NetJob"],cmap="RdYlGn"),
                     use_container_width=True,hide_index=True)
        fig=px.bar(df,x="Ngành",y="NetJob",color="NetJob",
                   color_continuous_scale=["#e63946","#f4a261","#2ec4b6"],
                   **_px_cfg("NetJob ròng theo ngành — đầu tư tối ưu 30.000 tỷ"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    with tab2s:
        i=1; net=a1[i]-c1[i]*risk[i]; rr=c1[i]*risk[i]/d1[i]
        st.markdown(f"**Ngành CN chế biến chế tạo** — risk = 42%, hệ số net AI = {net:.1f}")
        st.latex(r"x_H \geq \frac{c_1 \cdot risk}{d_1} \cdot x_{AI} \quad\Leftrightarrow\quad x_H \geq " + f"{rr:.3f}" + r"\cdot x_{AI}")
        xr=np.linspace(0,15000,100); xh_re=rr*xr; xh_nj=np.maximum(0,-net/b1[i]*xr)
        xh_min=np.maximum(xh_re,xh_nj)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=xr,y=xh_re,name=f"Ngưỡng retrain (≥{rr:.3f}·x_AI)",
                                 line=dict(color="#e63946",width=2.5)))
        fig.add_trace(go.Scatter(x=xr,y=xh_nj,name="Ngưỡng NetJob≥0",
                                 line=dict(color="#4895ef",width=2.5)))
        fig.add_trace(go.Scatter(x=xr,y=xh_min,name="Vùng khả thi x_H tối thiểu",
                                 fill="tozeroy",line=dict(color="#2ec4b6",width=0),
                                 fillcolor="rgba(46,196,182,0.15)"))
        fig.update_layout(**_plot_cfg("Ngưỡng đào tạo tối thiểu — CN chế biến chế tạo"),
                          xaxis_title="x_AI (tỷ VND)",yaxis_title="x_H tối thiểu (tỷ VND)")
        st.plotly_chart(fig,use_container_width=True)

    with tab3s:
        vuln=[0,2,3]; labels=list(np.array(sec)[vuln])+["Giữ việc","Đào tạo lại","Mất việc"]
        src,tgt,val,colr=[],[],[],[]
        for idx,k in enumerate(vuln):
            disp=Displaced[k]; retr=min(disp,RetrainCap[k]); lost=max(0,disp-retr)
            kept=L_sec[k]*1e6-disp
            for tnode,v,c in [(len(vuln),kept,"rgba(46,196,182,0.6)"),
                               (len(vuln)+1,retr,"rgba(244,162,97,0.6)"),
                               (len(vuln)+2,lost,"rgba(230,57,70,0.6)")]:
                if v>0: src.append(idx);tgt.append(tnode);val.append(v);colr.append(c)
        fig=go.Figure(go.Sankey(
            node=dict(label=labels,pad=18,thickness=16,
                      color=["#4895ef","#4895ef","#4895ef","#2ec4b6","#f4a261","#e63946"]),
            link=dict(source=src,target=tgt,value=val,color=colr)))
        fig.update_layout(template=PLOT_TMPL,height=440,
                          title="Luồng dịch chuyển lao động — nhóm dễ bị tổn thương (Nông-LT, XD, Bán buôn)")
        st.plotly_chart(fig,use_container_width=True)

    with tab4:
        res4=solve(cap5=True)
        if res4.success:
            st.success(f"✅ Khả thi với ràng buộc Displaced ≤ 5% lao động · NetJob = {-res4.fun:,.0f} "
                       f"(giảm {(-res.fun)-(-res4.fun):,.0f} = {((-res.fun)-(-res4.fun))/(-res.fun)*100:.1f}%)")
        else:
            st.error("❌ KHÔNG khả thi với ràng buộc 5%L — cần tăng đầu tư đào tạo lại mạnh hơn.")
        info_box("<b>Ràng buộc cốt lõi:</b> Displaced ≤ RetrainingCapacity (c₁·risk·x_AI ≤ d₁·x_H). "
                 "Ngành CN chế biến & Tài chính cần đầu tư x_H nhiều nhất vì risk cao nhưng d₁ thấp.")


# ============================================================
# BÀI 10 — STOCHASTIC SP
# ============================================================
def page_bai10():
    section_header("🎲","Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
                   "Here-and-now + Recourse · VSS · EVPI · Robust minimax regret")

    info_box("""
    <b>Cấu trúc:</b> Giai đoạn 1 (here-and-now): phân bổ x ≤ 65.000 tỷ trước khi biết kịch bản.
    Giai đoạn 2 (recourse): điều chỉnh y_s ≤ 15.000 tỷ sau khi kịch bản s hiện thực hóa.
    Ràng buộc: y_AI_s ≤ 0.5·x_H (AI bổ sung phụ thuộc nhân lực đã đào tạo).
    """)

    x_sp,y_sp,Z_sp,Z_ev,Z_ws,det = _solve_sp()
    VSS=Z_sp-Z_ev; EVPI=Z_ws-Z_sp

    c1,c2,c3,c4=st.columns(4)
    kpi_row([(c1,"Z* Stochastic (SP)",f"{Z_sp:,.0f} tỷ"),
             (c2,"Z* EV (deterministic)",f"{Z_ev:,.0f} tỷ"),
             (c3,"VSS (giá trị bất định)",f"{VSS:,.0f} tỷ"),
             (c4,"EVPI (giá trị thông tin)",f"{EVPI:,.0f} tỷ")])

    tab1,tab2,tab3=st.tabs(["📋 10.5.1 First & Second stage",
                             "🎯 10.5.2-3 Kịch bản · VSS · EVPI",
                             "🛡️ 10.5.4 Robust (minimax regret)"])
    with tab1:
        cc=st.columns(2)
        with cc[0]:
            st.markdown("**📌 First-stage (x*) — Tổng ≤ 65.000 tỷ:**")
            df1=pd.DataFrame({"Hạng mục":J10,"x* (tỷ)":x_sp.round(0),
                               "Tỷ lệ (%)":x_sp/x_sp.sum()*100})
            st.dataframe(df1.style.format({"Tỷ lệ (%)":"{:.1f}%"}),
                         use_container_width=True,hide_index=True)
            fig=go.Figure(go.Pie(labels=J10,values=x_sp,
                                 marker_colors=PALETTE[:4],hole=0.5))
            fig.update_layout(**_plot_cfg("Cơ cấu phân bổ First-stage"),height=280)
            st.plotly_chart(fig,use_container_width=True)
        with cc[1]:
            st.markdown("**🔄 Second-stage recourse (y_s*) — mỗi kịch bản ≤ 15.000 tỷ:**")
            ydf=pd.DataFrame({s:y_sp[s].round(0) for s in S10},index=J10).T
            ydf.index=[f"{s} (p={P_S[s]})" for s in S10]
            st.dataframe(ydf,use_container_width=True)

    with tab2:
        scn_names=["Lạc quan (s1)","Cơ sở (s2)","Bi quan (s3)","Khủng hoảng (s4)"]
        scn=pd.DataFrame({"Kịch bản":scn_names,"Xác suất":[0.30,0.45,0.20,0.05],
                          "Z*[s] (tỷ)":[round(det[s]["Z"],0) for s in S10]})
        cc=st.columns(2)
        with cc[0]:
            st.dataframe(scn,use_container_width=True,hide_index=True)
        with cc[1]:
            fig=px.bar(scn,x="Kịch bản",y="Z*[s] (tỷ)",color="Kịch bản",
                       color_discrete_sequence=["#2ec4b6","#4895ef","#f4a261","#e63946"],
                       **_px_cfg("Z* tối ưu theo từng kịch bản (deterministic)"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

        st.markdown(f"""
        <div class='info-box'>
          <b>VSS = {VSS:,.0f} tỷ</b> → Lợi ích của việc sử dụng mô hình ngẫu nhiên thay vì EV deterministic.
          <br><b>EVPI = {EVPI:,.0f} tỷ</b> → Giá trị tối đa sẵn sàng trả để có thông tin hoàn hảo về kịch bản.
          <br>SP đầu tư H (nhân lực) nhiều hơn EV — H là <b>"hàng hóa bảo hiểm"</b> giúp hấp thụ cú sốc kinh tế.
        </div>""", unsafe_allow_html=True)

    with tab3:
        from scipy.optimize import linprog
        nr=21; cr=np.zeros(nr); cr[20]=1.0; Ar=[]; br=[]
        rowb=[0]*nr
        for k in range(4): rowb[k]=1
        Ar.append(rowb); br.append(65000)
        for si in range(4):
            row=[0]*nr
            for k in range(4): row[4+si*4+k]=1
            Ar.append(row); br.append(15000)
        for si in range(4):
            row=[0]*nr; row[3]=-0.5; row[4+si*4+2]=1
            Ar.append(row); br.append(0)
        for si,s in enumerate(S10):
            row=[0]*nr
            for k,j in enumerate(J10): row[k]=-BETA_BASE[j]; row[4+si*4+k]=-BETA_S[(s,j)]
            row[20]=-1; Ar.append(row); br.append(-det[s]["Z"])
        rr=linprog(cr,A_ub=np.array(Ar),b_ub=np.array(br),
                   bounds=[(0,None)]*20+[(None,None)],method="highs")
        x_rob=rr.x[:4]; w_rob=rr.x[20]
        comp_df=pd.DataFrame({"Hạng mục":J10,"x* SP":x_sp.round(0),"x* Robust":x_rob.round(0)})
        st.dataframe(comp_df,use_container_width=True,hide_index=True)
        st.info(f"**Robust (minimax regret):** Max regret = {w_rob:,.0f} tỷ — chiến lược bảo thủ nhất, đầu tư H (nhân lực) nhiều hơn SP để giảm thiểu kịch bản xấu nhất.")


# ============================================================
# BÀI 11 — Q-LEARNING
# ============================================================
class VietnamEconomyEnv:
    action_names=["a0 Truyền thống","a1 Cân bằng","a2 Số hóa nhanh","a3 AI dẫn dắt","a4 Bao trùm"]
    allocation={0:np.array([.70,.10,.10,.10]),1:np.array([.40,.25,.15,.20]),
                2:np.array([.25,.45,.15,.15]),3:np.array([.20,.20,.45,.15]),
                4:np.array([.30,.20,.10,.40])}
    w=np.array([0.40,0.25,0.20,0.15])
    def __init__(self): self.T=10; self.rng=np.random.default_rng(0)
    def reset(self):
        self.state=np.array([1,1,0,1]); self.t=0
        self.K,self.D,self.AI,self.H=27500.,20.3,86.,30.; self.Y_prev=12847.6
        return self.state.copy()
    def step(self,action):
        a=self.allocation[action]; budget=2100.
        self.K=.95*self.K+a[0]*budget; self.D=.88*self.D+a[1]*budget*.01
        self.AI=.85*self.AI+a[2]*budget*.05; self.H=self.H+.8*a[3]*budget*.01-.02*self.H
        A_t=33.70*(1+.003*self.D/100+.002*self.AI/100+.004*self.H/100)**self.t
        L=53.9*1.009**self.t
        Y=A_t*self.K**.33*L**.42*self.D**.10*self.AI**.08*self.H**.07
        dg=(Y-self.Y_prev)/self.Y_prev; du=max(0,-dg*.5)
        cy=self.AI/(self.H+1)*.01; em=(self.K+self.AI)*.0001
        r=self.w[0]*dg*100-self.w[1]*du*100-self.w[2]*cy-self.w[3]*em
        self.Y_prev=Y; self.t+=1
        gl=0 if dg<.03 else(1 if dg<.06 else 2); dl=0 if self.D<25 else(1 if self.D<35 else 2)
        al=0 if self.AI<100 else(1 if self.AI<200 else 2); hl=0 if self.H<35 else(1 if self.H<50 else 2)
        self.state=np.array([gl,dl,al,hl])
        return self.state.copy(),r,self.t>=self.T

@st.cache_data(show_spinner=False)
def _train_q(n_ep=8000):
    env=VietnamEconomyEnv(); Q=np.zeros((3,3,3,3,5)); hist=[]
    for ep in range(n_ep):
        s=env.reset(); tot=0; eps=max(.05,1-ep/4000)
        while True:
            a=env.rng.integers(5) if env.rng.random()<eps else int(np.argmax(Q[tuple(s)]))
            s2,r,done=env.step(a)
            Q[tuple(s)+(a,)]+=.1*(r+.95*Q[tuple(s2)].max()*(1-done)-Q[tuple(s)+(a,)])
            tot+=r; s=s2
            if done: break
        hist.append(tot)
    return Q,hist

def _eval_policy(Q,pol,n=300):
    env=VietnamEconomyEnv(); rs=[]
    for _ in range(n):
        s=env.reset(); tot=0
        while True:
            a=pol(s,Q,env); s,r,done=env.step(a); tot+=r
            if done: break
        rs.append(tot)
    return np.mean(rs),np.std(rs)


def page_bai11():
    section_header("🤖","Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
                   "MDP 3⁴=81 trạng thái · 5 hành động · ε-greedy · 8.000 episodes")

    info_box("""
    <b>Cập nhật Q-table (Bellman equation):</b>
    Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') − Q(s,a)]
    <br>α=0.10 · γ=0.95 · ε: 1.0 → 0.05 giảm dần. Phần thưởng: R = 0.40·ΔGDP − 0.25·ΔUnemploy − 0.20·CyberRisk − 0.15·Emission
    """)

    with st.spinner("⚙️ Đang huấn luyện Q-learning (8.000 episodes)..."):
        Q,hist=_train_q()
    env=VietnamEconomyEnv()

    tab01,tab02,tab1,tab2=st.tabs(["🔧 11.3.1 Môi trường MDP","📉 11.3.2 Huấn luyện",
                                    "🧭 11.3.3 Chính sách π*","📊 11.3.4 So sánh chính sách"])
    with tab01:
        cc=st.columns(2)
        cc[0].markdown("**Không gian trạng thái S** (3⁴ = 81 trạng thái):")
        cc[0].dataframe(pd.DataFrame({
            "Yếu tố":["GDP growth","Digital index","AI capacity","Unemploy risk"],
            "Low (0)":["<3%","<25","<100 ng.DN","<35% H"],
            "Medium (1)":["3-6%","25-35","100-200","35-50%"],
            "High (2)":[">6%",">35",">200",">50%"]}),
            hide_index=True,use_container_width=True)
        cc[1].markdown("**Không gian hành động A** (5 chiến lược phân bổ K/D/AI/H):")
        cc[1].dataframe(pd.DataFrame({
            "Hành động":env.action_names,
            "K%":[70,40,25,20,30],"D%":[10,25,45,20,20],
            "AI%":[10,15,15,45,10],"H%":[10,20,15,15,40]}),
            hide_index=True,use_container_width=True)
        info_box("Chuyển trạng thái mô phỏng bằng hàm sản xuất Cobb-Douglas đã calibrate. Mỗi episode = 10 năm.")

    with tab02:
        c1,c2,c3,c4=st.columns(4)
        kpi_row([(c1,"Learning rate α","0,10"),(c2,"Discount factor γ","0,95"),
                 (c3,"Số episodes","8.000"),(c4,"Q-table shape",f"{Q.shape}")])
        w=200; sm=np.convolve(hist,np.ones(w)/w,mode="valid")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(hist))),y=hist,mode="lines",
                                 name="Phần thưởng từng episode",
                                 line=dict(color="#1e3050",width=1)))
        fig.add_trace(go.Scatter(x=list(range(w-1,len(hist))),y=sm,
                                 name=f"TB trượt {w} episode",
                                 line=dict(color="#2ec4b6",width=2.5)))
        fig.update_layout(**_plot_cfg("Hội tụ Q-learning — phần thưởng tích lũy theo episode"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"Q_max sau huấn luyện = {Q.max():.3f} · Q_min = {Q.min():.3f}")

    with tab1:
        tests=[([1,1,0,1],"🇻🇳 VN 2026 thực tế (GDP_med, D_med, AI_low, H_med)"),
               ([0,0,0,2],"😟 Kịch bản khó (GDP_low, D_low, AI_low, H_high)"),
               ([2,2,2,2],"🚀 Kịch bản tốt (GDP_high, D_high, AI_high, H_high)"),
               ([0,1,0,0],"⚡ Hậu khủng hoảng (GDP_low, D_med, AI_low, H_low)"),
               ([1,0,2,1],"🤖 AI mạnh, D yếu (GDP_med, D_low, AI_high, H_med)")]
        rows=[]
        for s,desc in tests:
            a=int(np.argmax(Q[tuple(s)]))
            q_vals=Q[tuple(s)]
            rows.append({"Trạng thái khởi đầu":desc,"π*(s) chọn":env.action_names[a],
                         "Q value":f"{q_vals[a]:.3f}"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        info_box("Khi GDP thấp, D thấp → π* chọn <b>Số hóa nhanh (a2)</b> — \"quick win\". "
                 "Khi mọi chỉ số cao → π* chọn <b>AI dẫn dắt (a3)</b> — consolidation.")

    with tab2:
        pols=[("π* (Q-learning)",lambda s,Q,e:int(np.argmax(Q[tuple(s)]))),
              ("Luôn Cân bằng (a1)",lambda s,Q,e:1),
              ("Luôn AI dẫn dắt (a3)",lambda s,Q,e:3),
              ("Random",lambda s,Q,e:e.rng.integers(5))]
        res_pol={n:_eval_policy(Q,p) for n,p in pols}
        dfp=pd.DataFrame({"Chính sách":list(res_pol.keys()),
                          "Phúc lợi BQ":[round(v[0],3) for v in res_pol.values()],
                          "Std Dev":[round(v[1],3) for v in res_pol.values()]})
        cc=st.columns(2)
        with cc[0]:
            st.dataframe(dfp,use_container_width=True,hide_index=True)
        with cc[1]:
            fig=px.bar(dfp,x="Chính sách",y="Phúc lợi BQ",color="Chính sách",
                       color_discrete_sequence=["#e63946","#4895ef","#2ec4b6","#5e7a99"],
                       error_y="Std Dev",**_px_cfg("So sánh các chính sách"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        warn_box("Lưu ý quan trọng (Mục 11 bài báo): <b>AI hỗ trợ ra quyết định, KHÔNG thay thế trách nhiệm chính trị-xã hội.</b> π* chỉ minh họa kỹ thuật học chính sách thích nghi — quyết định cuối thuộc về nhà hoạch định chính sách.")


# ============================================================
# BÀI 12 — AIDEOM-VN TÍCH HỢP
# ============================================================
def _m1_forecast(alloc,T=4,budget=3000):
    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    K,D,AI,H,A,L0=27500.,20.3,86.,30.,33.70,53.9
    traj=[A*K**a*L0**b*D**g*AI**d*H**th]
    for t in range(T):
        K=.95*K+alloc['K']*budget; D=.88*D+alloc['D']*budget*.01
        AI=.85*AI+alloc['AI']*budget*.05; H=H+.8*alloc['H']*budget*.01-.02*H
        A=A*(1+.003*D/100+.002*AI/100+.004*H/100); L=L0*1.009**(t+1)
        traj.append(A*K**a*L**b*D**g*AI**d*H**th)
    return traj

M12_SCEN={
    'S1 · Truyền thống':  {'K':.70,'D':.10,'AI':.10,'H':.10},
    'S2 · Số hóa nhanh':  {'K':.25,'D':.45,'AI':.15,'H':.15},
    'S3 · AI dẫn dắt':    {'K':.20,'D':.20,'AI':.45,'H':.15},
    'S4 · Bao trùm số':   {'K':.30,'D':.20,'AI':.10,'H':.40},
    'S5 · Tối ưu cân bằng':{'K':.25,'D':.25,'AI':.30,'H':.20}}


def page_bai12():
    section_header("🇻🇳","Bài 12 — Đồ án tích hợp AIDEOM-VN",
                   "6 module M1→M6 · 5 kịch bản chính sách · Dashboard tổng hợp")

    years=list(range(2026,2031))
    gdp_fc={n:_m1_forecast(al) for n,al in M12_SCEN.items()}

    a,b,g,d,th=0.33,0.42,0.10,0.08,0.07
    Y=MACRO["GDP_trillion_VND"].values
    A=Y/(K_HIST**a*L_HIST**b*D_HIST**g*AI_HIST**d*H_HIST**th)
    Y_hat=A.mean()*(K_HIST**a*L_HIST**b*D_HIST**g*AI_HIST**d*H_HIST**th)
    mape=np.mean(np.abs((Y-Y_hat)/Y))*100
    Y2030=gdp_fc['S5 · Tối ưu cân bằng'][-1]

    # Module status
    st.markdown("""
    <div style='display:flex;gap:10px;flex-wrap:wrap;margin:16px 0;'>
      <div style='background:rgba(46,196,182,.12);border:1px solid rgba(46,196,182,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#2ec4b6;font-weight:700;'>✅ M1 Cobb-Douglas</div>
      <div style='background:rgba(46,196,182,.12);border:1px solid rgba(46,196,182,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#2ec4b6;font-weight:700;'>✅ M2 TOPSIS</div>
      <div style='background:rgba(46,196,182,.12);border:1px solid rgba(46,196,182,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#2ec4b6;font-weight:700;'>✅ M3 LP Phân bổ</div>
      <div style='background:rgba(46,196,182,.12);border:1px solid rgba(46,196,182,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#2ec4b6;font-weight:700;'>✅ M4 NetJob</div>
      <div style='background:rgba(46,196,182,.12);border:1px solid rgba(46,196,182,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#2ec4b6;font-weight:700;'>✅ M5 Stochastic</div>
      <div style='background:rgba(72,149,239,.12);border:1px solid rgba(72,149,239,.3);
           border-radius:8px;padding:8px 14px;font-size:.8rem;color:#4895ef;font-weight:700;'>📊 M6 Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    kpi_row([(c1,"MAPE M1 (Cobb-Douglas)",f"{mape:.2f}%"),
             (c2,"TFP trung bình Ā",f"{A.mean():.4f}"),
             (c3,"GDP 2030 dự báo (S5)",f"{Y2030:,.0f} ng.tỷ")])

    t1,t2,t3,t4=st.tabs(["📊 Tổng quan M1-M2","💰 Phân bổ M3",
                          "🎬 5 Kịch bản M6","⚠️ Rủi ro M4-M5"])

    with t1:
        section_header("📐","M1 — Dự báo kinh tế (Cobb-Douglas)")
        gs={"TFP (A)":(np.log(A[-1])-np.log(A[0]))/5,
            "Vốn (K)":a*(np.log(K_HIST[-1])-np.log(K_HIST[0]))/5,
            "Lao động (L)":b*(np.log(L_HIST[-1])-np.log(L_HIST[0]))/5,
            "Số hóa (D)":g*(np.log(D_HIST[-1])-np.log(D_HIST[0]))/5,
            "AI":d*(np.log(AI_HIST[-1])-np.log(AI_HIST[0]))/5,
            "Nhân lực số (H)":th*(np.log(H_HIST[-1])-np.log(H_HIST[0]))/5}
        dec=pd.DataFrame({"Yếu tố":list(gs.keys()),
                          "Đóng góp (%/năm)":[v*100 for v in gs.values()]})
        fig=px.bar(dec,x="Yếu tố",y="Đóng góp (%/năm)",
                   color="Đóng góp (%/năm)",color_continuous_scale=["#1e3050","#e63946"],
                   **_px_cfg("Phân rã đóng góp tăng trưởng GDP 2020-2025"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

        section_header("🏆","M2 — Đánh giá sẵn sàng số (TOPSIS)")
        X=REGIONS[TOPSIS_CRIT].values.astype(float)
        w_exp=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
        C_exp=_topsis(X,w_exp,IS_BENEFIT); C_ent=_topsis(X,_entropy_w(X),IS_BENEFIT)
        rn=REGIONS[REG_NAME_COL].values if REG_NAME_COL else np.array(REGIONS_VI)
        m2=pd.DataFrame({"Vùng":rn,"C* Expert":C_exp.round(4),"C* Entropy":C_ent.round(4)})
        fig=go.Figure()
        fig.add_bar(y=m2["Vùng"],x=m2["C* Expert"],name="Expert",
                    orientation="h",marker_color="#4895ef",opacity=0.85)
        fig.add_bar(y=m2["Vùng"],x=m2["C* Entropy"],name="Entropy",
                    orientation="h",marker_color="#e63946",opacity=0.85)
        fig.update_layout(**_plot_cfg("TOPSIS Score — Expert vs Entropy weight"),barmode="group")
        st.plotly_chart(fig,use_container_width=True)

    with t2:
        section_header("🗺️","M3 — Tối ưu phân bổ ngành-vùng (LP, có ràng buộc công bằng C5)")
        x_opt,Z_lp=_solve_lp4(True)
        _c12, _c13 = st.columns(2)
        kpi_row([(_c12, "LP Z* GDP gain", f"{Z_lp:,.0f} tỷ VND")])
        fig=px.imshow(x_opt,x=ITEMS,y=REGIONS_VI,aspect="auto",text_auto=".0f",
                      color_continuous_scale="Blues",
                      **_px_cfg(f"Phân bổ tối ưu 6 vùng × 4 hạng mục (Z*={Z_lp:,.0f} tỷ)"))
        st.plotly_chart(fig,use_container_width=True)
        tot=x_opt.sum()
        info_box(f"Tỷ lệ: I={x_opt[:,0].sum()/tot*100:.1f}% · D={x_opt[:,1].sum()/tot*100:.1f}% · "
                 f"AI={x_opt[:,2].sum()/tot*100:.1f}% · H={x_opt[:,3].sum()/tot*100:.1f}%")

    with t3:
        section_header("🎬","M6 — So sánh 5 kịch bản chính sách 2026-2030")
        fig=go.Figure()
        for (n,traj),col in zip(gdp_fc.items(),PALETTE):
            fig.add_trace(go.Scatter(x=years,y=traj,mode="lines+markers",name=n,
                                     line=dict(color=col,width=2.5),marker=dict(size=7)))
        fig.add_hline(y=gdp_fc['S5 · Tối ưu cân bằng'][-1],line_dash="dot",
                      line_color="#5e7a99",annotation_text="GDP 2030 S5")
        fig.update_layout(**_plot_cfg("Quỹ đạo GDP 2026-2030 theo 5 kịch bản chính sách"),
                          xaxis_title="Năm",yaxis_title="GDP (ng.tỷ VND)")
        st.plotly_chart(fig,use_container_width=True)
        tbl=pd.DataFrame({"Kịch bản":list(gdp_fc.keys()),
                          "GDP 2026 (ng.tỷ)":[round(t[0]) for t in gdp_fc.values()],
                          "GDP 2030 (ng.tỷ)":[round(t[-1]) for t in gdp_fc.values()],
                          "CAGR (%/năm)":[round(((t[-1]/t[0])**(1/4)-1)*100,2) for t in gdp_fc.values()]})
        st.dataframe(tbl,use_container_width=True,hide_index=True)
        info_box("Kịch bản <b>S5 Tối ưu cân bằng (25%K+25%D+30%AI+20%H)</b> cho GDP 2030 cao nhất trong khi duy trì cân đối. <b>S3 AI dẫn dắt</b> cũng mạnh nhưng rủi ro nhân lực cao hơn.")

    with t4:
        from scipy.optimize import linprog
        section_header("👷","M4 — Mô phỏng thị trường lao động (NetJob ròng)")
        N_s=8
        sec_s=['Nông-LT','CN chế biến','Xây dựng','Bán buôn','Tài chính','Logistics','CNTT','Giáo dục']
        a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
        b1=np.array([45,28,35,32,22,30,20,55])
        c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
        d1=np.array([50,32,42,38,26,36,24,62])
        risk=np.array([18,42,25,38,52,35,28,22])/100; coeff=a1-c1*risk
        c_obj=np.concatenate([-coeff,-b1])
        A1=np.concatenate([np.ones(N_s),np.ones(N_s)]).reshape(1,-1)
        A2=np.zeros((N_s,2*N_s)); A3=np.zeros((N_s,2*N_s))
        for i in range(N_s):
            A2[i,i]=-coeff[i]; A2[i,N_s+i]=-b1[i]
            A3[i,i]=c1[i]*risk[i]; A3[i,N_s+i]=-d1[i]
        res=linprog(c_obj,A_ub=np.vstack([A1,A2,A3]),
                    b_ub=np.concatenate([[30000],np.zeros(N_s),np.zeros(N_s)]),
                    bounds=[(0,None)]*(2*N_s),method="highs")
        NJ=coeff*res.x[:N_s]+b1*res.x[N_s:]
        _cj1, _cj2 = st.columns(2)
        kpi_row([(_cj1, "Tổng NetJob ròng", f"{-res.fun:,.0f} việc")])
        fig=px.bar(x=sec_s,y=NJ.round(0),color=NJ,
                   color_continuous_scale=["#e63946","#f4a261","#2ec4b6"],
                   **_px_cfg("NetJob ròng theo ngành — tối ưu 30.000 tỷ"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

        section_header("🎲","M5 — Đánh giá rủi ro (Stochastic Programming)")
        x_sp,y_sp,Z_sp,Z_ev,Z_ws,det=_solve_sp()
        c1_m,c2_m,c3_m=st.columns(3)
        kpi_row([(c1_m,"Z* Stochastic",f"{Z_sp:,.0f} tỷ"),
                 (c2_m,"VSS (giá trị bất định)",f"{Z_sp-Z_ev:,.0f} tỷ"),
                 (c3_m,"EVPI (giá trị thông tin)",f"{Z_ws-Z_sp:,.0f} tỷ")])
        scn=pd.DataFrame({"Kịch bản":["Lạc quan","Cơ sở","Bi quan","Khủng hoảng"],
                          "Xác suất":[0.30,0.45,0.20,0.05],
                          "Z*[s] (tỷ)":[round(det[s]["Z"],0) for s in S10]})
        fig=px.bar(scn,x="Kịch bản",y="Z*[s] (tỷ)",color="Kịch bản",
                   color_discrete_sequence=["#2ec4b6","#4895ef","#f4a261","#e63946"],
                   **_px_cfg(f"Z* theo kịch bản · Z*_SP = {Z_sp:,.0f} tỷ"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        info_box("""<b>Khuyến nghị tích hợp:</b> Kịch bản <b>S5 (cân bằng)</b> cho GDP 2030 cao nhất trong khi
        giữ công bằng vùng và NetJob dương. SP đầu tư H (nhân lực) nhiều hơn EV — H là "hàng hóa bảo hiểm".
        <br><br><b>Lưu ý quan trọng:</b> AIDEOM-VN là <i>công cụ hỗ trợ</i> — quyết định cuối cùng thuộc về quy trình chính sách dân chủ.""")


# ============================================================
# ROUTER
# ============================================================
ROUTES={
    PAGES[0]:page_home,  PAGES[1]:page_bai1,  PAGES[2]:page_bai2,
    PAGES[3]:page_bai3,  PAGES[4]:page_bai4,  PAGES[5]:page_bai5,
    PAGES[6]:page_bai6,  PAGES[7]:page_bai7,  PAGES[8]:page_bai8,
    PAGES[9]:page_bai9,  PAGES[10]:page_bai10, PAGES[11]:page_bai11,
    PAGES[12]:page_bai12,
}
ROUTES[page]()
