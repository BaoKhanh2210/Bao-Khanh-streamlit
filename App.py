# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
=============================================================
Mỗi bài trình bày theo 5 trang con:
  Bối cảnh · Mô hình · Dữ liệu · Tính toán · Chính sách
Phần "Tính toán" được trình bày chi tiết: các bước giải, bảng số đầy đủ,
diễn giải kết quả — nhằm thuyết phục người chấm.

Bài tập lớn: Các mô hình ra quyết định
Họ và tên : Nguyễn Đình Bảo Nghĩa
Mã sinh viên: 23052345

Chạy:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="AIDEOM-VN · Mô hình ra quyết định",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------ palette
ACC1 = "#6366f1"   # indigo
ACC2 = "#0ea5e9"   # sky
ACC3 = "#f59e0b"   # amber
ACC4 = "#10b981"   # emerald
ACC5 = "#ef4444"   # red
EDGE = "#ffffff"
PALETTE = [ACC1, ACC2, ACC3, ACC4, "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]

st.markdown(
    """
    <style>
    :root{ --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --c1:#6366f1; --c2:#0ea5e9; }
    .stApp{ background:#f1f5f9; }
    .block-container{ padding-top:1.1rem; max-width:1300px; }
    h1,h2,h3,h4{ color:var(--ink); }

    .hero{ background:linear-gradient(120deg,#4f46e5,#6366f1 45%,#0ea5e9);
        border-radius:20px; padding:26px 32px; color:#fff;
        box-shadow:0 12px 30px rgba(79,70,229,.25); margin-bottom:18px;}
    .hero h1{ color:#fff; font-size:2.3rem; font-weight:800; margin:0; letter-spacing:-1px;}
    .hero .tag{ font-size:1rem; opacity:.92; }

    .kpi{ background:#fff; border-radius:16px; padding:15px 18px; border:1px solid var(--line);
        box-shadow:0 4px 14px rgba(2,6,23,.05); height:100%;}
    .kpi .lab{ color:var(--muted); font-size:.82rem; font-weight:600; }
    .kpi .val{ color:var(--ink); font-size:1.55rem; font-weight:800; line-height:1.15;}
    .kpi .dl{ display:inline-block; margin-top:5px; background:#ecfdf5; color:#059669;
        padding:1px 9px; border-radius:14px; font-size:.76rem; font-weight:700;}

    .softcard{ background:#fff; border-radius:16px; padding:16px 20px; border:1px solid var(--line);
        box-shadow:0 4px 14px rgba(2,6,23,.05); }
    .pill{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.8rem;}

    /* khối "các bước giải" */
    .step{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--c1);
        border-radius:10px; padding:12px 16px; margin:8px 0;
        box-shadow:0 2px 8px rgba(2,6,23,.04);}
    .step .h{ font-weight:800; color:#3730a3; font-size:.96rem; }
    .step .b{ color:#334155; font-size:.9rem; margin-top:3px;}

    /* khối diễn giải kết quả */
    .insight{ background:linear-gradient(135deg,#eef2ff,#e0f2fe); border:1px solid #c7d2fe;
        border-radius:12px; padding:14px 18px; margin:10px 0;}
    .insight .t{ font-weight:800; color:#3730a3; margin-bottom:4px;}
    .insight .b{ color:#1e293b; font-size:.92rem; }

    .badge{ display:inline-block;padding:2px 10px;border-radius:8px;font-weight:700;
        font-size:.78rem;background:#eef2ff;color:#4338ca;margin-right:6px;}

    .stTabs [data-baseweb="tab-list"]{ gap:4px; flex-wrap:wrap; }
    .stTabs [data-baseweb="tab"]{ background:#fff; border:1px solid var(--line); border-bottom:none;
        border-radius:11px 11px 0 0; padding:8px 15px; font-weight:600; color:var(--ink);}
    .stTabs [aria-selected="true"]{ background:var(--c1); color:#fff; border-color:var(--c1);}
    div[data-testid="stMetricValue"]{ color:var(--c1); }
    .sig{ background:linear-gradient(135deg,#eef2ff,#e0f2fe); border-radius:14px; padding:15px 18px;
        border:1px solid #c7d2fe;}
    .sig .nm{ font-weight:800; color:#3730a3; font-size:1.02rem;}
    .sig .mt{ color:#475569; font-size:.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    "text.color": "#0f172a", "axes.labelcolor": "#334155", "xtick.color": "#475569",
    "ytick.color": "#475569", "axes.edgecolor": "#cbd5e1", "grid.color": "#e2e8f0", "font.size": 10,
})

# ============================================================ DATA
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(f):
    for d in (DATA_DIR, os.path.join(DATA_DIR, "data"), "."):
        p = os.path.join(d, f)
        if os.path.exists(p):
            return p
    return f


@st.cache_data(show_spinner=False)
def load_macro():
    return pd.read_csv(_find("vietnam_macro_2020_2025.csv")).sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sectors():
    return pd.read_csv(_find("vietnam_sectors_2024.csv"))


@st.cache_data(show_spinner=False)
def load_regions():
    return pd.read_csv(_find("vietnam_regions_2024.csv"))


SECTOR_VI = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Khai khoáng",
             "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
             "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"]
REGION_VI = ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng", "Bắc Trung Bộ + DH Trung Bộ",
             "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"]
REGION_CODE = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
EN2CODE = {"Northern Midlands and Mountains": "NMM", "Red River Delta": "RRD",
           "North Central and South Central Coast": "NCC", "Central Highlands": "CH",
           "Southeast": "SE", "Mekong Delta": "MD"}

ITEMS = ["I", "D", "AI", "H"]
ITEM_NAMES = {"I": "Hạ tầng số", "D": "Chuyển đổi số DN", "AI": "Năng lực AI", "H": "Nhân lực số"}
BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}


# ============================================================ UI COMPONENTS
def section(title, desc=None):
    st.markdown(
        f'<div style="border-left:4px solid {ACC1};padding-left:12px;margin:16px 0 6px;">'
        f'<span style="font-weight:800;font-size:1.1rem;color:#0f172a;">{title}</span></div>',
        unsafe_allow_html=True)
    if desc:
        st.caption(desc)


def bai_header(emoji, title, sub, tags):
    chips = " ".join(f'<span class="badge">{t}</span>' for t in tags)
    st.markdown(
        f'<div class="hero" style="padding:20px 26px;"><h1 style="font-size:1.7rem;">{emoji} {title}</h1>'
        f'<div class="tag">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown(chips, unsafe_allow_html=True)
    st.write("")


def steps(items):
    """items: list of (header, body)."""
    for i, (h, b) in enumerate(items, 1):
        st.markdown(
            f'<div class="step"><div class="h">Bước {i}. {h}</div>'
            f'<div class="b">{b}</div></div>', unsafe_allow_html=True)


def insight(title, body):
    st.markdown(
        f'<div class="insight"><div class="t">💡 {title}</div><div class="b">{body}</div></div>',
        unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, delta or None)."""
    for col, it in zip(st.columns(len(items)), items):
        lab, val = it[0], it[1]
        dl = it[2] if len(it) > 2 and it[2] else ""
        dlh = f'<div class="dl">{dl}</div>' if dl else ""
        col.markdown(f'<div class="kpi"><div class="lab">{lab}</div>'
                     f'<div class="val">{val}</div>{dlh}</div>', unsafe_allow_html=True)


def render_tabs(ctx, model, data, calc, policy):
    """Khung 5 trang con cho mỗi bài."""
    t = st.tabs(["🧭 Bối cảnh", "📐 Mô hình", "🗃️ Dữ liệu", "🧮 Tính toán", "🏛️ Chính sách"])
    with t[0]:
        ctx()
    with t[1]:
        model()
    with t[2]:
        data()
    with t[3]:
        calc()
    with t[4]:
        policy()


# ============================================================ HÀM TÍNH TOÁN DÙNG CHUNG
def topsis(X, w, is_benefit):
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    return S_neg / (S_star + S_neg), R, V, A_star, A_neg, S_star, S_neg


def entropy_weights(X):
    P = X / X.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


# ============================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG
# ============================================================
def page_bai1():
    bai_header("🌱", "Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng (AI & số hóa)",
               "Growth accounting trên dữ liệu Việt Nam 2020-2025 · dự báo GDP 2030",
               ["Cấp độ: Dễ", "numpy · pandas", "Growth accounting"])

    df = load_macro()
    years = df["year"].values
    Y = df["GDP_trillion_VND"].values
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = df["digital_economy_share_GDP_pct"].values.astype(float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

    # ---------------- Bối cảnh ----------------
    def ctx():
        st.markdown("""
        Theo Cục Thống kê quốc gia, **GDP Việt Nam 2024 đạt 11.511,9 nghìn tỷ VND**
        (tăng 7,09%), năng suất lao động 221,9 triệu VND/người và đạt 245,0 triệu VND/người
        năm 2025. Đóng góp của khoa học – công nghệ vào GDP năm 2025 ước **2,49%**, kinh tế số
        chiếm khoảng **18,3–19,5% GDP**.
        """)
        st.markdown("**Câu hỏi nghiên cứu:**")
        st.markdown(
            "- Nếu mô hình hóa nền kinh tế bằng hàm Cobb-Douglas **mở rộng** thêm số hóa **D**, "
            "năng lực **AI**, vốn nhân lực số **H** thì dự báo có khớp số liệu thực tế?\n"
            "- Yếu tố nào đóng góp **lớn nhất** vào tăng trưởng giai đoạn 2020-2025?")
        kpi_row([("GDP 2025", "12.847,6 ngh.tỷ", "▲ 8,02%"),
                 ("Kinh tế số/GDP", "19,5%", "▲ 1,2 đpt"),
                 ("KH-CN đóng góp", "2,49% GDP", None),
                 ("NSLĐ 2025", "245 tr.VND/người", None)])

    # ---------------- Mô hình ----------------
    def model():
        st.markdown("**Hàm sản xuất Cobb-Douglas mở rộng** (lợi suất không đổi theo quy mô):")
        st.latex(r"Y_t = A_t \cdot K_t^{\alpha}\, L_t^{\beta}\, D_t^{\gamma}\, AI_t^{\delta}\, H_t^{\theta}, "
                 r"\quad \alpha+\beta+\gamma+\delta+\theta = 1")
        st.markdown("**Tuyến tính hóa (log)** để phân rã tăng trưởng (growth accounting):")
        st.latex(r"\Delta\ln Y_t = \Delta\ln A_t + \alpha\Delta\ln K_t + \beta\Delta\ln L_t "
                 r"+ \gamma\Delta\ln D_t + \delta\Delta\ln AI_t + \theta\Delta\ln H_t")
        st.markdown("**Giải ngược TFP** từ hàm sản xuất:")
        st.latex(r"A_t = \dfrac{Y_t}{K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta}}")
        st.markdown("Trong đó: **K** vốn vật chất · **L** lao động · **D** tỷ trọng kinh tế số · "
                    "**AI** số DN công nghệ số · **H** tỷ lệ lao động qua đào tạo · **A** TFP.")

    # ---------------- Dữ liệu ----------------
    def data():
        st.markdown("Trích từ `vietnam_macro_2020_2025.csv` và nguồn bổ sung Bộ KH-CN, Bộ TT-TT.")
        tb = pd.DataFrame({
            "Năm": years, "Y (GDP, ngh.tỷ)": Y, "K (vốn, ngh.tỷ)": K, "L (tr.LĐ)": L,
            "D (KTS/GDP %)": D, "AI (ngh.DN)": AI, "H (LĐ ĐT %)": H})
        st.dataframe(tb, width='stretch', hide_index=True)
        st.markdown("**Hệ số co giãn đề xuất** (Mục 6 bài báo):")
        st.dataframe(pd.DataFrame({
            "Tham số": ["α (vốn K)", "β (lao động L)", "γ (số hóa D)", "δ (AI)", "θ (nhân lực H)"],
            "Giá trị": [0.33, 0.42, 0.10, 0.08, 0.07],
            "Ý nghĩa": ["Độ co giãn theo vốn", "Độ co giãn theo lao động", "Độ co giãn số hóa",
                        "Độ co giãn AI", "Độ co giãn nhân lực số"]}),
            width='stretch', hide_index=True)
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(years, Y, "o-", color=ACC1, lw=2)
        ax.set_title("GDP thực tế 2020-2025"); ax.set_ylabel("ngh.tỷ VND"); ax.grid(alpha=.3)
        c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for s, lb, cc in [(D, "D (%)", ACC2), (H, "H (%)", ACC3)]:
            ax.plot(years, s, "o-", label=lb, color=cc)
        ax.set_title("Số hóa D & nhân lực H"); ax.legend(); ax.grid(alpha=.3)
        c2.pyplot(fig)

    # ---------------- Tính toán ----------------
    def calc():
        st.markdown("##### ⚙️ Tham số mô hình")
        with st.container():
            c = st.columns(5)
            alpha = c[0].number_input("α (K)", 0.0, 1.0, 0.33, 0.01)
            beta = c[1].number_input("β (L)", 0.0, 1.0, 0.42, 0.01)
            gamma = c[2].number_input("γ (D)", 0.0, 1.0, 0.10, 0.01)
            delta = c[3].number_input("δ (AI)", 0.0, 1.0, 0.08, 0.01)
            theta = c[4].number_input("θ (H)", 0.0, 1.0, 0.07, 0.01)
        s = alpha + beta + gamma + delta + theta
        (st.success if abs(s - 1) < 1e-9 else st.warning)(
            f"Σ hệ số = {s:.2f} " + ("✅ lợi suất không đổi theo quy mô" if abs(s-1) < 1e-9 else "⚠️ ≠ 1"))

        A = Y / (K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta)

        # ---- 1.4.1
        section("Câu 1.4.1 — Ước lượng TFP A_t (giải ngược hàm sản xuất)")
        steps([("Thay số liệu mỗi năm vào công thức giải ngược A_t",
                "Chia GDP thực tế cho tích các yếu tố đầu vào đã lũy thừa hệ số co giãn."),
               ("Tính tốc độ tăng TFP bình quân",
                "Dùng công thức tăng trưởng kép (CAGR) trên A_t từ 2020→2025.")])
        g_A_tb = ((A[-1] / A[0]) ** (1 / 5) - 1) * 100
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(years, A, "o-", color=ACC1, lw=2, ms=8)
        for x, a in zip(years, A):
            ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)
        ax.set_xlabel("Năm"); ax.set_ylabel("A_t (TFP)"); ax.grid(alpha=.3)
        ax.set_title("Năng suất nhân tố tổng hợp 2020-2025")
        c1.pyplot(fig)
        detail = pd.DataFrame({
            "Năm": years, "Y": Y.round(1),
            "Mẫu số": (K**alpha*L**beta*D**gamma*AI**delta*H**theta).round(2),
            "A_t": A.round(4)})
        c2.dataframe(detail, width='stretch', hide_index=True)
        c2.metric("TFP tăng bình quân", f"{g_A_tb:.2f}%/năm")
        insight("Diễn giải", f"TFP tăng từ {A[0]:.3f} (2020) lên {A[-1]:.3f} (2025), "
                f"bình quân <b>{g_A_tb:.2f}%/năm</b> → chất lượng tăng trưởng cải thiện, "
                "nền kinh tế dựa nhiều hơn vào năng suất thay vì chỉ tích lũy vốn.")

        # ---- 1.4.2
        section("Câu 1.4.2 — Dự báo Ŷ với A trung bình & độ chính xác (MAPE)")
        A_mean = A.mean()
        Y_hat = A_mean * (K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta)
        ape = np.abs((Y - Y_hat) / Y) * 100
        mape = ape.mean()
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(years, Y, "o-", color=ACC2, lw=2, label="Y thực tế")
        ax.plot(years, Y_hat, "s--", color=ACC1, lw=2, label="Ŷ dự báo")
        ax.fill_between(years, Y, Y_hat, alpha=.12, color=ACC1)
        ax.set_xlabel("Năm"); ax.set_ylabel("GDP (ngh.tỷ VND)"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title(f"Dự báo vs Thực tế — MAPE = {mape:.2f}%")
        c1.pyplot(fig)
        kpi_row([("MAPE", f"{mape:.2f}%", "rất tốt" if mape < 5 else "khá")])
        c2.dataframe(pd.DataFrame({"Năm": years, "Y": Y.round(0), "Ŷ": Y_hat.round(0),
                                   "Sai số %": ((Y_hat - Y) / Y * 100).round(2)}),
                     width='stretch', hide_index=True)
        insight("Diễn giải", f"Với A cố định = trung bình ({A_mean:.3f}), mô hình tái lập GDP "
                f"với sai số tuyệt đối trung bình chỉ <b>{mape:.2f}%</b> — hàm Cobb-Douglas mở rộng "
                "mô tả tốt động học tăng trưởng Việt Nam.")

        # ---- 1.4.3
        section("Câu 1.4.3 — Phân rã tăng trưởng (growth accounting)")
        n = 5
        g = {k: (np.log(v[-1]) - np.log(v[0])) / n
             for k, v in {"Y": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H, "A": A}.items()}
        contrib = {"TFP": g["A"], "K (Vốn)": alpha * g["K"], "L (Lao động)": beta * g["L"],
                   "D (Số hóa)": gamma * g["D"], "AI": delta * g["AI"], "H (Nhân lực)": theta * g["H"]}
        pct = {k: v / g["Y"] * 100 for k, v in contrib.items()}
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(list(pct.keys()), list(pct.values()), color=PALETTE[:6], edgecolor=EDGE)
        for b, v in zip(bars, pct.values()):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Đóng góp (%)"); ax.axhline(0, color="#94a3b8"); ax.grid(axis="y", alpha=.3)
        ax.set_title("Phân rã tăng trưởng GDP bình quân năm")
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
        c1.pyplot(fig)
        c2.dataframe(pd.DataFrame({
            "Yếu tố": list(contrib.keys()),
            "Δln/năm": [round(v, 4) for v in contrib.values()],
            "% đóng góp": [round(v, 1) for v in pct.values()]}),
            width='stretch', hide_index=True)
        c2.metric("Tăng trưởng GDP bình quân", f"{g['Y']*100:.2f}%/năm")
        top_new = max([("D", pct["D (Số hóa)"]), ("AI", pct["AI"]), ("H", pct["H (Nhân lực)"])],
                      key=lambda t: t[1])
        insight("Diễn giải", f"Trong ba yếu tố mới, <b>{top_new[0]}</b> đóng góp lớn nhất "
                f"(~{top_new[1]:.1f}% tăng trưởng). Vốn K vẫn là động lực chính, nhưng nhóm "
                "số hóa + AI + nhân lực số đã tạo đóng góp đáng kể, phản ánh chuyển dịch sang "
                "mô hình tăng trưởng dựa trên công nghệ.")

        # ---- 1.4.4
        section("Câu 1.4.4 — Kịch bản dự báo GDP 2030")
        c = st.columns(4)
        D30 = c[0].slider("D 2030 (% GDP)", 20.0, 40.0, 30.0, .5)
        AI30 = c[1].slider("AI 2030 (ngh.DN)", 80, 150, 100, 5)
        H30 = c[2].slider("H 2030 (%)", 30.0, 45.0, 35.0, .5)
        tfp_g = c[3].slider("TFP tăng (%/năm)", 0.0, 3.0, 1.2, .1)
        K30 = K[-1] * 1.06 ** 5
        L30 = L[-1] * 1.005 ** 5
        A30 = A[-1] * (1 + tfp_g / 100) ** 5
        Y30 = A30 * (K30 ** alpha * L30 ** beta * D30 ** gamma * AI30 ** delta * H30 ** theta)
        gr = ((Y30 / Y[-1]) ** (1 / 5) - 1) * 100
        usd = Y30 * 1e12 / (110e6) / 25500
        kpi_row([("GDP 2030 dự báo", f"{Y30:,.0f} ngh.tỷ", None),
                 ("Tăng trưởng 25-30", f"{gr:.2f}%/năm", None),
                 ("GDP/người 2030", f"≈ {usd:,.0f} USD", None)])
        proj_years = np.arange(2025, 2031)
        proj = Y[-1] * (1 + gr / 100) ** (proj_years - 2025)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(years, Y, "o-", color=ACC2, label="Thực tế 2020-25")
        ax.plot(proj_years, proj, "s--", color=ACC1, label="Dự báo 2025-30")
        ax.set_ylabel("GDP (ngh.tỷ VND)"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Quỹ đạo GDP đến 2030")
        st.pyplot(fig)

    # ---------------- Chính sách ----------------
    def policy():
        st.markdown("##### Câu hỏi thảo luận chính sách")
        with st.expander("a) TFP tăng hay giảm? Nói lên gì về chất lượng tăng trưởng?", expanded=True):
            st.markdown("TFP **tăng đều** trong 2020-2025 cho thấy tăng trưởng ngày càng dựa vào "
                        "năng suất và đổi mới công nghệ, không chỉ vào mở rộng vốn — dấu hiệu của "
                        "tăng trưởng **chất lượng và bền vững** hơn.")
        with st.expander("b) Yếu tố mới nào (D, AI, H) đóng góp nhiều nhất? Vì sao?"):
            st.markdown("**Số hóa D** thường dẫn đầu do tốc độ tăng nhanh (12% → 19,5% GDP) và "
                        "độ co giãn γ tương đối lớn. AI và H tăng chậm hơn nên đóng góp thấp hơn, "
                        "nhưng là nền tảng để D phát huy hiệu quả.")
        with st.expander("c) Mục tiêu 30% kinh tế số/GDP vào 2030 có khả thi không?"):
            st.markdown("Khả thi nếu duy trì đà tăng D ≈ 1,5 đpt/năm và TFP ≥ 1,2%/năm. Cần ràng "
                        "buộc đi kèm: đầu tư hạ tầng số, đào tạo nhân lực số (H) và khung pháp lý "
                        "dữ liệu — nếu không, D sẽ chững lại do thiếu năng lực hấp thụ.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 2 — LP NGÂN SÁCH 4 HẠNG MỤC
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    bai_header("💰", "Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số",
               "scipy.optimize.linprog · giá đối ngẫu (shadow price) · phân tích độ nhạy",
               ["Cấp độ: Dễ", "Linear Programming", "Shadow price"])

    COBJ = [-0.85, -1.20, -0.95, -1.35]
    names = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]

    def solve(B, x3min):
        A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
        b_ub = [B, -25, -15, -x3min, -10, 0]
        return linprog(COBJ, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    def ctx():
        st.markdown("""
        Theo **Quyết định 749/QĐ-TTg**, đến 2025 Việt Nam đặt mục tiêu kinh tế số đạt 20% GDP.
        Giả sử Bộ Kế hoạch – Đầu tư phân bổ **100.000 tỷ VND** ngân sách trung ương 2026 cho
        4 hạng mục: hạ tầng số (I), AI & dữ liệu (AI), nhân lực số (H), R&D công nghệ.
        Mỗi hạng mục có hệ số tác động khác nhau tới tăng GDP, phải tuân thủ tỷ lệ tối thiểu
        theo **Quyết định 411/QĐ-TTg**.
        """)
        st.markdown("**Mục tiêu:** phân bổ tối ưu để tối đa hóa tăng GDP kỳ vọng, đồng thời "
                    "hiểu ý nghĩa **giá đối ngẫu** trong phân tích chính sách.")
        kpi_row([("Ngân sách", "100.000 tỷ", None), ("Số hạng mục", "4", None),
                 ("Số ràng buộc", "6", None), ("Mục tiêu KTS", "20% GDP", None)])

    def model():
        st.markdown("**Biến quyết định** (nghìn tỷ VND): x₁ hạ tầng · x₂ AI · x₃ nhân lực · x₄ R&D")
        st.markdown("**Hàm mục tiêu** (tối đa hóa tăng GDP, nghìn tỷ VND):")
        st.latex(r"\max Z = 0{,}85\,x_1 + 1{,}20\,x_2 + 0{,}95\,x_3 + 1{,}35\,x_4")
        st.markdown("**Ràng buộc:**")
        st.latex(r"""\begin{aligned}
        & x_1+x_2+x_3+x_4 \le 100 \quad &\text{(ngân sách tổng)}\\
        & x_1 \ge 25,\; x_2 \ge 15,\; x_3 \ge 20,\; x_4 \ge 10 \quad &\text{(sàn mỗi hạng mục)}\\
        & x_2 + x_4 \ge 0{,}35(x_1+x_2+x_3+x_4) \quad &\text{(công nghệ chiến lược)}\\
        & x_1,x_2,x_3,x_4 \ge 0
        \end{aligned}""")
        st.caption("Hệ số 0,85–1,35 = số đồng GDP tăng thêm cho mỗi đồng đầu tư (World Bank 2024, OECD 2024).")

    def data():
        st.dataframe(pd.DataFrame({
            "Hạng mục": ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D công nghệ"],
            "Hệ số tác động": [0.85, 1.20, 0.95, 1.35],
            "Sàn tối thiểu (ngh.tỷ)": [25, 15, 20, 10],
            "Diễn giải": ["Thu hồi vốn chậm", "Thu hồi nhanh, cần nhân lực bổ trợ",
                          "Nền tảng hấp thụ công nghệ", "Lan tỏa dài hạn cao nhất"]}),
            width='stretch', hide_index=True)
        insight("Vì sao hệ số khác nhau?",
                "R&D có hệ số cao nhất (1,35) do tác động lan tỏa dài hạn; AI (1,20) cao hơn hạ tầng "
                "(0,85) vì thu hồi vốn nhanh hơn nhưng cần nhân lực số bổ trợ.")

    def calc():
        c = st.columns(2)
        B = c[0].slider("Ngân sách tổng (nghìn tỷ VND)", 100, 200, 100, 10)
        x3min = c[1].slider("Sàn nhân lực số x₃ ≥", 20, 40, 20, 1)
        res = solve(B, x3min)

        section("Câu 2.4.1 / 2.4.4 — Nghiệm tối ưu")
        steps([("Chuyển max → min", "linprog tối thiểu hóa nên đổi dấu hệ số mục tiêu."),
               ("Dựng ma trận ràng buộc A_ub·x ≤ b_ub", "Các sàn x_i ≥ c đổi thành −x_i ≤ −c."),
               ("Giải bằng phương pháp 'highs'", "Solver nội điểm/đơn hình hiệu năng cao.")])
        if res.success:
            x = res.x
            c1, c2 = st.columns([2, 3])
            kpi_row_data = [("Z* (GDP tăng thêm)", f"{-res.fun:.2f} ngh.tỷ", None),
                            ("Tỷ trọng AI+R&D", f"{(x[1]+x[3])/x.sum()*100:.1f}%", None)]
            with c1:
                kpi_row(kpi_row_data)
                st.dataframe(pd.DataFrame({"Hạng mục": names, "Phân bổ (ngh.tỷ)": x.round(2),
                                           "Tỷ trọng %": (x / x.sum() * 100).round(1)}),
                             width='stretch', hide_index=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(names, x, color=[ACC2, "#a855f7", ACC3, ACC1], edgecolor=EDGE)
            for i, v in enumerate(x):
                ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
            ax.set_ylabel("Nghìn tỷ VND"); ax.grid(axis="y", alpha=.3)
            plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
            c2.pyplot(fig)
            insight("Diễn giải", f"Sau khi thỏa các sàn tối thiểu, phần ngân sách còn lại được "
                    f"dồn vào <b>R&D (x₄)</b> — hạng mục có hệ số cao nhất (1,35). Đây là lời giải "
                    "góc điển hình của LP: tối ưu rơi vào đỉnh của miền khả thi.")
        else:
            st.error("Bài toán KHÔNG khả thi với cấu hình hiện tại.")

        section("Câu 2.4.2 — Giá đối ngẫu (shadow price) ràng buộc ngân sách")
        st.markdown("Shadow price = mức tăng Z* khi nới lỏng ràng buộc thêm 1 đơn vị.")

        section("Câu 2.4.3 — Phân tích độ nhạy: đường cong Z*(B)")
        Bs = np.arange(100, 201, 10)
        Zs = [(-solve(b, 20).fun if solve(b, 20).success else np.nan) for b in Bs]
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(Bs, Zs, "o-", color=ACC1, lw=2)
        ax.fill_between(Bs, Zs, alpha=.1, color=ACC1)
        ax.set_xlabel("Ngân sách tổng B (ngh.tỷ)"); ax.set_ylabel("Z* (GDP gain)")
        ax.grid(alpha=.3); ax.set_title("Z*(B) — độ dốc = shadow price ngân sách")
        c1.pyplot(fig)
        slope = (Zs[-1] - Zs[0]) / (Bs[-1] - Bs[0])
        c2.dataframe(pd.DataFrame({"B": Bs, "Z*": np.round(Zs, 2)}), width='stretch', hide_index=True)
        c2.metric("Shadow price ngân sách", f"{slope:.2f}")
        insight("Diễn giải", f"Độ dốc đường Z*(B) ≈ <b>{slope:.2f}</b>: mỗi nghìn tỷ ngân sách tăng "
                "thêm tạo ra ~1,35 nghìn tỷ GDP (bằng hệ số R&D), vì phần tăng được dồn hết vào R&D. "
                "Đây là <b>cận trên hợp lý của chi phí cơ hội vốn công</b>.")

    def policy():
        with st.expander("a) GDP tăng thêm bao nhiêu khi ngân sách +1 tỷ? Cận trên chi phí vốn?", expanded=True):
            st.markdown("Shadow price ≈ 1,35: chừng nào lợi suất biên còn > chi phí huy động vốn "
                        "thì việc mở rộng ngân sách vẫn hiệu quả về phúc lợi.")
        with st.expander("b) Vì sao R&D hệ số cao nhất nhưng sàn tối thiểu thấp nhất?"):
            st.markdown("Do **rủi ro và độ trễ tác động dài hạn** của R&D khiến nhà hoạch định "
                        "thận trọng khi đặt sàn bắt buộc, dù kỳ vọng lợi suất cao.")
        with st.expander("c) Tỷ lệ 35% công nghệ chiến lược có khả thi không?"):
            st.markdown("Tham vọng so với cơ cấu ngân sách thực tế đang ưu tiên hạ tầng giao thông "
                        "và an sinh xã hội — cần lộ trình tăng dần thay vì áp đặt ngay.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH
# ============================================================
def page_bai3():
    bai_header("📊", "Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành Việt Nam",
               "Chuẩn hóa min-max · weighted scoring · phân tích độ nhạy trọng số",
               ["Cấp độ: Dễ", "MCDM", "Min-max normalization"])

    df = load_sectors().copy()
    df["sector_vi"] = SECTOR_VI
    GDP24 = 11511.9
    df["productivity"] = (df["gdp_share_2024_pct"] / 100) * GDP24 / df["labor_million"]
    cols_good = ["growth_rate_2024_pct", "productivity", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    col_bad = "automation_risk_pct"

    def ng(x): return (x - x.min()) / (x.max() - x.min())
    def nb(x): return (x.max() - x) / (x.max() - x.min())
    Xg = df[cols_good].apply(ng)
    Xb = nb(df[col_bad])

    def ctx():
        st.markdown("""
        Cơ cấu kinh tế 2024: nông-lâm-thủy sản 11,86%, công nghiệp-xây dựng 37,64%,
        dịch vụ 42,36% GDP. **Câu hỏi:** trong các ngành lớn, ngành nào nên được ưu tiên
        đẩy mạnh chuyển đổi số và ứng dụng AI **trước** để tạo hiệu ứng lan tỏa tối đa?
        """)
        st.markdown("Cần xây dựng một **chỉ số ưu tiên định lượng** tổng hợp 7 tiêu chí.")
        kpi_row([("Số ngành", "10", None), ("Số tiêu chí", "7", None),
                 ("Tiêu chí 'xấu'", "Rủi ro TĐH", None), ("Chuẩn hóa", "Min-max", None)])

    def model():
        st.markdown("**Công thức chỉ số ưu tiên** (Mục 9 bài báo):")
        st.latex(r"Priority_i = a_1 G_i + a_2 P_i + a_3 S_i + a_4 X_i + a_5 E_i + a_6 AI_i - a_7 R_i")
        st.caption("G tăng trưởng · P năng suất · S lan tỏa · X xuất khẩu · E việc làm · "
                   "AI readiness · R rủi ro tự động hóa")
        st.markdown("**Chuẩn hóa min-max** về [0,1]:")
        st.latex(r"\tilde{x}_i = \frac{x_i - \min x}{\max x - \min x} \quad (\text{tiêu chí tốt})")
        st.latex(r"\tilde{x}_i = \frac{\max x - x_i}{\max x - \min x} \quad (\text{tiêu chí xấu: Risk})")

    def data():
        show = df[["sector_vi", "growth_rate_2024_pct", "productivity", "spillover_coef_0_1",
                   "export_billion_USD", "labor_million", "ai_readiness_0_100", "automation_risk_pct"]].copy()
        show.columns = ["Ngành", "Tăng trưởng%", "Năng suất", "Lan tỏa", "XK(tỷ$)",
                        "Việc làm(tr)", "AI ready", "Rủi ro TĐH%"]
        st.dataframe(show.round(2), width='stretch', hide_index=True)
        st.markdown("**Ma trận sau chuẩn hóa min-max** (Risk đã đảo dấu):")
        norm = Xg.copy(); norm.columns = ["Tăng trưởng", "Năng suất", "Lan tỏa", "XK", "Việc làm", "AI"]
        norm["Risk(đảo)"] = Xb
        norm.insert(0, "Ngành", SECTOR_VI)
        st.dataframe(norm.round(3), width='stretch', hide_index=True)

    def calc():
        section("Câu 3.4.1–3.4.2 — Tính Priority với bộ trọng số")
        labels = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI ready", "Rủi ro"]
        defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15]
        c = st.columns(7)
        raw = [c[i].number_input(labels[i], 0.0, 1.0, defaults[i], 0.01, key=f"w3_{i}") for i in range(7)]
        tot = sum(raw)
        st.caption(f"Tổng trọng số nhập = {tot:.2f} → tự chuẩn hóa về 1.")
        w = np.array(raw[:6]) / tot
        w_risk = raw[6] / tot
        steps([("Chuẩn hóa min-max 7 cột", "Đưa các tiêu chí về cùng thang [0,1], đảo dấu Risk."),
               ("Nhân trọng số & cộng", "Priority = Σ(trọng số × tiêu chí tốt) + w_risk × Risk_đảo."),
               ("Xếp hạng giảm dần", "Ngành Priority cao nhất được ưu tiên trước.")])
        priority = Xg.values @ w + w_risk * Xb.values
        df["Priority"] = priority
        rank = df[["sector_vi", "Priority"]].sort_values("Priority", ascending=False).reset_index(drop=True)
        rank.index += 1
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 5))
        order = df.sort_values("Priority")
        cc = [ACC1 if i >= 7 else ACC2 for i in range(10)]
        ax.barh(order["sector_vi"], order["Priority"], color=cc, edgecolor=EDGE)
        ax.set_xlabel("Priority"); ax.grid(axis="x", alpha=.3); ax.set_title("Xếp hạng ưu tiên 10 ngành")
        c1.pyplot(fig)
        r2 = rank.copy(); r2["Priority"] = r2["Priority"].round(4); r2.columns = ["Ngành", "Priority"]
        c2.dataframe(r2, width='stretch')
        c2.success(f"🥇 Top-3: {', '.join(rank['sector_vi'].head(3))}")
        insight("Diễn giải", f"Top-3 ngành ưu tiên: <b>{', '.join(rank['sector_vi'].head(3))}</b>. "
                "Các ngành dẫn đầu thường có lan tỏa và AI readiness cao — phù hợp tinh thần "
                "Nghị quyết 57-NQ/TW.")

        section("Câu 3.4.3 — Phân tích độ nhạy theo w_AI (heatmap)")
        w_ai_range = np.arange(0.05, 0.45, 0.05)
        base = np.array([0.15, 0.15, 0.20, 0.15, 0.10])
        hm = []
        for wai in w_ai_range:
            rem = 1.0 - wai - 0.15
            ws = base * (rem / base.sum())
            wf = np.append(ws, wai)
            hm.append(Xg.values @ wf + 0.15 * Xb.values)
        hm = np.array(hm)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(w_ai_range))); ax.set_yticklabels([f"{w:.2f}" for w in w_ai_range])
        ax.set_xticks(range(10)); ax.set_xticklabels([s[:10] for s in SECTOR_VI], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("w_AI"); ax.set_title("Priority theo trọng số AI Readiness")
        plt.colorbar(im, label="Priority"); fig.tight_layout()
        st.pyplot(fig)

        section("Câu 3.4.4 — So sánh 2 định hướng chính sách")
        w_growth = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05]); wg_risk = 0.05
        w_incl = np.array([0.05, 0.10, 0.25, 0.05, 0.25, 0.10]); wi_risk = 0.20
        pg = Xg.values @ w_growth + wg_risk * Xb.values
        pi = Xg.values @ w_incl + wi_risk * Xb.values
        rg = pd.Series(pg, index=SECTOR_VI).sort_values(ascending=False)
        ri = pd.Series(pi, index=SECTOR_VI).sort_values(ascending=False)
        c1, c2 = st.columns(2)
        c1.markdown("**🚀 Định hướng tăng trưởng**")
        c1.success("Top-3: " + ", ".join(rg.head(3).index))
        c1.dataframe(rg.round(4).rename("Priority"), width='stretch')
        c2.markdown("**🤝 Định hướng bao trùm**")
        c2.success("Top-3: " + ", ".join(ri.head(3).index))
        c2.dataframe(ri.round(4).rename("Priority"), width='stretch')

    def policy():
        with st.expander("a) Ba ngành nào nên ưu tiên? Có phù hợp Nghị quyết 57-NQ/TW?", expanded=True):
            st.markdown("Các ngành lan tỏa cao (CNTT-TT, Tài chính-Ngân hàng, CN chế biến) thường "
                        "dẫn đầu — phù hợp định hướng đột phá khoa học-công nghệ, đổi mới sáng tạo.")
        with st.expander("b) Vì sao Khai khoáng năng suất cao nhưng không ưu tiên?"):
            st.markdown("Khai khoáng có năng suất rất cao nhưng **tăng trưởng âm, lan tỏa thấp, rủi ro "
                        "tự động hóa lớn (55%)** → bị các tiêu chí khác kéo xuống.")
        with st.expander("c) Trọng số nên do ai quyết định?"):
            st.markdown("Nên qua **quy trình đối thoại công khai** giữa chuyên gia kỹ thuật và hội đồng "
                        "chính sách để bảo đảm tính chính danh, tránh thiên lệch một nhóm lợi ích.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 4 — LP NGÀNH-VÙNG
# ============================================================
@st.cache_data(show_spinner=False)
def _solve_lp4(B, with_equity, lam):
    import pulp
    gamma_val = 0.002
    reg = load_regions()
    D0 = dict(zip(reg["region_name_en"].map(EN2CODE), reg["digital_index_0_100"]))
    m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGION_CODE, ITEMS), lowBound=0)
    m += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGION_CODE for j in ITEMS)
    m += pulp.lpSum(x[r][j] for r in REGION_CODE for j in ITEMS) <= B
    for r in REGION_CODE:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]["H"] for r in REGION_CODE) >= 12000
    if with_equity:
        M = pulp.LpVariable("Dmax")
        for r in REGION_CODE:
            m += D0[r] + gamma_val * x[r]["D"] <= M
            m += D0[r] + gamma_val * x[r]["D"] >= lam * M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat = np.array([[x[r][j].value() for j in ITEMS] for r in REGION_CODE])
    return mat, float(pulp.value(m.objective))


def page_bai4():
    bai_header("🗺️", "Bài 4 — LP phân bổ ngân sách số theo ngành-vùng",
               "PuLP (CBC) · 24 biến · ràng buộc công bằng vùng miền (Mục 7.3)",
               ["Cấp độ: Trung bình", "PuLP / CBC", "Regional equity"])

    def ctx():
        st.markdown("""
        Theo **Quyết định 411/QĐ-TTg**, 6 vùng kinh tế-xã hội có mức sẵn sàng số rất khác nhau.
        Bài toán: phân bổ **50.000 tỷ VND** ngân sách kinh tế số cho 6 vùng × 4 hạng mục
        (I, D, AI, H) sao cho **tối đa hóa GDP gain** nhưng **bảo đảm công bằng vùng miền**.
        """)
        kpi_row([("Ngân sách", "50.000 tỷ", None), ("Số biến", "24", None),
                 ("Số vùng", "6", None), ("Sàn/vùng", "5.000 tỷ", None)])

    def model():
        st.latex(r"\max Z = \sum_r \sum_j \beta_{j,r}\, x_{j,r}")
        st.markdown("**Ràng buộc:**")
        st.latex(r"""\begin{aligned}
        &\text{(C1)}\ \textstyle\sum_{r,j} x_{j,r} \le 50000 &\text{(ngân sách tổng)}\\
        &\text{(C2)}\ \textstyle\sum_j x_{j,r} \ge 5000\ \forall r &\text{(sàn vùng)}\\
        &\text{(C3)}\ \textstyle\sum_j x_{j,r} \le 12000\ \forall r &\text{(trần vùng)}\\
        &\text{(C4)}\ \textstyle\sum_r x_{H,r} \ge 12000 &\text{(sàn nhân lực)}\\
        &\text{(C5)}\ D_r + \gamma x_{D,r} \ge \lambda \max_r(D_r+\gamma x_{D,r}) &\text{(công bằng)}
        \end{aligned}""")
        st.caption("γ = 0,002. C5 tuyến tính hóa bằng biến phụ M = max.")

    def data():
        st.markdown("**Hệ số tác động biên βⱼ,ᵣ:**")
        bm = pd.DataFrame([[BETA[(r, j)] for j in ITEMS] for r in REGION_CODE],
                          columns=["I (hạ tầng)", "D (CĐS DN)", "AI", "H (nhân lực)"], index=REGION_VI)
        st.dataframe(bm, width='stretch')
        reg = load_regions()
        st.markdown("**Chỉ số số hóa ban đầu Dᵣ:**")
        st.dataframe(pd.DataFrame({"Vùng": REGION_VI,
                                   "D₀ (digital_index)": reg["digital_index_0_100"].values}),
                     width='stretch', hide_index=True)

    def calc():
        c = st.columns(2)
        B = c[0].slider("Ngân sách tổng (tỷ VND)", 40000, 60000, 50000, 2000)
        lam = c[1].slider("Hệ số công bằng λ", 0.5, 0.9, 0.6, 0.05)
        mat, Z = _solve_lp4(B, True, lam)
        mat_ne, Z_ne = _solve_lp4(B, False, lam)

        section("Câu 4.4.1 / 4.4.3 — Phân bổ tối ưu & heatmap")
        steps([("Khai báo 24 biến x[vùng][hạng mục]", "Biến liên tục không âm trong PuLP."),
               ("Thêm ràng buộc C1–C5", "Bao gồm tuyến tính hóa ràng buộc công bằng bằng biến M."),
               ("Giải bằng CBC solver", "Trả về phân bổ tối ưu và giá trị Z*.")])
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(6)); ax.set_yticklabels([r[:18] for r in REGION_VI], fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
        ax.set_title(f"Phân bổ tối ưu (Z*={Z:,.0f} tỷ)")
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > 7000 else "black")
        plt.colorbar(im, label="tỷ VND"); fig.tight_layout()
        c1.pyplot(fig)
        dfm = pd.DataFrame(mat, columns=ITEMS, index=REGION_VI).round(0)
        dfm["Tổng"] = dfm.sum(axis=1)
        c2.dataframe(dfm, width='stretch')
        c2.metric("Z* (GDP gain)", f"{Z:,.0f} tỷ VND")

        section("Câu 4.4.4 — Chi phí kinh tế của công bằng vùng miền")
        kpi_row([("Z* CÓ công bằng", f"{Z:,.0f}", None),
                 ("Z* KHÔNG công bằng", f"{Z_ne:,.0f}", None),
                 ("Chi phí công bằng", f"{Z_ne - Z:,.0f} tỷ", f"-{(Z_ne-Z)/Z_ne*100:.2f}%")])
        insight("Diễn giải", f"Ràng buộc công bằng làm Z* giảm <b>{Z_ne-Z:,.0f} tỷ VND</b> "
                f"(~{(Z_ne-Z)/Z_ne*100:.2f}%). Đây là 'cái giá' của phân phối đều — đổi một phần "
                "hiệu quả lấy cân bằng phát triển giữa các vùng.")

    def policy():
        with st.expander("a) Bỏ ràng buộc công bằng, vốn chảy về đâu? Hậu quả?", expanded=True):
            st.markdown("Vốn dồn về **ĐB sông Hồng & Đông Nam Bộ** (hệ số AI 1,40–1,55) → đào sâu "
                        "chênh lệch vùng, di cư lao động, mất cân đối phát triển dài hạn.")
        with st.expander("b) Trần ngân sách mỗi vùng giảm Z* bao nhiêu? Chấp nhận được?"):
            st.markdown("Trần vùng đóng vai trò 'chính sách phân quyền', giảm Z* một phần nhưng đổi "
                        "lấy cân bằng — chấp nhận được nếu xã hội coi trọng công bằng vùng.")
        with st.expander("c) Tây Nguyên: đầu tư AI hay tập trung H, I trước?"):
            st.markdown("Hệ số AI thấp (0,45) nhưng H cao (1,35) → mô hình ưu tiên **nhân lực số (H) "
                        "và hạ tầng (I)** trước, tạo nền tảng rồi mới đẩy AI.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 5 — MIP CHỌN DỰ ÁN
# ============================================================
PROJ_NAMES = {
    1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc", 4: "VNeID 2.0",
    5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12", 8: "TT AI + supercomputing",
    9: "Fintech sandbox", 10: "Logistics thông minh", 11: "Nông nghiệp số ĐBSCL",
    12: "Đào tạo 50K kỹ sư AI", 13: "Khu CN bán dẫn BN-BG", 14: "An ninh mạng SOC", 15: "Open Data"}


@st.cache_data(show_spinner=False)
def _solve_mip(budget_total, budget_12, use_expected, force_p1p2):
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, value
    P = list(range(1, 16))
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    fields = {1: "ht", 2: "ht", 3: "ht", 4: "cp", 5: "cp", 6: "yt", 7: "gd", 8: "ai",
              9: "tc", 10: "lg", 11: "nn", 12: "nl", 13: "bd", 14: "an", 15: "dl"}
    prob = {"ht": .85, "cp": .75, "ai": .65, "bd": .65, "yt": .8, "gd": .8, "tc": .8,
            "lg": .8, "nn": .8, "nl": .8, "an": .8, "dl": .8}
    m = LpProblem("VN_Project_Selection", LpMaximize)
    y = LpVariable.dicts("y", P, cat="Binary")
    if use_expected:
        m += lpSum(prob[fields[i]] * B[i] * y[i] for i in P)
    else:
        m += lpSum(B[i] * y[i] for i in P)
    m += lpSum(C[i] * y[i] for i in P) <= budget_total
    m += lpSum(C1[i] * y[i] for i in P) <= budget_12
    if not force_p1p2:
        m += y[1] + y[2] <= 1
    else:
        m += y[1] >= 1; m += y[2] >= 1
    m += y[8] <= y[12]; m += y[13] <= y[12]
    m += y[4] + y[5] >= 1; m += y[14] >= 1
    m += lpSum(y[i] for i in P) >= 7; m += lpSum(y[i] for i in P) <= 11
    m.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[m.status]
    sel = [i for i in P if y[i].value() and y[i].value() > 0.5]
    Z = value(m.objective)
    rows = [{"Mã": f"P{i}", "Tên dự án": PROJ_NAMES[i], "Chi phí": C[i], "NPV": B[i],
             "NPV/C": round(B[i] / C[i], 2)} for i in sel]
    return status, sel, Z, sum(C[i] for i in sel), rows


def page_bai5():
    bai_header("🎯", "Bài 5 — MIP lựa chọn dự án chuyển đổi số (15 dự án)",
               "PuLP + CBC · knapsack tổng quát · ràng buộc loại trừ, tiên quyết, ngân sách đa năm",
               ["Cấp độ: Trung bình", "MIP / Binary", "Knapsack"])

    def ctx():
        st.markdown("""
        Bộ KH-CN xem xét **15 dự án** ứng cử cho chương trình chuyển đổi số quốc gia 2026-2030.
        Tổng ngân sách **80.000 tỷ VND**. Mỗi dự án có chi phí, lợi ích NPV và ràng buộc đặc thù
        (loại trừ, tiên quyết). Cần chọn **tập dự án tối ưu**.
        """)
        kpi_row([("Số dự án", "15", None), ("Ngân sách", "80.000 tỷ", None),
                 ("Biến nhị phân", "yᵢ ∈ {0,1}", None), ("Chọn", "7–11 dự án", None)])

    def model():
        st.latex(r"\max \sum_i B_i\, y_i, \quad y_i \in \{0,1\}")
        st.latex(r"""\begin{aligned}
        &\textstyle\sum_i C_i y_i \le 80000, \quad \sum_i C_{1,i} y_i \le 40000 &\text{(ngân sách)}\\
        & y_1 + y_2 \le 1 &\text{(loại trừ TT dữ liệu)}\\
        & y_8 \le y_{12},\ y_{13} \le y_{12} &\text{(tiên quyết đào tạo)}\\
        & y_4 + y_5 \ge 1,\ y_{14} \ge 1 &\text{(bắt buộc)}\\
        & 7 \le \textstyle\sum_i y_i \le 11 &\text{(số dự án)}
        \end{aligned}""")

    def data():
        _, _, _, _, _ = None, None, None, None, None
        C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
             9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
        B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
             9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
        st.dataframe(pd.DataFrame({
            "Mã": [f"P{i}" for i in range(1, 16)],
            "Dự án": [PROJ_NAMES[i] for i in range(1, 16)],
            "Chi phí (tỷ)": [C[i] for i in range(1, 16)],
            "NPV (tỷ)": [B[i] for i in range(1, 16)],
            "NPV/C": [round(B[i] / C[i], 2) for i in range(1, 16)]}),
            width='stretch', hide_index=True)

    def calc():
        c = st.columns(3)
        bt = c[0].slider("Ngân sách 5 năm (tỷ)", 80000, 120000, 80000, 5000)
        use_exp = c[1].toggle("Tối đa hóa lợi ích kỳ vọng (rủi ro)", value=False)
        force = c[2].toggle("Bắt buộc cả P1 & P2", value=False)
        status, sel, Z, cost, rows = _solve_mip(bt, 40000, use_exp, force)

        section("Câu 5.4.1–5.4.4 — Tập dự án tối ưu")
        steps([("Khai báo biến nhị phân y₁…y₁₅", "yᵢ = 1 nếu chọn dự án i."),
               ("Thêm ràng buộc ngân sách + logic", "Loại trừ, tiên quyết, bắt buộc, số lượng."),
               ("Giải bằng CBC (branch & bound)", "Tìm tổ hợp dự án tối ưu nguyên.")])
        if status == "Optimal":
            kpi_row([("Số dự án chọn", str(len(sel)), None),
                     ("Tổng chi phí", f"{cost:,} tỷ", None),
                     ("Tổng lợi ích Z*", f"{Z:,.0f} tỷ", None),
                     ("Hiệu suất Z*/C", f"{Z/cost:.2f}", None)])
            dfp = pd.DataFrame(rows)
            st.dataframe(dfp, width='stretch', hide_index=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(dfp["Mã"], dfp["NPV"], color=ACC4, label="NPV", edgecolor=EDGE)
            ax.bar(dfp["Mã"], dfp["Chi phí"], color=ACC1, alpha=.75, label="Chi phí")
            ax.set_ylabel("tỷ VND"); ax.legend(); ax.grid(axis="y", alpha=.3)
            ax.set_title("NPV vs Chi phí các dự án được chọn")
            st.pyplot(fig)
            insight("Diễn giải", f"Solver chọn <b>{len(sel)} dự án</b> với tổng lợi ích "
                    f"<b>{Z:,.0f} tỷ</b>, chi phí {cost:,} tỷ. Các ràng buộc tiên quyết "
                    "(đào tạo kỹ sư trước khi làm AI/bán dẫn) được tôn trọng tự động.")
        else:
            st.error(f"Bài toán {status} — không có nghiệm khả thi.")

    def policy():
        with st.expander("a) Vì sao mô hình có thể bỏ P15 (Open Data) dù NPV/C cao?", expanded=True):
            st.markdown("P15 có NPV/C ≈ 2,53 (cao nhất) nên thực tế thường **được** chọn. Nếu bị bỏ, "
                        "đó là do ràng buộc số lượng/ngân sách ràng buộc các dự án lớn khác — cần xem lại.")
        with st.expander("b) Bắt buộc P14 (an ninh mạng) có hợp lý không?"):
            st.markdown("Hợp lý: an ninh dữ liệu là **điều kiện nền** của chuyển đổi số, dù có thể "
                        "làm giảm Z* một chút.")
        with st.expander("c) Làm sao mô hình hóa cộng hưởng P8 (AI) & P13 (bán dẫn)?"):
            st.markdown("Thêm biến tích z = y₈·y₁₃ (tuyến tính hóa) và cộng phần lợi ích bổ sung khi "
                        "chọn cả hai dự án.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 6 — TOPSIS 6 VÙNG
# ============================================================
def page_bai6():
    bai_header("🏆", "Bài 6 — TOPSIS xếp hạng 6 vùng theo ưu tiên đầu tư AI",
               "Chuẩn hóa vector · trọng số chuyên gia vs Entropy · phân tích độ nhạy",
               ["Cấp độ: Trung bình", "TOPSIS / MCDM", "Entropy weight"])

    df = load_regions().copy()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
                "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
                "internet_penetration_pct", "gini_coef"]
    clabels = ["GRDP/N", "FDI", "Digital", "AI", "LĐĐT", "R&D", "Internet", "Gini"]
    is_ben = np.array([True, True, True, True, True, True, True, False])
    X = df[criteria].values.astype(float)

    def ctx():
        st.markdown("""
        Theo **Quyết định 127/QĐ-TTg**, Việt Nam đặt mục tiêu thành trung tâm AI của ASEAN.
        Ngân sách có hạn nên cần chọn vùng triển khai trung tâm AI và sandbox dữ liệu **trước**.
        Bài này dùng **TOPSIS** xếp hạng 6 vùng theo mức sẵn sàng AI trên 8 tiêu chí.
        """)
        kpi_row([("Số vùng", "6", None), ("Tiêu chí", "8", None),
                 ("Tiêu chí chi phí", "Gini", None), ("Mục tiêu", "3 trung tâm AI", None)])

    def model():
        st.markdown("**Quy trình TOPSIS 5 bước:**")
        st.latex(r"\text{B1: } r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}} \qquad "
                 r"\text{B2: } v_{ij} = w_j\, r_{ij}")
        st.latex(r"\text{B3: } A^* = \{\max_i v_{ij}\},\ A^- = \{\min_i v_{ij}\}")
        st.latex(r"\text{B4: } S_i^* = \sqrt{\textstyle\sum_j (v_{ij}-v_j^*)^2},\ "
                 r"S_i^- = \sqrt{\textstyle\sum_j (v_{ij}-v_j^-)^2}")
        st.latex(r"\text{B5: } C_i^* = \frac{S_i^-}{S_i^* + S_i^-} \in [0,1]")
        st.caption("C* càng lớn → vùng càng gần phương án lý tưởng → ưu tiên cao.")

    def data():
        show = df[["region_name_en"] + criteria].copy()
        show.insert(0, "Vùng", REGION_VI); show = show.drop(columns=["region_name_en"])
        show.columns = ["Vùng"] + clabels
        st.dataframe(show.round(3), width='stretch', hide_index=True)
        st.caption("7 tiêu chí lợi ích (benefit) + Gini là tiêu chí chi phí (càng thấp càng tốt).")

    def calc():
        section("Câu 6.4.1 — TOPSIS với trọng số chuyên gia")
        defaults = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
        cc = st.columns(8)
        raw = [cc[i].number_input(clabels[i], 0.0, 1.0, defaults[i], 0.01, key=f"w6_{i}") for i in range(8)]
        w_exp = np.array(raw) / sum(raw)
        steps([("Chuẩn hóa vector ma trận quyết định", "Chia mỗi cột cho chuẩn Euclid của cột đó."),
               ("Nhân trọng số → ma trận có trọng số V", "vᵢⱼ = wⱼ·rᵢⱼ."),
               ("Xác định A* (lý tưởng) & A⁻ (phản lý tưởng)", "Gini đảo vai trò do là chi phí."),
               ("Tính khoảng cách Euclid S*, S⁻ và C*", "C* = S⁻/(S*+S⁻).")])
        C_exp, R, V, As, An, Ss, Sn = topsis(X, w_exp, is_ben)
        w_ent = entropy_weights(X)
        C_ent = topsis(X, w_ent, is_ben)[0]

        res = pd.DataFrame({"Vùng": REGION_VI, "S*": Ss.round(4), "S⁻": Sn.round(4),
                            "C* chuyên gia": C_exp.round(4), "C* entropy": C_ent.round(4)})
        res["Hạng CG"] = res["C* chuyên gia"].rank(ascending=False).astype(int)
        res = res.sort_values("Hạng CG").reset_index(drop=True)
        c1, c2 = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = res.sort_values("C* chuyên gia")
        y = np.arange(6)
        ax.barh(y - 0.2, order["C* chuyên gia"], 0.4, color=ACC1, label="Chuyên gia")
        ax.barh(y + 0.2, order["C* entropy"], 0.4, color=ACC2, label="Entropy")
        ax.set_yticks(y); ax.set_yticklabels([r[:18] for r in order["Vùng"]], fontsize=8)
        ax.set_xlabel("C*"); ax.legend(); ax.grid(axis="x", alpha=.3)
        ax.set_title("TOPSIS: Chuyên gia vs Entropy")
        c1.pyplot(fig)
        c2.dataframe(res, width='stretch', hide_index=True)
        c2.success(f"🥇 Dẫn đầu: {res.iloc[0]['Vùng']}")
        insight("Diễn giải", f"<b>{res.iloc[0]['Vùng']}</b> đạt C* cao nhất "
                f"({res.iloc[0]['C* chuyên gia']:.3f}) — gần nhất với phương án lý tưởng, "
                "ứng viên hàng đầu cho trung tâm AI quốc gia.")

        section("Câu 6.4.2 — Trọng số khách quan (Entropy)")
        st.dataframe(pd.DataFrame({"Tiêu chí": clabels, "w Entropy": w_ent.round(4),
                                   "w Chuyên gia": w_exp.round(4)}),
                     width='stretch', hide_index=True)

        section("Câu 6.4.3 — Độ nhạy theo w_AI")
        w_ai_range = np.arange(0.10, 0.45, 0.05)
        hm = []
        for wai in w_ai_range:
            wg = 0.10; rem = 1 - wai - wg
            base = np.array([0.10, 0.10, 0.15, 0.15, 0.15, 0.05])
            ws = base * (rem / base.sum())
            wf = np.insert(ws, 3, wai); wf = np.append(wf, wg)
            hm.append(topsis(X, wf, is_ben)[0])
        hm = np.array(hm)
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(hm, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(w_ai_range))); ax.set_yticklabels([f"{w:.2f}" for w in w_ai_range])
        ax.set_xticks(range(6)); ax.set_xticklabels([r[:12] for r in REGION_VI], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("w_AI"); ax.set_title("C* theo trọng số AI Readiness")
        plt.colorbar(im, label="C*"); fig.tight_layout()
        st.pyplot(fig)

    def policy():
        with st.expander("a) Vùng nào nên đặt trung tâm AI đầu tiên?", expanded=True):
            st.markdown("**Đông Nam Bộ & ĐB sông Hồng** dẫn đầu TOPSIS — ứng viên cho trung tâm AI "
                        "quốc gia đầu tiên theo Quyết định 127/QĐ-TTg.")
        with st.expander("b) Entropy đổi thứ hạng vùng nào nhiều nhất? Vì sao?"):
            st.markdown("Entropy phản ánh **độ phân tán dữ liệu** thay vì ưu tiên chính sách, nên có "
                        "thể nâng/hạ các vùng giữa bảng nơi tiêu chí biến thiên mạnh.")
        with st.expander("c) AI Readiness & Internet tương quan cao — xử lý sao?"):
            st.markdown("Tương quan cao gây **trùng lặp tiêu chí**, làm lệch trọng số ngầm. Có thể "
                        "dùng PCA hoặc gộp tiêu chí trước khi chạy TOPSIS.")
        with st.expander("d) Chọn 3 vùng cho 3 trung tâm AI?"):
            st.markdown("2 vùng dẫn đầu + 1 vùng đại diện miền Trung (Bắc Trung Bộ) để cân bằng "
                        "yếu tố địa-chính trị và phát triển vùng.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 7 — NSGA-II PARETO
# ============================================================
@st.cache_data(show_spinner=False)
def _run_nsga(pop_size, n_gen, lam):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    beta_mat = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGION_CODE])
    D0 = np.array([38, 78, 55, 32, 82, 48], float)
    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    gamma_val = 0.002

    class Prob(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=20,
                             xl=np.zeros(24), xu=np.ones(24) * 12000)

        def _evaluate(self, x, out, *a, **k):
            X = x.reshape(6, 4)
            f1 = -(beta_mat * X).sum()
            sums = X.sum(axis=1); f2 = np.abs(sums - sums.mean()).mean()
            f3 = (e * (X[:, 0] + X[:, 1] + X[:, 2])).sum()
            f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            g = [X.sum() - 50000]
            for r in range(6): g.append(5000 - X[r].sum())
            for r in range(6): g.append(X[r].sum() - 12000)
            g.append(12000 - X[:, 3].sum())
            Dn = D0 + gamma_val * X[:, 1]; Dmax = Dn.max()
            for r in range(6): g.append(lam * Dmax - Dn[r])
            out["G"] = np.array(g)

    res = minimize(Prob(), NSGA2(pop_size=pop_size), get_termination("n_gen", n_gen),
                   seed=42, verbose=False)
    return res.F, res.X


def page_bai7():
    bai_header("🌐", "Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)",
               "4 mục tiêu: tăng trưởng · bao trùm · môi trường · an ninh dữ liệu",
               ["Cấp độ: Khá khó", "pymoo / NSGA-II", "Pareto front"])

    def ctx():
        st.markdown("""
        Phát triển kinh tế số & AI hướng tới **4 mục tiêu xung đột** (Mục 8.2):
        (i) tăng trưởng GDP nhanh; (ii) bao trùm xã hội (giảm bất bình đẳng vùng);
        (iii) net-zero 2050 (COP26); (iv) an ninh dữ liệu & chủ quyền số.
        Kết quả không phải một nghiệm duy nhất mà là **tập nghiệm Pareto**.
        """)
        kpi_row([("Số biến", "24", None), ("Số mục tiêu", "4", None),
                 ("Thuật toán", "NSGA-II", None), ("Quần thể", "100", None)])

    def model():
        st.latex(r"\max f_1 = \sum_{r,j}\beta_{j,r}x_{j,r} \quad (\text{tăng trưởng})")
        st.latex(r"\min f_2 = G(x) \approx \text{MAD ngân sách vùng} \quad (\text{bao trùm})")
        st.latex(r"\min f_3 = \sum_r e_r(x_{I,r}+x_{AI,r}) \quad (\text{phát thải})")
        st.latex(r"\min f_4 = \sum_r \rho_r x_{AI,r} - \sum_r \sigma_r x_{H,r} \quad (\text{rủi ro ròng})")
        st.caption("Ràng buộc C1–C5 giữ nguyên như Bài 4. eᵣ cường độ phát thải, ρ rủi ro AI, σ giảm rủi ro nhờ H.")

    def data():
        st.dataframe(pd.DataFrame({
            "Vùng": REGION_VI,
            "eᵣ (CO₂/tỷ)": [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
            "ρᵣ (rủi ro/AI)": [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
            "σᵣ (giảm rủi ro/H)": [0.32, 0.28, 0.30, 0.35, 0.25, 0.30]}),
            width='stretch', hide_index=True)

    def calc():
        c = st.columns(3)
        pop = c[0].select_slider("Pop size", [40, 60, 80, 100], 60)
        ngen = c[1].select_slider("Số thế hệ", [50, 100, 150, 200], 100)
        lam = c[2].slider("Hệ số công bằng λ", 0.5, 0.9, 0.6, 0.05)
        steps([("Định nghĩa Problem 24 biến, 4 mục tiêu", "Kế thừa ElementwiseProblem của pymoo."),
               ("Chạy NSGA-II qua nhiều thế hệ", "Non-dominated sorting + crowding distance."),
               ("Trích quần thể Pareto cuối", "Các nghiệm không bị trội."),
               ("Chọn nghiệm thỏa hiệp bằng TOPSIS", "Trọng số chính sách (0,40;0,25;0,20;0,15).")])
        with st.spinner("Đang chạy NSGA-II..."):
            F, X = _run_nsga(pop, ngen, lam)

        section("Câu 7.4.1–7.4.2 — Tập nghiệm Pareto")
        kpi_row([("Số nghiệm Pareto", str(len(F)), None),
                 ("GDP gain max", f"{-F[:,0].min():,.0f}", None),
                 ("Gini min", f"{F[:,1].min():.1f}", None),
                 ("Phát thải min", f"{F[:,2].min():,.0f}", None)])
        c1, c2 = st.columns(2)
        fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(-F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap="viridis", s=14, alpha=.8)
        ax.set_xlabel("GDP gain"); ax.set_ylabel("Gini/MAD"); ax.set_zlabel("Phát thải")
        ax.set_title("Pareto 3D (màu = rủi ro)"); fig.colorbar(sc, shrink=.6, label="f4")
        c1.pyplot(fig)
        Fn = F.copy()
        for i in range(4):
            lo, hi = F[:, i].min(), F[:, i].max()
            Fn[:, i] = (F[:, i] - lo) / (hi - lo) if hi > lo else 0.5
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(len(F)):
            ax.plot(range(4), Fn[i], color=ACC2, alpha=.06)
        ax.plot(range(4), Fn.mean(0), color=ACC1, lw=2, label="Trung bình")
        ax.set_xticks(range(4)); ax.set_xticklabels(["GDP", "Gini", "Phát thải", "Rủi ro"])
        ax.set_ylabel("Chuẩn hóa [0,1]"); ax.legend(); ax.set_title("Parallel coordinates")
        c2.pyplot(fig)

        section("Câu 7.4.3–7.4.4 — Nghiệm thỏa hiệp (TOPSIS trên Pareto)")
        w = np.array([0.40, 0.25, 0.20, 0.15])
        lo, hi = F.min(0), F.max(0); rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        R = (F - lo) / rng; V = R * w
        Ss = np.sqrt((V ** 2).sum(1)); Sn = np.sqrt(((V - w) ** 2).sum(1))
        Cs = Sn / (Ss + Sn); best = int(np.argmax(Cs)); bF = F[best]
        kpi_row([("GDP gain", f"{-bF[0]:,.0f}", None), ("Gini/MAD", f"{bF[1]:.1f}", None),
                 ("Phát thải", f"{bF[2]:,.0f}", None), ("Rủi ro ròng", f"{bF[3]:,.0f}", None)])
        bX = X[best].reshape(6, 4)
        st.dataframe(pd.DataFrame(bX.round(0), columns=ITEMS, index=REGION_VI), width='stretch')
        insight("Diễn giải", "Nghiệm thỏa hiệp cân bằng cả 4 mục tiêu thay vì cực đại hóa một mục tiêu. "
                "So với nghiệm GDP cao nhất, nghiệm này hy sinh một phần tăng trưởng để cải thiện "
                "bao trùm và giảm phát thải.")

    def policy():
        with st.expander("a) Đánh đổi tăng trưởng vs bao trùm có rõ không?", expanded=True):
            st.markdown("Đường biên Pareto cho thấy đánh đổi **rõ rệt**: nghiệm GDP cao nhất thường "
                        "có Gini xấu hơn — phản ánh tập trung vốn vào vùng phát triển.")
        with st.expander("b) Trọng số có phản ánh đúng ưu tiên VN?"):
            st.markdown("Trọng số (0,40;0,25;0,20;0,15) ưu tiên tăng trưởng. Để phù hợp COP26 nên "
                        "tăng trọng số môi trường (f3).")
        with st.expander("c) NSGA-II có thay thế quyết định chính trị không?"):
            st.markdown("Không. NSGA-II cung cấp **tập lựa chọn**; việc chọn điểm nào trên đường biên "
                        "là quyết định chính trị-xã hội.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 8 — TỐI ƯU ĐỘNG
# ============================================================
@st.cache_data(show_spinner=False)
def _run_dyn(rho_disc):
    from scipy.optimize import minimize
    a, b, gd, dai, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI, thH, mu = 0.05, 0.12, 0.15, 0.8, 0.02
    phi1, phi2, phi3, gcr = 0.003, 0.002, 0.004, 1.5
    T = 10
    K0, L0, D0, AI0, H0 = 27500., 53.9, 20.3, 86., 30.
    A0 = 12847.6 / (K0 ** a * L0 ** b * D0 ** gd * AI0 ** dai * H0 ** th)
    L = np.array([L0 * 1.009 ** t for t in range(T + 1)])

    def traj(u):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T + 1); D = np.zeros(T + 1); AI = np.zeros(T + 1)
        H = np.zeros(T + 1); A = np.zeros(T + 1); Y = np.zeros(T + 1); C = np.zeros(T)
        K[0], D[0], AI[0], H[0], A[0] = K0, D0, AI0, H0, A0
        for t in range(T):
            Y[t] = A[t] * K[t]**a * L[t]**b * D[t]**gd * AI[t]**dai * H[t]**th
            C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0: return None
            K[t+1] = (1-dK)*K[t]+IK[t]; D[t+1] = (1-dD)*D[t]+ID[t]
            AI[t+1] = (1-dAI)*AI[t]+IAI[t]; H[t+1] = H[t]+thH*IH[t]-mu*H[t]
            A[t+1] = A[t]*(1+phi1*(D[t]/100)+phi2*(AI[t]/100)+phi3*(H[t]/100))
        Y[T] = A[T]*K[T]**a*L[T]**b*D[T]**gd*AI[T]**dai*H[T]**th
        return K, D, AI, H, Y, C, A

    def welfare(u):
        r = traj(u)
        if r is None or np.any(r[5] <= 0): return 1e15
        C = r[5]
        return -sum(rho_disc**t * (C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    ti = 14000 * 0.15
    u0 = np.tile([ti*0.4, ti*0.25, ti*0.2, ti*0.15], T)
    cons = [{"type": "ineq", "fun": lambda u: (lambda r: -1e10 if r is None else min(r[5])-1)(traj(u))}]
    res = minimize(welfare, u0, method="SLSQP", bounds=[(0, None)]*(T*4),
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-8})
    return traj(res.x), -res.fun, np.arange(2026, 2037)


def page_bai8():
    bai_header("⏳", "Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
               "Cobb-Douglas động · hàm thỏa dụng CRRA · SLSQP · quỹ đạo K, D, AI, H, Y, C",
               ["Cấp độ: Khá khó", "Dynamic optimization", "CRRA utility"])

    def ctx():
        st.markdown("""
        Theo Văn kiện Đại hội XIII, Việt Nam đặt mục tiêu thu nhập trung bình cao 2030,
        thu nhập cao 2045. Cần thiết kế **chiến lược phân bổ vốn dài hạn** cân bằng giữa
        tăng trưởng, chuyển đổi số, AI và nhân lực, **tối đa hóa phúc lợi xã hội liên thời gian**.
        """)
        kpi_row([("Kỳ kế hoạch", "2026-2035", None), ("Biến/năm", "5 đầu tư", None),
                 ("Hàm thỏa dụng", "CRRA γ=1,5", None), ("Chiết khấu ρ", "0,97", None)])

    def model():
        st.latex(r"\max \sum_{t=2026}^{2035} \rho^{\,t-2026}\, U(C_t), \quad U(C)=\frac{C^{1-\gamma}}{1-\gamma}")
        st.latex(r"Y_t = A_t K_t^{0.33} L_t^{0.42} D_t^{0.10} AI_t^{0.08} H_t^{0.07}")
        st.markdown("**Động học tích lũy vốn:**")
        st.latex(r"K_{t+1}=(1-\delta_K)K_t+I_{K,t},\quad H_{t+1}=H_t+\theta_H I_{H,t}-\mu H_t")
        st.latex(r"A_{t+1}=A_t(1+\phi_1 D_t+\phi_2 AI_t+\phi_3 H_t) \quad (\text{TFP nội sinh})")
        st.markdown("**Ràng buộc ngân sách:** " + r"$C_t+I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t}\le Y_t$")

    def data():
        st.dataframe(pd.DataFrame({
            "Tham số": ["δ_K", "δ_D", "δ_AI", "θ_H", "μ", "φ₁", "φ₂", "φ₃", "ρ"],
            "Giá trị": [0.05, 0.12, 0.15, 0.8, 0.02, 0.003, 0.002, 0.004, 0.97],
            "Ý nghĩa": ["Khấu hao vốn", "Khấu hao hạ tầng số", "Khấu hao AI",
                        "Hiệu quả đào tạo→H", "Chảy máu chất xám", "TFP từ D", "TFP từ AI",
                        "TFP từ H", "Chiết khấu liên thời gian"]}), width='stretch', hide_index=True)
        st.markdown("**Điều kiện ban đầu 2026:** K₀=27.500 · L₀=53,9 · D₀=20,3% · AI₀=86 · H₀=30%")

    def calc():
        rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
        steps([("Tham số hóa quỹ đạo đầu tư", "10 năm × 4 hạng mục = 40 biến quyết định."),
               ("Mô phỏng động học K,D,AI,H,A,Y,C", "Cập nhật trạng thái từng năm, TFP nội sinh."),
               ("Tối đa hóa phúc lợi CRRA chiết khấu", "Giải bằng SLSQP với ràng buộc tiêu dùng > 0.")])
        with st.spinner("Đang tối ưu quỹ đạo..."):
            (K, D, AI, H, Y, C, A), W, years = _run_dyn(rho)

        section("Câu 8.3.1–8.3.2 — Quỹ đạo tối ưu")
        kpi_row([("Phúc lợi W*", f"{W:.3f}", None), ("GDP 2035", f"{Y[-1]:,.0f}", None),
                 ("K 2035", f"{K[-1]:,.0f}", None), ("H 2035", f"{H[-1]:.1f}%", None)])
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, dat, title, cc in [
            (axes[0, 0], K, "K (vốn vật chất)", ACC1), (axes[0, 1], D, "D (hạ tầng số %GDP)", ACC2),
            (axes[0, 2], AI, "AI (nghìn DN)", ACC3), (axes[1, 0], H, "H (nhân lực %)", ACC4),
            (axes[1, 2], A, "A (TFP)", "#a855f7")]:
            ax.plot(years, dat, "o-", color=cc, ms=4); ax.set_title(title); ax.grid(alpha=.3)
        axes[1, 1].plot(years, Y, "o-", color=ACC4, ms=4, label="Y (GDP)")
        axes[1, 1].plot(years[:10], C, "s-", color=ACC1, ms=4, label="C (tiêu dùng)")
        axes[1, 1].set_title("Y & C"); axes[1, 1].legend(); axes[1, 1].grid(alpha=.3)
        fig.suptitle("Quỹ đạo tối ưu 2026-2035", fontsize=13); fig.tight_layout()
        st.pyplot(fig)
        df = pd.DataFrame({"Năm": years, "K": K.round(0), "D": D.round(1), "AI": AI.round(1),
                           "H": H.round(1), "Y": Y.round(0)})
        df["C"] = list(C.round(0)) + [np.nan]
        st.dataframe(df, width='stretch', hide_index=True)
        insight("Diễn giải", "Quỹ đạo đầu tư có xu hướng <b>front-loaded</b> ở hạ tầng số & AI: "
                "đầu tư sớm sinh lợi suất kép qua TFP nội sinh, kéo GDP tăng nhanh dần về cuối kỳ.")

    def policy():
        with st.expander("a) Quỹ đạo front-loaded hay back-loaded? Vì sao?", expanded=True):
            st.markdown("**Front-loaded**: đầu tư mạnh hạ tầng số & AI giai đoạn đầu để tích lũy TFP "
                        "nội sinh, sinh lợi suất kép qua nhiều năm.")
        with st.expander("b) Đào tạo nhân lực nên đi trước hay đồng thời với AI?"):
            st.markdown("Nên **đi trước hoặc đồng thời** vì H là điều kiện hấp thụ công nghệ; thiếu H "
                        "thì đầu tư AI kém hiệu quả.")
        with st.expander("c) ρ thấp hơn (0,90) thì sao? Lý do 'dưới đầu tư' R&D?"):
            st.markdown("ρ thấp → ưu tiên ngắn hạn → giảm đầu tư R&D/AI. Đây là lý do nhiều chính phủ "
                        "'dưới đầu tư' vào công nghệ dài hạn.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 9 — LAO ĐỘNG & AI
# ============================================================
def page_bai9():
    from scipy.optimize import linprog
    bai_header("👷", "Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
               "LP tối đa hóa NetJob · ngưỡng đào tạo lại · luồng dịch chuyển lao động",
               ["Cấp độ: Khá khó", "LP", "NetJob ròng"])

    sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn-bán lẻ",
           "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục-ĐT"]
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], float)
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], float)
    N = 8

    def ctx():
        st.markdown("""
        Theo ILO Vietnam 2024 & OECD, **30-50% việc làm tại Việt Nam** có nguy cơ tự động hóa
        một phần trong 10 năm tới, đặc biệt chế biến chế tạo, bán buôn-bán lẻ, logistics.
        Nhưng AI cũng **tạo việc làm mới**. Bài toán: đầu tư bao nhiêu vào đào tạo lại để
        **NetJob ròng ≥ 0** cho mọi ngành?
        """)
        kpi_row([("Số ngành", "8", None), ("Ngân sách", "30.000 tỷ", None),
                 ("Hạng mục", "x_AI, x_H", None), ("Ràng buộc", "NetJob ≥ 0", None)])

    def model():
        st.latex(r"NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i")
        st.latex(r"NewJob_i=a_{1i}x_{AI,i},\ UpgradeJob_i=b_{1i}x_{H,i},\ "
                 r"DisplacedJob_i=c_{1i}\,risk_i\,x_{AI,i}")
        st.latex(r"\max \sum_i NetJob_i \quad \text{s.t.}\ \sum_i(x_{AI,i}+x_{H,i})\le 30000,\ "
                 r"NetJob_i\ge 0,\ Displaced_i\le d_{1i}x_{H,i}")
        st.caption("Ràng buộc cuối: 'tốc độ tự động hóa ≤ năng lực đào tạo lại'.")

    def data():
        st.dataframe(pd.DataFrame({
            "Ngành": sec, "LĐ(tr)": L, "Risk%": (risk * 100).astype(int),
            "a₁(việc/tỷ)": a1, "b₁": b1, "c₁": c1, "d₁": d1}), width='stretch', hide_index=True)
        st.caption("Đơn vị: số việc làm tạo ra/dịch chuyển trên mỗi tỷ VND đầu tư.")

    def calc():
        cap5 = st.toggle("Câu 9.4.4 — Thêm ràng buộc: không ngành nào mất > 5% lao động", value=False)
        coeff = a1 - c1 * risk
        cobj = np.concatenate([-coeff, -b1])
        A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
        A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)
        A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
        for i in range(N):
            A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
            A3[i, i] = c1[i] * risk[i]; A3[i, N + i] = -d1[i]
        A_ub = np.vstack([A1, A1b, A2, A3]); b_ub = np.concatenate([[30000], [-9000], np.zeros(N), np.zeros(N)])
        if cap5:
            A4 = np.zeros((N, 2 * N))
            for i in range(N): A4[i, i] = c1[i] * risk[i]
            A_ub = np.vstack([A_ub, A4]); b_ub = np.concatenate([b_ub, 0.05 * L * 1e6])
        res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (2 * N), method="highs")

        section("Câu 9.4.1 — Phân bổ tối ưu & NetJob")
        steps([("Tính hệ số ròng coeff = a₁ − c₁·risk", "Việc làm AI ròng sau khi trừ dịch chuyển."),
               ("Dựng LP với 16 biến (x_AI, x_H × 8 ngành)", "Mục tiêu max ΣNetJob."),
               ("Thêm ràng buộc NetJob ≥ 0 và năng lực đào tạo", "Bảo đảm an sinh từng ngành.")])
        if res.success:
            xA, xH = res.x[:N], res.x[N:]
            NJ = coeff * xA + b1 * xH; Disp = c1 * risk * xA
            dfp = pd.DataFrame({"Ngành": sec, "x_AI": xA.round(0), "x_H": xH.round(0),
                                "Displaced": Disp.round(0), "NetJob": NJ.round(0)})
            c1c, c2c = st.columns([3, 2])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(sec, NJ, color=ACC4, edgecolor=EDGE)
            ax.set_ylabel("NetJob (việc làm)"); ax.grid(axis="y", alpha=.3); ax.set_title("NetJob ròng theo ngành")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            c1c.pyplot(fig)
            c2c.metric("Tổng NetJob", f"{-res.fun:,.0f}")
            c2c.dataframe(dfp, width='stretch', hide_index=True)
            insight("Diễn giải", f"Tổng NetJob đạt <b>{-res.fun:,.0f} việc làm</b>. Ngành rủi ro cao "
                    "(CN chế biến, Bán buôn) cần nhiều x_H để bù dịch chuyển.")
        else:
            st.error("Bài toán KHÔNG khả thi với ràng buộc hiện tại.")

        section("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu (CN chế biến)")
        i = 1; ratio = c1[i] * risk[i] / d1[i]; net = a1[i] - c1[i] * risk[i]
        xr = np.linspace(0, 30000, 100)
        xh_re = ratio * xr; xh_nj = np.maximum(0, -net / b1[i] * xr); xh_min = np.maximum(xh_re, xh_nj)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xr, xh_re, "--", color=ACC1, label=f"Retrain: x_H ≥ {ratio:.3f}·x_AI")
        ax.plot(xr, xh_nj, "--", color=ACC2, label="NetJob ≥ 0")
        ax.fill_between(xr, xh_min, 30000, alpha=.2, color=ACC4, label="Vùng khả thi")
        ax.set_xlabel("x_AI (tỷ VND)"); ax.set_ylabel("x_H tối thiểu (tỷ VND)")
        ax.set_xlim(0, 30000); ax.set_ylim(0, 30000); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Ngưỡng đào tạo lại tối thiểu — CN chế biến")
        st.pyplot(fig)
        insight("Diễn giải", f"Mỗi 1 tỷ đầu tư AI cần <b>{ratio:.3f} tỷ</b> đầu tư đào tạo lại để giữ "
                "năng lực retraining và bảo đảm NetJob ≥ 0.")

    def policy():
        with st.expander("a) Ngành nào cần đào tạo lại nhiều nhất?", expanded=True):
            st.markdown("**CN chế biến & Bán buôn-bán lẻ** — rủi ro tự động hóa cao và lực lượng "
                        "lao động lớn, khớp với cảm nhận thực tế ở Việt Nam.")
        with st.expander("b) Tài chính-Ngân hàng: chiến lược gì?"):
            st.markdown("Rủi ro thay thế cao (52%) nhưng tạo việc làm mới cũng cao → đầu tư **song song** "
                        "AI và đào tạo chuyển đổi kỹ năng.")
        with st.expander("c) Có nên đầu tư AI vào Nông-Lâm-Thủy sản?"):
            st.markdown("Hệ số tạo việc làm AI thấp (8,5) nhưng rủi ro dịch chuyển thấp → ưu tiên đào "
                        "tạo (H) hơn là đẩy AI mạnh.")
        with st.expander("d) Ràng buộc 'tự động hóa ≤ năng lực đào tạo' nằm ở đâu?"):
            st.markdown("Chính là `Displaced_i ≤ RetrainingCapacity_i = d₁ᵢ·x_H,i` trong mô hình.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 10 — STOCHASTIC 2 GIAI ĐOẠN
# ============================================================
J10 = ["I", "D", "AI", "H"]
S10 = ["s1", "s2", "s3", "s4"]
P_S = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
BETA_BASE = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BETA_S = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10}
SCEN_NAMES = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}


@st.cache_data(show_spinner=False)
def _run_stochastic():
    import pyomo.environ as pyo

    def get_solver():
        for nm in ("appsi_highs", "glpk", "cbc"):
            s = pyo.SolverFactory(nm)
            try:
                if s.available():
                    return s
            except Exception:
                continue
        return pyo.SolverFactory("glpk")

    def build(scenarios, betas, fixed_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=J10); m.S = pyo.Set(initialize=scenarios)
        m.beta = pyo.Param(m.J, initialize=BETA_BASE)
        m.beta_s = pyo.Param(m.S, m.J, initialize={(s, j): betas[s, j] for s in scenarios for j in J10})
        m.p = pyo.Param(m.S, initialize={s: (P_S[s] if len(scenarios) > 1 else 1.0) for s in scenarios})
        if fixed_x is None:
            m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
            m.b1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)
        else:
            m.x = pyo.Param(m.J, initialize=fixed_x)
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        m.b2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= 15000)
        m.aic = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, "AI"] <= 0.5 * (m.x["H"] if fixed_x is None else fixed_x["H"]))
        m.obj = pyo.Objective(
            expr=sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S),
            sense=pyo.maximize)
        return m

    solver = get_solver()
    m_sp = build(S10, BETA_S); solver.solve(m_sp)
    x_sp = {j: pyo.value(m_sp.x[j]) for j in J10}; Z_SP = pyo.value(m_sp.obj)
    y_sp = {s: {j: pyo.value(m_sp.y[s, j]) for j in J10} for s in S10}
    det = {}
    for s in S10:
        md = build([s], BETA_S); solver.solve(md); det[s] = pyo.value(md.obj)
    beta_avg = {j: sum(P_S[s] * BETA_S[s, j] for s in S10) for j in J10}
    m_ev = pyo.ConcreteModel()
    m_ev.J = pyo.Set(initialize=J10); m_ev.x = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
    m_ev.b = pyo.Constraint(expr=sum(m_ev.x[j] for j in m_ev.J) <= 65000)
    m_ev.obj = pyo.Objective(expr=sum(beta_avg[j] * m_ev.x[j] for j in m_ev.J), sense=pyo.maximize)
    solver.solve(m_ev)
    x_ev = {j: pyo.value(m_ev.x[j]) for j in J10}
    Z_EV = sum(BETA_BASE[j] * x_ev[j] for j in J10)
    for s in S10:
        mt = build([s], BETA_S, fixed_x=x_ev); solver.solve(mt)
        Z_EV += P_S[s] * sum(BETA_S[s, j] * pyo.value(mt.y[s, j]) for j in J10)
    Z_WS = sum(P_S[s] * det[s] for s in S10)
    return x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det


def page_bai10():
    bai_header("🎲", "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
               "Pyomo · here-and-now vs recourse · VSS & EVPI",
               ["Cấp độ: Khó", "Pyomo / Stochastic", "VSS · EVPI"])

    def ctx():
        st.markdown("""
        Việt Nam có độ mở thương mại rất cao (XNK/GDP ≈ 180%), tăng trưởng phụ thuộc kịch bản
        toàn cầu. Khi hoạch định ngân sách số 2026-2030, Chính phủ phải quyết định **first-stage**
        (here-and-now) mà chưa biết kịch bản tương lai, rồi **điều chỉnh** (recourse) sau.
        """)
        kpi_row([("Ngân sách", "80.000 tỷ", None), ("First-stage", "≤ 65.000", None),
                 ("Dự phòng", "15.000", None), ("Kịch bản", "4", None)])

    def model():
        st.markdown("**Hàm mục tiêu hai giai đoạn:**")
        st.latex(r"\max \sum_j \beta_j x_j + \sum_{s\in S} p_s \sum_j \beta_j^s y_j^s")
        st.markdown("**Ràng buộc:**")
        st.latex(r"\sum_j x_j \le 65000,\quad \sum_j y_j^s \le 15000\ \forall s,\quad "
                 r"y_{AI}^s \le 0{,}5\,x_H\ \forall s")
        st.caption("x = quyết định first-stage; yˢ = điều chỉnh recourse theo kịch bản s.")
        st.markdown("**VSS** = Z_SP − Z_EV (giá trị tư duy xác suất) · "
                    "**EVPI** = Z_WS − Z_SP (giá trị thông tin hoàn hảo).")

    def data():
        st.markdown("**Cây kịch bản:**")
        st.dataframe(pd.DataFrame({
            "Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
            "Tăng trưởng TG%": [3.5, 2.8, 1.5, 0.2], "FDI (tỷ$)": [32.0, 27.0, 20.0, 12.0],
            "XK tăng%": [12.0, 8.0, 3.0, -5.0], "Xác suất": [0.30, 0.45, 0.20, 0.05]}),
            width='stretch', hide_index=True)
        st.markdown("**Hệ số βˢⱼ theo kịch bản:**")
        st.dataframe(pd.DataFrame({
            "Hạng mục": ["I", "D", "AI", "H"], "β cơ bản": [1.00, 1.10, 1.25, 0.95],
            "s1": [1.25, 1.35, 1.55, 1.05], "s2": [1.00, 1.10, 1.25, 0.95],
            "s3": [0.75, 0.85, 0.90, 1.00], "s4": [0.40, 0.50, 0.55, 1.10]}),
            width='stretch', hide_index=True)
        st.caption("Hệ số H cao trong kịch bản khủng hoảng vì lao động qua đào tạo hấp thụ cú sốc tốt hơn.")

    def calc():
        try:
            with st.spinner("Đang giải mô hình Pyomo..."):
                x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det = _run_stochastic()
        except Exception as e:
            st.error(f"Không giải được (cần solver GLPK/HiGHS/CBC): {e}")
            return
        steps([("Dựng mô hình 2 giai đoạn trong Pyomo", "Set, Param, Var cho first/second stage."),
               ("Giải lời giải SP (stochastic)", "Tối ưu kỳ vọng trên 4 kịch bản."),
               ("Giải lời giải EV và WS", "EV dùng kịch bản trung bình; WS giải từng kịch bản riêng."),
               ("Tính VSS và EVPI", "Đo giá trị của tư duy xác suất và thông tin hoàn hảo.")])

        section("Câu 10.5.1 — Quyết định first-stage tối ưu (SP)")
        c = st.columns([2, 3])
        c[0].metric("Z*_SP", f"{Z_SP:,.0f}")
        c[0].dataframe(pd.DataFrame({"Hạng mục": J10, "x* (ngh.tỷ)": [round(x_sp[j]) for j in J10]}),
                       width='stretch', hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(J10, [x_sp[j] for j in J10], color=ACC1, edgecolor=EDGE)
        ax.set_ylabel("ngh.tỷ VND"); ax.grid(axis="y", alpha=.3); ax.set_title("First-stage SP")
        c[1].pyplot(fig)

        section("Câu 10.5.2–10.5.3 — VSS & EVPI")
        VSS = Z_SP - Z_EV; EVPI = Z_WS - Z_SP
        kpi_row([("Z_EV", f"{Z_EV:,.0f}", None), ("Z_SP", f"{Z_SP:,.0f}", None),
                 ("VSS", f"{VSS:,.0f}", f"{VSS/Z_SP*100:.2f}%"),
                 ("EVPI", f"{EVPI:,.0f}", f"{EVPI/Z_SP*100:.2f}%")])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(SCEN_NAMES.values()), [det[s] for s in S10],
               color=[ACC4, ACC2, ACC3, ACC5], edgecolor=EDGE)
        ax.set_ylabel("Z* xác định"); ax.set_title("Z* tối ưu theo từng kịch bản (wait-and-see)")
        ax.grid(axis="y", alpha=.3)
        st.pyplot(fig)
        insight("Diễn giải", "<b>VSS &gt; 0</b> → bỏ qua bất định gây thiệt hại, nên cân nhắc xác suất "
                "khi quyết định. <b>EVPI</b> = giá trị tối đa nếu biết trước kịch bản — cận trên của "
                "chi cho dự báo/thông tin.")

    def policy():
        with st.expander("a) Lời giải SP đầu tư H nhiều hơn hay ít hơn?", expanded=True):
            st.markdown("**Nhiều hơn** — H đóng vai trò 'bảo hiểm': hệ số hiệu quả cao ngay cả trong "
                        "kịch bản khủng hoảng (β=1,10).")
        with st.expander("b) VSS dương nói lên điều gì?"):
            st.markdown("Tư duy xác suất có **giá trị thực tế** trong hoạch định chính sách Việt Nam, "
                        "đặc biệt với nền kinh tế mở phụ thuộc bên ngoài.")
        with st.expander("c) Bài học COVID-19 & bão Yagi?"):
            st.markdown("Việt Nam có thể đang **dưới đầu tư** vào nhân lực số như một hàng hóa bảo hiểm "
                        "trước các cú sốc.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 11 — Q-LEARNING
# ============================================================
@st.cache_data(show_spinner=False)
def _train_qlearning(n_episodes):
    import gymnasium as gym
    from gymnasium import spaces

    class Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
            self.T = 10
            self.alloc = {0: np.array([.70, .10, .10, .10]), 1: np.array([.40, .25, .15, .20]),
                          2: np.array([.25, .45, .15, .15]), 3: np.array([.20, .20, .45, .15]),
                          4: np.array([.30, .20, .10, .40])}
            self.w = np.array([.40, .25, .20, .15])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.state = (np.array(options["state"]) if options and "state" in options
                          else self.np_random.integers(0, 3, 4))
            self.t = 0; self.K = 27500.; self.D = 20.3; self.AI = 86.; self.H = 30.; self.Yp = 12847.6
            return self.state.copy(), {}

        def step(self, action):
            a = self.alloc[action]; bud = 2100.
            self.K = .95 * self.K + a[0] * bud
            self.D = .88 * self.D + a[1] * bud * .01
            self.AI = .85 * self.AI + a[2] * bud * .05
            self.H = self.H + .8 * a[3] * bud * .01 - .02 * self.H
            A = 33.70 * (1 + .003*(self.D/100) + .002*(self.AI/100) + .004*(self.H/100))**self.t
            L = 53.9 * 1.009**self.t
            Y = A * self.K**.33 * L**.42 * self.D**.10 * self.AI**.08 * self.H**.07
            dg = (Y - self.Yp) / self.Yp; du = max(0, -dg * .5)
            cyber = (self.AI / (self.H + 1)) * .01; emis = (self.K + self.AI) * .0001
            r = self.w[0]*dg*100 - self.w[1]*du*100 - self.w[2]*cyber - self.w[3]*emis
            self.Yp = Y; self.t += 1
            gl = 0 if dg < .03 else (1 if dg < .06 else 2)
            dl = 0 if self.D < 25 else (1 if self.D < 35 else 2)
            al = 0 if self.AI < 100 else (1 if self.AI < 200 else 2)
            hl = 0 if self.H < 35 else (1 if self.H < 50 else 2)
            self.state = np.array([gl, dl, al, hl])
            return self.state.copy(), r, self.t >= self.T, False, {}

    env = Env(); Q = np.zeros((3, 3, 3, 3, 5)); hist = []
    for ep in range(n_episodes):
        s, _ = env.reset(); tot = 0; eps = max(.05, 1 - ep / (n_episodes / 2))
        while True:
            a = env.action_space.sample() if np.random.rand() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, done, _, _ = env.step(a)
            Q[tuple(s) + (a,)] += .1 * (r + .95 * np.max(Q[tuple(s2)]) * (1 - done) - Q[tuple(s) + (a,)])
            tot += r; s = s2
            if done: break
        hist.append(tot)

    def ev(fn, n=300):
        rs = []
        for _ in range(n):
            s, _ = env.reset(); t = 0
            while True:
                s, r, d, _, _ = env.step(fn(s)); t += r
                if d: break
            rs.append(t)
        return np.mean(rs), np.std(rs)

    pol = {"π* (Q-learning)": ev(lambda s: int(np.argmax(Q[tuple(s)]))),
           "Luôn Cân bằng": ev(lambda s: 1), "Luôn AI dẫn dắt": ev(lambda s: 3),
           "Random": ev(lambda s: np.random.randint(5))}
    names = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
    return Q, hist, pol, names


def page_bai11():
    bai_header("🤖", "Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
               "MDP 81 trạng thái · 5 hành động · epsilon-greedy · so sánh rule-based",
               ["Cấp độ: Khó", "Reinforcement Learning", "Q-learning"])

    def ctx():
        st.markdown("""
        Nền kinh tế Việt Nam được xem như **môi trường (MDP)**, chính sách là **hành động**,
        phần thưởng phản ánh **phúc lợi xã hội**. Học tăng cường cho phép chính sách thích nghi
        theo trạng thái kinh tế, thay vì cố định như LP.
        """)
        st.warning("⚠️ AI hỗ trợ ra quyết định, **không thay thế** trách nhiệm chính trị (Mục 11).")
        kpi_row([("Trạng thái", "3⁴ = 81", None), ("Hành động", "5", None),
                 ("γ chiết khấu", "0,95", None), ("α học", "0,1", None)])

    def model():
        st.markdown("**Cập nhật Q-learning (Bellman):**")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\big]")
        st.markdown("**Phần thưởng (welfare):**")
        st.latex(r"R_t = w_1\Delta GDP - w_2\Delta unemploy - w_3 CyberRisk - w_4 Emission")
        st.markdown("**5 hành động** (cơ cấu phân bổ K/D/AI/H):")
        st.dataframe(pd.DataFrame({
            "Hành động": ["a0 Truyền thống", "a1 Cân bằng", "a2 Số hóa nhanh", "a3 AI dẫn dắt", "a4 Bao trùm"],
            "K": [70, 40, 25, 20, 30], "D": [10, 25, 45, 20, 20],
            "AI": [10, 15, 15, 45, 10], "H": [10, 20, 15, 15, 40]}), width='stretch', hide_index=True)

    def data():
        st.markdown("**Trạng thái sₜ** — mỗi yếu tố 3 mức {thấp, trung bình, cao}:")
        st.dataframe(pd.DataFrame({
            "Biến trạng thái": ["GDP growth", "Digital index", "AI capacity", "Unemployment risk"],
            "Mức": ["low / medium / high"] * 4}), width='stretch', hide_index=True)
        st.markdown("**Trọng số phần thưởng:** w = (0,40 ΔGDP; 0,25 thất nghiệp; 0,20 cyber; 0,15 phát thải)")
        st.markdown("**Trạng thái VN 2026:** GDP=medium, D=medium, AI=low, U=medium → [1,1,0,1]")

    def calc():
        n_ep = st.select_slider("Số episode huấn luyện", [2000, 5000, 10000], 5000)
        steps([("Xây môi trường VietnamEconomyEnv", "Gymnasium Env: reset, step, action/observation space."),
               ("Huấn luyện Q-table qua nhiều episode", "epsilon-greedy giảm dần 1,0 → 0,05."),
               ("Trích chính sách π*(s) = argmax_a Q", "Hành động tối ưu tại mỗi trạng thái."),
               ("So sánh với 3 chính sách rule-based", "Đánh giá phần thưởng tích lũy bình quân.")])
        with st.spinner("Đang huấn luyện Q-learning..."):
            Q, hist, pol, names = _train_qlearning(n_ep)

        section("Câu 11.3.3 — Chính sách π*(s) tại các trạng thái")
        test = [([1, 1, 0, 1], "VN 2026 thực tế"), ([0, 0, 0, 2], "GDP thấp, D thấp, H cao"),
                ([2, 2, 2, 2], "Tất cả cao"), ([0, 1, 0, 0], "Sau khủng hoảng"),
                ([1, 0, 2, 1], "AI mạnh, D yếu")]
        st.dataframe(pd.DataFrame([{"Trạng thái [G,D,AI,U]": str(s), "Mô tả": d,
                                    "π* (hành động)": names[int(np.argmax(Q[tuple(s)]))]} for s, d in test]),
                     width='stretch', hide_index=True)

        section("Câu 11.3.4 — Learning curve & so sánh chính sách")
        c1, c2 = st.columns(2)
        win = 200; sm = np.convolve(hist, np.ones(win) / win, mode="valid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sm, color=ACC1); ax.set_xlabel("Episode"); ax.set_ylabel("Tổng phúc lợi")
        ax.set_title("Learning curve"); ax.grid(alpha=.3)
        c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ns = list(pol.keys()); ms = [pol[n][0] for n in ns]; ss = [pol[n][1] for n in ns]
        ax.bar(range(len(ns)), ms, yerr=ss, capsize=5, color=[ACC1, ACC2, ACC4, "#9ca3af"], edgecolor=EDGE)
        ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Phúc lợi bình quân"); ax.set_title("So sánh chính sách"); ax.grid(axis="y", alpha=.3)
        c2.pyplot(fig)
        best = max(pol.items(), key=lambda kv: kv[1][0])
        insight("Diễn giải", f"Chính sách <b>{best[0]}</b> đạt phúc lợi cao nhất "
                f"({best[1][0]:.2f}). π* học được biết thích nghi theo trạng thái thay vì cố định.")

    def policy():
        with st.expander("a) GDP thấp, D thấp, U cao → π* chọn gì?", expanded=True):
            st.markdown("Thường chọn hành động đẩy số hóa / cân bằng — tương ứng chiến lược **'quick win'**.")
        with st.expander("b) GDP cao, AI cao, U thấp → π* chọn gì?"):
            st.markdown("Nghiêng về **'consolidation'** (cân bằng, ổn định) khi mọi chỉ số đã tốt.")
        with st.expander("c) Tích hợp π* vào quy trình chính sách thế nào?"):
            st.markdown("Dùng π* làm **công cụ tham mưu**; kết quả phải được hội đồng chính sách thẩm "
                        "định trước khi áp dụng — không tự động hóa hoàn toàn.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# BÀI 12 — AIDEOM-VN TÍCH HỢP (6 MODULE → 4 TAB)
# ============================================================
SCENARIOS_12 = {
    "S1 Truyền thống": {"K": .70, "D": .10, "AI": .10, "H": .10},
    "S2 Số hóa nhanh": {"K": .25, "D": .45, "AI": .15, "H": .15},
    "S3 AI dẫn dắt": {"K": .20, "D": .20, "AI": .45, "H": .15},
    "S4 Bao trùm số": {"K": .30, "D": .20, "AI": .10, "H": .40},
    "S5 Tối ưu cân bằng": {"K": .25, "D": .25, "AI": .30, "H": .20}}


@st.cache_data(show_spinner=False)
def _m1_forecast():
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    K0, L0, D0, AI0, H0, A0 = 27500, 53.9, 20.3, 86, 30, 33.70
    T, ba = 4, 3000

    def fc(al):
        K, D, AI, H, A = K0, D0, AI0, H0, A0
        tr = [A * K**a * L0**b * D**g * AI**d * H**th]
        for t in range(T):
            K = .95*K + al["K"]*ba; D = .88*D + al["D"]*ba*.01
            AI = .85*AI + al["AI"]*ba*.05; H = H + .8*al["H"]*ba*.01 - .02*H
            A = A*(1 + .003*(D/100) + .002*(AI/100) + .004*(H/100))
            L = L0 * 1.009**(t + 1)
            tr.append(A * K**a * L**b * D**g * AI**d * H**th)
        return tr
    return {n: fc(al) for n, al in SCENARIOS_12.items()}, list(range(2026, 2031))


def page_bai12():
    bai_header("🇻🇳", "Bài 12 — Nguyên mẫu AIDEOM-VN (tích hợp 6 module)",
               "Đồ án tổng kết · 6 module liên kết · dashboard 5 kịch bản chính sách",
               ["Cấp độ: Khó", "Tích hợp hệ thống", "6 module · 4 tab"])

    def ctx():
        st.markdown("""
        **Đồ án tổng kết môn học**: tích hợp các kỹ thuật Bài 1-11 thành hệ thống AIDEOM-VN
        đầy đủ **6 module**, với dashboard trực quan hỗ trợ ra quyết định trên **5 kịch bản
        chính sách**. Mục tiêu: từ dữ liệu thực → dự báo → đánh giá → tối ưu → khuyến nghị.
        """)
        kpi_row([("Số module", "6", None), ("Số tab dashboard", "4", None),
                 ("Kịch bản", "5", None), ("Nguồn dữ liệu", "≥ 3", None)])

    def model():
        st.markdown("**Kiến trúc 6 module liên kết (Mục 14 bài báo):**")
        st.dataframe(pd.DataFrame({
            "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "Tên": ["Dự báo kinh tế", "Đánh giá sẵn sàng số", "Tối ưu phân bổ",
                    "Mô phỏng lao động", "Đánh giá rủi ro", "Dashboard ra QĐ"],
            "Kỹ thuật chính": ["Cobb-Douglas (Bài 1)", "TOPSIS + entropy (Bài 6)",
                               "LP (Bài 4) + Dynamic (Bài 8)", "LP NetJob (Bài 9)",
                               "NSGA-II (Bài 7) + SP (Bài 10)", "Streamlit 4 tab"],
            "Hiển thị tại": ["Tab 1", "Tab 1", "Tab 2", "Tab 2", "Tab 3", "Tab 4"]}),
            width='stretch', hide_index=True)

    def data():
        st.markdown("**5 kịch bản chính sách (Mục 15):**")
        st.dataframe(pd.DataFrame({
            "Kịch bản": list(SCENARIOS_12.keys()),
            "K %": [int(v["K"] * 100) for v in SCENARIOS_12.values()],
            "D %": [int(v["D"] * 100) for v in SCENARIOS_12.values()],
            "AI %": [int(v["AI"] * 100) for v in SCENARIOS_12.values()],
            "H %": [int(v["H"] * 100) for v in SCENARIOS_12.values()]}),
            width='stretch', hide_index=True)
        st.caption("Dữ liệu nền: vietnam_macro / sectors / regions (2020-2025).")

    def calc():
        st.markdown("##### 🎛️ Dashboard AIDEOM-VN — 6 module trong 4 tab")
        tabs = st.tabs(["📊 Tổng quan (M1·M2)", "🗺️ Phân bổ (M3·M4)",
                        "⚖️ Rủi ro & Đa mục tiêu (M5)", "🎛️ So sánh kịch bản (M6)"])

        with tabs[0]:
            st.markdown("**M1 — Dự báo kinh tế Cobb-Douglas 2026-2030**")
            gdp_fc, years = _m1_forecast()
            fig, ax = plt.subplots(figsize=(9, 4))
            for (name, tr), cc in zip(gdp_fc.items(), PALETTE):
                ax.plot(years, tr, "o-", ms=4, label=name, color=cc)
            ax.set_xlabel("Năm"); ax.set_ylabel("GDP (ngh.tỷ VND)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
            ax.set_title("GDP theo 5 kịch bản")
            st.pyplot(fig)
            df_fc = pd.DataFrame({n: np.round(tr, 0) for n, tr in gdp_fc.items()}, index=years).T
            df_fc.columns = [str(y) for y in years]
            st.dataframe(df_fc, width='stretch')

            st.markdown("**M2 — Đánh giá sẵn sàng số (TOPSIS + Entropy)**")
            reg = load_regions()
            crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
                    "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
                    "internet_penetration_pct", "gini_coef"]
            is_ben = np.array([True] * 7 + [False]); X = reg[crit].values.astype(float)
            Ce = topsis(X, np.array([.10, .10, .15, .20, .15, .15, .05, .10]), is_ben)[0]
            Cn = topsis(X, entropy_weights(X), is_ben)[0]
            c1, c2 = st.columns([3, 2])
            fig, ax = plt.subplots(figsize=(7, 4)); y = np.arange(6)
            ax.barh(y - .2, Ce, .4, color=ACC1, label="Chuyên gia")
            ax.barh(y + .2, Cn, .4, color=ACC2, label="Entropy")
            ax.set_yticks(y); ax.set_yticklabels([r[:16] for r in REGION_VI], fontsize=8)
            ax.legend(); ax.grid(axis="x", alpha=.3); ax.set_title("Sẵn sàng AI 6 vùng")
            c1.pyplot(fig)
            tb = pd.DataFrame({"Vùng": REGION_VI, "C* CG": Ce.round(4), "C* Ent": Cn.round(4)})
            tb["Hạng"] = tb["C* CG"].rank(ascending=False).astype(int)
            c2.dataframe(tb.sort_values("Hạng"), width='stretch', hide_index=True)

        with tabs[1]:
            st.markdown("**M3 — Tối ưu phân bổ ngân sách ngành-vùng (LP)**")
            mat, Z = _solve_lp4(50000, True, 0.6)
            c1, c2 = st.columns([3, 2])
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
            ax.set_yticks(range(6)); ax.set_yticklabels([r[:16] for r in REGION_VI], fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS); ax.set_title(f"Phân bổ LP (Z*={Z:,.0f})")
            for i in range(6):
                for j in range(4):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > 7000 else "black")
            plt.colorbar(im, shrink=.8); fig.tight_layout()
            c1.pyplot(fig)
            c2.metric("Z* (GDP gain)", f"{Z:,.0f} tỷ")
            c2.metric("Tổng ngân sách", f"{mat.sum():,.0f} tỷ")
            c2.dataframe(pd.DataFrame(mat.round(0), columns=ITEMS, index=REGION_CODE), width='stretch')

            st.markdown("**M4 — Mô phỏng thị trường lao động (NetJob)**")
            from scipy.optimize import linprog
            sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn", "Tài chính", "Logistics", "CNTT", "Giáo dục"]
            risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
            a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
            b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55], float)
            c1v = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
            d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62], float)
            N = 8; coeff = a1 - c1v * risk
            A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
            A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
            for i in range(N):
                A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
                A3[i, i] = c1v[i] * risk[i]; A3[i, N + i] = -d1[i]
            r = linprog(np.concatenate([-coeff, -b1]), A_ub=np.vstack([A1, A2, A3]),
                        b_ub=np.concatenate([[30000], np.zeros(N), np.zeros(N)]),
                        bounds=[(0, None)] * (2 * N), method="highs")
            NJ = coeff * r.x[:N] + b1 * r.x[N:]
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.bar(sec, NJ, color=ACC4, edgecolor=EDGE)
            ax.set_ylabel("NetJob"); ax.grid(axis="y", alpha=.3); ax.set_title("NetJob ròng theo ngành")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            st.pyplot(fig)
            st.metric("Tổng NetJob", f"{-r.fun:,.0f} việc làm")

        with tabs[2]:
            st.markdown("**M5 — NSGA-II đa mục tiêu + Stochastic**")
            with st.spinner("Đang chạy NSGA-II..."):
                F, X = _run_nsga(60, 100, 0.6)
            w = np.array([.40, .25, .20, .15])
            lo, hi = F.min(0), F.max(0); rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
            V = ((F - lo) / rng) * w
            Cs = np.sqrt(((V - w) ** 2).sum(1)) / (np.sqrt((V ** 2).sum(1)) + np.sqrt(((V - w) ** 2).sum(1)))
            best = int(np.argmax(Cs))
            c1, c2 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sc = ax.scatter(-F[:, 0], F[:, 1], c=F[:, 2], cmap="viridis", s=14, alpha=.8)
            ax.scatter(-F[best, 0], F[best, 1], c=ACC5, s=160, marker="*", zorder=5, label="Thỏa hiệp")
            ax.set_xlabel("GDP gain"); ax.set_ylabel("Gini/MAD"); ax.legend(); ax.grid(alpha=.3)
            ax.set_title("Pareto (màu = phát thải)"); fig.colorbar(sc, shrink=.7)
            c1.pyplot(fig)
            c2.markdown("**Nghiệm thỏa hiệp (TOPSIS):**")
            c2.metric("GDP gain", f"{-F[best,0]:,.0f}")
            c2.metric("Gini/MAD", f"{F[best,1]:.1f}")
            c2.metric("Phát thải", f"{F[best,2]:,.0f}")
            st.markdown("**Stochastic (Pyomo) — quyết định first-stage dưới bất định**")
            try:
                x_sp, y_sp, Z_SP, x_ev, Z_EV, Z_WS, det = _run_stochastic()
                kpi_row([("Z*_SP", f"{Z_SP:,.0f}", None), ("VSS", f"{Z_SP - Z_EV:,.0f}", None),
                         ("EVPI", f"{Z_WS - Z_SP:,.0f}", None), ("x_H first-stage", f"{x_sp['H']:,.0f}", None)])
            except Exception as e:
                st.info(f"Bỏ qua phần stochastic (thiếu solver): {e}")

        with tabs[3]:
            st.markdown("**M6 — Dashboard so sánh 5 kịch bản chính sách 2030**")
            gdp_fc, years = _m1_forecast()
            summary = pd.DataFrame({
                "Kịch bản": list(gdp_fc.keys()),
                "GDP 2030 (ngh.tỷ)": [round(tr[-1], 0) for tr in gdp_fc.values()],
                "Tăng trưởng TB (%/năm)": [round(((tr[-1] / tr[0]) ** (1 / 4) - 1) * 100, 2)
                                           for tr in gdp_fc.values()]})
            summary = summary.sort_values("GDP 2030 (ngh.tỷ)", ascending=False).reset_index(drop=True)
            c1, c2 = st.columns([3, 2])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(summary["Kịch bản"], summary["GDP 2030 (ngh.tỷ)"], color=PALETTE[:5], edgecolor=EDGE)
            ax.set_xlabel("GDP 2030 (ngh.tỷ VND)"); ax.grid(axis="x", alpha=.3)
            ax.set_title("Xếp hạng kịch bản theo GDP 2030"); ax.invert_yaxis()
            c1.pyplot(fig)
            c2.dataframe(summary, width='stretch', hide_index=True)
            c2.success(f"🏆 Tốt nhất 2030: **{summary.iloc[0]['Kịch bản']}**")
            insight("Khuyến nghị tổng hợp",
                    "Lấy <b>S5 Tối ưu cân bằng</b> làm xương sống (cân bằng tăng trưởng - bao trùm - "
                    "rủi ro), bổ sung ràng buộc công bằng vùng (M3) và đào tạo lại lao động (M4) "
                    "để giảm thiểu rủi ro xã hội.")

    def policy():
        with st.expander("Khuyến nghị chính sách từ AIDEOM-VN", expanded=True):
            st.markdown(
                "- Kịch bản **AI dẫn dắt / Tối ưu cân bằng** cho GDP 2030 cao nhất nhờ TFP nội sinh "
                "của đầu tư AI và số hóa.\n"
                "- **S4 Bao trùm số** đánh đổi tăng trưởng lấy công bằng vùng và an sinh — phù hợp khi "
                "ưu tiên giảm bất bình đẳng.\n"
                "- AIDEOM-VN là **công cụ hỗ trợ tham mưu**; quyết định cuối thuộc về quy trình chính "
                "trị-xã hội, không thay thế trách nhiệm của nhà hoạch định.")
        with st.expander("Hướng mở rộng nghiên cứu"):
            st.markdown(
                "- Viết bài báo SCIE với use case cụ thể (ĐBSCL hoặc CN chế biến).\n"
                "- Mở rộng sang CGE/DSGE-AI để có cân bằng tổng thể.\n"
                "- Tích hợp dữ liệu thời gian thực; mở rộng RL thành Multi-Agent RL theo bộ-ngành.")

    render_tabs(ctx, model, data, calc, policy)


# ============================================================
# TRANG CHỦ
# ============================================================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1 style="font-size:2.3rem;">🧭 AIDEOM-VN</h1>
          <div class="tag"><b>AI-Driven Decision Optimization Model for Vietnam</b></div>
          <div class="tag" style="margin-top:4px;">12 bài toán mô hình ra quyết định · dữ liệu thực tế Việt Nam 2020–2025</div>
        </div>
        """, unsafe_allow_html=True)

    kpi_row([("GDP 2025", "514,0 tỷ USD", "▲ 8,02%"),
             ("Kinh tế số/GDP", "≈ 19,5%", "▲ 1,2 đpt"),
             ("FDI giải ngân 2025", "27,6 tỷ USD", "▲ 8,9%"),
             ("GDP/người 2025", "5.026 USD", "▲ 6,9%")])
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 📚 Lộ trình 12 bài · mỗi bài 5 trang (Bối cảnh · Mô hình · Dữ liệu · Tính toán · Chính sách)")
        levels = [
            ("Dễ", "#dcfce7", "#16a34a", [
                ("1", "Cobb-Douglas mở rộng + AI — growth accounting, dự báo 2030"),
                ("2", "LP ngân sách 4 hạng mục — shadow price"),
                ("3", "Chỉ số ưu tiên 10 ngành — min-max, sensitivity")]),
            ("Trung bình", "#fef9c3", "#ca8a04", [
                ("4", "LP ngân sách ngành-vùng — công bằng vùng"),
                ("5", "MIP chọn 15 dự án — knapsack, precedence"),
                ("6", "TOPSIS 6 vùng — entropy weight")]),
            ("Khá khó", "#ffedd5", "#ea580c", [
                ("7", "NSGA-II Pareto — 4 mục tiêu xung đột"),
                ("8", "Tối ưu động 2026-2035 — CRRA, SLSQP"),
                ("9", "AI & lao động — NetJob, đào tạo lại")]),
            ("Khó", "#fee2e2", "#dc2626", [
                ("10", "Stochastic 2 giai đoạn — VSS, EVPI"),
                ("11", "Q-learning chính sách thích nghi"),
                ("12", "AIDEOM tích hợp 6 module · dashboard")]),
        ]
        for name, bg, fg, rows in levels:
            st.markdown(f'<span class="pill" style="background:{bg};color:{fg};">{name}</span>',
                        unsafe_allow_html=True)
            for code, desc in rows:
                st.markdown(f'<div style="display:flex;gap:10px;padding:5px 0;">'
                            f'<div style="min-width:32px;font-weight:800;color:{ACC1};">B{code}</div>'
                            f'<div style="color:#334155;">{desc}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### 🧑‍🎓 Thông tin")
        st.markdown(
            '<div class="sig"><div class="nm">Nguyễn Đình Bảo Nghĩa</div>'
            '<div class="mt">Mã sinh viên: 23052345</div>'
            '<div class="mt">Bài tập lớn: Các mô hình ra quyết định</div></div>',
            unsafe_allow_html=True)
        st.markdown("#### 🛠️ Công cụ")
        st.markdown('<div class="softcard" style="color:#334155;">'
                    "NumPy · pandas · SciPy · PuLP · CVXPY · Pyomo · pymoo · Gymnasium · Streamlit</div>",
                    unsafe_allow_html=True)
        st.markdown("#### 📖 Cách dùng")
        st.markdown('<div class="softcard" style="color:#334155;font-size:.9rem;">'
                    "Chọn <b>cấp độ</b> rồi chọn <b>bài</b> ở thanh trên. Mỗi bài có 5 trang con — "
                    "xem kỹ trang <b>🧮 Tính toán</b> để thấy các bước giải và kết quả số.</div>",
                    unsafe_allow_html=True)

    st.markdown("#### 🗃️ Dữ liệu Việt Nam 2020–2025")
    t1, t2, t3, t4 = st.tabs(["📈 Vĩ mô", "🏭 10 ngành", "🗺️ 6 vùng", "📋 Tham chiếu"])
    with t1:
        st.dataframe(load_macro(), width='stretch', hide_index=True)
    with t2:
        sec = load_sectors().copy(); sec.insert(2, "Tên VN", SECTOR_VI)
        st.dataframe(sec, width='stretch', hide_index=True)
    with t3:
        reg = load_regions().copy(); reg.insert(2, "Tên VN", REGION_VI)
        st.dataframe(reg, width='stretch', hide_index=True)
    with t4:
        st.dataframe(pd.DataFrame({
            "Chỉ tiêu": ["GDP (ngh.tỷ VND)", "GDP (tỷ USD)", "GDP/người (USD)", "Tăng trưởng GDP (%)",
                          "Dân số (triệu)", "FDI giải ngân (tỷ USD)", "Xuất khẩu (tỷ USD)",
                          "Kinh tế số/GDP (%)", "DN công nghệ số", "Startup AI", "GII (/139)"],
            "2024": ["11.511,9", "476,3", "4.700", "7,09", "101,3", "25,35", "405,5", "18,3",
                      "73.788", "≈278", "44"],
            "2025": ["12.847,6", "514,0", "5.026", "8,02", "102,3", "27,60", "475,0", "≈19,5",
                      "80.052", "≈350+", "44"]}), width='stretch', hide_index=True)


# ============================================================
# ĐIỀU HƯỚNG + ROUTER
# ============================================================
NAV = {
    "🏠 Trang chủ": [("Trang chủ", "home")],
    "🟢 Dễ (Bài 1–3)": [("Bài 1 · Cobb-Douglas + AI", "b1"),
                        ("Bài 2 · LP ngân sách số", "b2"),
                        ("Bài 3 · Priority 10 ngành", "b3")],
    "🟡 Trung bình (Bài 4–6)": [("Bài 4 · LP ngành-vùng", "b4"),
                               ("Bài 5 · MIP 15 dự án", "b5"),
                               ("Bài 6 · TOPSIS 6 vùng", "b6")],
    "🟠 Khá khó (Bài 7–9)": [("Bài 7 · NSGA-II Pareto", "b7"),
                            ("Bài 8 · Tối ưu động", "b8"),
                            ("Bài 9 · Lao động & AI", "b9")],
    "🔴 Khó (Bài 10–12)": [("Bài 10 · Stochastic SP", "b10"),
                          ("Bài 11 · Q-learning RL", "b11"),
                          ("Bài 12 · AIDEOM tích hợp", "b12")],
}
ROUTES = {"home": page_home, "b1": page_bai1, "b2": page_bai2, "b3": page_bai3,
          "b4": page_bai4, "b5": page_bai5, "b6": page_bai6, "b7": page_bai7,
          "b8": page_bai8, "b9": page_bai9, "b10": page_bai10, "b11": page_bai11, "b12": page_bai12}

c1, c2 = st.columns([1, 2])
with c1:
    group = st.selectbox("Cấp độ", list(NAV.keys()), index=0)
with c2:
    opts = NAV[group]
    if len(opts) == 1:
        choice_key = opts[0][1]
        st.selectbox("Bài", [opts[0][0]], index=0, disabled=True)
    else:
        labels = [o[0] for o in opts]
        picked = st.selectbox("Bài", labels, index=0)
        choice_key = dict(zip(labels, [o[1] for o in opts]))[picked]

st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0 14px;'>",
            unsafe_allow_html=True)
ROUTES.get(choice_key, page_home)()
st.markdown(
    "<hr style='border:none;border-top:1px solid #e2e8f0;margin:28px 0 8px;'>"
    "<div style='text-align:center;color:#94a3b8;font-size:.82rem;'>"
    "AIDEOM-VN · Nguyễn Đình Bảo Nghĩa · Bài tập lớn: Các mô hình ra quyết định</div>",
    unsafe_allow_html=True)
