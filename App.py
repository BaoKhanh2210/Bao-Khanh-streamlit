"""
AIDEOM-VN Dashboard — Các mô hình ra quyết định
Sinh viên: Nguyễn Bảo Khánh | MSSV: 23051266
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# PAGE CONFIG
st.set_page_config(
    page_title="VN AIDEOM-VN | Mô hình ra quyết định",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e0f0ff !important; }
[data-testid="stSidebar"] .stSelectbox label {
    color: #90caf9 !important; font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
}

.student-card {
    position: fixed; bottom: 0; left: 0; width: 240px;
    background: rgba(0,0,0,0.45);
    border-top: 1px solid rgba(144,202,249,0.25);
    padding: 10px 14px; font-size: 11.5px; line-height: 1.85;
    color: #b0d4f1; font-family: 'IBM Plex Mono', monospace; z-index: 9999;
}
.student-card .lbl { color: #64b5f6; font-size: 9.5px; letter-spacing: 1px; text-transform: uppercase; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.page-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    padding: 22px 28px; border-radius: 12px; margin-bottom: 20px;
    border-left: 4px solid #42a5f5;
}
.page-header h1 { color: #fff; font-size: 1.6rem; font-weight: 700; margin: 0; }
.page-header p { color: #90caf9; margin: 5px 0 0; font-size: 0.9rem; }

.sec-title {
    font-size: 0.95rem; font-weight: 600; color: #1565c0;
    border-bottom: 2px solid #bbdefb;
    padding-bottom: 5px; margin: 18px 0 10px; letter-spacing: 0.3px;
}
.info-box {
    background: #e8f4f8; border-left: 4px solid #29b6f6;
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 10px 0; font-size: 0.88rem; color: #1a1a2e;
}
/* ── Box Bối cảnh Việt Nam ── */
.context-box {
    background: linear-gradient(135deg, #fff8e1 0%, #fffde7 100%);
    border-left: 5px solid #ffa726; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 14px 0; color: #4e342e; line-height: 1.7; font-size: 0.9rem;
}
.context-box h4 { color: #e65100; margin: 0 0 8px; font-size: 1rem; }
.context-box b { color: #bf360c; }
/* ── Box Mô hình toán học ── */
.model-box {
    background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
    border-left: 5px solid #5e35b1; border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 14px 0; color: #1a237e; line-height: 1.8; font-size: 0.9rem;
}
.model-box h4 { color: #4527a0; margin: 0 0 8px; font-size: 1rem; }
.model-box .formula {
    background: rgba(255,255,255,0.7); border: 1px dashed #9575cd; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem; color: #311b92; text-align: center;
}
.model-box ul { margin: 6px 0; padding-left: 20px; }
.model-box li { margin: 3px 0; }
/* ── Box Nhận xét/Diễn giải ── */
.note-box {
    background: #e8f5e9; border-left: 4px solid #43a047; border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 12px 0; font-size: 0.88rem; color: #1b5e20; line-height: 1.65;
}
.note-box b { color: #2e7d32; }

.sidebar-logo { text-align: center; padding: 18px 14px 14px; border-bottom: 1px solid rgba(144,202,249,0.15); margin-bottom: 10px; }
.sidebar-logo .title { font-size: 1.05rem; font-weight: 700; color: #e3f2fd; letter-spacing: 1px; }
.sidebar-logo .sub { font-size: 0.68rem; color: #90caf9; letter-spacing: 2px; text-transform: uppercase; margin-top: 3px; }

/* ── Hero card trang chủ (theo app gốc) ── */
.hero-card {
    background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(139,92,246,0.20), rgba(34,197,94,0.13));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px; padding: 30px 34px; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.hero-title { font-size: 42px; font-weight: 900; line-height: 1.15; margin-bottom: 10px; color: #0f2027; }
.hero-subtitle { font-size: 20px; font-weight: 650; opacity: 0.92; margin-bottom: 12px; color: #1565c0; }
.hero-note { font-size: 16px; line-height: 1.65; opacity: 0.88; color: #1a2332; }
.badge {
    display: inline-block; padding: 7px 12px; border-radius: 999px;
    background: rgba(21,101,192,0.10); border: 1px solid rgba(21,101,192,0.18);
    margin-right: 8px; margin-bottom: 8px; font-size: 14px; font-weight: 650; color: #1565c0;
}
.section-card {
    background: #f8fbff; border: 1px solid rgba(21,101,192,0.10);
    border-radius: 18px; padding: 18px 20px; margin-bottom: 16px;
}
.section-card h4 { color: #1a237e; margin: 6px 0; }
.small-muted { font-size: 14px; opacity: 0.85; line-height: 1.55; color: #455a64; }
.big-number { font-size: 30px; font-weight: 850; color: #1565c0; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2rem">🇻🇳</div>
        <div class="title">VN AIDEOM-VN</div>
        <div class="sub">Decision Optimization Model</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "CHỌN BÀI",
        [
            "🏠 Trang chủ",
            "🌱 Bài 1 — Cobb-Douglas + AI",
            "💰 Bài 2 — LP ngân sách số",
            "📊 Bài 3 — Priority 10 ngành",
            "🗺️ Bài 4 — LP ngành-vùng",
            "🎯 Bài 5 — MIP 15 dự án",
            "🏆 Bài 6 — TOPSIS 6 vùng",
            "🌐 Bài 7 — NSGA-II Pareto",
            "⏳ Bài 8 — Động 2026-2035",
            "👷 Bài 9 — Lao động & AI",
            "🎲 Bài 10 — Stochastic SP",
            "🤖 Bài 11 — Q-learning RL",
            "🧠 Bài 12 — AIDEOM tích hợp",
        ],
    )
    st.markdown("""
    <div class="student-card">
        <div class="lbl">Họ và tên</div>
        Nguyễn Bảo Khánh<br>
        <div class="lbl">Mã sinh viên</div>
        23051266<br>
        <div class="lbl">Bài tập lớn</div>
        Các mô hình ra quyết định
    </div>
    """, unsafe_allow_html=True)

# DATA
@st.cache_data
def get_macro():
    return pd.DataFrame({
        "year": [2020,2021,2022,2023,2024,2025],
        "GDP_trillion_VND": [8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
        "population_million": [97.3,98.5,99.5,100.3,101.3,102.3],
        "digital_economy_share_GDP_pct": [12.0,12.7,14.3,16.5,18.3,19.5],
        "labor_productivity_million_VND": [175.0,186.0,199.0,210.0,221.9,245.0],
    })

@st.cache_data
def get_sectors():
    return pd.DataFrame({
        "sector_name_vi": ["Nông-Lâm-TS","CN chế biến","Xây dựng","Khai khoáng",
                           "Bán buôn-BR","Tài chính-NH","Logistics","CNTT-TT","Giáo dục","Y tế"],
        "growth_rate_2024_pct": [3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
        "gdp_share_2024_pct":   [11.86,28.0,7.5,3.5,11.0,6.5,5.2,8.5,4.8,3.7],
        "spillover_coef_0_1":   [0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
        "export_billion_USD":   [40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
        "labor_million":        [13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
        "ai_readiness_0_100":   [15,55,20,30,48,72,42,88,38,45],
        "automation_risk_pct":  [18,42,25,55,38,52,35,28,22,18],
    })

@st.cache_data
def get_regions():
    return pd.DataFrame({
        "region_name_vi": ["Trung du MN Bắc","ĐB sông Hồng","Bắc Trung Bộ","Tây Nguyên","Đông Nam Bộ","ĐB sông CL"],
        "grdp_per_capita_million_VND": [57.0,152.3,87.5,68.9,158.9,80.5],
        "fdi_registered_billion_USD":  [3.5,20.0,8.2,0.8,18.5,2.1],
        "digital_index_0_100":         [38,78,55,32,82,48],
        "ai_readiness_0_100":          [22,68,40,18,75,30],
        "trained_labor_pct":           [21.5,36.8,27.5,18.2,42.5,16.8],
        "rd_intensity_pct":            [0.18,0.85,0.32,0.15,0.78,0.22],
        "internet_penetration_pct":    [72,92,84,68,94,78],
        "gini_coef":                   [0.405,0.358,0.372,0.412,0.385,0.392],
    })

df_macro   = get_macro()
df_sectors = get_sectors()
df_regions = get_regions()

def show_fig(fig):
    st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════
# TRANG CHỦ
# ══════════════════════════════════════════════════
if menu == "🏠 Trang chủ":
    # ── HERO SECTION ──
    st.markdown(
        '''
        <div class="hero-card">
            <div class="hero-title">🇻🇳 AIDEOM-VN</div>
            <div class="hero-subtitle">AI-Driven Decision Optimization Model for Vietnam</div>
            <div class="hero-note">
                Dashboard mô phỏng 12 bài toán ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI.
                Hệ thống kết hợp <b>Python</b>, <b>tối ưu hóa</b>, <b>học tăng cường</b>,
                <b>mô phỏng chính sách</b> để chuyển bài toán kinh tế thành mô hình định lượng có thể kiểm chứng.
            </div>
            <br>
            <span class="badge">🐍 Python</span>
            <span class="badge">📊 Streamlit Dashboard</span>
            <span class="badge">🧮 Optimization</span>
            <span class="badge">🤖 Reinforcement Learning</span>
            <span class="badge">🇻🇳 Vietnam 2020–2025 Data</span>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # ── QUICK MACRO METRICS ──
    st.subheader("📌 Bức tranh kinh tế Việt Nam tham chiếu nhanh 2024–2025")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GDP 2025","514,0 tỷ USD","+8,02%")
    c2.metric("Kinh tế số/GDP","≈19,5%","+1,2 điểm %")
    c3.metric("FDI giải ngân 2025","27,6 tỷ USD","+8,9%")
    c4.metric("GDP/người 2025","5.026 USD","+6,9%")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("GDP 2025","12.847,6 nghìn tỷ VND")
    c6.metric("DN công nghệ số","80,1 nghìn")
    c7.metric("GII 2025","Hạng 44/139")
    c8.metric("KH-CN/GDP","≈2,49%")
    st.caption("Các chỉ tiêu lấy theo bảng số liệu tham chiếu nhanh trong đề bài, làm tròn để phục vụ mô phỏng và giảng dạy.")

    # ── MỤC TIÊU WEB APP ──
    st.divider()
    st.subheader("🎯 Mục tiêu của web app")
    g1,g2,g3 = st.columns(3)
    with g1:
        st.markdown('''<div class="section-card"><div class="big-number">①</div>
        <h4>Chuyển chính sách thành mô hình</h4>
        <p class="small-muted">Mỗi bài biến một vấn đề phát triển kinh tế Việt Nam thành mô hình toán học:
        hàm sản xuất, LP, MIP, TOPSIS, Pareto, stochastic programming và Q-learning.</p></div>''', unsafe_allow_html=True)
    with g2:
        st.markdown('''<div class="section-card"><div class="big-number">②</div>
        <h4>Chạy thử kịch bản tương tác</h4>
        <p class="small-muted">Người dùng có thể chỉnh tham số, ngân sách, trọng số, ràng buộc và cú sốc;
        sau đó xem bảng kết quả, biểu đồ và thay đổi chính sách tương ứng.</p></div>''', unsafe_allow_html=True)
    with g3:
        st.markdown('''<div class="section-card"><div class="big-number">③</div>
        <h4>Giải nghĩa kết quả chính sách</h4>
        <p class="small-muted">Diễn giải kết quả theo góc nhìn chính sách công:
        tăng trưởng, bao trùm, rủi ro, công bằng vùng và trách nhiệm giải trình.</p></div>''', unsafe_allow_html=True)

    # ── DỮ LIỆU GỐC ──
    st.divider()
    st.markdown('<div class="sec-title">📋 Dữ liệu gốc Việt Nam 2020–2025</div>', unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["Vĩ mô 2020–2025","10 ngành 2024","6 vùng KT-XH"])
    with t1:
        st.dataframe(df_macro, use_container_width=True)
        fig,ax = plt.subplots(figsize=(9,4))
        ax.fill_between(df_macro["year"],df_macro["GDP_trillion_VND"],alpha=0.15,color="#1976d2")
        ax.plot(df_macro["year"],df_macro["GDP_trillion_VND"],"o-",color="#1976d2",lw=2.5)
        for y,v in zip(df_macro["year"],df_macro["GDP_trillion_VND"]):
            ax.annotate(f"{v:,.0f}",(y,v),textcoords="offset points",xytext=(0,10),ha="center",fontsize=9)
        ax.set_title("GDP Việt Nam 2020–2025",fontweight="bold"); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)
    with t2:
        st.dataframe(df_sectors, use_container_width=True)
    with t3:
        st.dataframe(df_regions, use_container_width=True)
    st.caption("⚠️ Lưu ý: Các con số trong tệp CSV được làm tròn để thuận tiện cho việc giảng dạy và lập trình. "
               "Vì vậy kết quả mô phỏng có thể lệch nhẹ so với số liệu gốc trong đề bài. "
               "Khi viết luận văn/bài báo, cần truy xuất số liệu gốc từ Cục Thống kê quốc gia (nso.gov.vn) "
               "và Tổng cục Hải quan để bảo đảm tính học thuật.")

    # ── BẢN ĐỒ 12 BÀI ──
    st.divider()
    st.subheader("🗺️ Bản đồ 12 bài theo 4 cấp độ")
    level_df = pd.DataFrame({
        "Cấp độ": ["🟢 DỄ","🟢 DỄ","🟢 DỄ","🟡 TRUNG BÌNH","🟡 TRUNG BÌNH","🟡 TRUNG BÌNH",
                   "🟠 KHÁ KHÓ","🟠 KHÁ KHÓ","🟠 KHÁ KHÓ","🔴 KHÓ","🔴 KHÓ","🔴 KHÓ"],
        "Bài": ["Bài 1","Bài 2","Bài 3","Bài 4","Bài 5","Bài 6","Bài 7","Bài 8","Bài 9","Bài 10","Bài 11","Bài 12"],
        "Tên bài": ["Cobb-Douglas + AI","LP ngân sách số","Priority 10 ngành","LP ngành-vùng","MIP 15 dự án",
                    "TOPSIS 6 vùng","NSGA-II Pareto","Tối ưu động 2026–2035","Lao động & AI",
                    "Stochastic Programming","Q-learning RL","AIDEOM tích hợp"],
        "Trọng tâm": ["TFP, growth accounting, GDP 2030","Phân bổ 100 nghìn tỷ cho I/AI/H/R&D",
                      "Xếp hạng ngành ưu tiên chuyển đổi số","Công bằng vùng miền trong phân bổ ngân sách",
                      "Chọn danh mục dự án số tối ưu","Xếp hạng vùng sẵn sàng AI",
                      "Đánh đổi tăng trưởng - bao trùm - môi trường - dữ liệu","Lộ trình đầu tư nhiều năm",
                      "Tác động AI đến lao động và kỹ năng","Tối ưu trong điều kiện bất định",
                      "Chính sách thích nghi bằng Q-learning","Dashboard tích hợp và khuyến nghị chính sách"],
        "Công cụ chính": ["numpy, pandas","scipy.optimize, PuLP","pandas, min-max, heatmap","PuLP, CVXPY",
                          "PuLP/CBC","TOPSIS, Entropy","pymoo","cvxpy, numpy","mô phỏng kịch bản",
                          "pyomo / scipy","gymnasium, RL","streamlit dashboard"],
    })
    st.dataframe(level_df, use_container_width=True, hide_index=True)

    info = [("DỄ","Bài 1–3","Cobb-Douglas, LP, MCDM","#42a5f5"),
            ("TRUNG BÌNH","Bài 4–6","LP đầy đủ, MIP, TOPSIS","#26c6da"),
            ("KHÁ KHÓ","Bài 7–9","NSGA-II, Tối ưu động, Lao động","#ffa726"),
            ("KHÓ","Bài 10–12","Stochastic LP, Q-learning, AIDEOM-VN","#ef5350")]
    cols = st.columns(4)
    for col,(level,bai,desc,color) in zip(cols,info):
        col.markdown(f'<div style="border-top:4px solid {color};background:#f8fbff;border-radius:8px;padding:12px 14px"><div style="font-size:9px;font-weight:700;color:{color};letter-spacing:2px">{level}</div><div style="font-size:1rem;font-weight:700;color:#1a237e;margin:3px 0">{bai}</div><div style="font-size:0.78rem;color:#546e7a">{desc}</div></div>', unsafe_allow_html=True)

    # ── DỮ LIỆU & CÁCH DÙNG ──
    st.divider()
    st.subheader("📁 Dữ liệu và cách sử dụng")
    d1,d2 = st.columns([1.2,1])
    with d1:
        st.markdown('''<div class="section-card"><h4>📦 Bộ dữ liệu đi kèm</h4>
        <ul>
        <li><b>vietnam_macro_2020_2025.csv</b>: GDP, FDI, xuất nhập khẩu, lạm phát, năng suất lao động, kinh tế số/GDP.</li>
        <li><b>vietnam_sectors_2024.csv</b>: 10 ngành, tăng trưởng, xuất khẩu, lao động, AI readiness, rủi ro tự động hóa.</li>
        <li><b>vietnam_regions_2024.csv</b>: 6 vùng, GRDP/người, FDI, digital index, AI readiness, Gini, R&D.</li>
        </ul></div>''', unsafe_allow_html=True)
    with d2:
        st.markdown('''<div class="section-card"><h4>🚀 Cách dùng nhanh</h4>
        <p class="small-muted">1. Chọn một bài ở menu bên trái.<br>
        2. Đọc bối cảnh và mô hình toán học.<br>
        3. Điều chỉnh tham số bằng slider.<br>
        4. Xem bảng kết quả, biểu đồ và nhận xét.</p></div>''', unsafe_allow_html=True)

    # ── LƯU Ý HỌC THUẬT (quan trọng) ──
    st.markdown('''<div class="context-box">
    <h4>⚠️ Lưu ý học thuật về số liệu</h4>
    <b>Các số liệu sử dụng trong dashboard này được làm tròn</b> nhằm phục vụ mục đích mô phỏng,
    giảng dạy và lập trình. Do đó, <b>kết quả tính toán có thể lệch nhẹ so với số liệu gốc trong đề bài</b>
    và so với báo cáo chính thức. Điều này KHÔNG ảnh hưởng đến tính đúng đắn của mô hình và phương pháp.
    <br><br>
    Khi sử dụng cho <b>luận văn, khóa luận hoặc bài báo khoa học</b>, người học cần:
    <ul>
      <li>Truy xuất số liệu gốc từ <b>Cục Thống kê quốc gia</b> (nso.gov.vn / gso.gov.vn) và <b>Tổng cục Hải quan</b>.</li>
      <li>Kiểm tra lại <b>đơn vị đo lường, thời điểm công bố và phương pháp tính</b> của từng chỉ tiêu.</li>
      <li>Ghi rõ nguồn dữ liệu và năm cơ sở khi trích dẫn.</li>
    </ul>
    Nguồn tham chiếu: NSO/GSO · Bộ KH-CN (MoST) · Bộ TT-TT (MIC) · Bộ KH-ĐT (MPI) · World Bank · Global Innovation Index 2025.
    </div>''', unsafe_allow_html=True)

    st.success("Gợi ý: bắt đầu từ Bài 1 → Bài 2 → Bài 3 để hiểu logic tăng trưởng, tối ưu ngân sách và xếp hạng ưu tiên trước khi sang các bài khó hơn.")

# ══════════════════════════════════════════════════
# BÀI 1 — COBB-DOUGLAS
# ══════════════════════════════════════════════════
elif menu == "🌱 Bài 1 — Cobb-Douglas + AI":
    st.markdown('<div class="page-header"><h1>Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng</h1><p>TFP · Dự báo GDP · Phân rã tăng trưởng · Kịch bản 2030</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 1.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo Cục Thống kê quốc gia, GDP Việt Nam 2024 đạt <b>11.511,9 nghìn tỷ VND</b> (tăng 7,09%),
    năng suất lao động đạt <b>221,9 triệu VND/người</b>, đến 2025 đạt 245,0 triệu VND/người.
    Đóng góp khoa học - công nghệ vào GDP 2025 ước <b>khoảng 2,49%</b>, kinh tế số chiếm <b>18,3-19,5% GDP</b>.
    <br><br>
    <b>Câu hỏi:</b> Nếu mô hình hóa nền kinh tế bằng hàm Cobb-Douglas mở rộng thêm yếu tố số hóa (D),
    năng lực AI và vốn nhân lực số (H) thì sản lượng dự báo có khớp thực tế không, và yếu tố nào
    đóng góp lớn nhất vào tăng trưởng?
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 1.2 — Mô hình toán học</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Hàm sản xuất Cobb-Douglas mở rộng</h4>
    <div class="formula">Y(t) = A(t) · K^0.33 · L^0.42 · D^0.10 · AI^0.08 · H^0.07</div>
    <b>Điều kiện lợi suất không đổi theo quy mô:</b>
    <div class="formula">α + β + γ + δ + θ = 1</div>
    <b>Phân rã tăng trưởng (lấy log và sai phân theo thời gian):</b>
    <div class="formula">ΔlnY = ΔlnA + α·ΔlnK + β·ΔlnL + γ·ΔlnD + δ·ΔlnAI + θ·ΔlnH</div>
    <b>Trong đó:</b>
    <ul>
      <li><b>Y</b> - GDP (nghìn tỷ VND); <b>A</b> - năng suất nhân tố tổng hợp (TFP)</li>
      <li><b>K</b> - vốn vật chất; <b>L</b> - lao động (triệu người)</li>
      <li><b>D</b> - chỉ số số hóa (tỷ trọng kinh tế số/GDP)</li>
      <li><b>AI</b> - năng lực AI (nghìn DN công nghệ số); <b>H</b> - vốn nhân lực số (% LĐ qua đào tạo)</li>
    </ul>
    <b>Hệ số đề xuất:</b> α=0,33 · β=0,42 · γ=0,10 · δ=0,08 · θ=0,07
    </div>""", unsafe_allow_html=True)

    years = df_macro["year"].values
    Y = df_macro["GDP_trillion_VND"].values
    K   = np.array([16500,17800,19600,21300,23500,25900])
    L   = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
    D   = df_macro["digital_economy_share_GDP_pct"].values
    AI  = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
    H   = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
    al,be,ga,de,th = 0.33,0.42,0.10,0.08,0.07
    A = Y/(K**al*L**be*D**ga*AI**de*H**th)

    st.markdown('<div class="sec-title">📋 1.3 — Dữ liệu đầu vào Việt Nam 2020-2025</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Năm": years, "Y (GDP ng.tỷ)": Y, "K (vốn t.lũy)": K, "L (triệu LĐ)": L,
        "D (KTS/GDP %)": D, "AI (ng.DN số)": AI, "H (LĐ ĐT %)": H,
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">🔬 1.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)

    # 1.4.1
    st.markdown('<div class="sec-title">📌 Câu 1.4.1 — TFP (Solow Residual)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Năm":years,"GDP thực tế":Y,"A_t/TFP":A.round(4)}), use_container_width=True)
    fig,ax = plt.subplots(figsize=(9,4))
    ax.plot(years,A,"o-",color="#1976d2",lw=2.5,markersize=8)
    for y,a in zip(years,A): ax.annotate(f"{a:.4f}",(y,a),textcoords="offset points",xytext=(0,10),ha="center",fontsize=9)
    ax.set_title("Xu hướng TFP 2020–2025",fontweight="bold"); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)
    c1,c2,c3 = st.columns(3)
    c1.metric("TFP 2020",f"{A[0]:.4f}"); c2.metric("TFP 2025",f"{A[-1]:.4f}")
    c3.metric("Tăng TFP TB",f"{((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm")
    st.markdown(f"""<div class="note-box">💡 <b>Nhận xét:</b> TFP có xu hướng tăng dần
    ({((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm) - phản ánh chất lượng tăng trưởng được cải thiện,
    đặc biệt giai đoạn hậu COVID-19 nhờ chuyển đổi số và ứng dụng công nghệ.</div>""", unsafe_allow_html=True)

    # 1.4.2
    st.markdown('<div class="sec-title">📌 Câu 1.4.2 — Dự báo & MAPE</div>', unsafe_allow_html=True)
    A_m = A.mean(); Y_hat = A_m*(K**al*L**be*D**ga*AI**de*H**th)
    mape = np.mean(np.abs((Y-Y_hat)/Y))*100
    st.dataframe(pd.DataFrame({"Năm":years,"Thực tế":Y,"Dự báo":Y_hat.round(1),"Sai số %":((Y_hat-Y)/Y*100).round(2)}), use_container_width=True)
    st.metric("MAPE",f"{mape:.4f}%")
    fig,ax = plt.subplots(figsize=(9,4))
    ax.plot(years,Y,"o-",label="Thực tế",color="#1976d2",lw=2.5); ax.plot(years,Y_hat,"s--",label="Dự báo",color="#e53935",lw=2)
    ax.set_title("GDP thực tế vs Dự báo",fontweight="bold"); ax.legend(); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)

    # 1.4.3
    st.markdown('<div class="sec-title">📌 Câu 1.4.3 — Phân rã tăng trưởng</div>', unsafe_allow_html=True)
    n=5
    g_Y=(np.log(Y[-1])-np.log(Y[0]))/n; g_K=(np.log(K[-1])-np.log(K[0]))/n; g_L=(np.log(L[-1])-np.log(L[0]))/n
    g_D=(np.log(D[-1])-np.log(D[0]))/n; g_AI=(np.log(AI[-1])-np.log(AI[0]))/n; g_H=(np.log(H[-1])-np.log(H[0]))/n
    g_A2=(np.log(A[-1])-np.log(A[0]))/n
    contrib={"TFP(A)":g_A2/g_Y*100,"K":al*g_K/g_Y*100,"L":be*g_L/g_Y*100,"D":ga*g_D/g_Y*100,"AI":de*g_AI/g_Y*100,"H":th*g_H/g_Y*100}
    st.dataframe(pd.DataFrame({"Yếu tố":list(contrib.keys()),"Đóng góp %":[round(v,2) for v in contrib.values()]}), use_container_width=True)
    fig,ax = plt.subplots(figsize=(9,4))
    cols7=["#42a5f5" if v>=0 else "#ef5350" for v in contrib.values()]
    bars=ax.bar(contrib.keys(),contrib.values(),color=cols7,edgecolor="white",lw=1.5)
    for bar,val in zip(bars,contrib.values()): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(1 if val>=0 else -2),f"{val:.1f}%",ha="center",fontsize=9,fontweight="bold")
    ax.set_title("Phân rã tăng trưởng GDP 2020–2025",fontweight="bold"); ax.axhline(0,color="gray",lw=0.8); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)

    # 1.4.4
    st.markdown('<div class="sec-title">📌 Câu 1.4.4 — Kịch bản GDP 2030</div>', unsafe_allow_html=True)
    K30=K[-1]*(1.06)**5; L30=L[-1]*(1.005)**5; A30=A[-1]*(1.012)**5; D30,AI30,H30=30.0,100.0,35.0
    Y30=A30*(K30**al*L30**be*D30**ga*AI30**de*H30**th)
    st.success(f"🎯 GDP dự báo 2030: **{Y30:,.0f} nghìn tỷ VND** | Tăng TB: **{((Y30/Y[-1])**(1/5)-1)*100:.2f}%/năm**")
    c1,c2 = st.columns(2); c1.metric("GDP 2025",f"{Y[-1]:,.0f}"); c2.metric("GDP 2030 (dự báo)",f"{Y30:,.0f}")

# ══════════════════════════════════════════════════
# BÀI 2 — PHÂN BỔ NGÂN SÁCH
# ══════════════════════════════════════════════════
elif menu == "💰 Bài 2 — LP ngân sách số":
    st.markdown('<div class="page-header"><h1>Bài 2 — Phân bổ ngân sách 4 hạng mục đầu tư số</h1><p>LP · Shadow price · Độ nhạy ngân sách</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 2.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo <b>Quyết định 749/QĐ-TTg</b> về Chương trình Chuyển đổi số quốc gia, đến 2025 Việt Nam đặt mục tiêu
    kinh tế số đạt <b>20% GDP</b>. Giả sử Bộ KH-ĐT đề xuất phân bổ <b>100.000 tỷ VND</b> ngân sách trung ương 2026
    cho 4 hạng mục: hạ tầng số (I), AI & dữ liệu (AI), nhân lực số (H), R&D công nghệ.
    Mỗi hạng mục có hệ số tác động khác nhau tới tăng GDP, đồng thời phải tuân thủ tỷ lệ tối thiểu theo QĐ 411/QĐ-TTg.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 2.2 — Mô hình toán học (Quy hoạch tuyến tính)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Biến quyết định (đơn vị: nghìn tỷ VND)</h4>
    x1 = hạ tầng số · x2 = AI & dữ liệu · x3 = nhân lực số · x4 = R&D công nghệ
    <h4>Hàm mục tiêu (tối đa hóa tăng GDP kỳ vọng)</h4>
    <div class="formula">max Z = 0,85·x1 + 1,20·x2 + 0,95·x3 + 1,35·x4</div>
    <b>Ràng buộc:</b>
    <ul>
      <li>x1 + x2 + x3 + x4 ≤ 100 (ngân sách tổng)</li>
      <li>x1 ≥ 25 (hạ tầng số tối thiểu); x2 ≥ 15 (AI); x3 ≥ 20 (nhân lực); x4 ≥ 10 (R&D)</li>
      <li>x2 + x4 ≥ 0,35·(x1+x2+x3+x4) (tỷ trọng công nghệ chiến lược)</li>
      <li>x1, x2, x3, x4 ≥ 0</li>
    </ul>
    <b>Diễn giải hệ số:</b> R&D có hệ số cao nhất (1,35) do tác động lan tỏa dài hạn;
    AI cao hơn hạ tầng do thu hồi vốn nhanh hơn nhưng cần nhân lực số bổ trợ.
    </div>""", unsafe_allow_html=True)

    from scipy.optimize import linprog
    c=[-0.85,-1.20,-0.95,-1.35]
    A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
    b_ub=[100,-25,-15,-20,-10,0]
    res=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method="highs")
    st.markdown('<div class="sec-title">🔬 2.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
    if res.success:
        x=res.x; Z=-res.fun
        labels=["Hạ tầng số","AI & dữ liệu","Nhân lực số","R&D"]
        coeffs=[0.85,1.20,0.95,1.35]
        # ── 2.4.1 ──
        st.markdown('<div class="sec-title">📌 Câu 2.4.1 — Giải LP, phân bổ tối ưu</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Hạng mục":labels,"Phân bổ (nghìn tỷ)":x.round(2),
                     "Hệ số tác động":coeffs,"GDP đóng góp":(x*coeffs).round(2),
                     "Tỷ lệ %":(x/x.sum()*100).round(1)}), use_container_width=True)
        st.success(f"**Z* = {Z:.2f} nghìn tỷ VND GDP kỳ vọng** | Ngân sách dùng: {x.sum():.0f}/100 nghìn tỷ")
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(11,4))
        ax1.bar(["x₁","x₂","x₃","x₄"],x,color=["#42a5f5","#26c6da","#66bb6a","#ffa726"],edgecolor="white")
        for i,v in enumerate(x): ax1.text(i,v+1,f"{v:.0f}",ha="center",fontsize=9,fontweight="bold")
        ax1.set_title("Phân bổ tối ưu (nghìn tỷ VND)",fontweight="bold"); ax1.grid(axis="y",alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
        ax2.pie(x,labels=labels,autopct="%1.1f%%",colors=["#42a5f5","#26c6da","#66bb6a","#ffa726"],startangle=90)
        ax2.set_title("Cơ cấu phân bổ",fontweight="bold"); fig.tight_layout(); show_fig(fig)

        # ── 2.4.2 Shadow price ──
        st.markdown('<div class="sec-title">📌 Câu 2.4.2 — Giá đối ngẫu (Shadow price)</div>', unsafe_allow_html=True)
        z0=Z; bu1=list(b_ub); bu1[0]=101
        r1=linprog(c,A_ub=A_ub,b_ub=bu1,bounds=[(0,None)]*4,method="highs")
        sp=(-r1.fun)-z0
        c1,c2=st.columns(2)
        c1.metric("Shadow price ngân sách tổng",f"{sp:.4f} ng.tỷ GDP / ng.tỷ ngân sách")
        c2.metric("Tức là +1 ng.tỷ ngân sách",f"→ +{sp*1000:,.0f} tỷ GDP")
        st.markdown(f"""<div class="note-box">💡 <b>Ý nghĩa chính sách:</b> Mỗi 1 nghìn tỷ VND ngân sách tăng thêm
        làm GDP kỳ vọng tăng <b>{sp:.4f} nghìn tỷ</b>. Đây là <b>chi phí cơ hội của vốn công</b> ngành số -
        cận trên hợp lý để so sánh với suất sinh lời các dự án đầu tư khác.</div>""", unsafe_allow_html=True)

        # ── 2.4.3 Sensitivity ngân sách ──
        st.markdown('<div class="sec-title">📌 Câu 2.4.3 — Phân tích độ nhạy ngân sách Z*(B)</div>', unsafe_allow_html=True)
        budgets=range(100,161,10); zvals=[]
        for B in budgets:
            bu=list(b_ub); bu[0]=B
            r=linprog(c,A_ub=A_ub,b_ub=bu,bounds=[(0,None)]*4,method="highs")
            zvals.append(-r.fun if r.success else None)
        st.dataframe(pd.DataFrame({"Ngân sách B (ng.tỷ)":list(budgets),"Z* (ng.tỷ GDP)":[round(z,2) for z in zvals]}), use_container_width=True)
        fig,ax = plt.subplots(figsize=(8,4))
        ax.plot(list(budgets),zvals,"o-",color="#1976d2",lw=2.5,markersize=8)
        for B,z in zip(budgets,zvals): ax.annotate(f"{z:.0f}",(B,z),textcoords="offset points",xytext=(0,10),ha="center",fontsize=8)
        ax.fill_between(list(budgets),zvals,alpha=0.12,color="#1976d2")
        ax.set_title("Z*(B) — GDP kỳ vọng theo ngân sách",fontweight="bold"); ax.set_xlabel("Ngân sách (ng.tỷ VND)"); ax.set_ylabel("Z* (ng.tỷ GDP)")
        ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

        # ── 2.4.4 Ưu tiên nhân lực x3>=30 ──
        st.markdown('<div class="sec-title">📌 Câu 2.4.4 — Kịch bản ưu tiên nhân lực số (x₃ ≥ 30)</div>', unsafe_allow_html=True)
        A_ub2=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
        b_ub2=[100,-25,-15,-30,-10,0]
        res2=linprog(c,A_ub=A_ub2,b_ub=b_ub2,bounds=[(0,None)]*4,method="highs")
        if res2.success:
            x2=res2.x; Z2=-res2.fun
            cc1,cc2=st.columns(2)
            cc1.metric("Z* (x₃≥20 gốc)",f"{Z:.2f}")
            cc2.metric("Z* (x₃≥30 mới)",f"{Z2:.2f}",f"{Z2-Z:.2f}")
            st.dataframe(pd.DataFrame({"Hạng mục":labels,"x₃≥20 (gốc)":x.round(1),"x₃≥30 (mới)":x2.round(1)}), use_container_width=True)
            st.markdown(f"""<div class="note-box">💡 <b>Nhận xét:</b> Bài toán <b>vẫn khả thi</b> khi buộc x₃≥30.
            Tuy nhiên Z* giảm <b>{Z-Z2:.2f} nghìn tỷ</b> ({(Z-Z2)/Z*100:.1f}%) vì phải chuyển vốn từ R&D/AI
            (hệ số cao) sang nhân lực số. Đây là chi phí của ưu tiên giải quyết thiếu hụt kỹ sư AI.</div>""", unsafe_allow_html=True)
        else:
            st.error("❌ Bài toán KHÔNG khả thi với x₃≥30.")

        # ── 2.5 Thảo luận chính sách ──
        st.markdown('<div class="sec-title">💬 2.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-box">
        <b>a)</b> Khi ngân sách +1 tỷ VND → GDP kỳ vọng +{sp*1000:,.0f} tỷ. Đây là cận trên hợp lý của chi phí cơ hội vốn công.<br>
        <b>b)</b> R&D có hệ số cao nhất nhưng ràng buộc tối thiểu thấp nhất (10) vì năng lực hấp thụ R&D thực tế còn hạn chế, cần thời gian.<br>
        <b>c)</b> Tỷ lệ 35% công nghệ chiến lược (AI+R&D) là tham vọng khi ngân sách 2025 còn ưu tiên hạ tầng giao thông & an sinh.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH
# ══════════════════════════════════════════════════
elif menu == "📊 Bài 3 — Priority 10 ngành":
    st.markdown('<div class="page-header"><h1>Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ</h1><p>Min-max · Trọng số · Phân tích độ nhạy</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 3.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Cơ cấu kinh tế 2024: nông-lâm-thủy sản <b>11,86%</b>, công nghiệp-xây dựng <b>37,64%</b>, dịch vụ <b>42,36% GDP</b>.
    Việt Nam đối diện câu hỏi: trong các ngành lớn, ngành nào nên được <b>ưu tiên đẩy mạnh chuyển đổi số và AI trước</b>
    để tạo hiệu ứng lan tỏa tối đa? Cần xây dựng một chỉ số ưu tiên định lượng.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 3.2 — Mô hình toán học (Weighted Scoring + Min-max)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Chỉ số ưu tiên ngành i</h4>
    <div class="formula">Priority(i) = a1·Growth + a2·Productivity + a3·Spillover + a4·Export + a5·Employment + a6·AIReadiness − a7·Risk</div>
    <b>Chuẩn hóa min-max về thang [0, 1]:</b>
    <div class="formula">x̃ = (x − min) / (max − min)</div>
    <b>Với chỉ số "xấu" như Risk (rủi ro tự động hóa) - đảo dấu:</b>
    <div class="formula">x̃ = (max − x) / (max − min)</div>
    <b>Bộ trọng số mặc định:</b> a1=0,15 · a2=0,15 · a3=0,20 · a4=0,15 · a5=0,10 · a6=0,20 · a7=0,15
    </div>""", unsafe_allow_html=True)

    df=df_sectors.copy(); df["labor_productivity"]=(df["gdp_share_2024_pct"]/100)*11511.9/df["labor_million"]
    def norm_good(x): return (x-x.min())/(x.max()-x.min())
    def norm_bad(x): return (x.max()-x)/(x.max()-x.min())
    cols_good=["growth_rate_2024_pct","labor_productivity","spillover_coef_0_1","export_billion_USD","labor_million","ai_readiness_0_100"]
    Xg=df[cols_good].apply(norm_good); Xb=norm_bad(df["automation_risk_pct"])
    w=np.array([0.15,0.15,0.20,0.15,0.10,0.20]); w_r=0.15
    p=Xg.values@w - w_r*Xb.values; df["Priority"]=p

    st.markdown('<div class="sec-title">🔬 3.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)

    # ── 3.4.1 Ma trận chuẩn hóa ──
    st.markdown('<div class="sec-title">📌 Câu 3.4.1 — Ma trận chuẩn hóa min-max</div>', unsafe_allow_html=True)
    norm_show=Xg.copy()
    norm_show.columns=["Tăng trưởng","Năng suất LĐ","Lan tỏa","Xuất khẩu","Việc làm","AI Ready"]
    norm_show.insert(0,"Ngành",df["sector_name_vi"].values)
    norm_show["Risk (đảo)"]=Xb.values
    st.dataframe(norm_show.round(4), use_container_width=True, hide_index=True)
    fig,ax=plt.subplots(figsize=(11,5))
    im=ax.imshow(np.column_stack([Xg.values,Xb.values]),cmap="RdYlGn",aspect="auto")
    ax.set_yticks(range(10)); ax.set_yticklabels(df["sector_name_vi"],fontsize=8)
    ax.set_xticks(range(7)); ax.set_xticklabels(["Tăng trg","Năng suất","Lan tỏa","XK","Việc làm","AI","Risk↓"],fontsize=8,rotation=20,ha="right")
    ax.set_title("Heatmap ma trận chuẩn hóa 10 ngành",fontweight="bold"); plt.colorbar(im,ax=ax)
    fig.tight_layout(); show_fig(fig)

    # ── 3.4.2 Xếp hạng ──
    ds=df[["sector_name_vi","Priority"]].sort_values("Priority",ascending=False).reset_index(drop=True)
    ds.index+=1
    st.markdown('<div class="sec-title">📌 Câu 3.4.2 — Xếp hạng Priority (trọng số mặc định)</div>', unsafe_allow_html=True)
    st.dataframe(ds, use_container_width=True)
    fig,ax = plt.subplots(figsize=(10,5))
    colors3=["#1976d2" if i<3 else "#90caf9" for i in range(len(ds))]
    ax.barh(ds["sector_name_vi"][::-1],ds["Priority"][::-1],color=colors3[::-1],edgecolor="white")
    for i,(name,v) in enumerate(zip(ds["sector_name_vi"][::-1],ds["Priority"][::-1])):
        ax.text(v,i,f" {v:.3f}",va="center",fontsize=8)
    ax.set_title("Chỉ số ưu tiên ngành (top-3 đậm)",fontweight="bold"); ax.grid(axis="x",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)
    top3_def=ds["sector_name_vi"].head(3).tolist()
    st.markdown(f"""<div class="note-box">💡 <b>Top-3 ưu tiên:</b> {' · '.join(top3_def)} -
    phù hợp Nghị quyết 57-NQ/TW về đột phá KH-CN và chuyển đổi số.</div>""", unsafe_allow_html=True)

    # ── 3.4.3 Sensitivity AI readiness ──
    st.markdown('<div class="sec-title">📌 Câu 3.4.3 — Độ nhạy trọng số AI Readiness</div>', unsafe_allow_html=True)
    rows=[]
    for a6 in np.arange(0.05,0.41,0.05):
        wn=np.array([0.15,0.15,0.20,0.15,0.10,a6]); r=1-w_r; wn=wn/wn.sum()*r
        pn=Xg.values@wn - w_r*Xb.values
        top3=df["sector_name_vi"].iloc[np.argsort(pn)[::-1][:3]].tolist()
        rows.append({"a₆":round(a6,2),"#1":top3[0],"#2":top3[1],"#3":top3[2]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("""<div class="note-box">💡 <b>Nhận xét:</b> Ngành CNTT-TT và Tài chính-NH duy trì top-3
    bất kể trọng số AI readiness - cho thấy thứ hạng <b>bền vững</b>, không quá nhạy với chủ quan trọng số.</div>""", unsafe_allow_html=True)

    # ── 3.4.4 Hai bộ trọng số ──
    st.markdown('<div class="sec-title">📌 Câu 3.4.4 — So sánh hai định hướng trọng số</div>', unsafe_allow_html=True)
    wg=np.array([0.25,0.25,0.10,0.25,0.05,0.05]); wg_r=0.05  # Tăng trưởng
    wi=np.array([0.05,0.10,0.25,0.05,0.25,0.10]); wi_r=0.20  # Bao trùm
    pg=Xg.values@wg - wg_r*Xb.values
    pi=Xg.values@wi - wi_r*Xb.values
    cmp=pd.DataFrame({"Ngành":df["sector_name_vi"],"Priority Tăng trưởng":pg.round(4),"Priority Bao trùm":pi.round(4)})
    cc1,cc2=st.columns(2)
    with cc1:
        st.markdown("**🏆 Định hướng Tăng trưởng** (ưu tiên tăng trưởng, năng suất, XK)")
        st.dataframe(cmp.nlargest(3,"Priority Tăng trưởng")[["Ngành","Priority Tăng trưởng"]], hide_index=True, use_container_width=True)
    with cc2:
        st.markdown("**🤝 Định hướng Bao trùm** (ưu tiên việc làm, lan tỏa, giảm rủi ro)")
        st.dataframe(cmp.nlargest(3,"Priority Bao trùm")[["Ngành","Priority Bao trùm"]], hide_index=True, use_container_width=True)
    fig,ax=plt.subplots(figsize=(11,5)); y=np.arange(10); h=0.38
    ax.barh(y-h/2,cmp["Priority Tăng trưởng"],h,label="Tăng trưởng",color="#42a5f5")
    ax.barh(y+h/2,cmp["Priority Bao trùm"],h,label="Bao trùm",color="#ef5350")
    ax.set_yticks(y); ax.set_yticklabels(df["sector_name_vi"],fontsize=8); ax.legend()
    ax.set_title("So sánh hai bộ trọng số chính sách",fontweight="bold"); ax.grid(axis="x",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)

    # ── 3.5 Thảo luận ──
    st.markdown('<div class="sec-title">💬 3.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <b>a)</b> Ba ngành ưu tiên đẩy mạnh CĐS/AI trước: {' · '.join(top3_def)} - phù hợp Nghị quyết 57-NQ/TW.<br>
    <b>b)</b> Khai khoáng có năng suất cao nhưng không ưu tiên vì rủi ro tự động hóa lớn (55%), lan tỏa thấp, tăng trưởng âm.<br>
    <b>c)</b> Bộ trọng số nên do <b>hội đồng chính sách + đối thoại công khai</b> quyết định để bảo đảm tính chính danh, không chỉ kỹ thuật.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 4 — LP VÙNG MIỀN
# ══════════════════════════════════════════════════
elif menu == "🗺️ Bài 4 — LP ngành-vùng":
    st.markdown('<div class="page-header"><h1>Bài 4 — LP phân bổ ngân sách theo vùng miền</h1><p>24 biến · Công bằng vùng · PuLP CBC</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 4.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo <b>Quyết định 411/QĐ-TTg</b> phê duyệt Chiến lược quốc gia phát triển kinh tế số,
    các vùng kinh tế xã hội Việt Nam có mức độ sẵn sàng số rất khác nhau. Bài toán: phân bổ
    <b>50.000 tỷ VND</b> cho <b>6 vùng × 4 hạng mục</b> (I - hạ tầng số, D - chuyển đổi số DN,
    AI - năng lực AI, H - nhân lực số) sao cho tối đa hóa GDP gain nhưng bảo đảm <b>công bằng vùng miền</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 4.2 — Mô hình toán học (LP 24 biến)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Biến quyết định</h4>
    x(j,r) ≥ 0 với j ∈ {I, D, AI, H}, r ∈ {1..6} → tổng <b>24 biến</b>
    <h4>Hàm mục tiêu (GDP gain kỳ vọng, tỷ VND)</h4>
    <div class="formula">max Z = Σ(r) Σ(j) β(j,r) · x(j,r)</div>
    <b>Các nhóm ràng buộc:</b>
    <ul>
      <li>(C1) Ngân sách tổng: Σ x(j,r) ≤ 50.000 tỷ</li>
      <li>(C2) Sàn mỗi vùng: Σ(j) x(j,r) ≥ 5.000 ∀r</li>
      <li>(C3) Trần mỗi vùng: Σ(j) x(j,r) ≤ 12.000 ∀r</li>
      <li>(C4) Sàn nhân lực số: Σ(r) x(H,r) ≥ 12.000 (24% ngân sách)</li>
      <li>(C5) Công bằng vùng: D(r) + γ·x(D,r) ≥ λ·max(D(r)+γ·x(D,r)), với γ=0,002, λ=0,6</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    try:
        import pulp
        regions=['NMM','RRD','NCC','CH','SE','MD']
        rnames=['Trung du MN','ĐB Hồng','Bắc Trung Bộ','Tây Nguyên','Đông Nam Bộ','ĐB Cửu Long']
        items=['I','D','AI','H']; inames=['Hạ tầng','CĐS','AI','Nhân lực']
        beta={'NMM':{'I':1.15,'D':0.85,'AI':0.55,'H':1.30},'RRD':{'I':0.95,'D':1.25,'AI':1.40,'H':1.05},
              'NCC':{'I':1.05,'D':0.95,'AI':0.85,'H':1.15},'CH':{'I':1.20,'D':0.75,'AI':0.45,'H':1.35},
              'SE':{'I':0.90,'D':1.30,'AI':1.55,'H':1.00},'MD':{'I':1.10,'D':0.85,'AI':0.65,'H':1.25}}
        D0={'NMM':38,'RRD':78,'NCC':55,'CH':32,'SE':82,'MD':48}
        gv,lm=0.002,0.6
        st.markdown('<div class="sec-title">📋 4.3 — Bảng hệ số tác động biên β(j,r)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(
            {inames[k]:[beta[r][items[k]] for r in regions] for k in range(4)},
            index=rnames).assign(**{"D0 (số hóa ban đầu)":[D0[r] for r in regions]}),
            use_container_width=True)
        st.markdown('<div class="sec-title">🔬 4.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
        m=pulp.LpProblem('B4',pulp.LpMaximize)
        x=pulp.LpVariable.dicts('x',(regions,items),lowBound=0)
        m+=pulp.lpSum(beta[r][j]*x[r][j] for r in regions for j in items)
        m+=pulp.lpSum(x[r][j] for r in regions for j in items)<=50000
        for r in regions:
            m+=pulp.lpSum(x[r][j] for j in items)>=5000
            m+=pulp.lpSum(x[r][j] for j in items)<=12000
        m+=pulp.lpSum(x[r]['H'] for r in regions)>=12000
        M=pulp.LpVariable('Dm',lowBound=0)
        for r in regions: m+=D0[r]+gv*x[r]['D']<=M
        for r in regions: m+=D0[r]+gv*x[r]['D']>=lm*M
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[m.status]=="Optimal":
            alloc=np.array([[x[r][j].value() for j in items] for r in regions])
            Z_eq=pulp.value(m.objective)
            # ── 4.4.1 PuLP CBC ──
            st.markdown('<div class="sec-title">📌 Câu 4.4.1 — Giải bằng PuLP (CBC), phân bổ tối ưu</div>', unsafe_allow_html=True)
            st.success(f"**Z* = {Z_eq:,.0f} tỷ VND GDP gain** (có ràng buộc công bằng C5)")
            dfm=pd.DataFrame(alloc.round(0),index=rnames,columns=inames); dfm["Tổng"]=dfm.sum(axis=1)
            st.dataframe(dfm, use_container_width=True)
            # ── 4.4.3 Heatmap ──
            st.markdown('<div class="sec-title">📌 Câu 4.4.3 — Heatmap phân bổ & ngân sách mỗi vùng</div>', unsafe_allow_html=True)
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
            im=ax1.imshow(alloc,cmap="Blues",aspect="auto")
            ax1.set_yticks(range(6)); ax1.set_yticklabels(rnames)
            ax1.set_xticks(range(4)); ax1.set_xticklabels(inames)
            ax1.set_title("Heatmap phân bổ tối ưu",fontweight="bold"); plt.colorbar(im,ax=ax1)
            for i in range(6):
                for j in range(4): ax1.text(j,i,f"{alloc[i,j]:.0f}",ha="center",va="center",fontsize=8,color="white" if alloc[i,j]>alloc.max()*0.6 else "black")
            ax2.bar(rnames,alloc.sum(axis=1),color="#42a5f5",edgecolor="white")
            ax2.axhline(5000,color="red",ls="--",label="Sàn"); ax2.axhline(12000,color="orange",ls="--",label="Trần")
            ax2.set_title("Tổng ngân sách mỗi vùng",fontweight="bold"); ax2.set_xticklabels(rnames,rotation=20,ha="right"); ax2.legend()
            for ax in [ax1,ax2]: ax.grid(alpha=0.3,axis="y" if ax==ax2 else "x"); ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); show_fig(fig)

            # ── 4.4.2 CVXPY đối chiếu ──
            st.markdown('<div class="sec-title">📌 Câu 4.4.2 — Đối chiếu PuLP với solver thứ hai</div>', unsafe_allow_html=True)
            Z_v2=None; solver_name=""
            # Ưu tiên CVXPY; nếu không có thì fallback sang scipy.linprog (luôn sẵn)
            try:
                import cvxpy as cp
                bm=np.array([[beta[r][j] for j in items] for r in regions])
                xv=cp.Variable((6,4),nonneg=True); rs=cp.sum(xv,axis=1)
                D0v=np.array([D0[r] for r in regions]); Dn=D0v+gv*xv[:,1]; Mc=cp.Variable()
                cons=[cp.sum(xv)<=50000,rs>=5000,rs<=12000,cp.sum(xv[:,3])>=12000,Dn<=Mc,Dn>=lm*Mc]
                prob=cp.Problem(cp.Maximize(cp.sum(cp.multiply(bm,xv))),cons)
                for sv in ["CLARABEL","ECOS","SCS"]:
                    try:
                        prob.solve(solver=getattr(cp,sv))
                        if prob.status and "optimal" in prob.status: break
                    except: continue
                Z_v2=prob.value; solver_name="CVXPY"
            except Exception:
                # Fallback: giải lại bằng scipy.linprog (24 biến x + 1 biến M)
                from scipy.optimize import linprog as _lp
                # biến: [x(0..23), M]; thứ tự x theo (region r, item j): idx = r*4 + j
                nvar=25
                cobj=np.zeros(nvar)
                for r in range(6):
                    for j in range(4): cobj[r*4+j]=-beta[regions[r]][items[j]]
                A_ub=[]; b_ub=[]
                # tổng <= 50000
                row=[0]*nvar
                for k in range(24): row[k]=1
                A_ub.append(row); b_ub.append(50000)
                # mỗi vùng: >=5000 (=> -sum<=-5000) và <=12000
                for r in range(6):
                    row=[0]*nvar
                    for j in range(4): row[r*4+j]=-1
                    A_ub.append(row); b_ub.append(-5000)
                    row=[0]*nvar
                    for j in range(4): row[r*4+j]=1
                    A_ub.append(row); b_ub.append(12000)
                # tổng H >= 12000
                row=[0]*nvar
                for r in range(6): row[r*4+1+2]=1   # j=2 là 'AI'? -> H là j=3
                # sửa: H là items index 3
                row=[0]*nvar
                for r in range(6): row[r*4+3]=-1
                A_ub.append(row); b_ub.append(-12000)
                # công bằng C5: D0+gv*x_D <= M  và  D0+gv*x_D >= lm*M
                # x_D là j=1
                for r in range(6):
                    row=[0]*nvar; row[r*4+1]=gv; row[24]=-1
                    A_ub.append(row); b_ub.append(-D0[regions[r]])           # gv*xD - M <= -D0
                    row=[0]*nvar; row[r*4+1]=-gv; row[24]=lm
                    A_ub.append(row); b_ub.append(D0[regions[r]])            # -gv*xD + lm*M <= D0
                bounds=[(0,None)]*24+[(0,None)]
                rsp=_lp(cobj,A_ub=np.array(A_ub),b_ub=np.array(b_ub),bounds=bounds,method="highs")
                if rsp.success:
                    Z_v2=-rsp.fun; solver_name="SciPy (HiGHS)"
            if Z_v2 is not None:
                cc1,cc2=st.columns(2)
                cc1.metric("Z* PuLP (CBC)",f"{Z_eq:,.0f} tỷ")
                cc2.metric(f"Z* {solver_name}",f"{Z_v2:,.0f} tỷ")
                if abs(Z_eq-Z_v2)<50:
                    st.success(f"✅ PuLP và {solver_name} cho kết quả TRÙNG KHỚP (chênh {abs(Z_eq-Z_v2):.1f} tỷ < 50) — xác nhận tính đúng đắn của lời giải LP.")
                else:
                    st.info(f"Hai solver chênh {abs(Z_eq-Z_v2):.1f} tỷ (do dung sai số học của thuật toán điểm trong khác simplex).")
            else:
                st.info("Không giải được bằng solver thứ hai. Kết quả PuLP đã có ở trên.")

            # ── 4.4.4 Chi phí công bằng (bỏ C5) ──
            st.markdown('<div class="sec-title">📌 Câu 4.4.4 — Chi phí của công bằng vùng miền</div>', unsafe_allow_html=True)
            m2=pulp.LpProblem('B4_noC5',pulp.LpMaximize)
            x2=pulp.LpVariable.dicts('x2',(regions,items),lowBound=0)
            m2+=pulp.lpSum(beta[r][j]*x2[r][j] for r in regions for j in items)
            m2+=pulp.lpSum(x2[r][j] for r in regions for j in items)<=50000
            for r in regions:
                m2+=pulp.lpSum(x2[r][j] for j in items)>=5000
                m2+=pulp.lpSum(x2[r][j] for j in items)<=12000
            m2+=pulp.lpSum(x2[r]['H'] for r in regions)>=12000
            m2.solve(pulp.PULP_CBC_CMD(msg=False))
            alloc2=np.array([[x2[r][j].value() for j in items] for r in regions])
            Z_noeq=pulp.value(m2.objective); cost_eq=Z_noeq-Z_eq
            cc1,cc2,cc3=st.columns(3)
            cc1.metric("Z* có C5",f"{Z_eq:,.0f} tỷ")
            cc2.metric("Z* bỏ C5",f"{Z_noeq:,.0f} tỷ")
            cc3.metric("Chi phí công bằng",f"{cost_eq:,.0f} tỷ",f"{cost_eq/Z_eq*100:.1f}%",delta_color="inverse")
            cmp4=pd.DataFrame({"Vùng":rnames,"Có C5 (công bằng)":alloc.sum(axis=1).round(0),"Bỏ C5":alloc2.sum(axis=1).round(0)})
            fig,ax=plt.subplots(figsize=(11,4)); xpos=np.arange(6); ww=0.38
            ax.bar(xpos-ww/2,cmp4["Có C5 (công bằng)"],ww,label="Có công bằng C5",color="#42a5f5")
            ax.bar(xpos+ww/2,cmp4["Bỏ C5"],ww,label="Bỏ công bằng",color="#ef5350")
            ax.set_xticks(xpos); ax.set_xticklabels(rnames,rotation=20,ha="right",fontsize=8); ax.legend()
            ax.set_title("Ngân sách mỗi vùng: có vs không có ràng buộc công bằng",fontweight="bold"); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); show_fig(fig)
            st.markdown(f"""<div class="note-box">💡 <b>Chi phí công bằng = {cost_eq:,.0f} tỷ VND GDP gain</b> ({cost_eq/Z_eq*100:.1f}% của Z*).
            Bỏ C5, vốn dồn về ĐB sông Hồng & Đông Nam Bộ (β_AI cao nhất). C5 bảo đảm Tây Nguyên và Trung du MN
            được nâng cấp số hóa tối thiểu - đánh đổi hiệu quả lấy công bằng.</div>""", unsafe_allow_html=True)

            # ── 4.5 Thảo luận ──
            st.markdown('<div class="sec-title">💬 4.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="info-box">
            <b>a)</b> Bỏ công bằng, vốn chảy về ĐB sông Hồng & Đông Nam Bộ (sẵn sàng số cao) - hậu quả dài hạn: gia tăng chênh lệch vùng, "bẫy ngoại vi số".<br>
            <b>b)</b> Trần ngân sách mỗi vùng (C3) là "chính sách phân quyền", làm giảm Z* nhưng chống tập trung quá mức.<br>
            <b>c)</b> Tây Nguyên có hệ số AI thấp (0,45) - mô hình ưu tiên đầu tư H (nhân lực) và I (hạ tầng) trước, tạo nền tảng hấp thụ AI sau.
            </div>""", unsafe_allow_html=True)
    except ImportError: st.error("Cần cài: `pip install pulp`")
    except Exception as e: st.error(f"Lỗi: {e}")

# ══════════════════════════════════════════════════
# BÀI 5 — MIP
# ══════════════════════════════════════════════════
elif menu == "🎯 Bài 5 — MIP 15 dự án":
    st.markdown('<div class="page-header"><h1>Bài 5 — MIP lựa chọn dự án chuyển đổi số</h1><p>Biến nhị phân · Ràng buộc tiên quyết · PuLP CBC</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 5.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Bộ KH-CN (sau hợp nhất 2025) xem xét <b>15 dự án ứng cử</b> cho chương trình chuyển đổi số quốc gia 2026-2030.
    Tổng ngân sách <b>80.000 tỷ VND</b> (năm 1-2 tối đa 40.000 tỷ). Mỗi dự án có chi phí, lợi ích NPV và
    ràng buộc đặc thù (loại trừ, tiên quyết). Cần xây dựng mô hình MIP để chọn tập dự án tối ưu.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 5.2 — Mô hình toán học (MIP nhị phân)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Biến quyết định</h4>
    y(i) ∈ {0, 1}, i = 1..15 → y(i)=1 nếu chọn dự án i
    <h4>Hàm mục tiêu</h4>
    <div class="formula">max Σ B(i) · y(i)</div>
    <b>Ràng buộc:</b>
    <ul>
      <li>(C1) Ngân sách 5 năm: Σ C(i)·y(i) ≤ 80.000</li>
      <li>(C2) Ngân sách năm 1-2: Σ C1(i)·y(i) ≤ 40.000</li>
      <li>(C3) Loại trừ trung tâm dữ liệu: y1 + y2 ≤ 1</li>
      <li>(C4) Tiên quyết AI cần đào tạo: y8 ≤ y12; (C5) Bán dẫn cần đào tạo: y13 ≤ y12</li>
      <li>(C6) Cân đối: y4 + y5 ≥ 1 (chính phủ số); y14 ≥ 1 (an ninh mạng bắt buộc)</li>
      <li>(C7) Số lượng: 7 ≤ Σ y(i) ≤ 11</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    try:
        from pulp import *
        P=list(range(1,16))
        C={1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
        C1={1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
        B={1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
        names={1:'TT DL Hòa Lạc',2:'TT DL phía Nam',3:'5G toàn quốc',4:'VNeID 2.0',5:'Cổng DVCQG v3',
               6:'Y tế số',7:'GD K-12',8:'TT AI+Super',9:'Fintech sandbox',10:'Logistics TM',
               11:'NN số ĐBSCL',12:'ĐT 50K KS AI',13:'KCN bán dẫn',14:'ANMQG SOC',15:'Open Data'}
        fields={1:'Hạ tầng',2:'Hạ tầng',3:'Hạ tầng',4:'CP số',5:'CP số',6:'Y tế',7:'Giáo dục',8:'AI',
                9:'Tài chính',10:'Logistics',11:'Nông nghiệp',12:'Nhân lực',13:'Bán dẫn',14:'An ninh',15:'Dữ liệu'}
        prob_p={'Hạ tầng':0.85,'CP số':0.75,'AI':0.65,'Bán dẫn':0.65,'Y tế':0.8,'Giáo dục':0.8,
                'Tài chính':0.8,'Logistics':0.8,'Nông nghiệp':0.8,'Nhân lực':0.8,'An ninh':0.8,'Dữ liệu':0.8}

        # ── 5.3 Bảng danh mục 15 dự án ──
        st.markdown('<div class="sec-title">📋 5.3 — Danh mục 15 dự án ứng cử</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Mã":[f"P{i}" for i in P],"Tên dự án":[names[i] for i in P],"Lĩnh vực":[fields[i] for i in P],
            "Chi phí (tỷ)":[C[i] for i in P],"Năm 1-2 (tỷ)":[C1[i] for i in P],
            "NPV (tỷ)":[B[i] for i in P],"B/C":[round(B[i]/C[i],2) for i in P]
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-title">🔬 5.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
        def solve_mip(bt=80000,b12=40000,use_exp=False,force12=False):
            md=LpProblem('MIP5',LpMaximize); y=LpVariable.dicts('y',P,cat='Binary')
            md+=lpSum((prob_p[fields[i]] if use_exp else 1)*B[i]*y[i] for i in P)
            md+=lpSum(C[i]*y[i] for i in P)<=bt; md+=lpSum(C1[i]*y[i] for i in P)<=b12
            if force12: md+=y[1]>=1; md+=y[2]>=1
            else: md+=y[1]+y[2]<=1
            md+=y[8]<=y[12]; md+=y[13]<=y[12]; md+=y[4]+y[5]>=1; md+=y[14]>=1
            md+=lpSum(y[i] for i in P)>=7; md+=lpSum(y[i] for i in P)<=11
            md.solve(PULP_CBC_CMD(msg=False))
            sel=[i for i in P if y[i].value() and y[i].value()>0.5]
            return sel,value(md.objective),LpStatus[md.status]

        # ── 5.4.1 Cơ sở ──
        st.markdown('<div class="sec-title">📌 Câu 5.4.1 — Giải MIP (CBC), tập dự án tối ưu</div>', unsafe_allow_html=True)
        sel,Z5,_=solve_mip()
        tc=sum(C[i] for i in sel)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Dự án chọn",f"{len(sel)}/15"); c2.metric("Tổng chi phí",f"{tc:,} tỷ")
        c3.metric("Tổng NPV Z*",f"{Z5:,.0f} tỷ"); c4.metric("B/C trung bình",f"{Z5/tc:.2f}×")
        st.dataframe(pd.DataFrame({"Dự án":[names[i] for i in sel],"Lĩnh vực":[fields[i] for i in sel],
                     "Chi phí":[C[i] for i in sel],"NPV":[B[i] for i in sel],"B/C":[round(B[i]/C[i],2) for i in sel]}), use_container_width=True, hide_index=True)
        fig,ax=plt.subplots(figsize=(10,5))
        ax.barh([names[i] for i in sel],[B[i]/1000 for i in sel],color="#42a5f5",label="NPV")
        ax.barh([names[i] for i in sel],[-C[i]/1000 for i in sel],color="#ef5350",label="Chi phí")
        ax.axvline(0,color="gray",lw=0.8); ax.set_title("Lợi ích & Chi phí dự án được chọn (nghìn tỷ VND)",fontweight="bold"); ax.legend(); ax.grid(axis="x",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

        # Scatter tất cả 15
        fig,ax=plt.subplots(figsize=(9,5))
        for i in P:
            col="#2e7d32" if i in sel else "#bdbdbd"
            ax.scatter(C[i]/1000,B[i]/1000,s=120,color=col,edgecolor="white",zorder=3)
            ax.annotate(f"P{i}",(C[i]/1000,B[i]/1000),fontsize=8,ha="center",va="center",color="white",fontweight="bold")
        ax.set_xlabel("Chi phí (nghìn tỷ)"); ax.set_ylabel("NPV (nghìn tỷ)")
        ax.set_title("Ma trận Chi phí–NPV 15 dự án (xanh = được chọn)",fontweight="bold"); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

        # ── 5.4.2 Nới ngân sách ──
        st.markdown('<div class="sec-title">📌 Câu 5.4.2 — Nới ngân sách lên 100.000 tỷ</div>', unsafe_allow_html=True)
        sel2,Z5_2,_=solve_mip(bt=100000)
        added=set(sel2)-set(sel); removed=set(sel)-set(sel2)
        c1,c2=st.columns(2)
        c1.metric("Z* (80.000 tỷ)",f"{Z5:,.0f}"); c2.metric("Z* (100.000 tỷ)",f"{Z5_2:,.0f}",f"+{Z5_2-Z5:,.0f}")
        if added: st.info("🆕 Thêm: "+", ".join(f"P{i} ({names[i]})" for i in sorted(added)))
        if removed: st.warning("➖ Bỏ: "+", ".join(f"P{i} ({names[i]})" for i in sorted(removed)))

        # ── 5.4.3 Bắt buộc P1+P2 ──
        st.markdown('<div class="sec-title">📌 Câu 5.4.3 — Bắt buộc cả P1 và P2 (redundancy)</div>', unsafe_allow_html=True)
        sel3,Z5_3,stt3=solve_mip(force12=True)
        if stt3=="Optimal":
            st.success(f"✅ Vẫn khả thi · Z* = {Z5_3:,.0f} tỷ (giảm {Z5-Z5_3:,.0f} = {(Z5-Z5_3)/Z5*100:.1f}% do phải chọn cả 2 trung tâm dữ liệu)")
        else:
            st.error("❌ KHÔNG khả thi khi bắt buộc cả P1 và P2.")

        # ── 5.4.4 Lợi ích kỳ vọng có rủi ro ──
        st.markdown('<div class="sec-title">📌 Câu 5.4.4 — Lợi ích kỳ vọng E[Z] với rủi ro tiến độ</div>', unsafe_allow_html=True)
        sel4,Z5_4,_=solve_mip(use_exp=True)
        c1,c2=st.columns(2)
        c1.metric("Z* Deterministic",f"{Z5:,.0f}"); c2.metric("E[Z] có rủi ro",f"{Z5_4:,.0f}")
        st.dataframe(pd.DataFrame({"Dự án":[names[i] for i in P],"Lĩnh vực":[fields[i] for i in P],
                     "p(đúng tiến độ)":[prob_p[fields[i]] for i in P],"B×p":[round(B[i]*prob_p[fields[i]],0) for i in P]}).sort_values("B×p",ascending=False), use_container_width=True, hide_index=True)
        st.markdown("""<div class="note-box">💡 <b>Nhận xét:</b> Dự án AI & Bán dẫn có xác suất hoàn thành đúng tiến độ
        thấp nhất (0,65) - rủi ro cao nhất. Khi tính lợi ích kỳ vọng, cần tăng năng lực quản trị dự án và đối tác quốc tế.</div>""", unsafe_allow_html=True)

        # ── 5.5 Thảo luận ──
        st.markdown('<div class="sec-title">💬 5.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        <b>a)</b> P15 (Open Data) có B/C rất cao nhưng có thể bị bỏ do ràng buộc số lượng ≤11 và ưu tiên dự án NPV tuyệt đối lớn.<br>
        <b>b)</b> Bắt buộc P14 (an ninh mạng) làm giảm Z* nhẹ nhưng hợp lý - an ninh số là điều kiện nền tảng, không thể đánh đổi.<br>
        <b>c)</b> Hiệu ứng cộng hưởng P8 (AI) + P13 (bán dẫn) có thể mô hình hóa bằng biến tích y8·y13 (tuyến tính hóa qua biến phụ z ≤ y8, z ≤ y13).
        </div>""", unsafe_allow_html=True)
    except ImportError: st.error("Cần cài: `pip install pulp`")
    except Exception as e: st.error(f"Lỗi: {e}")

# ══════════════════════════════════════════════════
# BÀI 6 — TOPSIS
# ══════════════════════════════════════════════════
elif menu == "🏆 Bài 6 — TOPSIS 6 vùng":
    st.markdown('<div class="page-header"><h1>Bài 6 — TOPSIS xếp hạng 6 vùng KT-XH</h1><p>Chuẩn hoá vector · Ideal solution · Entropy weight</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 6.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo <b>Quyết định 127/QĐ-TTg</b> về Chiến lược quốc gia về AI đến 2030, Việt Nam đặt mục tiêu
    trở thành <b>trung tâm AI của ASEAN</b>. Ngân sách có hạn nên cần lựa chọn vùng nào triển khai
    trung tâm AI và sandbox dữ liệu trước. Bài toán áp dụng <b>TOPSIS</b> để xếp hạng 6 vùng theo mức độ sẵn sàng AI.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 6.2 — Mô hình toán học (TOPSIS)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Quy trình 5 bước TOPSIS</h4>
    <b>B1.</b> Chuẩn hóa vector:
    <div class="formula">r(ij) = x(ij) / sqrt(Σ x(ij)²)</div>
    <b>B2.</b> Ma trận có trọng số: v(ij) = w(j) · r(ij)
    <b>B3.</b> Lời giải lý tưởng dương A* (max lợi ích/min chi phí) và âm A⁻ (ngược lại)
    <b>B4.</b> Khoảng cách Euclid:
    <div class="formula">S*(i) = sqrt(Σ(v(ij) − v*(j))²) ; S⁻(i) = sqrt(Σ(v(ij) − v⁻(j))²)</div>
    <b>B5.</b> Hệ số gần gũi tương đối (càng lớn càng tốt):
    <div class="formula">C*(i) = S⁻(i) / (S*(i) + S⁻(i)), 0 ≤ C* ≤ 1</div>
    <b>Tiêu chí lợi ích:</b> GRDP/người, FDI, Digital Index, AI Readiness, LĐ đào tạo, R&D, Internet.
    <b>Tiêu chí chi phí:</b> Gini (càng thấp càng tốt).
    </div>""", unsafe_allow_html=True)

    crit=['grdp_per_capita_million_VND','fdi_registered_billion_USD','digital_index_0_100','ai_readiness_0_100','trained_labor_pct','rd_intensity_pct','internet_penetration_pct','gini_coef']
    lbls=['GRDP/N','FDI','Digital','AI','LĐQĐ','R&D','Internet','Gini']; ib=[True,True,True,True,True,True,True,False]
    X=df_regions[crit].values.astype(float)
    def topsis_fn(X,w,ib):
        R=X/np.sqrt((X**2).sum(axis=0)); V=R*w
        As=np.where(ib,V.max(axis=0),V.min(axis=0)); An=np.where(ib,V.min(axis=0),V.max(axis=0))
        Ss=np.sqrt(((V-As)**2).sum(axis=1)); Sn=np.sqrt(((V-An)**2).sum(axis=1))
        return Sn/(Ss+Sn)
    def entropy_w(X):
        Xp=X/X.sum(axis=0); k=1/np.log(len(X)); E=-k*np.nansum(Xp*np.log(Xp+1e-12),axis=0); d=1-E; return d/d.sum()

    # ── 6.3 Bảng dữ liệu 6 vùng ──
    st.markdown('<div class="sec-title">📋 6.3 — Dữ liệu 6 vùng kinh tế xã hội</div>', unsafe_allow_html=True)
    df6_show=df_regions[["region_name_vi"]+crit].copy()
    df6_show.columns=["Vùng","GRDP/N (tr)","FDI (tỷ$)","Digital","AI Ready","LĐ ĐT %","R&D %","Internet %","Gini"]
    st.dataframe(df6_show, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">🔬 6.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
    we=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    wen=entropy_w(X)
    Ce=topsis_fn(X,we,ib); Cen=topsis_fn(X,wen,ib)

    # ── 6.4.1 Expert ──
    st.markdown('<div class="sec-title">📌 Câu 6.4.1 — TOPSIS với trọng số chuyên gia</div>', unsafe_allow_html=True)
    df_exp=pd.DataFrame({"Hạng":Ce.argsort()[::-1].argsort()+1,"Vùng":df_regions["region_name_vi"],"C* Score":Ce.round(4)}).sort_values("C* Score",ascending=False)
    st.dataframe(df_exp, use_container_width=True, hide_index=True)
    fig,ax=plt.subplots(figsize=(10,4)); o1=np.argsort(Ce)[::-1]
    ax.barh([df_regions["region_name_vi"].iloc[i] for i in o1[::-1]],[Ce[i] for i in o1[::-1]],color=["#1976d2" if j==len(o1)-1 else "#90caf9" for j in range(6)],edgecolor="white")
    for j,i in enumerate(o1[::-1]): ax.text(Ce[i],j,f" {Ce[i]:.3f}",va="center",fontsize=8)
    ax.set_title("TOPSIS Score — Trọng số chuyên gia (w_AI=0,20)",fontweight="bold"); ax.grid(axis="x",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)
    best6=df_regions["region_name_vi"].iloc[int(np.argmax(Ce))]
    st.markdown(f"""<div class="note-box">💡 <b>Kết quả:</b> <b>{best6}</b> dẫn đầu - phù hợp để ưu tiên đặt
    Trung tâm AI quốc gia đầu tiên theo Quyết định 127/QĐ-TTg.</div>""", unsafe_allow_html=True)

    # ── 6.4.2 Entropy ──
    st.markdown('<div class="sec-title">📌 Câu 6.4.2 — Trọng số khách quan Entropy</div>', unsafe_allow_html=True)
    df_cmp=pd.DataFrame({"Vùng":df_regions["region_name_vi"],"C* Expert":Ce.round(4),
                         "Rank Expert":Ce.argsort()[::-1].argsort()+1,"C* Entropy":Cen.round(4),
                         "Rank Entropy":Cen.argsort()[::-1].argsort()+1})
    st.dataframe(df_cmp.sort_values("C* Expert",ascending=False), use_container_width=True, hide_index=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    o1=np.argsort(Ce)[::-1]; o2=np.argsort(Cen)[::-1]
    ax1.barh([df_regions["region_name_vi"].iloc[i] for i in o1[::-1]],[Ce[i] for i in o1[::-1]],color=["#1976d2" if j==5 else "#90caf9" for j in range(6)],edgecolor="white"); ax1.set_title("Expert weights",fontweight="bold")
    ax2.barh([df_regions["region_name_vi"].iloc[i] for i in o2[::-1]],[Cen[i] for i in o2[::-1]],color=["#388e3c" if j==5 else "#a5d6a7" for j in range(6)],edgecolor="white"); ax2.set_title("Entropy weights",fontweight="bold")
    for ax in [ax1,ax2]: ax.grid(axis="x",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)
    st.dataframe(pd.DataFrame({"Tiêu chí":lbls,"w Expert":we.round(4),"w Entropy":wen.round(4)}), use_container_width=True, hide_index=True)

    # ── 6.4.3 Sensitivity w_AI ──
    st.markdown('<div class="sec-title">📌 Câu 6.4.3 — Độ nhạy trọng số w_AI</div>', unsafe_allow_html=True)
    rng=np.arange(0.10,0.41,0.05); heat=[]
    for wai in rng:
        wb=np.array([0.10,0.10,0.15,0.15,0.15,0.05]); rem=1-wai-0.10
        wsc=wb*(rem/wb.sum()); wfull=np.concatenate([wsc[:3],[wai],wsc[3:],[0.10]]); wfull=wfull/wfull.sum()
        heat.append(topsis_fn(X,wfull,ib))
    heat=np.array(heat)
    fig,ax=plt.subplots(figsize=(10,4))
    im=ax.imshow(heat,cmap="RdYlGn",aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(df_regions["region_name_vi"],fontsize=8,rotation=20,ha="right")
    ax.set_yticks(range(len(rng))); ax.set_yticklabels([f"{w:.2f}" for w in rng])
    ax.set_ylabel("Trọng số w_AI"); ax.set_title("C* Score theo w_AI (0,10→0,40)",fontweight="bold"); plt.colorbar(im,ax=ax)
    for i in range(len(rng)):
        for j in range(6): ax.text(j,i,f"{heat[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.tight_layout(); show_fig(fig)
    st.markdown("""<div class="note-box">💡 <b>Nhận xét:</b> Đông Nam Bộ & ĐB sông Hồng giữ top-2 ổn định
    khi tăng w_AI - thứ hạng các vùng dẫn đầu KHÔNG nhạy với trọng số, củng cố độ tin cậy kết quả.</div>""", unsafe_allow_html=True)

    # ── 6.4.4 AHP ──
    st.markdown('<div class="sec-title">📌 Câu 6.4.4 — So sánh AHP với TOPSIS</div>', unsafe_allow_html=True)
    ahp=np.array([[1,1,1/3,1/5,1/3,1/3,3,3],[1,1,1/3,1/5,1/3,1/3,3,3],[3,3,1,1/2,1,1,5,5],
                  [5,5,2,1,2,2,7,7],[3,3,1,1/2,1,1,5,5],[3,3,1,1/2,1,1,5,5],
                  [1/3,1/3,1/5,1/7,1/5,1/5,1,1],[1/3,1/3,1/5,1/7,1/5,1/5,1,1]])
    n=8; gm=np.prod(ahp,axis=1)**(1/n); w_ahp=gm/gm.sum()
    lam=np.mean((ahp@w_ahp)/w_ahp); CI=(lam-n)/(n-1); CR=CI/1.41
    C_ahp=topsis_fn(X,w_ahp,ib)
    status="✅ Nhất quán (CR<0,10)" if CR<0.10 else f"⚠️ CR={CR:.3f}>0,10"
    st.info(f"**Kiểm tra nhất quán AHP:** λmax={lam:.3f} · CI={CI:.3f} · CR={CR:.3f} → {status}")
    st.dataframe(pd.DataFrame({"Vùng":df_regions["region_name_vi"],"C* Expert":Ce.round(4),
                 "C* Entropy":Cen.round(4),"C* AHP":C_ahp.round(4)}), use_container_width=True, hide_index=True)

    # ── 6.5 Thảo luận ──
    st.markdown('<div class="sec-title">💬 6.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
    top3_6=[df_regions["region_name_vi"].iloc[i] for i in np.argsort(Ce)[::-1][:3]]
    st.markdown(f"""<div class="info-box">
    <b>a)</b> Vùng dẫn đầu TOPSIS: <b>{best6}</b> - nên triển khai trung tâm AI quốc gia đầu tiên.<br>
    <b>b)</b> Khi dùng Entropy, vùng có thay đổi hạng lớn nhất thường do trọng số dữ liệu tự nhiên khác chủ quan chuyên gia.<br>
    <b>c)</b> AI Readiness & Internet tương quan cao → có thể gây trùng lặp thông tin; nên dùng PCA hoặc giảm chiều trước TOPSIS.<br>
    <b>d)</b> 3 trung tâm AI nên đặt tại: {' · '.join(top3_6)} - cân nhắc thêm yếu tố địa-chính trị (an ninh biên giới, vùng động lực).
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 7 — NSGA-II
# ══════════════════════════════════════════════════
elif menu == "🌐 Bài 7 — NSGA-II Pareto":
    st.markdown('<div class="page-header"><h1>Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)</h1><p>4 mục tiêu · Pareto front · Nghiệm thỏa hiệp TOPSIS</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 7.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Phát triển kinh tế số và AI ở Việt Nam đồng thời hướng tới <b>4 mục tiêu chiến lược</b> thường xung đột nhau:
    (i) thúc đẩy tăng trưởng GDP nhanh; (ii) bảo đảm bao trùm xã hội, giảm bất bình đẳng vùng;
    (iii) mục tiêu phát thải ròng bằng 0 vào 2050 (cam kết COP26); (iv) tăng cường an ninh dữ liệu, chủ quyền số.
    Kết quả không phải nghiệm duy nhất mà là <b>tập nghiệm Pareto</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 7.2 — Mô hình toán học (Đa mục tiêu NSGA-II)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>24 biến quyết định x(j,r), 4 hàm mục tiêu</h4>
    <div class="formula">max f1 = Σ β(j,r)·x(j,r)   (GDP gain)</div>
    <div class="formula">min f2 = G(x)   (bất bình đẳng vùng - xấp xỉ Gini/MAD)</div>
    <div class="formula">min f3 = Σ e(r)·(x(I,r) + x(AI,r))   (phát thải CO2)</div>
    <div class="formula">min f4 = Σ ρ(r)·x(AI,r) − Σ σ(r)·x(H,r)   (rủi ro an ninh dữ liệu ròng)</div>
    <b>Thuật toán:</b> NSGA-II (Fast Non-dominated Sorting GA II), pop_size=80, n_gen tùy chọn.
    <b>Chọn nghiệm thỏa hiệp:</b> áp dụng TOPSIS với trọng số chính sách (0,40 tăng trưởng;
    0,25 bao trùm; 0,20 môi trường; 0,15 an ninh).
    </div>""", unsafe_allow_html=True)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as moo_min
        from pymoo.termination import get_termination
        ng=st.slider("Số thế hệ NSGA-II",50,200,100,25)
        b7=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
        e7=np.array([0.42,0.55,0.48,0.32,0.62,0.38]); rho7=np.array([0.18,0.45,0.28,0.12,0.52,0.22]); sig7=np.array([0.32,0.28,0.30,0.35,0.25,0.30])
        D07=np.array([38,78,55,32,82,48]); g7,l7=0.002,0.6
        class P7(ElementwiseProblem):
            def __init__(self): super().__init__(n_var=24,n_obj=4,n_ieq_constr=20,xl=np.zeros(24),xu=np.ones(24)*12000)
            def _evaluate(self,x,out,*args,**kwargs):
                X=x.reshape(6,4); f1=-(b7*X).sum(); sums=X.sum(axis=1); f2=np.abs(sums-sums.mean()).mean()
                f3=(e7*(X[:,0]+X[:,2])).sum(); f4=(rho7*X[:,2]).sum()-(sig7*X[:,3]).sum()
                out["F"]=[f1,f2,f3,f4]; cons=[X.sum()-50000]
                for r in range(6): cons.append(5000-X[r].sum())
                for r in range(6): cons.append(X[r].sum()-12000)
                cons.append(12000-X[:,3].sum()); Dn=D07+g7*X[:,1]; Dm=Dn.max()
                for r in range(6): cons.append(l7*Dm-Dn[r])
                out["G"]=np.array(cons)
        st.markdown('<div class="sec-title">🔬 7.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
        with st.spinner("Chạy NSGA-II (pop=80)..."):
            res7=moo_min(P7(),NSGA2(pop_size=80),get_termination("n_gen",ng),seed=42,verbose=False)
        F=res7.F; Xsol=res7.X
        if F is not None and len(np.atleast_2d(F))>0:
            F=np.atleast_2d(F); Xsol=np.atleast_2d(Xsol)
            wp=np.array([0.40,0.25,0.20,0.15]); fm,fx=F.min(axis=0),F.max(axis=0); fr=np.where(fx-fm>1e-12,fx-fm,1)
            Rp=(F-fm)/fr; Vp=Rp*wp; Sb=np.sqrt(((Vp)**2).sum(axis=1)); Sw=np.sqrt(((Vp-wp)**2).sum(axis=1)); Cs=Sw/(Sb+Sw); best=int(np.argmax(Cs)); mg=int(np.argmin(F[:,0]))

            # ── 7.4.1 & 7.4.2 Pareto front ──
            st.markdown('<div class="sec-title">📌 Câu 7.4.1-7.4.2 — Tập Pareto & biểu đồ đánh đổi</div>', unsafe_allow_html=True)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Nghiệm Pareto",len(F)); c2.metric("GDP gain max",f"{-F[mg,0]:,.0f}")
            c3.metric("GDP thỏa hiệp",f"{-F[best,0]:,.0f}"); c4.metric("C* thỏa hiệp",f"{Cs[best]:.4f}")
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            sc=axes[0].scatter(-F[:,0],F[:,1],c=F[:,2],s=20,alpha=0.6,cmap="viridis")
            axes[0].scatter(-F[best,0],F[best,1],s=180,marker="*",color="red",label="Thỏa hiệp",zorder=5)
            axes[0].scatter(-F[mg,0],F[mg,1],s=140,marker="D",color="#ff9800",label="Tăng trưởng max",zorder=5)
            axes[0].set_xlabel("f1: GDP gain"); axes[0].set_ylabel("f2: Gini/MAD"); axes[0].set_title("Pareto Front f1 vs f2",fontweight="bold"); axes[0].legend(); plt.colorbar(sc,ax=axes[0],label="f3 Phát thải")
            axes[1].scatter(-F[:,0],F[:,3],c=F[:,1],s=20,alpha=0.6,cmap="plasma")
            axes[1].scatter(-F[best,0],F[best,3],s=180,marker="*",color="red",zorder=5)
            axes[1].set_xlabel("f1: GDP gain"); axes[1].set_ylabel("f4: Rủi ro ròng"); axes[1].set_title("Pareto Front f1 vs f4",fontweight="bold")
            for ax in axes: ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); show_fig(fig)

            # ── 7.4.3 Nghiệm thỏa hiệp TOPSIS ──
            st.markdown('<div class="sec-title">📌 Câu 7.4.3 — Nghiệm thỏa hiệp (TOPSIS trên Pareto)</div>', unsafe_allow_html=True)
            bX=Xsol[best].reshape(6,4)
            rn7=['Trung du MN','ĐB Hồng','Bắc Trung Bộ','Tây Nguyên','Đông Nam Bộ','ĐB Cửu Long']
            dfb=pd.DataFrame(bX.round(0),index=rn7,columns=['I','D','AI','H']); dfb["Tổng"]=dfb.sum(axis=1)
            st.dataframe(dfb, use_container_width=True)
            st.markdown(f"""<div class="note-box">💡 Nghiệm thỏa hiệp (trọng số 0,40/0,25/0,20/0,15) cho GDP gain
            <b>{-F[best,0]:,.0f} tỷ</b>, cân bằng giữa 4 mục tiêu - phù hợp định hướng phát triển bao trùm + xanh của Việt Nam.</div>""", unsafe_allow_html=True)

            # ── 7.4.4 Chi phí cơ hội ──
            st.markdown('<div class="sec-title">📌 Câu 7.4.4 — Chi phí cơ hội của các mục tiêu</div>', unsafe_allow_html=True)
            oc=pd.DataFrame({"Mục tiêu":["f1 GDP gain (tỷ)","f2 Gini/MAD","f3 Phát thải","f4 Rủi ro ròng"],
                             "Nghiệm thỏa hiệp":[round(-F[best,0],1),round(F[best,1],2),round(F[best,2],2),round(F[best,3],2)],
                             "Tăng trưởng max":[round(-F[mg,0],1),round(F[mg,1],2),round(F[mg,2],2),round(F[mg,3],2)]})
            st.dataframe(oc, use_container_width=True, hide_index=True)
            st.markdown(f"""<div class="note-box">💡 <b>Chi phí cơ hội:</b> Nghiệm tăng trưởng max đạt GDP cao hơn
            {(-F[mg,0])-(-F[best,0]):,.0f} tỷ nhưng phải hi sinh về bao trùm (f2 cao hơn {(F[mg,1]-F[best,1])/max(F[best,1],1e-6)*100:.0f}%)
            và môi trường. Nghiệm thỏa hiệp cân đối hơn cho phát triển bền vững.</div>""", unsafe_allow_html=True)

            # ── 7.5 Thảo luận ──
            st.markdown('<div class="sec-title">💬 7.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
            st.markdown("""<div class="info-box">
            <b>a)</b> Đánh đổi tăng trưởng-bao trùm rõ rệt trên biên Pareto: tối đa GDP thường dồn vốn về vùng giàu, tăng bất bình đẳng.<br>
            <b>b)</b> Trọng số (0,40/0,25/0,20/0,15) ưu tiên tăng trưởng - có thể điều chỉnh tăng trọng số môi trường để phù hợp cam kết COP26.<br>
            <b>c)</b> NSGA-II cung cấp <b>tập lựa chọn</b> cho nhà hoạch định, KHÔNG thay thế quyết định chính trị - chỉ hỗ trợ định lượng đánh đổi.
            </div>""", unsafe_allow_html=True)
    except ImportError: st.error("Cần cài: `pip install pymoo`")
    except Exception as e: st.error(f"Lỗi: {e}")

# ══════════════════════════════════════════════════
# BÀI 8 — TỐI ƯU ĐỘNG
# ══════════════════════════════════════════════════
elif menu == "⏳ Bài 8 — Động 2026-2035":
    st.markdown('<div class="page-header"><h1>Bài 8 — Tối ưu động 2026–2035</h1><p>Cobb-Douglas · Động học vốn · CRRA welfare · SLSQP</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 8.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo Văn kiện Đại hội XIII, Việt Nam đặt mục tiêu trở thành nước <b>thu nhập trung bình cao vào 2030</b>
    và <b>thu nhập cao vào 2045</b>. Cần thiết kế chiến lược phân bổ vốn dài hạn (2026-2035), cân bằng giữa
    tăng trưởng, chuyển đổi số, AI và chất lượng nhân lực - có xét hiệu ứng tích lũy và độ trễ.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 8.2 — Mô hình toán học (Quy hoạch động NLP)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Hàm mục tiêu - tối đa hóa phúc lợi xã hội liên thời gian</h4>
    <div class="formula">max Σ(t) ρ^t · U(C(t)), với U(C) = (C^(1−γ) − 1)/(1−γ), γ=1,5</div>
    <h4>Hàm sản xuất</h4>
    <div class="formula">Y(t) = A(t)·K^0.33·L^0.42·D^0.10·AI^0.08·H^0.07</div>
    <h4>Động học tích lũy vốn</h4>
    <ul>
      <li>K(t+1) = (1−δK)·K(t) + I_K(t), δK=0,05</li>
      <li>D(t+1) = (1−δD)·D(t) + I_D(t), δD=0,12</li>
      <li>AI(t+1) = (1−δAI)·AI(t) + I_AI(t), δAI=0,15</li>
      <li>H(t+1) = H(t) + θH·I_H(t) − μ·H(t), θH=0,8, μ=0,02</li>
    </ul>
    <h4>TFP nội sinh</h4>
    <div class="formula">A(t+1) = A(t)·(1 + φ1·D + φ2·AI + φ3·H)</div>
    <b>Ràng buộc ngân sách:</b> C(t) + I_K + I_D + I_AI + I_H ≤ Y(t); ρ=0,97; giải bằng SLSQP.
    </div>""", unsafe_allow_html=True)

    from scipy.optimize import minimize as spmin
    al8,be8,ga8,de8,th8=0.33,0.42,0.10,0.08,0.07; dK,dD,dAI=0.05,0.12,0.15; thH,mu=0.8,0.02
    phi1,phi2,phi3=0.003,0.002,0.004; rho8=0.97; gcr=1.5; T8=10
    K0,L0,D0v,AI0,H0=27500.0,53.9,20.3,86.0,30.0; Y0=12847.6
    A0v=Y0/(K0**al8*L0**be8*D0v**ga8*AI0**de8*H0**th8); L8=np.array([L0*1.009**t for t in range(T8+1)])
    def traj8(u,sh_t=None,sh_v=0.0):
        IK=u[0::4]; ID=u[1::4]; IAI=u[2::4]; IH=u[3::4]
        K=np.zeros(T8+1); D=np.zeros(T8+1); AI=np.zeros(T8+1); H=np.zeros(T8+1); A=np.zeros(T8+1); Y=np.zeros(T8+1); C=np.zeros(T8)
        K[0]=K0; D[0]=D0v; AI[0]=AI0; H[0]=H0; A[0]=A0v
        for t in range(T8):
            if sh_t is not None and t==sh_t: A[t]*=(1-sh_v)
            Y[t]=A[t]*K[t]**al8*L8[t]**be8*D[t]**ga8*AI[t]**de8*H[t]**th8
            C[t]=Y[t]-IK[t]-ID[t]-IAI[t]-IH[t]
            if C[t]<=0: return None
            K[t+1]=(1-dK)*K[t]+IK[t]; D[t+1]=(1-dD)*D[t]+ID[t]; AI[t+1]=(1-dAI)*AI[t]+IAI[t]; H[t+1]=H[t]+thH*IH[t]-mu*H[t]
            A[t+1]=A[t]*(1+phi1*(D[t]/100)+phi2*(AI[t]/100)+phi3*(H[t]/100))
        Y[T8]=A[T8]*K[T8]**al8*L8[T8]**be8*D[T8]**ga8*AI[T8]**de8*H[T8]**th8
        return K,D,AI,H,Y,C,A
    def neg_W(u):
        r=traj8(u)
        if r is None: return 1e15
        C=r[5]
        if np.any(C<=0): return 1e15
        return -sum(rho8**t*((C[t]**(1-gcr)-1)/(1-gcr)) for t in range(T8))
    st.markdown('<div class="sec-title">🔬 8.3 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
    with st.spinner("Tối ưu SLSQP 10 năm..."):
        opt=spmin(neg_W,np.ones(4*T8)*500,method="SLSQP",bounds=[(0,None)]*(4*T8),options={"maxiter":150})
    r8=traj8(opt.x)
    if r8:
        K8,D8,AI8,H8,Y8,C8,A8=r8; yrs8=np.arange(2026,2037)
        # ── 8.3.1 & 8.3.2 Quỹ đạo ──
        st.markdown('<div class="sec-title">📌 Câu 8.3.1-8.3.2 — Quỹ đạo tối ưu K, D, AI, H, Y, C</div>', unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.metric("GDP 2035",f"{Y8[-1]:,.0f} ng.tỷ"); c2.metric("Tăng trưởng BQ",f"{((Y8[-1]/Y8[0])**(1/10)-1)*100:.2f}%/năm")
        c3.metric("Tiêu dùng BQ",f"{C8.mean():,.0f} ng.tỷ/năm")
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        data8=[("GDP (Y)",Y8,"#1976d2"),("Tiêu dùng (C)",np.append(C8,np.nan),"#e53935"),
               ("Vốn (K)",K8,"#2e7d32"),("Hạ tầng số (D)",D8,"#f57c00"),("AI",AI8,"#7b1fa2"),("Nhân lực (H)",H8,"#00838f")]
        for (lbl,arr,col),ax in zip(data8,axes.flatten()):
            ax.plot(yrs8,arr,"o-",color=col,lw=2.2,markersize=5); ax.set_title(lbl,fontweight="bold"); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Quỹ đạo tối ưu 2026–2035",fontsize=13,fontweight="bold",y=1.01); fig.tight_layout(); show_fig(fig)
        st.markdown("""<div class="note-box">💡 <b>Nhận xét:</b> Mô hình đề xuất <b>front-load</b> đầu tư AI & nhân lực số
        giai đoạn đầu để tích lũy năng lực hấp thụ. Nhân lực (H) nên đi trước/đồng thời với AI - phù hợp Nghị quyết 57-NQ/TW.</div>""", unsafe_allow_html=True)

        # ── 8.3.3 Cú sốc 2028 ──
        st.markdown('<div class="sec-title">📌 Câu 8.3.3 — Phân tích cú sốc TFP -8% năm 2028 (như bão Yagi)</div>', unsafe_allow_html=True)
        r_shock=traj8(opt.x,sh_t=2,sh_v=0.08)
        if r_shock:
            Y_sh=r_shock[4]
            fig,ax=plt.subplots(figsize=(10,4))
            ax.plot(yrs8,Y8,"o-",label="Không sốc",color="#2e7d32",lw=2.5)
            ax.plot(yrs8,Y_sh,"s--",label="Có sốc 2028 (-8%)",color="#e53935",lw=2)
            ax.axvline(2028,color="gray",ls=":",label="Cú sốc")
            ax.set_title("Tác động cú sốc TFP -8% năm 2028",fontweight="bold"); ax.legend(); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); show_fig(fig)
            st.markdown(f"""<div class="note-box">💡 GDP 2035 giảm còn {Y_sh[-1]:,.0f} ng.tỷ (mất {Y8[-1]-Y_sh[-1]:,.0f} ng.tỷ).
            Bài học: cần quỹ dự phòng chính sách để tái phân bổ khi xảy ra cú sốc kinh tế.</div>""", unsafe_allow_html=True)

        # ── 8.3.4 So sánh chiến lược ──
        st.markdown('<div class="sec-title">📌 Câu 8.3.4 — So sánh chiến lược đầu tư</div>', unsafe_allow_html=True)
        u_even=np.ones(4*T8)*500
        u_front=np.zeros(4*T8)
        for t in range(T8):
            f=1.5 if t<3 else 0.7
            u_front[t*4:(t+1)*4]=500*f
        W_opt=-neg_W(opt.x); W_even=-neg_W(u_even); W_front=-neg_W(u_front)
        st.dataframe(pd.DataFrame({"Chiến lược":["Tối ưu (SLSQP)","Đầu tư đều","Front-load (150%+70%)"],
                     "Phúc lợi W":[round(W_opt,2),round(W_even,2),round(W_front,2)]}), use_container_width=True, hide_index=True)
        st.markdown("""<div class="note-box">💡 <b>Nhận xét:</b> Chiến lược tối ưu (SLSQP) cho phúc lợi cao nhất.
        Front-load tốt hơn đầu tư đều khi hệ số chiết khấu ρ cao (chính phủ quan tâm dài hạn) vì tích lũy vốn sớm tạo lan tỏa TFP.</div>""", unsafe_allow_html=True)

        # ── 8.4 Thảo luận ──
        st.markdown('<div class="sec-title">💬 8.4 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        <b>a)</b> Quỹ đạo tối ưu thường front-loaded vì đầu tư sớm tạo hiệu ứng tích lũy + lan tỏa TFP dài hạn.<br>
        <b>b)</b> Tỷ lệ đầu tư AI/H ngụ ý nhân lực nên đi trước hoặc đồng thời với AI để bảo đảm năng lực hấp thụ.<br>
        <b>c)</b> ρ=0,97 (quan tâm dài hạn) khuyến khích đầu tư R&D; nếu ρ=0,90 (ngắn hạn) chính phủ dễ "dưới đầu tư" vào R&D.
        </div>""", unsafe_allow_html=True)
    else: st.warning("Chưa hội tụ — thử tăng maxiter.")

# ══════════════════════════════════════════════════
# BÀI 9 — LAO ĐỘNG
# ══════════════════════════════════════════════════
elif menu == "👷 Bài 9 — Lao động & AI":
    st.markdown('<div class="page-header"><h1>Bài 9 — Tác động AI tới thị trường lao động</h1><p>NetJob · Thay thế · Đào tạo lại · LP tối ưu</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 9.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Theo ILO Vietnam 2024 và OECD AI Employment Report 2024, khoảng <b>30-50% việc làm</b> tại Việt Nam
    có nguy cơ bị tự động hóa một phần trong 10 năm tới (đặc biệt chế biến chế tạo, bán buôn-bán lẻ, logistics).
    Nhưng AI cũng tạo việc làm mới. Bài toán: phân bổ <b>30.000 tỷ</b> cho đào tạo lại lao động để
    bảo đảm <b>NetJob ròng dương</b> cho tất cả ngành.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 9.2 — Mô hình toán học (LP NetJob)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Việc làm ròng theo ngành i</h4>
    <div class="formula">NetJob(i) = NewJob(i) + UpgradeJob(i) − DisplacedJob(i)</div>
    <ul>
      <li>NewJob = a1·x_AI + a2·x_D (việc làm mới từ AI & số hóa)</li>
      <li>UpgradeJob = b1·x_H (nâng cấp nhờ đào tạo)</li>
      <li>DisplacedJob = c1·risk·x_AI (việc bị thay thế)</li>
      <li>RetrainCapacity = d1·x_H (năng lực đào tạo lại)</li>
    </ul>
    <h4>Bài toán tối ưu</h4>
    <div class="formula">max Σ NetJob(i)</div>
    <b>Ràng buộc:</b> Σ(x_AI + x_H) ≤ 30.000; NetJob(i) ≥ 0 ∀i; Displaced(i) ≤ RetrainCapacity(i) ∀i
    <br>(nguyên tắc: "tốc độ tự động hóa không vượt quá năng lực đào tạo lại")
    </div>""", unsafe_allow_html=True)

    from scipy.optimize import linprog
    N=8; secs=['Nông-LT','CN cbct','Xây dựng','BBL','TC-NH','Logistics','CNTT','GD-ĐT']
    Lsec=np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
    risk=np.array([18,42,25,38,52,35,28,22])/100; a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    b1=np.array([45,28,35,32,22,30,20,55]); c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5]); d1=np.array([50,32,42,38,26,36,24,62])

    # ── 9.3 Bảng tham số ──
    st.markdown('<div class="sec-title">📋 9.3 — Tham số 8 ngành (việc/tỷ đồng đầu tư)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Ngành":secs,"LĐ (triệu)":Lsec,"Risk %":(risk*100).astype(int),
                 "a1 (việc mới/AI)":a1,"b1 (nâng cấp/H)":b1,"c1 (thay thế)":c1,"d1 (đào tạo)":d1}), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">🔬 9.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
    coeff=a1-c1*risk; c_obj=np.concatenate([-coeff,-b1])
    A1l=np.ones((1,2*N)); A2=np.zeros((N,2*N))
    for i in range(N): A2[i,i]=-coeff[i]; A2[i,N+i]=-b1[i]
    A3=np.zeros((N,2*N))
    for i in range(N): A3[i,i]=c1[i]*risk[i]; A3[i,N+i]=-d1[i]
    res9=linprog(c_obj,A_ub=np.vstack([A1l,A2,A3]),b_ub=np.concatenate([[30000],np.zeros(N),np.zeros(N)]),bounds=[(0,None)]*(2*N),method="highs")
    if res9.success:
        xA=res9.x[:N]; xH=res9.x[N:]; NJ=coeff*xA+b1*xH
        Displaced=c1*risk*xA; RetrainCap=d1*xH
        # ── 9.4.1 ──
        st.markdown('<div class="sec-title">📌 Câu 9.4.1 — Phân bổ tối ưu & NetJob ròng</div>', unsafe_allow_html=True)
        c1c,c2c,c3c=st.columns(3)
        c1c.metric("Tổng NetJob",f"{NJ.sum():,.0f} việc"); c2c.metric("Ngân sách AI",f"{xA.sum():,.0f} tỷ"); c3c.metric("Ngân sách Đào tạo",f"{xH.sum():,.0f} tỷ")
        st.dataframe(pd.DataFrame({"Ngành":secs,"x_AI (tỷ)":xA.round(0),"x_H (tỷ)":xH.round(0),
                     "Displaced":Displaced.round(0),"Retrain Cap":RetrainCap.round(0),"NetJob":NJ.round(0)}), use_container_width=True, hide_index=True)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
        ax1.bar(secs,NJ,color=["#66bb6a" if v>=0 else "#ef5350" for v in NJ],edgecolor="white")
        ax1.set_xticklabels(secs,rotation=30,ha="right"); ax1.axhline(0,color="gray",lw=0.8); ax1.set_title("NetJob ròng theo ngành",fontweight="bold"); ax1.grid(axis="y",alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
        x9=np.arange(N); w9=0.25
        ax2.bar(x9-w9,a1*xA,w9,label="Việc làm mới",color="#42a5f5")
        ax2.bar(x9,b1*xH,w9,label="Nâng cấp",color="#66bb6a")
        ax2.bar(x9+w9,c1*risk*xA,w9,label="Bị thay thế",color="#ef5350")
        ax2.set_xticks(x9); ax2.set_xticklabels(secs,rotation=30,ha="right"); ax2.legend(); ax2.set_title("Phân rã tác động lao động",fontweight="bold"); ax2.grid(axis="y",alpha=0.3); ax2.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

        # ── 9.4.2 Ngưỡng đào tạo ──
        st.markdown('<div class="sec-title">📌 Câu 9.4.2 — Ngưỡng đào tạo tối thiểu ngành CN chế biến</div>', unsafe_allow_html=True)
        i_cn=1; rr=c1[i_cn]*risk[i_cn]/d1[i_cn]; net_ai=a1[i_cn]-c1[i_cn]*risk[i_cn]
        st.markdown(f"""<div class="note-box">Ngành <b>CN chế biến chế tạo</b> (risk=42%): để Displaced ≤ RetrainCapacity,
        cần <b>x_H ≥ {rr:.3f}·x_AI</b>. Hệ số net AI = {net_ai:.1f} (âm → mỗi đồng AI làm mất việc ròng nếu không đào tạo kèm).
        Đây là ngành cần đầu tư đào tạo lại nhiều nhất.</div>""", unsafe_allow_html=True)

        # ── 9.4.3 Sankey luồng lao động ──
        st.markdown('<div class="sec-title">📌 Câu 9.4.3 — Luồng dịch chuyển lao động nhóm dễ tổn thương</div>', unsafe_allow_html=True)
        vuln=[0,2,3]; vnames=[secs[i] for i in vuln]
        kept=[Lsec[i]*1e6-Displaced[i] for i in vuln]
        retr=[min(Displaced[i],RetrainCap[i]) for i in vuln]
        lost=[max(0,Displaced[i]-RetrainCap[i]) for i in vuln]
        fig,ax=plt.subplots(figsize=(10,4)); xp=np.arange(3); ww=0.6
        ax.bar(xp,kept,ww,label="Giữ việc",color="#66bb6a")
        ax.bar(xp,retr,ww,bottom=kept,label="Đào tạo lại",color="#ffa726")
        ax.bar(xp,lost,ww,bottom=[k+r for k,r in zip(kept,retr)],label="Mất việc",color="#ef5350")
        ax.set_xticks(xp); ax.set_xticklabels(vnames); ax.legend()
        ax.set_title("Luồng lao động nhóm dễ tổn thương (Nông-LT, Xây dựng, Bán buôn-bán lẻ)",fontweight="bold"); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

        # ── 9.4.4 Ràng buộc 5%L ──
        st.markdown('<div class="sec-title">📌 Câu 9.4.4 — Ràng buộc Displaced ≤ 5% lao động</div>', unsafe_allow_html=True)
        A4=np.zeros((N,2*N))
        for i in range(N): A4[i,i]=c1[i]*risk[i]
        res9b=linprog(c_obj,A_ub=np.vstack([A1l,A2,A3,A4]),
                      b_ub=np.concatenate([[30000],np.zeros(N),np.zeros(N),0.05*Lsec*1e6]),bounds=[(0,None)]*(2*N),method="highs")
        if res9b.success:
            NJ2=coeff*res9b.x[:N]+b1*res9b.x[N:]
            st.success(f"✅ Khả thi với ràng buộc 5%L · Tổng NetJob = {NJ2.sum():,.0f} (giảm {NJ.sum()-NJ2.sum():,.0f} = {(NJ.sum()-NJ2.sum())/NJ.sum()*100:.1f}%)")
        else:
            st.error("❌ KHÔNG khả thi - cần tăng đầu tư đào tạo lại mạnh hơn.")

        # ── 9.5 Thảo luận ──
        st.markdown('<div class="sec-title">💬 9.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        <b>a)</b> CN chế biến & Tài chính-NH cần đào tạo lại nhiều nhất (risk cao 42%-52%) - khớp thực tế Việt Nam.<br>
        <b>b)</b> Tài chính-NH risk 52% nhưng hệ số tạo việc mới cao (45,8) - chiến lược: đầu tư AI mạnh + đào tạo kèm.<br>
        <b>c)</b> Nông-Lâm-TS hệ số tạo việc AI thấp (8,5) nhưng lao động dịch chuyển lớn - mô hình ưu tiên đào tạo (H) hơn AI.<br>
        <b>d)</b> Ràng buộc "Displaced ≤ RetrainCapacity" biểu diễn "tốc độ tự động hóa không vượt năng lực đào tạo lại".
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 10 — QUY HOẠCH NGẪU NHIÊN
# ══════════════════════════════════════════════════
elif menu == "🎲 Bài 10 — Stochastic SP":
    st.markdown('<div class="page-header"><h1>Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn</h1><p>First-stage · Recourse · VSS · EVPI · Pyomo</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 10.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Việt Nam có độ mở thương mại rất cao (XNK/GDP ≈ 180% năm 2025), tăng trưởng phụ thuộc lớn vào
    kịch bản kinh tế toàn cầu. Khi hoạch định chính sách đầu tư số 2026-2030, Chính phủ phải đưa ra
    <b>quyết định first-stage</b> (kế hoạch ngân sách) khi <b>chưa biết chắc kịch bản tương lai</b>.
    Bài toán áp dụng quy hoạch ngẫu nhiên 2 giai đoạn.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 10.2 — Mô hình toán học (Two-stage Stochastic Programming)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Cấu trúc 2 giai đoạn</h4>
    <b>Giai đoạn 1 (here-and-now):</b> quyết định x = (x_I, x_D, x_AI, x_H) trước khi biết kịch bản
    <div class="formula">Σ x(j) ≤ 65.000 (giữ 15.000 dự phòng)</div>
    <b>Giai đoạn 2 (recourse):</b> điều chỉnh y(s) sau khi kịch bản s hiện thực hóa
    <div class="formula">Σ y(s,j) ≤ 15.000 ∀s ; y(AI,s) ≤ 0,5·x_H ∀s</div>
    <h4>Hàm mục tiêu</h4>
    <div class="formula">max Σ β(j)·x(j) + Σ p(s)·[Σ β(s,j)·y(s,j)]</div>
    <b>4 kịch bản:</b> Lạc quan (p=0,30) · Cơ sở (p=0,45) · Bi quan (p=0,20) · Khủng hoảng (p=0,05)
    <h4>Chỉ số đánh giá</h4>
    <ul>
      <li><b>VSS</b> (Value of Stochastic Solution): lợi ích khi cân nhắc bất định</li>
      <li><b>EVPI</b> (Expected Value of Perfect Information): giá trị thông tin hoàn hảo</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    from scipy.optimize import linprog
    J=['I','D','AI','H']; S=['s1','s2','s3','s4']
    ps={'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05}
    bb={'I':1.00,'D':1.10,'AI':1.25,'H':0.95}
    bs={('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
        ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
        ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
        ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10}

    # ── 10.3 Bảng kịch bản ──
    st.markdown('<div class="sec-title">📋 10.4 — Bảng hệ số β theo kịch bản & xác suất</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Hạng mục":["I (hạ tầng)","D (CĐS)","AI","H (nhân lực)"],
                 "β cơ bản":[bb[j] for j in J],
                 "s1 Lạc quan":[bs[('s1',j)] for j in J],"s2 Cơ sở":[bs[('s2',j)] for j in J],
                 "s3 Bi quan":[bs[('s3',j)] for j in J],"s4 Khủng hoảng":[bs[('s4',j)] for j in J]}), use_container_width=True, hide_index=True)
    st.caption("Xác suất: s1=0,30 · s2=0,45 · s3=0,20 · s4=0,05. Hệ số H cao trong khủng hoảng vì LĐ qua đào tạo hấp thụ cú sốc tốt hơn.")

    st.markdown('<div class="sec-title">🔬 10.5 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)

    # Giải SP bằng scipy (đảm bảo chạy được không cần solver ngoài)
    n=4+16; cobj=np.zeros(n)
    for k,j in enumerate(J): cobj[k]=-bb[j]
    for si,s in enumerate(S):
        for k,j in enumerate(J): cobj[4+si*4+k]=-ps[s]*bs[(s,j)]
    Aub=[]; bub=[]
    Aub.append([1,1,1,1]+[0]*16); bub.append(65000)
    for si in range(4):
        row=[0]*n
        for k in range(4): row[4+si*4+k]=1
        Aub.append(row); bub.append(15000)
    for si in range(4):
        row=[0]*n; row[3]=-0.5; row[4+si*4+2]=1
        Aub.append(row); bub.append(0)
    res10=linprog(cobj,A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*n,method="highs")
    x_sp=res10.x[:4]; Z_sp=-res10.fun

    # EV deterministic
    beta_avg={j:sum(ps[s]*bs[(s,j)] for s in S) for j in J}
    cev=[-beta_avg[j] for j in J]
    rev=linprog(cev,A_ub=[[1,1,1,1]],b_ub=[65000],bounds=[(0,None)]*4,method="highs")
    x_ev=rev.x; Z_ev=sum(bb[j]*x_ev[k] for k,j in enumerate(J))
    for s in S:
        cs=[-bs[(s,j)] for j in J]
        r=linprog(cs,A_ub=[[1,1,1,1],[0,0,1,0]],b_ub=[15000,0.5*x_ev[3]],bounds=[(0,None)]*4,method="highs")
        Z_ev+=ps[s]*(-r.fun)
    # Wait-and-see
    det={}
    for s in S:
        cs=np.zeros(8)
        for k,j in enumerate(J): cs[k]=-bb[j]; cs[4+k]=-bs[(s,j)]
        r=linprog(cs,A_ub=[[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1],[0,0,0,-0.5,0,0,1,0]],b_ub=[65000,15000,0],bounds=[(0,None)]*8,method="highs")
        det[s]=-r.fun
    Z_ws=sum(ps[s]*det[s] for s in S)
    VSS=Z_sp-Z_ev; EVPI=Z_ws-Z_sp

    # ── 10.5.1 First-stage ──
    st.markdown('<div class="sec-title">📌 Câu 10.5.1 — Quyết định first-stage tối ưu</div>', unsafe_allow_html=True)
    st.success(f"**Z* Stochastic = {Z_sp:,.0f} tỷ GDP kỳ vọng**")
    st.dataframe(pd.DataFrame({"Hạng mục":["Hạ tầng (I)","CĐS (D)","AI","Nhân lực (H)"],
                 "First-stage x* (tỷ)":x_sp.round(0),"Tỷ lệ %":(x_sp/x_sp.sum()*100).round(1)}), use_container_width=True, hide_index=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    ax1.bar(["I","D","AI","H"],x_sp,color=["#42a5f5","#26c6da","#ffa726","#66bb6a"],edgecolor="white")
    ax1.set_title("First-stage x* (tỷ VND)",fontweight="bold"); ax1.grid(axis="y",alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
    ax2.bar(["s1 Lạc quan","s2 Cơ sở","s3 Bi quan","s4 K.hoảng"],[det[s] for s in S],color=["#1976d2","#42a5f5","#ffa726","#ef5350"],edgecolor="white")
    ax2.set_title("Z* tối ưu theo từng kịch bản",fontweight="bold"); ax2.grid(axis="y",alpha=0.3); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); show_fig(fig)

    # ── 10.5.2 So sánh EV vs SP ──
    st.markdown('<div class="sec-title">📌 Câu 10.5.2 — So sánh lời giải EV vs SP</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Hạng mục":["I","D","AI","H"],"x* SP (stochastic)":x_sp.round(0),"x* EV (deterministic)":x_ev.round(0)}), use_container_width=True, hide_index=True)

    # ── 10.5.3 VSS & EVPI ──
    st.markdown('<div class="sec-title">📌 Câu 10.5.3 — VSS và EVPI</div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Z* Stochastic (SP)",f"{Z_sp:,.0f}"); c2.metric("Z* EV",f"{Z_ev:,.0f}")
    c3.metric("VSS",f"{VSS:,.0f} tỷ"); c4.metric("EVPI",f"{EVPI:,.0f} tỷ")
    st.markdown(f"""<div class="note-box">💡 <b>VSS = {VSS:,.0f} tỷ</b>: lợi ích của việc dùng mô hình ngẫu nhiên thay vì EV.
    <b>EVPI = {EVPI:,.0f} tỷ</b>: giá trị tối đa sẵn sàng trả để có thông tin hoàn hảo về kịch bản.
    SP đầu tư H (nhân lực) nhiều hơn EV - H là "hàng hóa bảo hiểm" giúp hấp thụ cú sốc.</div>""", unsafe_allow_html=True)

    # ── 10.6 Thảo luận ──
    st.markdown('<div class="sec-title">💬 10.6 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>a)</b> So với lời giải xác định, SP đầu tư H nhiều hơn vì nhân lực giúp linh hoạt thích ứng mọi kịch bản.<br>
    <b>b)</b> VSS dương khẳng định giá trị của tư duy xác suất trong hoạch định chính sách Việt Nam.<br>
    <b>c)</b> COVID-19 và bão Yagi là cú sốc thực tế - Việt Nam nên xem nhân lực số như "hàng hóa bảo hiểm", tránh "dưới đầu tư".
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 11 — Q-LEARNING
# ══════════════════════════════════════════════════
elif menu == "🤖 Bài 11 — Q-learning RL":
    st.markdown('<div class="page-header"><h1>Bài 11 — Q-learning chính sách kinh tế thích nghi</h1><p>MDP · 81 trạng thái · 5 hành động · Epsilon-greedy</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 11.1 — Bối cảnh Việt Nam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Vấn đề đặt ra</h4>
    Nền kinh tế Việt Nam có thể được xem như <b>môi trường</b>, chính sách là <b>hành động</b>,
    phần thưởng phản ánh <b>phúc lợi xã hội</b>. Học tăng cường cho phép chính sách <b>thích nghi</b>
    theo trạng thái kinh tế hiện tại, thay vì cố định như LP.
    <br><br>
    <b>Lưu ý:</b> AI hỗ trợ ra quyết định KHÔNG thay thế trách nhiệm chính trị - bài này minh họa kỹ thuật.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 11.2 — Mô hình toán học (MDP + Q-learning)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>Markov Decision Process</h4>
    <b>Trạng thái (3^4 = 81):</b> GDP growth, Digital index, AI capacity, Unemployment risk - mỗi cái 3 mức {low, medium, high}
    <b>5 hành động (chiến lược phân bổ K/D/AI/H):</b>
    <ul>
      <li>a0 Truyền thống (70/10/10/10) · a1 Cân bằng (40/25/15/20)</li>
      <li>a2 Số hóa nhanh (25/45/15/15) · a3 AI dẫn dắt (20/20/45/15) · a4 Bao trùm (30/20/10/40)</li>
    </ul>
    <h4>Phần thưởng (welfare)</h4>
    <div class="formula">R = 0,40·ΔGDP − 0,25·ΔUnemploy − 0,20·CyberRisk − 0,15·Emission</div>
    <h4>Cập nhật Q-table (Bellman)</h4>
    <div class="formula">Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]</div>
    <b>Tham số:</b> α=0,10 · γ=0,95 · ε-greedy giảm dần · 10 năm/episode
    </div>""", unsafe_allow_html=True)

    # ── 11.3 Bảng MDP ──
    st.markdown('<div class="sec-title">📋 11.3 — Không gian trạng thái & hành động</div>', unsafe_allow_html=True)
    cc1,cc2=st.columns(2)
    with cc1:
        st.markdown("**Trạng thái (3⁴=81):**")
        st.dataframe(pd.DataFrame({"Yếu tố":["GDP growth","Digital index","AI capacity","Unemploy risk"],
                     "Low (0)":["<3%","<34","<34","<14% H"],"Med (1)":["3-6%","34-68","34-68","14-28%"],"High (2)":[">6%",">68",">68",">28%"]}), hide_index=True, use_container_width=True)
    with cc2:
        st.markdown("**5 hành động (K/D/AI/H %):**")
        st.dataframe(pd.DataFrame({"Hành động":["a0 Truyền thống","a1 Cân bằng","a2 Số hóa nhanh","a3 AI dẫn dắt","a4 Bao trùm"],
                     "K":[70,40,25,20,30],"D":[10,25,45,20,20],"AI":[10,15,15,45,10],"H":[10,20,15,15,40]}), hide_index=True, use_container_width=True)

    st.markdown('<div class="sec-title">🔬 11.4 — Yêu cầu lập trình & Kết quả</div>', unsafe_allow_html=True)
    nep=st.slider("Số episodes",2000,10000,5000,1000)
    alloc11={0:np.array([0.70,0.10,0.10,0.10]),1:np.array([0.40,0.25,0.15,0.20]),2:np.array([0.25,0.45,0.15,0.15]),3:np.array([0.20,0.20,0.45,0.15]),4:np.array([0.30,0.20,0.10,0.40])}
    anames=["Truyền thống","Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm"]
    w11=np.array([0.40,0.25,0.20,0.15])
    def disc(gdp,D,AI,H): return (min(2,max(0,int(gdp/3))),min(2,max(0,int(D/34))),min(2,max(0,int(AI/34))),min(2,max(0,int(H/14))))
    def step11(K,D,AI,H,Yp,a):
        al=alloc11[a]; B=2100.0
        K2=(1-0.05)*K+al[0]*B; D2=(1-0.12)*D+al[1]*B*0.01; AI2=(1-0.15)*AI+al[2]*B*0.05; H2=H+0.8*al[3]*B*0.01-0.02*H
        Av=33.7*(1+0.003*(D2/100)+0.002*(AI2/100)+0.004*(H2/100))
        Y2=Av*K2**0.33*54.0**0.42*D2**0.10*AI2**0.08*H2**0.07
        gg=(Y2-Yp)/Yp*100; cy=0.02*al[2]*B; em=0.01*al[0]*B; ud=-al[3]*B*0.001
        r=w11[0]*gg-w11[1]*max(0,ud)-w11[2]*cy*0.001-w11[3]*em*0.001
        return K2,D2,AI2,H2,Y2,disc(gg,D2,AI2,H2),r
    def run_policy(policy_fn,Q,episodes=200):
        tot=[]
        for _ in range(episodes):
            K,D,AI,H,Y=27500.0,20.3,86.0,30.0,12847.6; s=disc(6,D,AI,H); er=0
            for _ in range(10):
                a=policy_fn(s,Q); K,D,AI,H,Y,s2,r=step11(K,D,AI,H,Y,a); s=s2; er+=r
            tot.append(er)
        return np.mean(tot)

    # ── 11.4.1-11.4.2 Training ──
    st.markdown('<div class="sec-title">📌 Câu 11.4.1-11.4.2 — Huấn luyện Q-learning</div>', unsafe_allow_html=True)
    with st.spinner(f"Huấn luyện {nep} episodes..."):
        Q11=np.zeros((3,3,3,3,5)); rh=[]
        for ep in range(nep):
            K,D,AI,H,Y=27500.0,20.3,86.0,30.0,12847.6; s=disc(6,D,AI,H); er=0
            eps=0.05+(1.0-0.05)*np.exp(-3*ep/nep)
            for _ in range(10):
                a=(np.random.randint(5) if np.random.rand()<eps else int(np.argmax(Q11[s])))
                K,D,AI,H,Y,s2,r=step11(K,D,AI,H,Y,a)
                Q11[s+(a,)]+=0.1*(r+0.95*Q11[s2].max()-Q11[s+(a,)]); s=s2; er+=r
            rh.append(er)
    c1,c2,c3=st.columns(3)
    c1.metric("α (learning rate)","0,10"); c2.metric("γ (discount)","0,95"); c3.metric("Reward TB (200 ep cuối)",f"{np.mean(rh[-200:]):.2f}")
    fig,ax=plt.subplots(figsize=(10,4))
    ww=max(1,nep//50); sm=[np.mean(rh[max(0,i-ww):i+1]) for i in range(len(rh))]
    ax.plot(rh,alpha=0.2,color="#90caf9",lw=0.8); ax.plot(sm,color="#1976d2",lw=2,label=f"TB trượt (w={ww})")
    ax.set_title("Learning Curve — hội tụ Q-learning",fontweight="bold"); ax.set_xlabel("Episode"); ax.set_ylabel("Phần thưởng tích lũy"); ax.legend(); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); show_fig(fig)

    # ── 11.4.3 Chính sách π* ──
    st.markdown('<div class="sec-title">📌 Câu 11.4.3 — Chính sách tối ưu π*(s)</div>', unsafe_allow_html=True)
    rows11=[]
    for desc,s in [("🇻🇳 VN 2026 thực tế",disc(6,20,86,30)),("😟 GDP thấp, D thấp, U cao",(0,0,0,2)),
                   ("🚀 GDP cao, AI cao, U thấp",(2,2,2,0)),("⚡ Hậu khủng hoảng",(0,1,0,0)),("📊 Trung bình",(1,1,1,1))]:
        ba=int(np.argmax(Q11[s]))
        rows11.append({"Trạng thái":desc,"π*(s)":anames[ba],"Q value":f"{Q11[s][ba]:.3f}"})
    st.dataframe(pd.DataFrame(rows11), use_container_width=True, hide_index=True)
    st.markdown("""<div class="note-box">💡 Khi GDP thấp & D thấp → π* chọn "Số hóa nhanh" (quick win).
    Khi mọi chỉ số cao → π* chọn "AI dẫn dắt" (consolidation).</div>""", unsafe_allow_html=True)

    # ── 11.4.4 So sánh chính sách ──
    st.markdown('<div class="sec-title">📌 Câu 11.4.4 — So sánh π* với chính sách rule-based</div>', unsafe_allow_html=True)
    r_star=run_policy(lambda s,Q:int(np.argmax(Q[s])),Q11)
    r_a1=run_policy(lambda s,Q:1,Q11); r_a3=run_policy(lambda s,Q:3,Q11)
    r_rand=run_policy(lambda s,Q:np.random.randint(5),Q11)
    dfp=pd.DataFrame({"Chính sách":["π* (Q-learning)","Luôn Cân bằng (a1)","Luôn AI dẫn dắt (a3)","Random"],
                      "Phúc lợi TB":[round(r_star,3),round(r_a1,3),round(r_a3,3),round(r_rand,3)]})
    cc1,cc2=st.columns([1,1.2])
    with cc1: st.dataframe(dfp, use_container_width=True, hide_index=True)
    with cc2:
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(dfp["Chính sách"],dfp["Phúc lợi TB"],color=["#1976d2","#42a5f5","#66bb6a","#bdbdbd"],edgecolor="white")
        ax.set_xticklabels(dfp["Chính sách"],rotation=20,ha="right",fontsize=8)
        ax.set_title("So sánh phúc lợi các chính sách",fontweight="bold"); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)

    # ── 11.5 Thảo luận ──
    st.markdown('<div class="sec-title">💬 11.5 — Câu hỏi thảo luận chính sách</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>a)</b> GDP thấp/D thấp/U cao → π* chọn "Số hóa nhanh" - khớp tư duy "quick win" để tạo động lực sớm.<br>
    <b>b)</b> GDP cao/AI cao/U thấp → π* chọn "AI dẫn dắt" - phù hợp giai đoạn consolidation, tối ưu năng lực sẵn có.<br>
    <b>c)</b> Tích hợp π* vào quy trình hoạch định Việt Nam như <b>công cụ tư vấn</b>, KHÔNG thay quyết định chính trị - giữ trách nhiệm giải trình con người.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# BÀI 12 — AIDEOM-VN TỔNG HỢP
# ══════════════════════════════════════════════════
elif menu == "🧠 Bài 12 — AIDEOM tích hợp":
    st.markdown('<div class="page-header"><h1>Bài 12 — AIDEOM-VN Nguyên mẫu tổng hợp</h1><p>6 Module · 5 Kịch bản chính sách · Dashboard ra quyết định</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🇻🇳 12.1 — Bối cảnh & Mục tiêu đồ án</div>', unsafe_allow_html=True)
    st.markdown("""<div class="context-box">
    <h4>Đồ án tổng kết môn học</h4>
    Tích hợp các kỹ thuật đã học (Bài 1-11) thành hệ thống <b>AIDEOM-VN</b> đầy đủ <b>6 module</b> liên kết,
    với dashboard hỗ trợ ra quyết định trên <b>5 kịch bản chính sách</b>. Hệ thống cho phép nhà hoạch định
    so sánh định lượng các phương án phát triển kinh tế số Việt Nam giai đoạn 2026-2030.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">📐 12.2 — Kiến trúc mô hình AIDEOM-VN</div>', unsafe_allow_html=True)
    st.markdown("""<div class="model-box">
    <h4>6 Module tích hợp theo pipeline</h4>
    <ul>
      <li><b>M1 Dự báo kinh tế:</b> Cobb-Douglas + TFP (Bài 1) → GDP, TFP 2026-2030</li>
      <li><b>M2 Đánh giá sẵn sàng số:</b> TOPSIS + Entropy (Bài 6) → bản đồ Digital Index + AI Readiness</li>
      <li><b>M3 Tối ưu phân bổ:</b> LP 24 biến + công bằng (Bài 4) → phân bổ ngành-vùng-thời gian</li>
      <li><b>M4 Mô phỏng lao động:</b> NetJob (Bài 9) → việc làm ròng từng ngành</li>
      <li><b>M5 Đánh giá rủi ro:</b> Đa mục tiêu (Bài 7) + Stochastic (Bài 10) → cyber/môi trường/phụ thuộc</li>
      <li><b>M6 Dashboard ra QĐ:</b> trực quan 5 kịch bản, cảnh báo, khuyến nghị</li>
    </ul>
    <h4>5 Kịch bản chính sách</h4>
    <div class="formula">S1 Truyền thống · S2 Số hóa nhanh · S3 AI dẫn dắt · S4 Bao trùm số · S5 Tối ưu cân bằng</div>
    </div>""", unsafe_allow_html=True)

    # Dữ liệu kịch bản
    scen12={"S1 Truyền thống":{"K":0.70,"D":0.10,"AI":0.10,"H":0.10},
            "S2 Số hóa nhanh":{"K":0.25,"D":0.45,"AI":0.15,"H":0.15},
            "S3 AI dẫn dắt":  {"K":0.20,"D":0.20,"AI":0.45,"H":0.15},
            "S4 Bao trùm số": {"K":0.30,"D":0.20,"AI":0.10,"H":0.40},
            "S5 Tối ưu":      {"K":0.25,"D":0.25,"AI":0.30,"H":0.20}}
    al12,be12,ga12,de12,th12=0.33,0.42,0.10,0.08,0.07
    K0_,L0_,D0_,AI0_,H0_,A0_=27500,53.9,20.3,86,30,33.70; T12=4; yrs12=list(range(2026,2031)); bg12=3000
    def fc12(alloc):
        K,D,AI,H,A=K0_,D0_,AI0_,H0_,A0_; traj=[A*K**al12*L0_**be12*D**ga12*AI**de12*H**th12]
        for t in range(T12):
            K=(1-0.05)*K+alloc["K"]*bg12; D=(1-0.12)*D+alloc["D"]*bg12*0.01; AI=(1-0.15)*AI+alloc["AI"]*bg12*0.05; H=H+0.8*alloc["H"]*bg12*0.01-0.02*H
            A=A*(1+0.003*(D/100)+0.002*(AI/100)+0.004*(H/100)); L=L0_*1.009**(t+1)
            traj.append(A*K**al12*L**be12*D**ga12*AI**de12*H**th12)
        return traj
    gdp12={n:fc12(al) for n,al in scen12.items()}

    tab1,tab2,tab3,tab4 = st.tabs(["🏗️ Tổng quan & Module","📊 M1–M4 Dự báo & Phân bổ","🔀 M5 So sánh kịch bản","⚠️ M6 Rủi ro & Khuyến nghị"])

    # ── TAB 1: TỔNG QUAN ──────────────────────────
    with tab1:
        st.markdown('<div class="sec-title">🗺️ Kiến trúc 6 Module AIDEOM-VN</div>', unsafe_allow_html=True)
        mods=[("M1","Dự báo kinh tế","Cobb-Douglas + TFP","#1976d2"),("M2","Sẵn sàng số","TOPSIS + Entropy","#0288d1"),
              ("M3","Tối ưu phân bổ","LP 24 biến + Công bằng","#00838f"),("M4","Lao động","NetJob + Đào tạo lại","#2e7d32"),
              ("M5","Đa mục tiêu","NSGA-II + Stochastic","#e65100"),("M6","Dashboard QĐ","5 kịch bản + Rủi ro","#6a1b9a")]
        cols12t=st.columns(3)
        for i,(code,name,tech,color) in enumerate(mods):
            cols12t[i%3].markdown(f'<div style="border-left:4px solid {color};background:#f8fbff;border-radius:8px;padding:12px 14px;margin-bottom:10px"><div style="font-size:9px;font-weight:700;color:{color};letter-spacing:2px">{code}</div><div style="font-size:0.95rem;font-weight:700;color:#1a237e;margin:3px 0">{name}</div><div style="font-size:0.76rem;color:#546e7a">{tech}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">📋 5 Kịch bản & GDP 2030</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Kịch bản":n,"K%":int(al["K"]*100),"D%":int(al["D"]*100),"AI%":int(al["AI"]*100),"H%":int(al["H"]*100),"GDP 2030":round(gdp12[n][-1],1)} for n,al in scen12.items()]), use_container_width=True)

    # ── TAB 2: M1–M4 ──────────────────────────────
    with tab2:
        st.markdown('<div class="sec-title">M1 — Dự báo GDP 2026–2030</div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(10,5))
        c12=["#1976d2","#00838f","#e65100","#2e7d32","#7b1fa2"]
        for (n,traj),col in zip(gdp12.items(),c12): ax.plot(yrs12,traj,"o-",label=n,color=col,lw=2,markersize=6)
        ax.set_title("GDP 2026–2030 theo 5 kịch bản",fontweight="bold"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); show_fig(fig)

        st.markdown('<div class="sec-title">M2 — TOPSIS 6 vùng</div>', unsafe_allow_html=True)
        Xm2=df_regions[['grdp_per_capita_million_VND','fdi_registered_billion_USD','digital_index_0_100','ai_readiness_0_100','trained_labor_pct','rd_intensity_pct','internet_penetration_pct','gini_coef']].values.astype(float)
        ibm2=[True,True,True,True,True,True,True,False]; wm2=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
        Rm2=Xm2/np.sqrt((Xm2**2).sum(axis=0)); Vm2=Rm2*wm2; Asm2=np.where(ibm2,Vm2.max(axis=0),Vm2.min(axis=0)); Anm2=np.where(ibm2,Vm2.min(axis=0),Vm2.max(axis=0))
        Ssm2=np.sqrt(((Vm2-Asm2)**2).sum(axis=1)); Snm2=np.sqrt(((Vm2-Anm2)**2).sum(axis=1)); Cm2=Snm2/(Ssm2+Snm2)
        fig,ax=plt.subplots(figsize=(9,4)); om2=np.argsort(Cm2)[::-1]
        ax.bar([df_regions["region_name_vi"].iloc[i] for i in om2],[Cm2[i] for i in om2],color=["#1976d2" if j==0 else "#90caf9" for j in range(6)],edgecolor="white")
        ax.set_title("M2: Sẵn sàng AI — 6 vùng",fontweight="bold"); ax.set_xticklabels([df_regions["region_name_vi"].iloc[i] for i in om2],rotation=20,ha="right"); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); show_fig(fig)

        st.markdown('<div class="sec-title">M3 & M4 — Phân bổ LP & NetJob</div>', unsafe_allow_html=True)
        ca,cb=st.columns(2)
        with ca:
            # M3 heatmap placeholder
            alloc_demo=np.array([[3500,2500,1500,5000],[2500,5000,6000,3500],[3000,2500,2500,5000],[4500,1500,1000,5500],[2000,4500,7000,2500],[3000,2000,2000,5000]])
            fig2,ax2=plt.subplots(figsize=(6,4)); im=ax2.imshow(alloc_demo,cmap="Blues",aspect="auto")
            ax2.set_yticks(range(6)); ax2.set_yticklabels(['MN Bắc','Hồng','BTrung','TN','ĐNBộ','CL'],fontsize=8)
            ax2.set_xticks(range(4)); ax2.set_xticklabels(['I','D','AI','H'],fontsize=9)
            ax2.set_title("M3: Phân bổ LP (tỷ VND)",fontsize=10,fontweight="bold")
            for i in range(6):
                for j in range(4): ax2.text(j,i,f"{alloc_demo[i,j]:.0f}",ha="center",va="center",fontsize=8,color="white" if alloc_demo[i,j]>alloc_demo.max()*0.6 else "black")
            fig2.tight_layout(); show_fig(fig2)
        with cb:
            from scipy.optimize import linprog as lp9
            N9=8; scs9=['Nông-LT','CN cbct','Xây dựng','BBL','TC-NH','Logistics','CNTT','GD-ĐT']
            r9=np.array([18,42,25,38,52,35,28,22])/100; a19=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5]); b19=np.array([45,28,35,32,22,30,20,55]); c19=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
            cf9=a19-c19*r9; co9=np.concatenate([-cf9,-b19]); A1n=np.ones((1,2*N9)); A2n=np.zeros((N9,2*N9))
            for i in range(N9): A2n[i,i]=-cf9[i]; A2n[i,N9+i]=-b19[i]
            rr9=lp9(co9,A_ub=np.vstack([A1n,A2n]),b_ub=np.concatenate([[30000],np.zeros(N9)]),bounds=[(0,None)]*(2*N9),method="highs")
            NJ9=cf9*rr9.x[:N9]+b19*rr9.x[N9:] if rr9.success else np.zeros(N9)
            fig3,ax3=plt.subplots(figsize=(6,4))
            ax3.bar(scs9,NJ9,color=["#66bb6a" if v>=0 else "#ef5350" for v in NJ9],edgecolor="white")
            ax3.set_xticklabels(scs9,rotation=30,ha="right",fontsize=8); ax3.axhline(0,color="gray",lw=0.8); ax3.set_title("M4: NetJob theo ngành",fontsize=10,fontweight="bold"); ax3.grid(axis="y",alpha=0.3); ax3.spines[["top","right"]].set_visible(False); fig3.tight_layout(); show_fig(fig3)

    # ── TAB 3: KỊCH BẢN ───────────────────────────
    with tab3:
        st.markdown('<div class="sec-title">M5 — So sánh 5 kịch bản chính sách 2030</div>', unsafe_allow_html=True)
        g2030=[gdp12[n][-1] for n in scen12]
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
        ax1.bar(list(scen12.keys()),g2030,color=c12,edgecolor="white")
        for i,(v,col) in enumerate(zip(g2030,c12)): ax1.text(i,v+20,f"{v:,.0f}",ha="center",fontsize=8,fontweight="bold")
        ax1.set_title("GDP 2030 (nghìn tỷ VND)",fontweight="bold"); ax1.set_xticklabels(list(scen12.keys()),rotation=20,ha="right"); ax1.grid(axis="y",alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
        for (n,traj),col in zip(gdp12.items(),c12): ax2.plot(yrs12,traj,"o-",label=n[:12],color=col,lw=2,markersize=5)
        ax2.set_title("Quỹ đạo GDP",fontweight="bold"); ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3); ax2.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); show_fig(fig)
        st.dataframe(pd.DataFrame([{"Kịch bản":n,"GDP 2030":f"{gdp12[n][-1]:,.0f}","TB/năm":f"{((gdp12[n][-1]/gdp12[n][0])**(1/4)-1)*100:.2f}%"} for n in scen12]), use_container_width=True)
        best12=list(scen12.keys())[np.argmax(g2030)]
        st.info(f"💡 Kịch bản GDP cao nhất: **{best12}** — {max(g2030):,.0f} nghìn tỷ VND")

    # ── TAB 4: RỦI RO ──────────────────────────────
    with tab4:
        st.markdown('<div class="sec-title">M6 — Bản đồ rủi ro & Khuyến nghị</div>', unsafe_allow_html=True)
        df_risk=pd.DataFrame({"Loại rủi ro":["An ninh mạng","Phụ thuộc CN","Khoảng cách số","Thất nghiệp","Phát thải","Địa chính trị"],
                              "Mức độ (1-10)":[7,6,8,5,6,7],"Xác suất (%)":  [35,40,60,45,50,30]})
        df_risk["Chỉ số"]=(df_risk["Mức độ (1-10)"]*df_risk["Xác suất (%)"]/10).round(1)
        st.dataframe(df_risk, use_container_width=True)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
        rc=["#ef5350" if v>=5 else "#ffa726" for v in df_risk["Mức độ (1-10)"]]
        ax1.barh(df_risk["Loại rủi ro"],df_risk["Chỉ số"],color=rc,edgecolor="white")
        ax1.axvline(25,color="gray",ls="--",label="Ngưỡng cảnh báo"); ax1.legend(); ax1.set_title("Chỉ số rủi ro tổng hợp",fontweight="bold"); ax1.grid(axis="x",alpha=0.3); ax1.spines[["top","right"]].set_visible(False)
        angles=np.linspace(0,2*np.pi,6,endpoint=False).tolist(); angles+=angles[:1]; vals=df_risk["Mức độ (1-10)"].tolist(); vals+=vals[:1]
        ax2=plt.subplot(1,2,2,projection="polar"); ax2.plot(angles,vals,color="#1976d2",lw=2); ax2.fill(angles,vals,alpha=0.2,color="#1976d2")
        ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(["Mạng","CN","Số","Việc làm","Phát thải","ĐCT"],fontsize=8); ax2.set_title("Radar rủi ro",fontweight="bold",pad=15)
        fig.tight_layout(); show_fig(fig)
        st.markdown('<div class="sec-title">📋 Khuyến nghị chính sách</div>', unsafe_allow_html=True)
        for title,desc,color in [
            ("🎯 Ưu tiên S5 — Tối ưu cân bằng","AI 30% + Nhân lực 20% + Hạ tầng 25% + CĐS 25%. Tránh cực đoan một chiều.","#1976d2"),
            ("🛡️ An ninh số bắt buộc","Tối thiểu 5% ngân sách ICT cho SOC quốc gia và bảo mật dữ liệu.","#7b1fa2"),
            ("📚 Nhân lực số đi trước","Tăng H ≥25% để hấp thụ AI, tránh thất nghiệp ròng ngành chế biến (risk=42%).","#2e7d32"),
            ("⚖️ Công bằng vùng miền","Duy trì sàn/trần ngân sách để thu hẹp khoảng cách số Tây Nguyên & Miền núi.","#e65100")]:
            st.markdown(f'<div style="border-left:4px solid {color};background:#f8fbff;border-radius:8px;padding:12px 16px;margin-bottom:8px"><div style="font-weight:700;color:{color};font-size:0.9rem">{title}</div><div style="color:#37474f;font-size:0.82rem;margin-top:3px">{desc}</div></div>', unsafe_allow_html=True)
