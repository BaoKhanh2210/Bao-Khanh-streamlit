# AIDEOM-VN — Mô hình ra quyết định phát triển kinh tế Việt Nam

**Họ và tên:** Nguyễn Bảo Khánh · **Mã sinh viên:** 23051266
**Bài tập lớn:** Các mô hình ra quyết định

## Chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```
App tự sinh 3 file CSV dữ liệu VN 2020-2025 nếu chưa có, nên chạy được ngay.
(Nếu có file CSV gốc, đặt cùng thư mục để dùng số liệu thật.)

## Cấu trúc
- **Trang chủ** — KPI 2025, 12 bài theo 4 cấp độ, dữ liệu gốc (vĩ mô / 10 ngành / 6 vùng)
- **Bài 1-11** — mỗi bài là 1 trang, lời giải mô hình + biểu đồ Plotly
- **Bài 12** — AIDEOM-VN tích hợp 6 module chia thành **4 tab**:
  - Tổng quan (M1 Cobb-Douglas + M2 TOPSIS)
  - Phân bổ (M3 LP ngành-vùng)
  - 5 Kịch bản (M6 so sánh GDP 2030)
  - Cảnh báo rủi ro (M4 lao động + M5 Stochastic)

Nội dung tính toán bám sát notebook `bai_tap_cuoi_ki.ipynb`.
