# AIDEOM-VN — Web app 12 bài toán mô hình ra quyết định

**Bài tập lớn: Các mô hình ra quyết định**
Họ và tên: Nguyễn Đình Bảo Nghĩa · Mã sinh viên: 23052345

Web app Streamlit giải 12 bài toán phát triển kinh tế Việt Nam trong kỉ nguyên AI
trên dữ liệu thực tế 2020-2025.

## Cài đặt & chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```
App mở tại http://localhost:8501

## Điều hướng
Chọn **cấp độ** rồi chọn **bài** ở thanh trên cùng. **Mỗi bài gồm 5 trang con:**
🧭 Bối cảnh · 📐 Mô hình · 🗃️ Dữ liệu · 🧮 **Tính toán** · 🏛️ Chính sách.
Trang *Tính toán* trình bày các bước giải, bảng số đầy đủ, biểu đồ và phần
diễn giải kết quả.

## Nội dung
| Cấp độ | Bài | Kỹ thuật |
|--------|-----|----------|
| Dễ | 1–3 | Cobb-Douglas, LP scipy, MCDM weighted scoring |
| Trung bình | 4–6 | LP PuLP/CVXPY, MIP knapsack, TOPSIS + entropy |
| Khá khó | 7–9 | NSGA-II Pareto, tối ưu động SLSQP, LP NetJob |
| Khó | 10–12 | Stochastic 2-stage (Pyomo, VSS/EVPI), Q-learning, AIDEOM tích hợp |

**Bài 12** tích hợp 6 module (M1 dự báo · M2 TOPSIS · M3 phân bổ · M4 lao động ·
M5 đa mục tiêu & rủi ro · M6 dashboard) trình bày trong **4 tab** tại trang Tính toán.

## Tệp dữ liệu (đặt cùng thư mục app.py hoặc trong data/)
vietnam_macro_2020_2025.csv · vietnam_sectors_2024.csv · vietnam_regions_2024.csv
