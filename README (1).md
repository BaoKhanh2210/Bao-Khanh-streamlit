# AIDEOM-VN · Analytics Terminal

Nội dung & số liệu của bộ 12 bài tập Mô hình ra quyết định, trình bày theo phong
cách **command-center tối** với accent neon: mục lục **sidebar bên trái**, lưới nền
kỹ thuật và phông Space Grotesk / JetBrains Mono. Mặc định nền tối, có nút ☀/☾ để
chuyển sáng/tối.

## Chạy
```bash
cd aideom_terminal
python -m http.server 5000        # cách 1: không cần cài gì
# hoặc
pip install -r requirements.txt && python app.py   # cách 2: Flask
```
Mở http://127.0.0.1:5000 — hoặc nhấp đúp `index.html`.

## Điều hướng
Trang chủ `#home`, từng bài `#bai1` … `#bai12`.

Mọi tính toán & biểu đồ chạy bằng JavaScript trong trình duyệt (Chart.js + KaTeX
qua CDN), không cần backend. Deploy: chỉ cần `index.html`.
