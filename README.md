# Chợ Xe Máy Cũ --- Ứng dụng dự đoán giá & duyệt tin

Ứng dụng **Streamlit** hỗ trợ:

-   🔮 **Dự đoán giá xe máy cũ**
-   🚨 **Phát hiện xe có giá bất thường**
-   📢 **Đăng tin bán / mua xe**
-   🔧 **Quản trị viên duyệt tin**
-   👤 **Trang thông tin tác giả**

Toàn bộ ứng dụng chạy bằng 1 file duy nhất `gui_project1.py`.
## ⚙️ Yêu cầu môi trường

### 🧩 Cài đặt thư viện cần thiết
```bash
pip install pandas numpy scikit-learn lightgbm xgboost pyspark matplotlib seaborn joblib
```

Hoặc, nếu bạn dùng `requirements.txt`:
```bash
pip install -r requirements.txt
```
▶️ 2. Chạy ứng dụng

Từ thư mục chứa gui_project1.py, chạy:
```bash
streamlit run gui_project1.py
```

Ứng dụng sẽ mở tại:

https://guiprojec1-5t2i6nz849hazftzu4d79x.streamlit.app/

🗂 3. Cấu trúc file dữ liệu (tự động tạo khi chạy)

Ứng dụng sẽ tự tạo các file Excel sau:

`posts_sell.xlsx`
`posts_buy.xlsx`
`approved_posts_for_sale.xlsx`
`approved_posts_for_buy.xlsx`
`rejected_posts.xlsx`


Model machine learning:

`model_randomforest.pkl`


Dữ liệu mẫu để load model:

`data_motobikes.xlsx`

🧠 4. Các chức năng chính
🔮 1. Dự đoán giá xe máy

   🔹 Người dùng chọn thông số xe (hãng, dòng xe, km đã đi, tình trạng…)

   🔹 Model RandomForest dự đoán giá thị trường

   🔹 Gợi ý:

             🔹 Giá bán hợp lý

             🔹 Giá mua hợp lý

   🔹 Hiển thị cả:

             🔹 Giá dự đoán

             🔹 Khoảng giá gợi ý

   🔹 Risk Score (độ rủi ro)

   🔹 Lý do bất thường (nếu có)

📢 2. Đăng tin bán / đăng tin mua

 🔹 Sau khi dự đoán giá, người dùng có thể:

 🔹 Chọn giá đăng: giá nhập hoặc giá dự đoán

 🔹 Nhập ID người đăng + ghi chú

 🔹 Tin được lưu vào file Excel và nằm trong trạng thái pending.

🔧 3. Quản trị viên duyệt tin (QTV)

     🔹 Có đăng nhập (tài khoản được khai báo trong code)

     🔹 QTV có thể:

                  ✔ Duyệt tin (tự động chuyển sang file approved)

                  ❌ Từ chối tin (chuyển sang file rejected)

      🔹Chọn nhiều dòng một lúc

      🔹 UI sử dụng st.data_editor mới

🚨 4. Phát hiện giá bất thường

       🔹 Kiểm tra:

               🔹 giá quá chênh lệch so với giá dự đoán

               🔹 km không hợp lý so với tuổi xe

               🔹 độ rủi ro theo model

               🔹 dòng xe không phù hợp với hãng

       🔹 Tính mức:

                ⚠ An toàn

                🚧 Đáng chú ý

                🔥 Nguy hiểm

       🔹 Hiển thị lý do chi tiết

👤 5. Trang tác giả

     🔹 Chứa thông tin thành viên nhóm.

🧩 5. Cấu trúc mã nguồn

        **Phần dự đoán giá**
        Chuẩn hóa input → chuẩn hóa features → chạy model → hiển thị kết quả.
        
        **Phần đăng tin**
        Ghi tin đăng vào file Excel → hiển thị trong màn user → chờ QTV duyệt.
        
        **Phần QTV**
        Dùng st.data_editor để cho phép tick chọn → duyệt/từ chối → lưu file tương ứng.
        
        **Phần anomaly**
        Dùng logic trong utils_anomaly.py + risk scoring.

🔑 6. Tài khoản admin mẫu
admin / 123456
qtv1  / password1
qtv2  / abc123

⭐ 7. Đóng góp / Cải tiến (gợi ý)

Thêm API endpoint để nhận tin đăng từ ứng dụng mobile

Thêm upload hình ảnh xe

Thêm logging duyệt tin

Thêm trang hồ sơ người dùng

📬 8. Liên hệ tác giả

Hồ Thị Quỳnh Như  
Nguyễn Văn Cường  
Nguyễn Thị Tuyết Anh
