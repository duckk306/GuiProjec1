# gui_project1.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random

# ==============================
#  IMPORT CÁC HÀM CLEAN + ANOMALY
# ==============================

from utils_clean_data import clean_motobike_data 
from utils_anomaly import run_price_anomaly_detection_with_reason
# ==============================
#  KHAI BÁO CỘT DÙNG CHUNG
# ==============================

num_cols = ["km_driven", "cc_numeric", "age", "price_segment_code"]

flag_cols = [
    "is_moi",
    "is_do_xe",
    "is_su_dung_nhieu",
    "is_bao_duong",
    "is_do_ben",
    "is_phap_ly"
]

cat_cols = ["brand", "vehicle_type", "model", "origin", "segment"]

# ==============================
#  CONFIG GIAO DIỆN
# ==============================
st.set_page_config(
    page_title="Dự đoán & Phát hiện giá xe máy",
    layout="centered",
)

st.image("xe_may_cu.jpg", use_container_width=True)
st.title("🔮 Dự đoán giá xe máy")
st.markdown("Ứng dụng có thể đọc `data_motobikes.xlsx` và xử lý đầy đủ.")


# ==============================
#  MENU BÊN PHẢI
# ==============================
menu = ["Home", "Dự đoán giá xe máy", "Phát hiện xe máy bất thường", "Thông tin tác giả"]
choice = st.sidebar.selectbox("📌 MENU", menu)


# ==============================
#  HÀM LOAD MODEL
# ==============================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return joblib.load(f)


# ============================================================
#  1️⃣ HOME
# ============================================================
if choice == "Home":
    st.header("🏍️ Hệ thống dự đoán & phát hiện giá xe máy bất thường")
    st.write("""
    ✔ Dự đoán giá xe dựa trên RandomForest  
    ✔ Phát hiện xe đăng bán với giá bất thường  
    ✔ Tự động phân tích mô tả & phát hiện dấu hiệu đáng ngờ  
    ✔ Hỗ trợ file CSV và cả nhập tay  

    👉 Chọn menu bên phải để bắt đầu!
    """)


# ============================================================
#  2️⃣ DỰ ĐOÁN GIÁ XE MÁY
# ============================================================
elif choice == "Dự đoán giá xe máy":
    st.header("📈 Dự đoán giá xe máy bằng mô hình RandomForest")

    st.subheader("1️⃣ Upload file dữ liệu để dự đoán toàn bộ dataset")

    uploaded = st.file_uploader("Tải lên file data_motobikes.xlsx", type=["xlsx"])

    if uploaded:
        df_raw = pd.read_excel(uploaded)
        st.success("✔ Đã đọc file Excel thành công!")

        df_clean = clean_motobike_data(df_raw)

        st.info(f"📊 Dữ liệu sau khi clean: {df_clean.shape[0]} dòng")

        # load model
        model_path = "best_model_randomforest.pkl"
        model = load_model(model_path)

        # Các cột X
        #num_cols = ["km_driven", "cc_numeric", "age", "price_segment_code"]
        #flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu",
             #"is_bao_duong", "is_do_ben", "is_phap_ly"]
        #cat_cols = ["brand", "vehicle_type", "model", "origin", "segment"]
        rename_map = {"Số Km đã đi": "km_driven",
                        "Thương hiệu": "brand",
                        "Dòng xe": "model",
                        "Loại xe": "vehicle_type",
                        "Xuất xứ": "origin",
                         "Phân khúc giá": "segment",
                        }

        df_clean = df_clean.rename(columns=rename_map)
        df_clean["price_segment_code"] = df_clean["segment"].astype("category").cat.codes

        X = df_clean[num_cols + flag_cols + cat_cols]
        y_pred = model.predict(X)

        df_clean["Giá_dự_đoán"] = y_pred

        st.subheader("📌 10 kết quả dự đoán ngẫu nhiên")

        df_sample = df_clean.sample(10, random_state=42)[
            ["brand", "model", "Giá", "Giá_dự_đoán", "km_driven", "age"]
        ]

        st.dataframe(df_sample,column_config={
        "Giá": st.column_config.NumberColumn(format="%.1f"),
        "Giá_dự_đoán": st.column_config.NumberColumn(format="%.1f"),})


    st.subheader("2️⃣ Người dùng tự nhập thông tin xe")

    with st.form("predict_form"):

        brand = st.text_input("Thương hiệu")
        model_name = st.text_input("Dòng xe")
        year = st.number_input("Năm đăng ký", min_value=1980, max_value=2025, value=2020)
        km = st.number_input("Số Km đã đi", min_value=0, value=10000)
        vehicle_type = st.selectbox("Loại xe", ["Tay ga", "Xe số"])
        price_min = st.number_input("Khoảng giá min (triệu)", 0.0, 2000.0, 20.0)
        price_max = st.number_input("Khoảng giá max (triệu)", 0.0, 3000.0, 30.0)

        submit = st.form_submit_button("🚀 Dự đoán giá bán")

    if submit:
        if brand == "" or model_name == "":
            st.error("⚠ Vui lòng nhập đầy đủ Thương hiệu và Dòng xe")
        else:
            # Tạo bản ghi nhập tay
            single = pd.DataFrame([{
                "Thương hiệu": brand.title(),
                "Dòng xe": model_name.title(),
                "Loại xe": vehicle_type.title(),
                "Năm đăng ký": year,
                "Số Km đã đi": km,
                "Giá": np.nan,     # user không nhập
                "Khoảng giá min": price_min,
                "Khoảng giá max": price_max,
                "Mô tả chi tiết": "",
                "Dung tích xe": "100 - 175",
                "Xuất xứ": "Việt Nam",
                "Phân khúc giá": "Phổ thông",
            }])

            df_single = clean_motobike_data(single)

            model = load_model("best_model_randomforest.pkl")

            X_single = df_single[num_cols + flag_cols + cat_cols]
            pred = model.predict(X_single)[0]

            st.success(f"💰 **Giá bán xe gợi ý: {pred:.2f} triệu**")
            st.info(f"💵 Giá mua vào gợi ý: {(pred * 0.92):.2f} triệu")


# ============================================================
#  3️⃣ PHÁT HIỆN XE MÁY BẤT THƯỜNG
# ============================================================
elif choice == "Phát hiện xe máy bất thường":

    st.header("🚨 Phát hiện xe đăng bán bất thường")

    uploaded2 = st.file_uploader("Tải lên file CSV để kiểm tra bất thường", type=["xlsx"])

    if uploaded2:
        df_raw = pd.read_excel(uploaded2)
        df_clean = clean_motobike_data(df_raw)

        model = load_model("best_model_randomforest.pkl")

        num_cols = ["Giá", "Khoảng giá min", "Khoảng giá max", "Số Km đã đi", "age", "cc_numeric"]
        flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu", "is_bao_duong", "is_do_ben", "is_phap_ly"]
        #cat_cols = ["Thương hiệu", "Loại xe", "Dòng xe", "Xuất xứ", "Phân khúc giá"]
        cat_cols = ["brand", "vehicle_type", "model", "origin", "segment"]

        df_detect = run_price_anomaly_detection_with_reason(
            df_clean, model,
            num_cols=num_cols, flag_cols=flag_cols, cat_cols=cat_cols
        )

        st.subheader("📌 Kết quả phát hiện bất thường")

        df_detect["color"] = df_detect["highlight_style"]

        st.dataframe(df_detect,column_config={
        "color": st.column_config.TextColumn("Cảnh báo (color code)"),})


# ============================================================
#  4️⃣ THÔNG TIN TÁC GIẢ
# ============================================================
elif choice == "Thông tin tác giả":
    st.header("👤 Nhóm tác giả dự án")

    st.write("""
    **Hồ Thị Quỳnh Như**  
    **Nguyễn Văn Cường**  
    **Nguyễn Thị Tuyết Anh**  
    """)

