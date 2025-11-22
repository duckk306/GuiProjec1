# gui_project1.py
# File path: gui_project1.py
"""
Streamlit app:
- Dự đoán giá xe máy (file upload + manual input)
- Phát hiện giá bất thường (file upload + manual check)
Requirements:
- utils_clean_data.clean_motobike_data
- utils_anomaly.run_price_anomaly_detection_with_reason
- model_randomforest.pkl (pipeline chứa preprocessing + model)
- xe_may_cu.jpg
"""

from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# local utils (ensure these files are in same folder)
from utils_clean_data import clean_motobike_data
from utils_anomaly import run_price_anomaly_detection_with_reason

# ================== CONFIG ==================
st.set_page_config(page_title="Dự đoán giá & Phát hiện giá bất thường - Xe máy cũ", layout="centered")
st.image("xe_may_cu.jpg", use_container_width=True)
st.title("🔮 Dự đoán giá & Phát hiện giá bất thường — Xe máy cũ")
st.markdown("Upload file  hoặc nhập tay để dự đoán.")

# ================== Feature lists ==================
num_cols = ['price_min', 'price_max', 'year_reg', 'km_driven', 'cc_numeric', 'price_segment_code', 'age']
flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu", "is_bao_duong", "is_do_ben", "is_phap_ly"]
cat_cols = ["brand", "vehicle_type", "model", "origin", "segment",'engine_size']

# Default brands fallback (in case user doesn't upload a dataset)
BRANDS = ['Aprilia','Bmw','Bazan','Benelli','Brixton','Cr&S','Daelim','Detech','Ducati','Gpx','Halim',
          'Harley Davidson','Honda','Hyosung','Hãng Khác','Ktm','Kawasaki','Keeway','Kengo','Kymco',
          'Moto Guzzi','Nioshima','Peugeot','Piaggio','Rebelusa','Royal Enfield','Sym','Sachs','Sanda',
          'Suzuki','Taya','Triumph','Vento','Victory','Vinfast','Visitor','Yamaha']

# ================== Helpers ==================
@st.cache_resource
def load_pipeline(path="model_randomforest.pkl"):
    try:
        p = joblib.load(path)
        return p
    except Exception as e:
        st.error(f"Không load được model từ `{path}`: {e}")
        return None

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def safe_prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has all required columns and correct dtypes for pipeline.predict"""
    dfc = df.copy()
    # ensure columns exist
    for c in num_cols + flag_cols + cat_cols:
        if c not in dfc.columns:
            if c in flag_cols:
                dfc[c] = 0
            elif c in num_cols:
                dfc[c] = 0.0
            else:
                dfc[c] = ""
    # numeric conversions
    for n in ["km_driven", "cc_numeric", "age", "price_segment_code"]:
        if n in dfc.columns:
            dfc[n] = pd.to_numeric(dfc[n], errors="coerce").fillna(0.0)
    for f in flag_cols:
        if f in dfc.columns:
            # convert truthy to 1/0
            dfc[f] = dfc[f].apply(lambda x: 1 if (str(x) in ["1","True","true","True ","Yes","yes","Có","1.0"] or x==1 or x is True) else 0).astype(int)
    # keep original indexing
    return dfc

def style_prediction_table(df: pd.DataFrame):
    """
    Trả về DataFrame đã thêm cột highlight_color để hiển thị thủ công.
    Không sử dụng pandas Styler để tránh lỗi Jinja2.
    """
    out = df.copy()

    # highlight theo residual, tránh dùng .style
    if {"price", "price_pred"}.issubset(out.columns):
        out["residual_pct"] = ((out["price"] - out["price_pred"]) / (out["price"] + 1e-9)).abs()
        out["highlight_color"] = out["residual_pct"].apply(
            lambda v: "#ffcccc" if v > 0.3 else ("#fff2cc" if v > 0.15 else "")
        )
        out.drop(columns=["residual_pct"], inplace=True)

    return out

# ================== Load pipeline ==================
pipeline = load_pipeline("model_randomforest.pkl")

# ================== Menu ==================
menu = ["Home", "Dự đoán giá xe máy", "Phát hiện xe máy bất thường", "Thông tin tác giả"]
choice = st.sidebar.selectbox("📌 MENU", menu)

# Keep last cleaned dataset in session_state (used for dropdown options)
if "last_clean" not in st.session_state:
    st.session_state["last_clean"] = None

# ------------------ PAGES ------------------
if choice == "Home":
    st.header("🏠 Home")
    st.write("""
    ✔ Dự đoán giá xe dựa trên RandomForest  
    ✔ Phát hiện xe đăng bán với giá bất thường  
    ✔ Tự động phân tích mô tả & phát hiện dấu hiệu đáng ngờ  
    ✔ Hỗ trợ file và cả nhập tay  

    👉 Chọn menu bên trái để bắt đầu!
    """)
    st.write("Mẹo: upload file Excel để lấy dropdown tự động cho nhập tay.")

# ------------------ PREDICTION PAGE ------------------
elif choice == "Dự đoán giá xe máy":
    st.header("📈 Dự đoán giá xe máy")

    st.subheader("A. Dự đoán từ file `data_motobikes.xlsx`")
    uploaded_file = st.file_uploader("Tải file Excel/CSV (data_motobikes.xlsx)", type=["xlsx","csv"], key="pred_file")
    if uploaded_file is not None:
        try:
            if str(uploaded_file.name).lower().endswith(".csv"):
                raw = pd.read_csv(uploaded_file)
            else:
                raw = pd.read_excel(uploaded_file)
            st.success(f"Đã đọc file: {uploaded_file.name} — {raw.shape[0]} hàng")
            with st.spinner("Tiền xử lý dữ liệu..."):
                data_clean = clean_motobike_data(raw)
                # ensure age float to avoid future warnings
                if "age" in data_clean.columns:
                    data_clean["age"] = data_clean["age"].astype(float, errors="ignore")
                st.session_state["last_clean"] = data_clean.copy()
            st.write("Kích thước sau khi clean:", data_clean.shape)
            # Prepare X and predict
            X_df = safe_prepare_X(data_clean)
            X = X_df[num_cols + flag_cols + cat_cols]
            if pipeline is None:
                st.error("Model chưa được load (model_randomforest.pkl).")
            else:
                try:
                    preds = pipeline.predict(X)
                    data_clean = data_clean.copy()
                    data_clean["price_pred"] = np.round(preds, 2)
                    # show top 10 (use highest residual or simply first 10)
                    show_cols = [c for c in ["brand", "model", "year_reg", "km_driven", "cc_numeric", "price", "price_pred"] if c in data_clean.columns]
                    st.subheader("Top 10 bản ghi (kèm dự đoán giá)")
                    df_show = data_clean[show_cols].head(10).reset_index(drop=True)
                    st.dataframe(df_show)
                    # nicer styled table (with gradient/residual highlight)
                    styled_df = style_prediction_table(data_clean[show_cols].head(10))
                    st.markdown("### Bảng mẫu với highlight sai số dự đoán")
                    st.dataframe(styled_df)
                    # download
                    st.download_button("⬇️ Tải toàn bộ kết quả dự đoán (CSV)", df_to_csv_bytes(data_clean), file_name="predictions_motobikes.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
        except Exception as e:
            st.error(f"Lỗi khi đọc/tiền xử lý file: {e}")

    st.markdown("---")
    st.subheader("B. Nhập tay để gợi ý giá")
    # Dropdown options: prefer values from uploaded+clean dataset
    last = st.session_state.get("last_clean")
    price = st.number_input("Giá mong muốn (triệu VND)", min_value=0.0, value=10.0, step=0.1)
    price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1)
    price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1)
    brands_opts = sorted(last["brand"].dropna().unique().tolist()) if last is not None and "brand" in last.columns else BRANDS
    models_opts = sorted(last["model"].dropna().unique().tolist()) if last is not None and "model" in last.columns else ["Wave","Exciter","Sirius"]
    vehicle_types_opts = sorted(last["vehicle_type"].dropna().unique().tolist()) if last is not None and "vehicle_type" in last.columns else ["Xe số","Xe tay ga","Xe côn"]
    origin_opts = sorted(last["origin"].dropna().unique().tolist()) if last is not None and "origin" in last.columns else ["Việt Nam","Nhập Khẩu"]
    segment_opts = sorted(last["segment"].dropna().unique().tolist()) if last is not None and "segment" in last.columns else ["Phổ thông","Cận cao cấp","Cao cấp"]
    engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2)
    col1, col2 = st.columns(2)
    with col1:
        brand_inp = st.selectbox("Thương hiệu (brand)", options=brands_opts)
        model_inp = st.selectbox("Dòng xe (model)", options=models_opts)
        vehicle_type_inp = st.selectbox("Loại xe (vehicle_type)", options=vehicle_types_opts)
    with col2:
        km_driven = st.number_input("Số Km đã đi (km_driven)", min_value=0, step=1, value=1000)
        cc_numeric = st.number_input("Dung tích numeric (cc_numeric)", min_value=0, step=1, value=137)
        age = st.number_input("Tuổi xe (age)", min_value=0.1, step=0.1, value=3.0, format="%.1f")

    # flags 2x3
    st.markdown("**Tình trạng (Tick = Có / Không = Không)**")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        is_moi = st.checkbox("Xe mới", value=False)
    with r1c2:
        is_do_xe = st.checkbox("Có độ xe", value=False)
    with r1c3:
        is_su_dung_nhieu = st.checkbox("iXe đi nhiều", value=False)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        is_bao_duong = st.checkbox("Xe có bão dưỡng", value=False)
    with r2c2:
        is_do_ben = st.checkbox("Xe máy còn êm", value=False)
    with r2c3:
        is_phap_ly = st.checkbox("Pháp lý rõ ràng", value=True)
    origin_inp = st.selectbox("Xuất xứ (origin)", options=origin_opts)
    segment_inp = st.selectbox("Phân khúc (segment)", options=segment_opts)
    segment_map = {
    "Phổ Thông": 1,
    "Tầm Trung": 2,
    "Cao Cấp": 3,}
    price_segment_code = segment_map.get(segment_inp, 1) 
    suggestion_type = st.radio("Chọn loại gợi ý", ("Gợi ý giá bán", "Gợi ý giá mua hợp lý"))
    
    if st.button("🔍 Dự đoán / Gợi ý"):
        
        row = {
            "price": price,
            "price_min": price_min,
            "price_max": price_max,
            "km_driven": km_driven,
            "engine_size": engine_size_sel,
            "cc_numeric": cc_numeric,
            "age": age,
            "year_reg": 2025 - age,
            "price_segment_code": price_segment_code,
            "is_moi": int(is_moi),
            "is_do_xe": int(is_do_xe),
            "is_su_dung_nhieu": int(is_su_dung_nhieu),
            "is_bao_duong": int(is_bao_duong),
            "is_do_ben": int(is_do_ben),
            "is_phap_ly": int(is_phap_ly),
            "brand": brand_inp,
            "vehicle_type": vehicle_type_inp,
            "model": model_inp,
            "origin": origin_inp,
            "segment": segment_inp
        }
        df_row = pd.DataFrame([row])
        df_row_prep = safe_prepare_X(df_row)
        X_row = df_row_prep[num_cols + flag_cols + cat_cols]
        if pipeline is None:
            st.error("Model chưa được load (model_randomforest.pkl).")
        else:
            try:
                pred = float(pipeline.predict(X_row)[0])
                if suggestion_type == "Gợi ý giá bán":
                    st.success(f"📦 Gợi ý giá bán: **{pred:,.2f} triệu VND**")
                    st.info(f"Khoảng tham khảo: {pred*0.95:,.2f} — {pred*1.05:,.2f} triệu")
                else:
                    buy_price = pred * 0.92
                    st.success(f"🛒 Gợi ý giá mua hợp lý: **{buy_price:,.2f} triệu VND**")
                    st.info(f"(Giá model dự đoán = {pred:,.2f} triệu)")
                st.download_button("⬇️ Tải kết quả (CSV)", df_to_csv_bytes(df_row), file_name="suggestion_single.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

# ------------------ ANOMALY PAGE ------------------
elif choice == "Phát hiện xe máy bất thường":
    st.header("🚨 Phát hiện xe máy bất thường")

    st.subheader("A. Phát hiện từ file `data_motobikes.xlsx`")
    uploaded_file2 = st.file_uploader("Tải file Excel/CSV để kiểm tra bất thường", type=["xlsx","csv"], key="anomaly_file")
    if uploaded_file2 is not None:
        try:
            if str(uploaded_file2.name).lower().endswith(".csv"):
                raw2 = pd.read_csv(uploaded_file2)
            else:
                raw2 = pd.read_excel(uploaded_file2)
            st.success(f"Đã đọc file: {uploaded_file2.name} — {raw2.shape[0]} hàng")
            with st.spinner("Tiền xử lý dữ liệu..."):
                data_clean2 = clean_motobike_data(raw2)
                if "age" in data_clean2.columns:
                    data_clean2["age"] = data_clean2["age"].astype(float, errors="ignore")
                st.session_state["last_clean"] = data_clean2.copy()
            st.write("Kích thước sau khi clean:", data_clean2.shape)
            X_df2 = safe_prepare_X(data_clean2)
            X2 = X_df2[num_cols + flag_cols + cat_cols]
            if pipeline is None:
                st.error("Model chưa được load (model_randomforest.pkl).")
            else:
                try:
                    data_clean2 = data_clean2.copy()
                    data_clean2["price_pred_final"] = np.round(pipeline.predict(X2), 2)
                    with st.spinner("Chạy thuật toán phát hiện bất thường..."):
                        result_df = run_price_anomaly_detection_with_reason(
                            data=data_clean2,
                            trained_model=pipeline,
                            num_cols=num_cols,
                            flag_cols=flag_cols,
                            cat_cols=cat_cols,
                            seg_col="price_segment_code",
                            k=0.05
                        )
                    anomalies = result_df[result_df["anomaly_reason"] != "Không có dấu hiệu bất thường"].copy()
                    if anomalies.empty:
                        st.info("Không tìm thấy bản ghi bất thường trong file.")
                    else:
                        anomalies_sorted = anomalies.sort_values(by="anomaly_score", ascending=False)
                        st.subheader("Top 10 bản ghi bất thường")
                        show_cols = [c for c in ["brand", "model", "year_reg", "km_driven", "price", "price_pred_final", "anomaly_score", "anomaly_reason", "anomaly_level"] if c in anomalies_sorted.columns]
                        anomaly_view = anomalies_sorted[show_cols].head(10).reset_index(drop=True)
                        st.dataframe(anomaly_view)
                        st.download_button("⬇️ Tải kết quả bất thường (CSV)", df_to_csv_bytes(anomalies_sorted), file_name="anomalies_motobikes.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán/check anomaly: {e}")
        except Exception as e:
            st.error(f"Lỗi khi đọc/tiền xử lý file: {e}")

    st.markdown("---")
    st.subheader("B. Nhập tay để kiểm tra 1 xe")
    # Manual inputs for anomaly check
    price = st.number_input("Giá (triệu VND)", min_value=0.0, value=10.0, step=0.1)
    price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1)
    price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1)

    # Dropdown options from last_clean if present
    last = st.session_state.get("last_clean")
    brands_opts = sorted(last["brand"].dropna().unique().tolist()) if last is not None and "brand" in last.columns else BRANDS
    brand_sel = st.selectbox("Thương hiệu", options=brands_opts)
    model_sel = st.text_input("Dòng xe (Dòng xe)", value="Wave")
    year_reg = st.number_input("Năm đăng ký", min_value=1900, max_value=2025, value=2020, step=1)
    if 2025 - year_reg == 0:
        age = 0.5
    else:
        age = 2025 - year_reg
    km_driven_an = st.number_input("Số Km đã đi", min_value=0, value=5000, step=1)
    vehicle_type_sel = st.text_input("Loại xe", value="Xe số")
    engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2)
    origin_sel = st.selectbox("Xuất xứ", options=["Việt Nam","Nhập Khẩu"])
    segment_sel = st.selectbox("Phân khúc giá", options=["Phổ thông","Cận cao cấp","Cao cấp"])
    segment_map = {
    "Phổ Thông": 1,
    "Tầm Trung": 2,
    "Cao Cấp": 3,}
    price_segment_code = segment_map.get(segment_sel, 1) 
    # flags 2x3
    st.markdown("**Tình trạng (Tick = Có / Không = Không)**")
    a1, a2, a3 = st.columns(3)
    with a1:
        an_is_moi = st.checkbox("is_moi", value=False, key="an_is_moi")
    with a2:
        an_is_do_xe = st.checkbox("is_do_xe", value=False, key="an_is_do_xe")
    with a3:
        an_is_su_dung_nhieu = st.checkbox("is_su_dung_nhieu", value=False, key="an_is_su_dung_nhieu")
    b1, b2, b3 = st.columns(3)
    with b1:
        an_is_bao_duong = st.checkbox("is_bao_duong", value=False, key="an_is_bao_duong")
    with b2:
        an_is_do_ben = st.checkbox("is_do_ben", value=False, key="an_is_do_ben")
    with b3:
        an_is_phap_ly = st.checkbox("is_phap_ly", value=True, key="an_is_phap_ly")

    if st.button("Kiểm tra"):
        row = {
            "price": price,
            "price_min": price_min,
            "price_max": price_max,
            "brand": brand_sel,
            "model": model_sel,
            "year_reg": year_reg,
            "age": age,
            "km_driven": km_driven_an,
            "vehicle_type": vehicle_type_sel,
            "engine_size": engine_size_sel,
            "cc_numeric": 137,
            "origin": origin_sel,
            "segment": segment_sel,
            "is_moi": int(an_is_moi),
            "is_do_xe": int(an_is_do_xe),
            "is_su_dung_nhieu": int(an_is_su_dung_nhieu),
            "is_bao_duong": int(an_is_bao_duong),
            "is_do_ben": int(an_is_do_ben),
            "is_phap_ly": int(an_is_phap_ly),
            "price_segment_code": price_segment_code
        }
        df_row = pd.DataFrame([row])
        df_row_prep = safe_prepare_X(df_row)
        if pipeline is None:
            st.error("Model chưa được load (model_randomforest.pkl).")
        else:
            try:
                df_row_prep["price_pred_final"] = pipeline.predict(df_row_prep[num_cols + flag_cols + cat_cols])
                res = run_price_anomaly_detection_with_reason(
                    data=df_row_prep,
                    trained_model=pipeline,
                    num_cols=num_cols,
                    flag_cols=flag_cols,
                    cat_cols=cat_cols,
                    seg_col="price_segment_code",
                    k=0.05
                )
                st.markdown("### Kết quả kiểm tra")
                st.write("**Anomaly reason:**", res.loc[0, "anomaly_reason"])
                st.write("**Anomaly level:**", res.loc[0, "anomaly_level"])
                st.download_button("⬇️ Tải kết quả kiểm tra (CSV)", df_to_csv_bytes(df_row), file_name="anomaly_check_single.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi khi kiểm tra bất thường: {e}")

# ------------------ AUTHOR PAGE ------------------
elif choice == "Thông tin tác giả":
    st.header("👤 Nhóm tác giả dự án")
    st.write("""
    **Hồ Thị Quỳnh Như**  
    **Nguyễn Văn Cường**  
    **Nguyễn Thị Tuyết Anh**  
    """)
