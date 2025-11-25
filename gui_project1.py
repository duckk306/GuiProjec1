# GUI_Project1_fixed.py
# Fixed Streamlit app with robust AgGrid handling, post_time normalization,
# ensured post_id on load, and fully fixed admin approve/reject flows.
from io import BytesIO
import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# local utils (must be in same folder)
from utils_clean_data import clean_motobike_data
from utils_anomaly import run_price_anomaly_detection_with_reason

# ================== CONFIG & PATHS ==================
st.set_page_config(page_title="Chợ xe máy cũ", layout="centered")
st.image("mua-ban-xe-may-cu-0.png", use_container_width=True)
st.title("🔮 Dự đoán giá & Phát hiện giá bất thường — Xe máy cũ")
st.markdown("Chợ xe máy cũ tích hợp đầy đủ chức năng dự đoán giá xe máy cũ, phát hiện giá bất thường, đăng tin bán/mua.")

DEFAULT_DATA = "data_motobikes.xlsx"
MODEL_PATH = "model_randomforest.pkl"

# Persist as Excel files (user requested Excel)
POSTS_SELL_XLSX = "posts_sell.xlsx"
POSTS_BUY_XLSX = "posts_buy.xlsx"
APPROVED_SELL_XLSX = "approved_posts_for_sale.xlsx"
APPROVED_BUY_XLSX = "approved_posts_for_buy.xlsx"
REJECTED_XLSX = "rejected_posts.xlsx"
QTV_ACCOUNTS = {
    "admin": "123456",
    "qtv1": "password1",
    "qtv2": "abc123"
}
# feature lists
num_cols = ['price_min', 'price_max', 'year_reg', 'km_driven', 'cc_numeric', 'price_segment_code', 'age']
flag_cols = ["is_moi", "is_do_xe", "is_su_dung_nhieu", "is_bao_duong", "is_do_ben", "is_phap_ly"]
cat_cols = ["brand", "vehicle_type", "model", "origin", "segment", 'engine_size']

BRANDS = ['Aprilia','Bmw','Bazan','Benelli','Brixton','Cr&S','Daelim','Detech','Ducati','Gpx','Halim',
          'Harley Davidson','Honda','Hyosung','Hãng Khác','Ktm','Kawasaki','Keeway','Kengo','Kymco',
          'Moto Guzzi','Nioshima','Peugeot','Piaggio','Rebelusa','Royal Enfield','Sym','Sachs','Sanda',
          'Suzuki','Taya','Triumph','Vento','Victory','Vinfast','Visitor','Yamaha']

# ================== HELPERS ==================
@st.cache_resource
def load_pipeline(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Không load được model từ `{path}`: {e}")
        return None

def qtv_login():
    st.subheader("🔐 Đăng nhập quản trị viên (QTV)")

    user = st.text_input("ID QTV:", key="qtv_user")
    pw = st.text_input("Mật khẩu:", type="password", key="qtv_pw")

    if st.button("Đăng nhập", key="qtv_login_btn"):
        if user in QTV_ACCOUNTS and pw == QTV_ACCOUNTS[user]:
            st.session_state["qtv_logged_in"] = True
            st.success("Đăng nhập thành công!")
            st.rerun()
        else:
            st.error("Sai ID hoặc mật khẩu!")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="posts")
        return bio.getvalue()
    except Exception:
        return df_to_csv_bytes(df)


def _read_xlsx_if_exists(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_xlsx(df, path):
    try:
        # Ensure directory exists
        dirp = os.path.dirname(path)
        if dirp and not os.path.exists(dirp):
            os.makedirs(dirp, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu file {path}: {e}")
        return False


def safe_prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    for c in num_cols + flag_cols + cat_cols:
        if c not in dfc.columns:
            if c in flag_cols:
                dfc[c] = 0
            elif c in num_cols:
                dfc[c] = 0.0
            else:
                dfc[c] = ""
    for n in ["km_driven", "cc_numeric", "age", "price_segment_code", "year_reg", "price_min", "price_max"]:
        if n in dfc.columns:
            dfc[n] = pd.to_numeric(dfc[n], errors="coerce").fillna(0.0)
    for f in flag_cols:
        if f in dfc.columns:
            dfc[f] = dfc[f].apply(lambda x: 1 if (str(x).lower() in ["1","true","yes","có","co"]) or x==1 or x is True else 0).astype(int)
    return dfc


def compute_risk_score_strict(row, last_clean_brand_models=None, anomaly_reason=None):
    score = 0.0
    try:
        price = float(row.get("price", 0.0))
        pred = float(row.get("predicted_price", 0.0))
        if pred > 0:
            diff_pct = abs(price - pred) / pred
            score += min(50.0, diff_pct * 100.0 * 0.5)
    except Exception:
        pass
    if anomaly_reason and isinstance(anomaly_reason, str) and anomaly_reason != "Không có dấu hiệu bất thường":
        score += 25.0
    try:
        km = float(row.get("km_driven", 0.0))
        age = float(row.get("age", 0.0))
        if age >= 5 and km < 2000:
            score += 20.0
        elif age >= 8 and km < 5000:
            score += 30.0
    except Exception:
        pass
    try:
        if int(row.get("is_moi", 0)) == 1 and float(row.get("age", 0.0)) > 3:
            score += 10.0
    except Exception:
        pass
    if last_clean_brand_models and isinstance(last_clean_brand_models, dict):
        brand = str(row.get("brand", "")).strip()
        model = str(row.get("model", "")).strip()
        if brand in last_clean_brand_models:
            known_models = last_clean_brand_models.get(brand, [])
            if model and (model not in known_models):
                score += 30.0
    score = min(100.0, score)
    return round(score, 2)


def risk_level_from_score(score):
    if score >= 70:
        return "Nguy hiểm"
    elif score >= 40:
        return "Đáng chú ý"
    else:
        return "An toàn"


def make_post_record(df_row: pd.DataFrame, post_type: str, chosen_price: float, user_id: str = "anonymous", note: str = ""):
    rec = df_row.iloc[0].to_dict()
    rec["post_id"] = str(uuid.uuid4())[:8]
    rec["post_time"] = pd.Timestamp.now()
    rec["post_type"] = post_type
    rec["price_input"] = rec.get("price", np.nan)
    rec["price_pred"] = rec.get("predicted_price", np.nan)
    rec["price_final"] = chosen_price
    rec["status"] = "pending"
    rec["user_id"] = user_id
    rec["note"] = note
    rec["anomaly_reason"] = rec.get("anomaly_reason", "")
    rec["risk_score"] = rec.get("risk_score", np.nan)
    return rec


def ensure_post_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    if "post_id" not in out.columns:
        out["post_id"] = [str(uuid.uuid4())[:8] for _ in range(len(out))]
    # ensure dtype str
    out["post_id"] = out["post_id"].astype(str)
    return out


def normalize_datetime_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    for col in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        except Exception:
            # ignore problematic columns
            pass
    return out


def prepare_for_aggrid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out is None or out.empty:
        return out
    # convert datetime-like columns to string
    out = normalize_datetime_like_columns(out)
    # convert object columns to str (except post_id)
    for col in out.columns:
        if col != "post_id" and out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


def save_post_record(record: dict):
    df_new = pd.DataFrame([record])
    if record.get("post_type") == "sell":
        posts = _read_xlsx_if_exists(POSTS_SELL_XLSX)
        posts = pd.concat([posts, df_new], ignore_index=True)
        posts = ensure_post_id(posts)
        _save_xlsx(posts, POSTS_SELL_XLSX)
        st.session_state["posts_sell"] = posts.copy()
    else:
        posts = _read_xlsx_if_exists(POSTS_BUY_XLSX)
        posts = pd.concat([posts, df_new], ignore_index=True)
        posts = ensure_post_id(posts)
        _save_xlsx(posts, POSTS_BUY_XLSX)
        st.session_state["posts_buy"] = posts.copy()
    st.session_state.setdefault("pending_notifications", [])
    st.session_state["pending_notifications"].append(record.get("post_id"))


def rename_columns_vn(df: pd.DataFrame, mode="general"):
    """
    mode = "sell"  -> Giá bán
    mode = "buy"   -> Giá mua
    mode = "general" -> Giá bán / Giá mua (dùng cho QTV)
    """

    if mode == "sell":
        price_name = "Giá bán"
    elif mode == "buy":
        price_name = "Giá mua"
    else:
        price_name = "Giá bán / Giá mua"

    col_map = {
        "selected": "Chọn",
        "user_id": "ID người dùng",
        "note": "Mô tả",
        "price_final": price_name,
        "year_reg": "Năm đăng ký",
        "km_driven": "Km đã đi",
        "brand": "Hãng xe",
        "model": "Dòng xe",
        "cc_numeric": "Dung tích xe (cc)",
        "origin": "Xuất xứ",
        "vehicle_type": "Loại xe",
    }
    df = df.rename(columns=col_map)
    return df

def reorder_columns(df: pd.DataFrame):
    front_cols = ["selected", "user_id", "note"]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]

# ================== LOAD PIPELINE ==================
pipeline = load_pipeline(MODEL_PATH)

# ================== SESSION STATE & PERSISTENT LOAD ==================
if "last_clean" not in st.session_state:
    st.session_state["last_clean"] = None
if "predicted_df" not in st.session_state:
    st.session_state["predicted_df"] = None
if "last_predict" not in st.session_state:
    st.session_state["last_predict"] = None

# load persisted posts from excel if exist (ensure post_id & normalize datetimes)
if "posts_sell" not in st.session_state:
    posts = _read_xlsx_if_exists(POSTS_SELL_XLSX)
    posts = ensure_post_id(posts)
    posts = normalize_datetime_like_columns(posts)
    st.session_state["posts_sell"] = posts
if "posts_buy" not in st.session_state:
    posts = _read_xlsx_if_exists(POSTS_BUY_XLSX)
    posts = ensure_post_id(posts)
    posts = normalize_datetime_like_columns(posts)
    st.session_state["posts_buy"] = posts
if "pending_notifications" not in st.session_state:
    st.session_state["pending_notifications"] = []

# ================== AUTO LOAD DEFAULT DATA + PREDICT (runs once per session) ==================
if st.session_state.get("predicted_df") is None:
    if os.path.exists(DEFAULT_DATA):
        try:
            raw = pd.read_excel(DEFAULT_DATA)
            data_clean = clean_motobike_data(raw)
            if "age" in data_clean.columns:
                data_clean["age"] = data_clean["age"].astype(float, errors="ignore")
            st.session_state["last_clean"] = data_clean.copy()
            X_df = safe_prepare_X(data_clean)
            feats = [c for c in (num_cols + flag_cols + cat_cols) if c in X_df.columns]
            if pipeline is not None and len(feats) > 0:
                preds = pipeline.predict(X_df[feats])
                data_clean = data_clean.copy()
                data_clean["price_pred"] = np.round(preds, 2)
                st.session_state["predicted_df"] = data_clean
            else:
                st.warning("Model chưa được load hoặc thiếu features; predicted_df không có.")
        except Exception as e:
            st.error(f"Lỗi khi auto load/clean/predict default data: {e}")
    else:
        st.warning(f"Không tìm thấy file mặc định: {DEFAULT_DATA}.")

# ================== MENU ==================
menu = [
    "Home",
    "Dự đoán giá xe máy",
    "Đăng bán",
    "Đăng mua",
    "Phát hiện xe máy bất thường",
    "Duyệt tin (QTV)",
    "Thông tin tác giả"
]
choice = st.sidebar.selectbox("📌 MENU", menu)

# ------------------ PAGES ------------------
if choice == "Home":
    st.header("🏠 Home")
    st.write("""
            ✔ Dự đoán giá xe - Gợi ý giá bán/mua hợp lý
             
            ✔ Cho phép người dùng đăng tin bán/mua xe máy cũ
                      
            ✔ Phát hiện xe đăng bán với giá bất thường
             
            ✔ Tự động phân tích mô tả & phát hiện dấu hiệu đáng ngờ  
    """)

# ------------------ PREDICTION PAGE ------------------
elif choice == "Dự đoán giá xe máy":
    st.header("📈 Dự đoán giá xe máy")

    st.subheader("A. Kết quả model (dữ liệu mẫu tự load)")
    pred_df = st.session_state.get("predicted_df")
    if pred_df is None:
        st.warning("Dữ liệu mẫu chưa được load hoặc model chưa được tính.")
    else:
        if st.button("📄 Hiển thị 10 xe máy mẫu đã được model dự đoán", key="show_sample_10"):
            show_cols = [c for c in ["brand", "model", "year_reg", "km_driven", "cc_numeric", "price", "price_pred"] if c in pred_df.columns]
            st.dataframe(pred_df[show_cols].head(10).reset_index(drop=True))
            st.download_button("⬇️ Tải toàn bộ kết quả dự đoán (Excel)", df_to_excel_bytes(pred_df), file_name="predicted_sample.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.subheader("B. Nhập tay để gợi ý giá (dựa trên model đã load)")
    last = st.session_state.get("last_clean")

    brands_opts = sorted(last["brand"].dropna().unique().tolist()) if last is not None and "brand" in last.columns else BRANDS
    models_opts = sorted(last["model"].dropna().unique().tolist()) if last is not None and "model" in last.columns else ["Wave","Exciter","Sirius"]
    vehicle_types_opts = sorted(last["vehicle_type"].dropna().unique().tolist()) if last is not None and "vehicle_type" in last.columns else ["Xe số","Xe tay ga","Xe côn"]
    origin_opts = sorted(last["origin"].dropna().unique().tolist()) if last is not None and "origin" in last.columns else ["Việt Nam","Nhập Khẩu"]
    segment_opts = sorted(last["segment"].dropna().unique().tolist()) if last is not None and "segment" in last.columns else ["Phổ thông","Cận cao cấp","Cao cấp"]

    price = st.number_input("Giá mong muốn (triệu VND)", min_value=0.0, value=10.0, step=0.1, key="inp_price")
    price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1, key="inp_price_min")
    price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1, key="inp_price_max")
    engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2, key="inp_engine_size")
    col1, col2 = st.columns(2)
    with col1:
        brand_inp = st.selectbox("Thương hiệu (brand)", options=brands_opts, key="inp_brand")
        model_inp = st.selectbox("Dòng xe (model)", options=models_opts, key="inp_model")
        vehicle_type_inp = st.selectbox("Loại xe (vehicle_type)", options=vehicle_types_opts, key="inp_vehicle_type")
    with col2:
        km_driven = st.number_input("Số Km đã đi (km_driven)", min_value=0, step=1, value=1000, key="inp_km")
        cc_numeric = st.number_input("Dung tích numeric (cc_numeric)", min_value=0, step=1, value=137, key="inp_cc")
        age = st.number_input("Tuổi xe (age)", min_value=0.1, step=0.1, value=3.0, format="%.1f", key="inp_age")

    st.markdown("**Tình trạng (Tick = Có / Không = Không)**")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        is_moi = st.checkbox("Xe còn mới", value=False, key="inp_is_moi")
    with r1c2:
        is_do_xe = st.checkbox("Có độ xe", value=False, key="inp_is_do_xe")
    with r1c3:
        is_su_dung_nhieu = st.checkbox("Xe đi nhiều", value=False, key="inp_is_su_dung_nhieu")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        is_bao_duong = st.checkbox("Xe có bảo dưỡng", value=False, key="inp_is_bao_duong")
    with r2c2:
        is_do_ben = st.checkbox("Máy xe còn tốt", value=False, key="inp_is_do_ben")
    with r2c3:
        is_phap_ly = st.checkbox("Pháp lý rõ ràng", value=True, key="inp_is_phap_ly")

    origin_inp = st.selectbox("Xuất xứ (origin)", options=origin_opts, key="inp_origin")
    segment_inp = st.selectbox("Phân khúc (segment)", options=segment_opts, key="inp_segment")
    segment_map = {"Phổ thông": 1, "Cận cao cấp": 2, "Cao cấp": 3}
    price_segment_code = segment_map.get(segment_inp, 1)
    suggestion_type = st.radio("Chọn loại gợi ý", ("Gợi ý giá bán", "Gợi ý giá mua hợp lý"), key="inp_suggestion_type")

    # Save inputs to session so they're not lost on rerun
    st.session_state["inputs_last"] = {
        "price": price, "price_min": price_min, "price_max": price_max,
        "engine_size_sel": engine_size_sel, "brand_inp": brand_inp, "model_inp": model_inp,
        "vehicle_type_inp": vehicle_type_inp, "km_driven": km_driven, "cc_numeric": cc_numeric,
        "age": age, "is_moi": is_moi, "is_do_xe": is_do_xe, "is_su_dung_nhieu": is_su_dung_nhieu,
        "is_bao_duong": is_bao_duong, "is_do_ben": is_do_ben, "is_phap_ly": is_phap_ly,
        "origin_inp": origin_inp, "segment_inp": segment_inp, "price_segment_code": price_segment_code,
        "suggestion_type": suggestion_type
    }

    # --- PREDICT BUTTON: compute and STORE prediction in session_state ---
    predict_clicked = st.button("🔍 Dự đoán / Gợi ý", key="btn_predict")
    if predict_clicked:
        row = {
            "price": price,
            "price_min": price_min,
            "price_max": price_max,
            "km_driven": km_driven,
            "engine_size": engine_size_sel,
            "cc_numeric": cc_numeric,
            "age": age,
            "year_reg": int(max(1900, 2025 - age)),
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
        # Save the row to session_state right away — prevents loss after rerun
        st.session_state["last_predict"] = df_row.copy()

        df_row_prep = safe_prepare_X(df_row)
        X_row = df_row_prep[[c for c in (num_cols + flag_cols + cat_cols) if c in df_row_prep.columns]]
        if pipeline is None:
            st.error("Model chưa được load (model_randomforest.pkl).")
        else:
            try:
                pred = float(pipeline.predict(X_row)[0])
                st.session_state["last_predict"].loc[0, "predicted_price"] = round(pred, 2)
                try:
                    anomaly_res = run_price_anomaly_detection_with_reason(
                        data=df_row_prep.assign(price=df_row.loc[0,"price"]),
                        trained_model=pipeline,
                        num_cols=num_cols,
                        flag_cols=flag_cols,
                        cat_cols=cat_cols,
                        seg_col="price_segment_code",
                        k=0.05
                    )
                    anomaly_reason = anomaly_res.loc[0, "anomaly_reason"] if "anomaly_reason" in anomaly_res.columns else "Không có dấu hiệu bất thường"
                except Exception:
                    anomaly_reason = "Không có dấu hiệu bất thường"
                st.session_state["last_predict"].loc[0, "anomaly_reason"] = anomaly_reason
                last = st.session_state.get("last_clean")
                brand_model_map = {}
                if last is not None:
                    for b, g in last.groupby("brand"):
                        brand_model_map[b] = sorted(g["model"].dropna().unique().tolist())
                risk = compute_risk_score_strict(st.session_state["last_predict"].loc[0].to_dict(), last_clean_brand_models=brand_model_map, anomaly_reason=anomaly_reason)
                st.session_state["last_predict"].loc[0, "risk_score"] = risk
                st.session_state["last_predict"].loc[0, "risk_level"] = risk_level_from_score(risk)

                if suggestion_type == "Gợi ý giá bán":
                    st.success(f"📦 Gợi ý giá bán: **{pred:,.2f} triệu VND**")
                    st.info(f"Khoảng tham khảo: {pred*0.95:,.2f} — {pred*1.05:,.2f} triệu")
                else:
                    buy_price = pred * 0.92
                    st.success(f"🛒 Gợi ý giá mua hợp lý: **{buy_price:,.2f} triệu VND**")
                    st.info(f"(Giá model dự đoán = {pred:,.2f} triệu)")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

    # --- OUTSIDE predict button: show Đăng tin UI if there's a prediction stored ---
    saved = st.session_state.get("last_predict")
    if saved is not None:
        st.subheader("Tóm tắt (bản ghi sẽ lưu nếu bạn Xác nhận đăng)")
        st.write(saved.T)

        st.markdown("### Đăng tin")
        post_type_choice = st.radio("Bạn muốn:", ("Đăng bán", "Đăng mua"), key="post_type_choice")
        price_choice = st.radio("Chọn giá để đăng:", ("Giữ giá đã nhập", "Dùng giá dự đoán"), key="price_choice")
        chosen_price = float(saved.loc[0, "price"]) if price_choice == "Giữ giá đã nhập" else float(saved.loc[0, "predicted_price"])

        # text_input outside confirm button, persistent via key
        user_id = st.text_input("ID người đăng", value="", key="user_id")
        user_note= st.text_input("Ghi chú", value="", key="user_note")

        if st.button("✅ Xác nhận và gửi tin lên hệ thống", key="confirm_send"):
            ptype = "sell" if post_type_choice == "Đăng bán" else "buy"
            record = make_post_record(saved, post_type=ptype, chosen_price=chosen_price, user_id=(user_id or "anonymous"), note=user_note)
            save_post_record(record)
            st.success("✅ Tin của bạn đã được gửi lên hệ thống và chờ QTV duyệt.")
            st.info("Bạn có thể vào menu 'Đăng bán' hoặc 'Đăng mua' để xem lại tin đã gửi (được lưu trên server).")
            # do not rerun; keep last_predict for review/edit

# ------------------ Đăng bán / Đăng mua (user view) ------------------
elif choice == "Đăng bán":
    st.header("📢 Tin đăng bán (Người dùng)")
    # Show approved posts only
    posts = _read_xlsx_if_exists(APPROVED_SELL_XLSX)
    posts = normalize_datetime_like_columns(posts)
    if posts.empty:
        st.info("Hiện chưa có tin đăng bán.")
    else:
        st.write(f"Tổng: {len(posts)} tin")
        show_cols = [
            "user_id", "note", "price_final", "year_reg",
            "km_driven", "brand", "model", "cc_numeric",
            "origin", "vehicle_type"
        ]

        posts_show = posts.copy()

        # Giữ đúng cột + đổi tên tiếng Việt
        posts_show = posts_show[show_cols]
        posts_show = rename_columns_vn(posts_show)

        st.dataframe(posts_show.reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Tải tin đăng bán (Excel)", df_to_excel_bytes(posts), file_name="posts_sell.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif choice == "Đăng mua":
    st.header("📣 Tin đăng mua (Người dùng)")
    posts = _read_xlsx_if_exists(APPROVED_BUY_XLSX)
    posts = normalize_datetime_like_columns(posts)
    if posts.empty:
        st.info("Hiện chưa có tin đăng mua.")
    else:
        st.write(f"Tổng: {len(posts)} tin")
        show_cols = [
            "user_id", "note", "price_final", "year_reg",
            "km_driven", "brand", "model", "cc_numeric",
            "origin", "vehicle_type"
        ]

        posts_show = posts.copy()
        posts_show = posts_show[show_cols]
        posts_show = rename_columns_vn(posts_show)

        st.dataframe(posts_show.reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Tải tin đăng mua (Excel)", df_to_excel_bytes(posts), file_name="posts_buy.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------ ANOMALY PAGE ------------------
elif choice == "Phát hiện xe máy bất thường":
    st.header("🚨 Phát hiện xe máy bất thường")
    pred_df = st.session_state.get("predicted_df")
    st.subheader("A. Kết quả anomaly trên dữ liệu mẫu (auto-run)")
    if pred_df is None:
        st.warning("Dữ liệu mẫu chưa load hoặc model chưa predict.")
    else:
        if st.button("📄 Hiển thị 10 bản ghi có dấu hiệu bất thường", key="show_anom_10"):
            try:
                df_for_anom = pred_df.copy()
                if "price" not in df_for_anom.columns:
                    st.error("Dữ liệu mẫu thiếu cột `price` để kiểm tra anomaly.")
                else:
                    result_df = run_price_anomaly_detection_with_reason(
                        data=df_for_anom,
                        trained_model=pipeline,
                        num_cols=num_cols,
                        flag_cols=flag_cols,
                        cat_cols=cat_cols,
                        seg_col="price_segment_code",
                        k=0.05
                    )
                    anomalies = result_df[result_df["anomaly_reason"] != "Không có dấu hiệu bất thường"].copy()
                    if anomalies.empty:
                        st.info("Không tìm thấy bản ghi bất thường trong dữ liệu mẫu.")
                    else:
                        anomalies_sorted = anomalies.sort_values(by="anomaly_score", ascending=False)
                        show_cols = [c for c in ["brand","model","year_reg","km_driven","price","price_pred_final","anomaly_score","anomaly_reason","anomaly_level"] if c in anomalies_sorted.columns]
                        st.dataframe(anomalies_sorted[show_cols].head(10).reset_index(drop=True))
                        st.download_button("⬇️ Tải kết quả bất thường (Excel)", df_to_excel_bytes(anomalies_sorted), file_name="anomalies_sample.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Lỗi khi chạy anomaly trên dữ liệu mẫu: {e}")

    st.markdown("---")
    st.subheader("B. Nhập tay để kiểm tra 1 xe")
    price = st.number_input("Giá (triệu VND)", min_value=0.0, value=10.0, step=0.1, key="an_price")
    price_min = st.number_input("Khoảng giá min (triệu VND)", min_value=0.0, value=8.0, step=0.1, key="an_price_min")
    price_max = st.number_input("Khoảng giá max (triệu VND)", min_value=0.0, value=12.0, step=0.1, key="an_price_max")
    last = st.session_state.get("last_clean")
    brands_opts = sorted(last["brand"].dropna().unique().tolist()) if last is not None and "brand" in last.columns else BRANDS
    brand_sel = st.selectbox("Thương hiệu", options=brands_opts, key="an_brand")
    model_sel = st.text_input("Dòng xe (Dòng xe)", value="Wave", key="an_model")
    year_reg = st.number_input("Năm đăng ký", min_value=1900, max_value=2025, value=2020, step=1, key="an_year_reg")
    age = 0.5 if 2025 - year_reg == 0 else 2025 - year_reg
    km_driven_an = st.number_input("Số Km đã đi", min_value=0, value=5000, step=1, key="an_km")
    vehicle_type_sel = st.text_input("Loại xe", value="Xe số", key="an_vehicle_type")
    engine_size_sel = st.selectbox("Dung tích xe (nhãn)", options=["Dưới 50","50 - 100","100 - 175","Trên 175"], index=2, key="an_engine_size")
    origin_sel = st.selectbox("Xuất xứ", options=["Việt Nam","Nhập Khẩu"], key="an_origin")
    segment_sel = st.selectbox("Phân khúc giá", options=["Phổ thông","Cận cao cấp","Cao cấp"], key="an_segment")
    segment_map = {"Phổ Thông": 1, "Cận cao cấp": 2, "Cao Cấp": 3}
    price_segment_code = segment_map.get(segment_sel, 1)

    st.markdown("**Tình trạng (Tick = Có / Không = Không)**")
    a1, a2, a3 = st.columns(3)
    with a1:
        an_is_moi = st.checkbox("Xe còn mới", value=False, key="an_is_moi")
    with a2:
        an_is_do_xe = st.checkbox("Có độ xe", value=False, key="an_is_do_xe")
    with a3:
        an_is_su_dung_nhieu = st.checkbox("Xe đi nhiều", value=False, key="an_is_su_dung_nhieu")
    b1, b2, b3 = st.columns(3)
    with b1:
        an_is_bao_duong = st.checkbox("Xe có bảo dưỡng", value=False, key="an_is_bao_duong")
    with b2:
        an_is_do_ben = st.checkbox("Máy xe còn tốt", value=False, key="an_is_do_ben")
    with b3:
        an_is_phap_ly = st.checkbox("Pháp lý rõ ràng", value=True, key="an_is_phap_ly")

    if st.button("Kiểm tra", key="an_check"):
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
                df_row["predicted_price"] = float(pipeline.predict(df_row_prep[[c for c in (num_cols + flag_cols + cat_cols) if c in df_row_prep.columns]])[0])
                res = run_price_anomaly_detection_with_reason(
                    data=df_row_prep.assign(price=row["price"]),
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
                st.download_button("⬇️ Tải kết quả kiểm tra (Excel)", df_to_excel_bytes(df_row), file_name="anomaly_check_single.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Lỗi khi kiểm tra bất thường: {e}")

# ------------------ ADMIN (QTV) ------------------
elif choice == "Duyệt tin (QTV)":
    if "qtv_logged_in" not in st.session_state or st.session_state["qtv_logged_in"] is False:
        qtv_login()
        st.stop()
    
    st.header("🔧 Duyệt tin — Quản trị viên")

    pending = len(st.session_state.get("pending_notifications", []))
    st.markdown(f"**Tin chờ duyệt:** {pending}")

    manage_sell = st.checkbox("Quản lý tin đăng bán", value=True)
    manage_buy = st.checkbox("Quản lý tin đăng mua", value=False)

    # ===========================
    #     XỬ LÝ TIN ĐĂNG BÁN
    # ===========================
    if manage_sell:

        st.subheader("📦 Tin đăng bán (chờ duyệt)")

        df_sell = st.session_state.get("posts_sell", pd.DataFrame()).copy()

        if df_sell.empty:
            st.info("Không có tin đăng bán nào.")
        else:
            # Thêm cột checkbox
            df_sell_display = df_sell.copy()
            df_sell_display["selected"] = False

            df_sell_display = reorder_columns(df_sell_display)
            df_sell_display = rename_columns_vn(df_sell_display, mode="sell")

            edited_sell = st.data_editor(
                df_sell_display,
                use_container_width=True,
                hide_index=True,
                key="editor_sell"
            )

            # Những dòng được chọn
            selected_sell = edited_sell[edited_sell["Chọn"] == True]

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✔️ Duyệt tin bán"):
                    if selected_sell.empty:
                        st.warning("Chưa chọn dòng để duyệt.")
                    else:
                        post_ids = selected_sell["post_id"].tolist()

                        # Lưu vào approved
                        approved = _read_xlsx_if_exists(APPROVED_SELL_XLSX)
                        approved = pd.concat(
                            [approved, df_sell[df_sell["post_id"].isin(post_ids)]],
                            ignore_index=True
                        )
                        _save_xlsx(approved, APPROVED_SELL_XLSX)

                        # xóa khỏi pending
                        df_sell_new = df_sell[~df_sell["post_id"].isin(post_ids)]
                        st.session_state["posts_sell"] = df_sell_new
                        _save_xlsx(df_sell_new, POSTS_SELL_XLSX)

                        # Gỡ pending_notifications
                        for pid in post_ids:
                            if pid in st.session_state["pending_notifications"]:
                                st.session_state["pending_notifications"].remove(pid)

                        st.success(f"Đã duyệt {len(post_ids)} tin bán.")

            with col2:
                if st.button("❌ Từ chối tin bán"):
                    if selected_sell.empty:
                        st.warning("Chưa chọn dòng để từ chối.")
                    else:
                        post_ids = selected_sell["post_id"].tolist()

                        # Lưu rejected
                        rejected = _read_xlsx_if_exists(REJECTED_XLSX)
                        rejected = pd.concat(
                            [rejected, df_sell[df_sell["post_id"].isin(post_ids)]],
                            ignore_index=True
                        )
                        _save_xlsx(rejected, REJECTED_XLSX)

                        df_sell_new = df_sell[~df_sell["post_id"].isin(post_ids)]
                        st.session_state["posts_sell"] = df_sell_new
                        _save_xlsx(df_sell_new, POSTS_SELL_XLSX)

                        for pid in post_ids:
                            if pid in st.session_state["pending_notifications"]:
                                st.session_state["pending_notifications"].remove(pid)

                        st.success(f"Đã từ chối {len(post_ids)} tin bán.")

    st.markdown("---")

    # ===========================
    #     XỬ LÝ TIN ĐĂNG MUA
    # ===========================
    if manage_buy:

        st.subheader("🛒 Tin đăng mua (chờ duyệt)")

        df_buy = st.session_state.get("posts_buy", pd.DataFrame()).copy()

        if df_buy.empty:
            st.info("Không có tin đăng mua nào.")
        else:
            df_buy_display = df_buy.copy()
            df_buy_display["selected"] = False

            df_buy_display = reorder_columns(df_buy_display)
            df_buy_display = rename_columns_vn(df_buy_display, mode="buy")

            edited_buy = st.data_editor(
                df_buy_display,
                use_container_width=True,
                hide_index=True,
                key="editor_buy"
            )

            selected_buy = edited_buy[edited_buy["Chọn"] == True]

            col3, col4 = st.columns(2)

            with col3:
                if st.button("✔️ Duyệt tin mua"):
                    if selected_buy.empty:
                        st.warning("Chưa chọn dòng để duyệt.")
                    else:
                        post_ids = selected_buy["post_id"].tolist()

                        approved = _read_xlsx_if_exists(APPROVED_BUY_XLSX)
                        approved = pd.concat(
                            [approved, df_buy[df_buy["post_id"].isin(post_ids)]],
                            ignore_index=True
                        )
                        _save_xlsx(approved, APPROVED_BUY_XLSX)

                        df_buy_new = df_buy[~df_buy["post_id"].isin(post_ids)]
                        st.session_state["posts_buy"] = df_buy_new
                        _save_xlsx(df_buy_new, POSTS_BUY_XLSX)

                        for pid in post_ids:
                            if pid in st.session_state["pending_notifications"]:
                                st.session_state["pending_notifications"].remove(pid)

                        st.success(f"Đã duyệt {len(post_ids)} tin mua.")

            with col4:
                if st.button("❌ Từ chối tin mua"):
                    if selected_buy.empty:
                        st.warning("Chưa chọn dòng để từ chối.")
                    else:
                        post_ids = selected_buy["post_id"].tolist()

                        rejected = _read_xlsx_if_exists(REJECTED_XLSX)
                        rejected = pd.concat(
                            [rejected, df_buy[df_buy["post_id"].isin(post_ids)]],
                            ignore_index=True
                        )
                        _save_xlsx(rejected, REJECTED_XLSX)

                        df_buy_new = df_buy[~df_buy["post_id"].isin(post_ids)]
                        st.session_state["posts_buy"] = df_buy_new
                        _save_xlsx(df_buy_new, POSTS_BUY_XLSX)

                        for pid in post_ids:
                            if pid in st.session_state["pending_notifications"]:
                                st.session_state["pending_notifications"].remove(pid)

                        st.success(f"Đã từ chối {len(post_ids)} tin mua.")

# ------------------ AUTHOR PAGE ------------------
elif choice == "Thông tin tác giả":
    st.header("👤 Nhóm tác giả dự án")
    st.write("""
    **Hồ Thị Quỳnh Như**  
    **Nguyễn Văn Cường**  
    **Nguyễn Thị Tuyết Anh**  
    """)
