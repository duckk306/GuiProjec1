# utils_anomaly.py
import pandas as pd
import numpy as np


def run_price_anomaly_detection_with_reason(
    data, trained_model,
    num_cols, flag_cols, cat_cols,
    seg_col="price_segment_code", k=0.05
):
    """
    Trả về anomaly_score (0-100), anomaly_level, lý do, và highlight_style cho GUI.
    """

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    df = data.copy()

    # -------------------------
    # Chuẩn hoá tên cột
    # -------------------------
    rename_dict = {
        "Giá": "price",
        "Khoảng giá min": "price_min",
        "Khoảng giá max": "price_max",
    }
    df = df.rename(columns=rename_dict)

    # -------------------------
    # Kiểm tra phân khúc
    # -------------------------
    if seg_col not in df.columns:
        seg_col = None

    # -------------------------
    # 1) Dự đoán giá
    # -------------------------
    X = df[num_cols + flag_cols + cat_cols]
    df["price_pred_final"] = trained_model.predict(X)

    # -------------------------
    # 2) Residual & Z-score
    # -------------------------
    df["residual"] = df["price"] - df["price_pred_final"]

    if seg_col:
        df["residual_z"] = df.groupby(seg_col)["residual"].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1)
        )
    else:
        std = df["residual"].std(ddof=0) or 1
        df["residual_z"] = (df["residual"] - df["residual"].mean()) / std

    # -------------------------
    # 3) Vi phạm min/max
    # -------------------------
    df["viol_min"] = (df["price"] < df["price_min"]).astype(int)
    df["viol_max"] = (df["price"] > df["price_max"]).astype(int)
    df["viol_minmax"] = (df["viol_min"] | df["viol_max"]).astype(int)

    # -------------------------
    # 4) Khoảng tin cậy P10–P90
    # -------------------------
    if seg_col:
        df["P10"] = df.groupby(seg_col)["price"].transform(lambda x: np.percentile(x, 10))
        df["P90"] = df.groupby(seg_col)["price"].transform(lambda x: np.percentile(x, 90))
    else:
        p10, p90 = np.percentile(df["price"], [10, 90])
        df["P10"], df["P90"] = p10, p90

    df["viol_confidence"] = (~((df["price"] >= df["P10"]) & (df["price"] <= df["P90"]))).astype(int)

    # -------------------------
    # 5) Unsupervised ensemble
    # -------------------------
    extra_features = [c for c in ["age", "km_driven", "is_do_ben"] if c in df.columns]
    feat_cols = ["residual_z"] + num_cols + flag_cols + extra_features
    X_feat = df[feat_cols].fillna(0)

    n_samples = len(df)

    # Auto-adjust n_neighbors
    if n_samples > 1:
        lof_neighbors = max(1, min(20, n_samples - 1))
    else:
        lof_neighbors = None

    models_unsup = {
        "IForest": IsolationForest(contamination=k, random_state=42),
        "OCSVM": OneClassSVM(nu=k, kernel="rbf"),
    }

    if lof_neighbors:
        models_unsup["LOF"] = LocalOutlierFactor(
            contamination=k,
            novelty=True,
            n_neighbors=lof_neighbors
        )

    # Fit & score
    for name, model in models_unsup.items():
        try:
            model.fit(X_feat)

            try:
                score = -model.decision_function(X_feat)
            except Exception:
                if hasattr(model, "negative_outlier_factor_"):
                    score = -model.negative_outlier_factor_
                else:
                    score = np.zeros(n_samples)
        except Exception:
            score = np.zeros(n_samples)

        df[f"score_{name}"] = score

    # -------------------------
    # 6) Chọn mô hình unsup tốt nhất
    # -------------------------
    ref_signals = ["residual_z", "viol_minmax", "viol_confidence"]
    corr_results = {}

    for name in models_unsup.keys():
        valid = [c for c in ref_signals if c in df.columns]
        corr = df[[f"score_{name}"] + valid].corr(method="spearman")[f"score_{name}"].abs().mean()
        corr_results[name] = corr

    best_unsup = max(corr_results, key=corr_results.get)
    df["unsup_score"] = df[f"score_{best_unsup}"].fillna(0)

    # -------------------------
    # 7) Score thô → scale 0–100
    # -------------------------
    score_raw = (
        0.6 * df["residual_z"].abs().fillna(0) +
        0.15 * df["viol_minmax"] +
        0.15 * df["viol_confidence"] +
        0.10 * df["unsup_score"]
    )

    if len(df) == 1:
    # Khi chỉ có 1 bản ghi: không dùng MinMax → luôn trả giá trị >0 nếu có bất thường
        df["anomaly_score"] = (score_raw * 100).clip(0, 100)
    else:
        scaler = MinMaxScaler(feature_range=(0, 100))
        df["anomaly_score"] = scaler.fit_transform(score_raw.values.reshape(-1, 1))

    # -------------------------
    # 8) Lý do
    # -------------------------
    unsup_thr = df["unsup_score"].quantile(0.95)

    def explain(row):
        reasons = []
        if abs(row["residual_z"]) > 2:
            reasons.append("Giá lệch xa mô hình dự đoán")
        if row["viol_min"] == 1:
            reasons.append("Giá thấp hơn mức min")
        if row["viol_max"] == 1:
            reasons.append("Giá cao hơn mức max")
        if row["viol_confidence"] == 1:
            reasons.append("Nằm ngoài khoảng tin cậy P10–P90")
        if row["unsup_score"] > unsup_thr:
            reasons.append("Mô hình unsupervised đánh giá bất thường")
        return ", ".join(reasons) if reasons else "Không có dấu hiệu bất thường"

    df["anomaly_reason"] = df.apply(explain, axis=1)

    # -------------------------
    # 9) Phân cấp mức độ (ĐÃ FIX LỖI)
    # -------------------------
    if len(df) < 5:
        # Dataset nhỏ → không dùng percentile
        def lvl_small(row):
            if row["anomaly_reason"] == "Không có dấu hiệu bất thường":
                return "Không có dấu hiệu bất thường"
            else:
                return "Phát hiện giá bất thường – Mức độ: Thấp"

        df["anomaly_level"] = df.apply(lvl_small, axis=1)

    else:
        # Dataset đủ lớn → dùng percentile
        p70, p90 = np.percentile(df["anomaly_score"], [70, 90])

        def lvl(s):
            if s >= p90:
                return "Cao"
            elif s >= p70:
                return "Trung bình"
            else:
                return "Thấp"

        df["anomaly_level_raw"] = df["anomaly_score"].apply(lvl)

        def final_lvl(row):
            if row["anomaly_reason"] == "Không có dấu hiệu bất thường":
                return "Không có dấu hiệu bất thường"
            return f"Phát hiện giá bất thường – Mức độ: {row['anomaly_level_raw']}"

        df["anomaly_level"] = df.apply(final_lvl, axis=1)

    # -------------------------
    # 10) Highlight
    # -------------------------
    def highlight(level):
        if "Cao" in level:
            return "background-color: #ff4d4d; color: white;"
        if "Trung bình" in level:
            return "background-color: #ffa64d; color: black;"
        if "Không có dấu hiệu" in level:
            return "background-color: #e8f5e9; color: black;"
        return "background-color: #f0f0f0; color: black;"

    df["highlight_style"] = df["anomaly_level"].apply(highlight)

    return df
