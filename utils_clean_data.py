import pandas as pd
import numpy as np
import re
import string

def clean_motobike_data(df):
    import pandas as pd
    import numpy as np
    import re
    import string

    # ==== COPY NGUYÊN CODE CỦA BẠN ====

    data = df.drop(columns=['Địa chỉ','Tình trạng', 'Chính sách bảo hành', 'Trọng lượng', 'Href'])

    # Sort
    data = data.sort_values(by=['Thương hiệu', 'Dòng xe', 'Loại xe'], ascending=[True, True, True]).reset_index(drop=True)

    # Chuẩn hóa giá
    data['Giá'] = (
        data['Giá']
        .astype(str)
        .str.replace(r'[^\d]', '', regex=True)
    )
    data.loc[data['Giá'] == '', 'Giá'] = np.nan
    data['Giá'] = data['Giá'].astype(float) / 1_000_000

    for col in ['Khoảng giá min', 'Khoảng giá max']:
        data[col] = (
            data[col].astype(str)
            .str.replace('tr', '', case=False, regex=False)
            .str.replace(',', '.')
            .str.strip()
        )
        data.loc[data[col] == '', col] = np.nan
        data[col] = data[col].astype(float)

    # Clean
    data_clean = data.copy()
    data_clean = data_clean.dropna(subset=['Tiêu đề', 'Giá'])
    data_clean['Khoảng giá min'] = data_clean['Khoảng giá min'].fillna(data_clean['Giá'])
    data_clean['Khoảng giá max'] = data_clean['Khoảng giá max'].fillna(data_clean['Giá'])
    data_clean['Khoảng giá min'] = data_clean.groupby('Thương hiệu')['Khoảng giá min'].transform(lambda x: x.fillna(x.median()))
    data_clean['Khoảng giá max'] = data_clean.groupby('Thương hiệu')['Khoảng giá max'].transform(lambda x: x.fillna(x.median()))

    # Phân khúc giá
    def price_segment(price):
        if price < 70: return "Phổ thông"
        elif price < 200: return "Cận cao cấp"
        else: return "Cao cấp"
    data_clean["Phân khúc giá"] = data_clean["Giá"].apply(price_segment)

    # Numeric
    data_clean[['Giá', 'Khoảng giá min', 'Khoảng giá max']] = data_clean[['Giá', 'Khoảng giá min', 'Khoảng giá max']].astype(float)
    data_clean = data_clean[(data_clean['Giá'] > 1) & (data_clean['Giá'] < 5000)]

    # KM
    data_clean.loc[data_clean['Số Km đã đi'] > 99999, 'Số Km đã đi'] = 99999

    # Text normalize
    for col in ['Thương hiệu', 'Dòng xe', 'Loại xe', 'Dung tích xe', 'Xuất xứ', 'Phân khúc giá']:
        data_clean[col] = data_clean[col].str.strip().str.title()

    # Dung tích xe
    def parse_cc(val):
        if 'Dưới' in val: return 40
        if '50 - 100' in val: return 75
        if '100 - 175' in val: return 137
        if 'Trên 175' in val: return 200
        return np.nan
    data_clean['cc_numeric'] = data_clean['Dung tích xe'].apply(parse_cc)

    # Price segment map
    price_segment_map = {'Phổ Thông': 1, 'Cận Cao Cấp': 2, 'Cao Cấp': 3}
    data_clean['price_segment_code'] = data_clean['Phân khúc giá'].map(price_segment_map)

    # Năm đăng ký
    data_clean['Năm đăng ký'] = data_clean['Năm đăng ký'].replace({
        'trước năm 1980': '1979',
        'Đang cập nhật': np.nan,
        'Không rõ': np.nan
    })
    data_clean['Năm đăng ký'] = pd.to_numeric(data_clean['Năm đăng ký'], errors='coerce')
    data_clean['Năm đăng ký'] = data_clean['Năm đăng ký'].astype(int)
    min_age = 0.5
    data_clean['age'] = 2025 - data_clean['Năm đăng ký']
    data_clean["age"] = data_clean["age"].astype(float)
    data_clean.loc[data_clean['age'] <= 0, 'age'] = min_age

    # Missing cc_numeric
    data_clean['cc_numeric'] = data_clean['cc_numeric'].fillna(data_clean['cc_numeric'].median())

    numeric_cols = ["Giá", "Khoảng giá min", "Khoảng giá max", "Số Km đã đi", "age", "cc_numeric"]

    premium_brands = ['BMW', 'Harley Davidson', 'Ducati', 'Triumph', 'Kawasaki', 'Benelli']

    data_clean.loc[
        (~data_clean['Thương hiệu'].isin(premium_brands)) & (data_clean['Giá'] > 300),
        'Giá'
    ] = 300

    # Outlier detection
    Q1 = data_clean[numeric_cols].quantile(0.25)
    Q3 = data_clean[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (data_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | (data_clean[numeric_cols] > (Q3 + 1.5 * IQR))
    outlier_counts = outlier_mask.sum().sort_values(ascending=False)

    # Keyword groups
    kw_moi = ["mới","còn mới","như mới","mới 95","mới 99","mới tinh","xe lướt","xe ít đi","ít sử dụng","xe để không",
              "để kho","keng","leng keng","nguyên zin","zin 100%","zin nguyên bản","dán keo","dán ppf","ngoại hình đẹp",
              "dàn áo liền lạc","đẹp như hình"]
    kw_do_xe = ["độ","đồ chơi","full đồ","pô độ","pô móc","phuộc rcb","tay thắng","lên đồ","tem độ","lên full đồ",
                "đồ zin còn đủ","kính gió","thùng givi","ốc titan","mão gió","bao tay","trợ lực","độ máy"]
    kw_su_dung = ["ít đi","đi làm","đi học","đi phượt","đi cà phê","để không","ít sử dụng","xe gia đình","xe công ty",
                  "dư xe","đi lại nhẹ nhàng","xe nữ dùng","xe nữ chạy","xe để lâu","ít chạy","đi gần"]
    kw_bao_duong = ["bảo dưỡng","bảo trì","thay nhớt","vệ sinh","bao test","đi bảo dưỡng","bảo dưỡng định kỳ",
                     "mới thay bình","mới làm nồi","đã làm lại máy","thay bố thắng","thay lọc","bảo dưỡng lớn",
                     "chỉnh sên","xe kỹ"]
    kw_do_ben = ["máy êm","nổ êm","chạy êm","máy mạnh","máy bốc","tiết kiệm xăng","ổn định","chạy ngon",
                  "không xì nhớt","không rò rỉ","không lỗi","máy khô ráo","máy tốt","chạy mượt","vận hành ổn định",
                  "êm ái","bền bỉ","máy móc zin","chạy bình thường","hoạt động tốt"]
    kw_phap_ly = ["chính chủ","ủy quyền","bao sang tên","cà vẹt","giấy tờ đầy đủ","giấy tờ hợp lệ",
                  "hồ sơ gốc","bstp","bao công chứng","bao tranh chấp","ra tên","cavet","hợp pháp"]

    # NLP helpers
    def keyword_flag(text, keywords):
        if pd.isna(text): return 0
        text = text.lower()
        return int(any(re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text) for kw in keywords))

    def clean_text(text):
        if pd.isna(text): return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    vietnamese_stopwords = {
        "xe","máy","bán","cần","mua","báo","liên","hệ","anh","chị","em","mn","mọi","người","xin",
        "cảm","ơn","chợ","tốt","đầy","đủ","điện","thoại","địa","chỉ","số","của","và","với","còn",
        "thì","nên","rất","đã","được","ko","kg","thật","là","thôi","nha","nhé","ạ","nhưng","bởi",
        "vì","thì","nào","vậy"
    }

    def remove_stopwords(text):
        words = text.split()
        return " ".join([w for w in words if w not in vietnamese_stopwords])

    data_clean["desc_clean"] = data_clean["Mô tả chi tiết"].apply(clean_text)
    data_clean["desc_clean"] = data_clean["desc_clean"].apply(remove_stopwords)

    # keyword features
    data_clean["is_moi"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_moi))
    data_clean["is_do_xe"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_do_xe))
    data_clean["is_su_dung_nhieu"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_su_dung))
    data_clean["is_bao_duong"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_bao_duong))
    data_clean["is_do_ben"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_do_ben))
    data_clean["is_phap_ly"] = data_clean["desc_clean"].apply(lambda x: keyword_flag(x, kw_phap_ly))

        # Rename
    rename_map = {
            "Giá": "price",
            "Khoảng giá min": "price_min",
            "Khoảng giá max": "price_max",
            "Thương hiệu": "brand",
            "Dòng xe": "model",
            "Năm đăng ký": "year_reg",
            "Số Km đã đi": "km_driven",
            "Loại xe": "vehicle_type",
            "Dung tích xe": "engine_size",
            "Xuất xứ": "origin",
            "Phân khúc giá": "segment",}
    data_clean = data_clean.rename(columns=rename_map)

    # Trả về clean dataframe
    return data_clean
