"""
ESOGU Probability Project - Credit Risk Decision Panel
Updated version:
- Adds loan_purpose as the 17th feature
- Uses CatBoost as the main tree-based model
- Keeps categorical variables as categorical for CatBoost
- Uses KNNImputer + StandardScaler for numerical variables
- Produces a SHAP-based XAI explanation for each single prediction

Run:
    pip install streamlit pandas numpy scikit-learn catboost shap matplotlib
    streamlit run app_streamlit_updated.py

Put one of these files in the same folder as this script:
    loans_full_schema.csv
    loans_full_schema.csv.zip
"""

import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(page_title="ESOGU FinTech - Kredi Risk Paneli", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size:32px;
        font-weight:bold;
        color:#1E3A8A;
        text-align:center;
        margin-bottom:20px;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .small-note {
        color: #475569;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🚀 KREDİ RİSK ANALİZ MOTORU</div>", unsafe_allow_html=True)
st.sidebar.header("📋 Müşteri Kredi Başvuru Bilgileri")

# =========================================================
# CONFIG
# =========================================================
SAFE_STATUS = ["Fully Paid", "Current"]
RISK_STATUS = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"]

NUM_COLS = [
    "loan_amount",
    "interest_rate",
    "installment",
    "term",
    "annual_income",
    "emp_length",
    "debt_to_income",
    "delinq_2y",
    "total_credit_lines",
    "total_credit_utilized",
    "months_since_last_credit_inquiry",
    "total_debit_limit",
]

CAT_COLS = [
    "grade",
    "state",
    "homeownership",
    "verified_income",
    "loan_purpose",  # 17th feature added after instructor feedback
]

FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET_COL = "loan_status"


def find_dataset_file():
    """Find csv or zipped csv in the current directory."""
    candidates = [
        Path("loans_full_schema.csv"),
        Path("loans_full_schema.csv.zip"),
        Path("data/loans_full_schema.csv"),
        Path("data/loans_full_schema.csv.zip"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_dataset() -> pd.DataFrame:
    """Read the dataset from csv or zip."""
    dataset_path = find_dataset_file()
    if dataset_path is None:
        st.error(
            "❌ Dataset bulunamadı. Script ile aynı klasöre 'loans_full_schema.csv' "
            "veya 'loans_full_schema.csv.zip' koymalısınız."
        )
        st.stop()

    if dataset_path.suffix == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                st.error("❌ Zip dosyasının içinde CSV dosyası bulunamadı.")
                st.stop()
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f, low_memory=False)

    return pd.read_csv(dataset_path, low_memory=False)


def build_target(series: pd.Series) -> pd.Series:
    """
    Convert loan_status into a binary risk label.

    0 = Safe / regular or paid loan
    1 = Risky / delayed or collection-problem loan

    Important report note:
    This is not only strict default prediction. It is a broader credit risk indicator.
    """
    clean_status = series.astype(str).str.strip()
    return np.where(clean_status.isin(RISK_STATUS), 1, 0)


# =========================================================
# MODEL AND PIPELINE CACHING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_and_train_pipeline():
    df_raw = read_dataset()

    missing_columns = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df_raw.columns]
    if missing_columns:
        st.error(f"❌ Dataset içinde eksik sütunlar var: {missing_columns}")
        st.stop()

    df = df_raw[FEATURE_COLS + [TARGET_COL]].copy()
    df = df[df[TARGET_COL].astype(str).str.strip().isin(SAFE_STATUS + RISK_STATUS)].copy()
    df["risk_label"] = build_target(df[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    y = df["risk_label"].astype(int)

    # 1) Numerical transformation: StandardScaler before KNNImputer.
    # KNN uses distance, so scaling before KNN prevents large-scale variables from dominating.
    num_scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        num_scaler.fit_transform(X[NUM_COLS]),
        columns=NUM_COLS,
        index=X.index,
    )

    num_imputer = KNNImputer(n_neighbors=5)
    # Full KNN on 10k rows can be slow. We fit the imputer on a representative sample,
    # then transform the full dataset. This keeps the KNN approach while making the app usable.
    knn_fit_sample = X_num_scaled.sample(n=min(2500, len(X_num_scaled)), random_state=42)
    num_imputer.fit(knn_fit_sample)
    X_num_imputed = pd.DataFrame(
        num_imputer.transform(X_num_scaled),
        columns=NUM_COLS,
        index=X.index,
    )

    # 2) Categorical transformation: keep categorical variables as string for CatBoost.
    X_cat = X[CAT_COLS].fillna("Unknown").astype(str)
    X_model = pd.concat([X_num_imputed, X_cat], axis=1)[FEATURE_COLS]

    # 3) CatBoost model: strong for categorical variables and tree-based explanation.
    model = CatBoostClassifier(
        iterations=160,
        learning_rate=0.06,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=1,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(X_model, y, cat_features=CAT_COLS)

    unique_cats = {
        col: sorted(X[col].dropna().astype(str).unique().tolist())
        for col in CAT_COLS
    }

    class_counts = y.value_counts().sort_index().to_dict()

    return model, num_scaler, num_imputer, unique_cats, X_model, class_counts


with st.spinner("🌲 17 Değişkenli CatBoost + XAI motoru eğitiliyor..."):
    model, num_scaler, num_imputer, unique_cats, X_train_template, class_counts = load_and_train_pipeline()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
loan_amount = st.sidebar.slider("Kredi Tutarı ($)", 1000, 40000, 15000, step=500)
interest_rate = st.sidebar.slider("Faiz Oranı (%)", 5.3, 31.0, 12.0, step=0.1)
installment = st.sidebar.slider("Aylık Taksit ($)", 30.0, 1400.0, 400.0, step=10.0)
term = st.sidebar.selectbox("Vade Süresi (Ay)", [36, 60])
annual_income = st.sidebar.slider("Yıllık Gelir ($)", 10000, 300000, 65000, step=5000)
debt_to_income = st.sidebar.slider("Borç / Gelir Oranı (DTI)", 0.0, 60.0, 17.5, step=0.5)
emp_length = st.sidebar.slider("Çalışma Süresi (Yıl)", 0, 10, 6, step=1)
delinq_2y = st.sidebar.slider("Son 2 Yıldaki Gecikme Sayısı", 0, 5, 0, step=1)
total_credit_lines = st.sidebar.slider("Toplam Kredi Hesap Sayısı", 2, 80, 21, step=1)
total_credit_utilized = st.sidebar.slider("Kullanılan Toplam Kredi Limiti ($)", 0, 260000, 37000, step=5000)
months_since_last_credit_inquiry = st.sidebar.slider("Son Kredi Sorgusundan Beri Geçen Süre (Ay)", 0, 24, 6, step=1)
total_debit_limit = st.sidebar.slider("Toplam Banka Kartı / Vadesiz Hesap Limiti ($)", 0, 100000, 25000, step=5000)

homeownership = st.sidebar.selectbox("Ev Sahipliği Durumu", unique_cats["homeownership"])
verified_income = st.sidebar.selectbox("Gelir Doğrulama Durumu", unique_cats["verified_income"])
grade = st.sidebar.selectbox("Kredi Skoru Derecesi (Grade)", unique_cats["grade"])
state = st.sidebar.selectbox("Yaşanılan Eyalet", unique_cats["state"])
loan_purpose = st.sidebar.selectbox("Kredi Kullanım Amacı", unique_cats["loan_purpose"])

# =========================================================
# INPUT PREPROCESSING
# =========================================================
input_dict = {
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "installment": installment,
    "term": term,
    "annual_income": annual_income,
    "emp_length": emp_length,
    "debt_to_income": debt_to_income,
    "delinq_2y": delinq_2y,
    "total_credit_lines": total_credit_lines,
    "total_credit_utilized": total_credit_utilized,
    "months_since_last_credit_inquiry": months_since_last_credit_inquiry,
    "total_debit_limit": total_debit_limit,
    "grade": grade,
    "state": state,
    "homeownership": homeownership,
    "verified_income": verified_income,
    "loan_purpose": loan_purpose,
}

input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

input_num_scaled = pd.DataFrame(
    num_scaler.transform(input_df[NUM_COLS]),
    columns=NUM_COLS,
    index=input_df.index,
)
input_num_imputed = pd.DataFrame(
    num_imputer.transform(input_num_scaled),
    columns=NUM_COLS,
    index=input_df.index,
)
input_cat = input_df[CAT_COLS].fillna("Unknown").astype(str)
input_model = pd.concat([input_num_imputed, input_cat], axis=1)[FEATURE_COLS]

# =========================================================
# PREDICTION + XAI
# =========================================================
col1, col2 = st.columns([1, 2])

prob_risk = float(model.predict_proba(input_model)[0][1])
prob_percentage = prob_risk * 100
threshold = 50.0

if prob_percentage <= threshold:
    factor = prob_percentage / threshold
    r = int(110 + (255 - 110) * factor)
    g = int(231 + (255 - 231) * factor)
    b = int(183 + (255 - 183) * factor)
    border_r = int(16 + (220 - 16) * factor)
    border_g = int(185 + (220 - 185) * factor)
    border_b = int(129 + (220 - 129) * factor)
    text_color = "#047857"
    status_text = "ONAY / DÜŞÜK RİSK"
else:
    factor = (prob_percentage - threshold) / threshold
    r = int(255 - (255 - 252) * factor)
    g = int(255 - (255 - 120) * factor)
    b = int(255 - (255 - 120) * factor)
    border_r = int(220 - (220 - 239) * factor)
    border_g = int(220 - (220 - 68) * factor)
    border_b = int(220 - (220 - 68) * factor)
    text_color = "#991B1B"
    status_text = "RED / YÜKSEK RİSK"

dynamic_bg = f"rgb({r}, {g}, {b})"
dynamic_border = f"rgb({border_r}, {border_g}, {border_b})"

with col1:
    st.subheader("🎯 Risk Analiz Sonucu")
    st.markdown(
        f"""
        <div class='metric-box' style='background-color:{dynamic_bg}; border: 2px solid {dynamic_border}; color:{text_color};'>
            <p style='margin:0; font-size:16px; font-weight:bold; color:{text_color};'>KREDİ DURUMU: {status_text}</p>
            <p style='margin:0; font-size:42px; font-weight:black; color:{text_color};'>% {prob_percentage:.2f}</p>
            <p style='margin:0; font-size:14px; font-weight:normal; color:{text_color}; opacity:0.95;'>Gecikme / Ödeme Riski Olasılığı</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <p class='small-note'>
        Eğitim verisinde güvenli sınıf: <b>{class_counts.get(0, 0)}</b>,
        riskli sınıf: <b>{class_counts.get(1, 0)}</b>. Karar eşiği: <b>%50</b>.
        </p>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("🧠 Bu Müşteriye Özel XAI (SHAP) Karar Gerekçesi")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_model, check_additivity=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values[0], max_display=10, show=False)
    plt.title("Hangi Değişken Riski Ne Kadar Etkiledi?", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.success("✅ Sistem aktif: Sol panelden değerleri değiştirerek modeli canlı test edebilirsiniz.")

st.info(
    "Rapor notu: Bu çıktı kesin bankacılık kararı değildir. Dataset içinde 'Current' krediler henüz tamamen kapanmadığı için "
    "hedef değişken 'kesin temerrüt' yerine daha geniş anlamda 'kredi ödeme riski' olarak yorumlanmalıdır."
)
