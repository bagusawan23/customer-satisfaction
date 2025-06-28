
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import requests
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📊 Customer Satisfaction Dashboard", layout="wide")
st.title("📊 Customer Satisfaction Dashboard")

# --- AI Function ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_ai_insight(prompt):
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "Kamu adalah analis AI ahli logistik dan pelayanan pelanggan."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ Gagal mengambil insight AI (kode: {response.status_code})"

# --- File Upload ---
uploaded_file = st.file_uploader("Upload file Excel (customer_satisfaction.xlsx)", type=["xlsx"])
if uploaded_file:
    data_full = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data_full.head())

    data_full['order_date'] = pd.to_datetime(data_full['order_date'])
    data_full['order_month'] = data_full['order_date'].dt.to_period('M').astype(str)
    monthly_trend = data_full.groupby('order_month')['customer_satisfaction'].mean().reset_index()

    st.subheader("📈 Rata-rata Kepuasan Pelanggan per Bulan")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly_trend, x='order_month', y='customer_satisfaction', marker='o', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("📊 Korelasi antar Variabel Numerik")
    numeric_cols = [
        "sla_fulfillment",
        "num_complaints_per_100_orders",
        "response_time_hours",
        "delivery_time_diff_days",
        "customer_satisfaction"
    ]
    correlation_matrix = data_full[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📉 Distribusi Variabel Numerik")
    fig3, ax3 = plt.subplots(len(numeric_cols), 1, figsize=(10, 20))
    for i, col in enumerate(numeric_cols):
        sns.histplot(data_full[col], kde=True, bins=30, ax=ax3[i])
        ax3[i].set_title(f'Distribusi: {col}')
    st.pyplot(fig3)

    st.subheader("📍 Rata-rata & Distribusi Kepuasan Pelanggan per Wilayah")
    region_stats = data_full.groupby("region")[numeric_cols].mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=region_stats, x="region", y="customer_satisfaction", ax=ax4)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data_full, x="region", y="customer_satisfaction", ax=ax5)
    st.pyplot(fig5)

    # Train Model
    X = data_full[["sla_fulfillment", "num_complaints_per_100_orders", "response_time_hours", "delivery_time_diff_days", "region", "product_type"]]
    y = data_full["customer_satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ["sla_fulfillment", "num_complaints_per_100_orders", "response_time_hours", "delivery_time_diff_days"]
    categorical_features = ["region", "product_type"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor())
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📉 Scatter Plot: Prediksi vs Aktual")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax6)
    ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    st.pyplot(fig6)

    st.subheader("📊 Distribusi Error Absolut Prediksi")
    comparison_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    comparison_df["absolute_error"] = (comparison_df["actual"] - comparison_df["predicted"]).abs()
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    sns.histplot(comparison_df["absolute_error"], bins=30, kde=True, ax=ax7)
    st.pyplot(fig7)

    st.subheader("📌 Feature Importance")
    model = pipeline.named_steps["model"]
    feature_names = (
        numeric_features +
        list(pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features))
    )
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax8)
    st.pyplot(fig8)

    st.subheader("🧪 Simulasi Prediksi Kepuasan Pelanggan")
    with st.form("simulasi_form"):
        sla = st.slider("SLA Fulfillment (%)", 0, 100, 92)
        complaints = st.number_input("Jumlah Komplain per 100 Order", min_value=0.0, value=2.0)
        response_time = st.number_input("Response Time (jam)", min_value=0.0, value=3.0)
        delivery_diff = st.number_input("Perbedaan Waktu Pengiriman (hari)", min_value=0.0, value=0.0)
        region = st.selectbox("Wilayah", data_full["region"].unique())
        product = st.selectbox("Tipe Produk", data_full["product_type"].unique())
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            simulasi_input = pd.DataFrame([{
                "sla_fulfillment": sla,
                "num_complaints_per_100_orders": complaints,
                "response_time_hours": response_time,
                "delivery_time_diff_days": delivery_diff,
                "region": region,
                "product_type": product
            }])
            hasil = pipeline.predict(simulasi_input)[0]
            st.success(f"Prediksi Skor Kepuasan Pelanggan: {hasil:.2f}")

    # Insight AI
    if st.button("🔎 Minta Insight AI Berdasarkan Hasil Model"):
        prompt = f'''
Data menunjukkan performa prediksi model customer satisfaction:
- MAE: {mae:.2f}
- R2 Score: {r2:.2f}
Fitur: {", ".join(numeric_features + categorical_features)}

Tolong berikan:
1. Penjelasan performa model
2. Insight dari data
3. Rekomendasi peningkatan layanan
Jawab dalam Bahasa Indonesia.
'''
        with st.spinner("Meminta insight dari AI..."):
            ai_response = get_ai_insight(prompt)
            st.markdown("### 🤖 Insight dari AI")
            st.markdown(ai_response)
