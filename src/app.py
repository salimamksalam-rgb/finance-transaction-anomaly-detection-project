"""
app.py
Professional Anomaly Detection Dashboard
Manufacturing Transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==================================================
# PAGE SETTINGS
# ==================================================
st.set_page_config(
    page_title="Finance Transaction Anomaly Detection Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# PROFESSIONAL DARK THEME CSS
# ==================================================
st.markdown("""
<style>

/* ===============================
   BLACK THEME STREAMLIT UI
   =============================== */

.stApp {
    background: #0a0a0a;
    color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #222222;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Title */
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
}

.sub-text {
    color: #9ca3af;
    font-size: 15px;
    margin-bottom: 20px;
}

/* Cards */
.metric-box {
    background: #111111;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #222222;
    text-align: center;
}

.metric-title {
    color: #9ca3af;
    font-size: 13px;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #3b82f6;
}

/* Blocks */
.block {
    background: #111111;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #222222;
    margin-bottom: 18px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #1d4ed8 !important;
}

/* Tables */
thead tr th {
    background: #111111 !important;
    color: white !important;
}

tbody tr td {
    background: #1a1a1a !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown('<div class="main-title">📊 Finance Transaction Anomaly Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Detect unusual transactions using Z-Score analysis</div>', unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("System Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Z-Score Threshold",
    2.0, 5.0, 3.0, 0.1
)

# ==================================================
# STOP IF NO FILE
# ==================================================
if uploaded_file is None:
    st.info("👈 Upload a CSV dataset from the sidebar to begin.")
    st.stop()

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(uploaded_file)

# Clean column names
df.columns = df.columns.str.strip()

# ==================================================
# DATA PREVIEW
# ==================================================
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("📌 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# SELECT NUMERIC COLUMN
# ==================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) == 0:
    st.error("No numeric columns found in dataset.")
    st.stop()

column = st.selectbox("Select Column for Detection", numeric_cols)

# ==================================================
# Z-SCORE CALCULATION
# ==================================================
mean = df[column].mean()
std = df[column].std()

if std == 0:
    st.error("Standard deviation is zero. Cannot compute Z-score.")
    st.stop()

df["Z_Score"] = (df[column] - mean) / std

df["Anomaly"] = df["Z_Score"].apply(
    lambda x: "Yes" if abs(x) > threshold else "No"
)

anomalies = df[df["Anomaly"] == "Yes"]

# ==================================================
# DASHBOARD METRICS
# ==================================================
st.subheader("📊 Dashboard Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(df))
c2.metric("Anomalies", len(anomalies))
c3.metric("Mean", round(mean, 2))
c4.metric("Std Dev", round(std, 2))

# ==================================================
# VISUALIZATIONS
# ==================================================
left, right = st.columns(2)

with left:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("📈 Distribution")

    fig1 = px.histogram(
        df,
        x=column,
        nbins=30,
        color_discrete_sequence=["#2563eb"]
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("🚨 Anomaly Detection")

    fig2 = px.scatter(
        df,
        x=df.index,
        y=column,
        color="Anomaly",
        color_discrete_map={
            "Yes": "red",
            "No": "#2563eb"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# ANOMALY TABLE
# ==================================================
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("🚨 Detected Anomalies")

if len(anomalies) > 0:
    st.dataframe(anomalies, use_container_width=True)
else:
    st.success("No anomalies detected.")

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# DOWNLOAD BUTTON
# ==================================================
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Results CSV",
    data=csv,
    file_name="anomaly_results.csv",
    mime="text/csv"
)