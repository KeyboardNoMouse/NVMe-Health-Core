import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Lenovo Telemetry Diagnostics", page_icon="💽", layout="wide")

custom_css = """
<style>
    .stApp { background-color: #121212; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .main { animation: fadeIn 1.2s ease-in-out; }
    .stButton>button {
        background: linear-gradient(45deg, #E2231A, #9A1812);
        color: white; font-weight: bold; border: none; border-radius: 4px;
        padding: 10px 24px; transition: all 0.3s ease 0s;
        box-shadow: 0px 4px 10px rgba(226, 35, 26, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(226, 35, 26, 0.5);
    }
    .css-1n76uvr {
        border-radius: 8px !important; border: 2px dashed #E2231A !important;
        background-color: rgba(226, 35, 26, 0.05) !important;
    }
    h1, h2, h3 { color: #F4F4F4; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetricValue"] { color: #E2231A; }
    .stSidebar { background-color: #1c1c1c !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_server = load_lottieurl("https://lottie.host/5b28d093-68d8-4a52-9653-a55e2edae9e2/vA0mZOhxXh.json")

@st.cache_resource
def load_model():
    return joblib.load('nvme_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Model file 'nvme_model.pkl' not found.")
    st.stop()

# --- SIDEBAR: SIMULATION LAB ---
st.sidebar.title("🔬 Simulation Lab")
st.sidebar.write("Manually adjust metrics to test the AI's real-time response.")

s_temp = st.sidebar.slider("Temperature (°C)", 20, 100, 40)
s_media = st.sidebar.slider("Media Errors", 0, 50, 0)
s_unsafe = st.sidebar.slider("Unsafe Shutdowns", 0, 20, 2)
s_read = st.sidebar.slider("Read Error Rate", 0, 50, 5)
s_write = st.sidebar.slider("Write Error Rate", 0, 50, 5)
s_life = st.sidebar.slider("Percent Life Used", 0, 100, 20)

# Real-time Prediction for Simulation
sim_data = pd.DataFrame([[s_temp, s_media, s_unsafe, s_read, s_write, s_life]], 
                        columns=['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                                 'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used'])

sim_pred = model.predict(sim_data)[0]
sim_proba = model.predict_proba(sim_data).max() * 100

status_map = {
    0: "✅ Healthy",
    1: "⏳ Wear-Out Failure (Pattern 1)",
    4: "⚙️ Controller/Firmware Failure (Pattern 4)",
    5: "📉 Rapid Error Accumulation (Pattern 5)"
}

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Result:")
st.sidebar.write(f"**Status:** {status_map.get(sim_pred, 'Unknown')}")
st.sidebar.write(f"**Confidence:** {sim_proba:.1f}%")

# --- MAIN PAGE ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("⚡ Lenovo NVMe Health Core")
    st.markdown("### ThinkSystem Predictive Maintenance Engine")
with col2:
    if lottie_server:
        st_lottie(lottie_server, height=120)

uploaded_file = st.file_uploader("Upload NVMe Telemetry (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']
    
    if all(f in df.columns for f in features):
        preds = model.predict(df[features])
        probas = model.predict_proba(df[features]).max(axis=1) * 100
        
        df['Predicted_Status'] = [status_map.get(p, "Unknown") for p in preds]
        df['Confidence_%'] = probas.round(1)
        
        total = len(df)
        anomalies = len(df[df['Predicted_Status'] != "✅ Healthy"])
        
        st.markdown("### 📊 Fleet Diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Drives Scanned", total)
        m2.metric("Anomalies Detected", anomalies, delta="- Action Required" if anomalies > 0 else None, delta_color="inverse")
        m3.metric("System Health", f"{((total-anomalies)/total)*100:.1f}%")
        
        st.markdown("### 🗄️ Detailed Telemetry Report")
        st.dataframe(df[['Drive_ID', 'Predicted_Status', 'Confidence_%'] + features].sort_values(by='Confidence_%', ascending=False), use_container_width=True)
    else:
        st.error("CSV must contain: " + ", ".join(features))
