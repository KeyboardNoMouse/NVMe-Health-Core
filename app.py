import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Lenovo Telemetry Diagnostics", page_icon="💽", layout="wide")

custom_css = """
<style>
    .stApp {
        background-color: #121212;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .main {
        animation: fadeIn 1.2s ease-in-out;
    }
    .stButton>button {
        background: linear-gradient(45deg, #E2231A, #9A1812);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
        box-shadow: 0px 4px 10px rgba(226, 35, 26, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(226, 35, 26, 0.5);
        color: white;
    }
    .css-1n76uvr {
        border-radius: 8px !important;
        border: 2px dashed #E2231A !important;
        background-color: rgba(226, 35, 26, 0.05) !important;
    }
    h1, h2, h3 {
        color: #F4F4F4;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div[data-testid="stMetricValue"] {
        color: #E2231A;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_server = load_lottieurl("https://lottie.host/5b28d093-68d8-4a52-9653-a55e2edae9e2/vA0mZOhxXh.json")

col1, col2 = st.columns([2, 1])
with col1:
    st.title("⚡ Lenovo NVMe Health Core")
    st.markdown("### ThinkSystem Telemetry Analysis & Anomaly Detection")
    st.write("Upload raw SMART logs below. The AI engine will proactively isolate degradation signatures and identify high-risk drives based on official Lenovo failure parameters.")
with col2:
    if lottie_server:
        st_lottie(lottie_server, height=150, key="server_anim")

st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('nvme_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Neural Engine Offline: `nvme_model.pkl` not found. Please run the backend training script.")
    st.stop()

uploaded_file = st.file_uploader("Drop NVMe_Drive_Failure_Dataset.csv here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']
    
    if not all(feature in df.columns for feature in features):
        st.error("Data mismatch. Please ensure the CSV contains standard SMART telemetry columns.")
    else:
        X_new = df[features]
        
        with st.spinner('Analyzing drive metrics via Random Forest Engine...'):
            predictions = model.predict(X_new)
            
        df['Predicted_Status'] = predictions
        
        total_drives = len(df)
        healthy_drives = len(df[df['Predicted_Status'] == 0])
        failing_drives = total_drives - healthy_drives
        
        st.markdown("### 📊 Fleet Diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Total Drives Scanned", value=total_drives)
        m2.metric(label="✅ Healthy Drives", value=healthy_drives)
        m3.metric(label="🚨 Anomalies Detected", value=failing_drives, delta="- Action Required", delta_color="inverse")
        
        st.markdown("---")
        
        status_map = {
            0: "✅ Healthy",
            1: "⏳ Wear-Out Failure (Pattern 1)",
            4: "⚙️ Controller/Firmware Failure (Pattern 4)",
            5: "📉 Rapid Error Accumulation / Early-Life Defect (Pattern 5)"
        }
        df['Status_Label'] = df['Predicted_Status'].map(status_map)
        
        display_df = df[['Drive_ID', 'Vendor', 'Status_Label'] + features].sort_values(by='Status_Label', ascending=False)
        
        st.markdown("### 🗄️ Detailed Telemetry Report")
        st.dataframe(display_df, use_container_width=True, height=400)
