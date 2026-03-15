import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NVMe Health Core", page_icon="💽", layout="wide")

# --- CUSTOM CSS INJECTION ---
# Flex those front-end skills here! You can tweak these hex codes and animations.
custom_css = """
<style>
    /* Main background fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .main {
        animation: fadeIn 1.2s ease-in-out;
    }
    /* Stylish glowing button */
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
        box-shadow: 0px 5px 15px rgba(0, 242, 254, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 20px rgba(0, 242, 254, 0.6);
        color: white;
    }
    /* Style the file uploader box */
    .css-1n76uvr {
        border-radius: 15px !important;
        border: 2px dashed #4facfe !important;
        background-color: rgba(79, 172, 254, 0.05) !important;
    }
    /* Headers */
    h1, h2, h3 {
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- HELPER FUNCTION FOR ANIMATIONS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a cool tech/server animation from LottieFiles
lottie_server = load_lottieurl("https://lottie.host/5b28d093-68d8-4a52-9653-a55e2edae9e2/vA0mZOhxXh.json")

# --- HEADER SECTION ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("⚡ NVMe Drive Health Core")
    st.markdown("### AI-Powered Telemetry Analysis & Anomaly Detection")
    st.write("Upload your raw server logs below. The neural engine will isolate degradation signatures and identify top failure patterns instantly.")
with col2:
    if lottie_server:
        st_lottie(lottie_server, height=150, key="server_anim")

st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return joblib.load('nvme_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Neural Engine Offline: `nvme_model.pkl` not found. Please run the backend training script.")
    st.stop()

# --- MAIN DASHBOARD AREA ---
uploaded_file = st.file_uploader("Drop NVMe_Drive_Failure_Dataset.csv here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Run predictions silently in the background
    features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']
    
    # Catch missing columns to prevent ugly errors
    if not all(feature in df.columns for feature in features):
        st.error("Data mismatch. Please ensure the CSV contains standard SMART telemetry columns.")
    else:
        X_new = df[features]
        predictions = model.predict(X_new)
        df['Predicted_Status'] = predictions
        
        # Calculate dashboard metrics
        total_drives = len(df)
        healthy_drives = len(df[df['Predicted_Status'] == 0])
        failing_drives = total_drives - healthy_drives
        
        # Display Top-Level Metrics
        st.markdown("### 📊 Fleet Diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Total Drives Scanned", value=total_drives)
        m2.metric(label="✅ Healthy Drives", value=healthy_drives)
        m3.metric(label="🚨 Anomalies Detected", value=failing_drives, delta="- Action Required", delta_color="inverse")
        
        st.markdown("---")
        
        # Map labels and display the data
        status_map = {
            0: "✅ Healthy",
            1: "🚨 End of Life (Wear-Out)",
            4: "⚠️ Media Corruption",
            5: "❌ Controller Error"
        }
        df['Status_Label'] = df['Predicted_Status'].map(status_map)
        
        # Reorder columns for a better user experience
        display_df = df[['Drive_ID', 'Vendor', 'Status_Label'] + features].sort_values(by='Status_Label', ascending=False)
        
        st.markdown("### 🗄️ Detailed Telemetry Report")
        st.dataframe(display_df, use_container_width=True, height=400)