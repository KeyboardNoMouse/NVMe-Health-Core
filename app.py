import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lenovo | NVMe Health Core", page_icon="🔴", layout="wide")

# --- CSS: Minimalist Dashboard + Original Sidebar Style ---
custom_css = """
<style>
    /* Main Dashboard Background */
    .stApp { background-color: #0E0E0E; color: #E0E0E0; }
    
    /* Glassmorphism Cards for Main Content */
    div[data-testid="stMetricBlock"], .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #333;
    }
    
    /* FORCE REMOVE TOP PADDING IN SIDEBAR */
    section[data-testid="stSidebar"] .stSidebarContent, 
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }

    /* ThinkPad Red Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #E2231A, #9A1812);
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(226, 35, 26, 0.4);
    }

    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #FFFFFF; }
    
    /* FIX: Changed metric numbers back to white */
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    
    /* FIX: Changed the small labels back to white */
    .tech-label {
        font-size: 0.75rem;
        opacity: 0.6;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 5px;
        color: #FFFFFF;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- UTILITIES ---
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_pulse = load_lottieurl("https://lottie.host/8e7f8373-c15b-486a-912c-96b612140889/t2XG2SIsGZ.json")

@st.cache_resource
def load_model():
    return joblib.load('nvme_model.pkl')

model = load_model()

status_map = {
    0: "✅ Healthy",
    1: "⏳ Wear-Out Failure (Pattern 1)",
    4: "⚙️ Controller/Firmware Failure (Pattern 4)",
    5: "📉 Rapid Error Accumulation (Pattern 5)"
}

# --- SIDEBAR: SIMULATION LAB ---
with st.sidebar:
    st.title("🔬 Simulation Lab")
    st.write("Manually adjust metrics to test the AI's real-time response.")
    
    s_temp = st.slider("Temperature (°C)", 20, 100, 40)
    s_media = st.slider("Media Errors", 0, 50, 0)
    s_unsafe = st.slider("Unsafe Shutdowns", 0, 20, 2)
    s_read = st.slider("Read Error Rate", 0, 50, 5)
    s_write = st.slider("Write Error Rate", 0, 50, 5)
    s_life = st.slider("Percent Life Used", 0, 100, 20)
    
    # Model Calculation
    sim_data = pd.DataFrame([[s_temp, s_media, s_unsafe, s_read, s_write, s_life]], 
                            columns=['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                                     'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used'])
    
    sim_pred = model.predict(sim_data)[0]
    sim_proba = model.predict_proba(sim_data).max() * 100
    
    st.markdown("---")
    st.subheader("Simulation Result:")
    st.write(f"**Status:** {status_map.get(sim_pred, 'Unknown')}")
    st.write(f"**Confidence:** {sim_proba:.1f}%")
    st.progress(sim_proba/100)

# --- MAIN DASHBOARD ---
c1, c2 = st.columns([4, 1])
with c1:
    st.title("Lenovo NVMe Health Core")
    st.markdown("<p style='opacity:0.6; font-size:1.1rem;'>ThinkSystem AI Telemetry Monitoring Service</p>", unsafe_allow_html=True)
with c2:
    if lottie_pulse:
        st_lottie(lottie_pulse, height=100)

st.markdown("<br>", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload Fleet Telemetry (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
                'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']
    
    if all(f in df.columns for f in features):
        # AI Logic
        preds = model.predict(df[features])
        probas = model.predict_proba(df[features]).max(axis=1) * 100
        
        df['Predicted_Status'] = [status_map.get(p, "Unknown") for p in preds]
        df['Confidence_%'] = probas.round(1)
        
        # Dashboard Overview
        st.markdown("<p class='tech-label'>Fleet Health Overview</p>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        total, fails = len(df), len(df[df['Predicted_Status'] != "✅ Healthy"])
        
        m1.metric("Nodes Scanned", total)
        m2.metric("Critical Alerts", fails, delta="- Action Required" if fails > 0 else None, delta_color="inverse")
        m3.metric("Uptime Probability", f"{((total-fails)/total)*100:.1f}%")
        
        st.markdown("<br><p class='tech-label'>Telemetry Diagnostic Logs</p>", unsafe_allow_html=True)
        # Displaying main columns for cleanliness
        show_cols = ['Drive_ID', 'Predicted_Status', 'Confidence_%'] + features
        st.dataframe(df[show_cols].sort_values(by='Confidence_%', ascending=False), use_container_width=True)
    else:
        st.error("CSV Schema mismatch. Please use standard Lenovo telemetry logs.")
else:
    st.info("System Ready. Please ingest CSV telemetry for batch diagnostic analysis.")

st.markdown("---")
st.caption("Developed for Lenovo Official Demo | v1.1 Pro")
