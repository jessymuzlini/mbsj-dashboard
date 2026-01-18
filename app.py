import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="MBSJ Command Center",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling (MBSJ Light Theme) ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f1f5f9; /* slate-100 */
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric Cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-label {
        color: #64748b; /* slate-500 */
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #0f172a; /* slate-900 */
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }

    /* Custom Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .header-title {
        color: #0f172a;
        font-weight: 800;
        font-size: 1.5rem;
    }
    .system-badge {
        background-color: #dcfce7; /* green-100 */
        color: #166534; /* green-800 */
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a; /* Dark Blue */
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Majlis_Bandaraya_Subang_Jaya_%28MBSJ%29_Logo.svg/1200px-Majlis_Bandaraya_Subang_Jaya_%28MBSJ%29_Logo.svg.png", width=80)
    st.title("MBSJ SDDS v2.0")
    
    st.markdown("---")
    st.subheader("Navigation")
    st.button("Dashboard", use_container_width=True, type="primary")
    st.button("Cameras", use_container_width=True)
    st.button("Map View", use_container_width=True)
    st.button("Reports", use_container_width=True)
    
    st.markdown("---")
    st.subheader("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# --- Header ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
        <div class="dashboard-header">
            <div>
                <div class="header-title">MBSJ COMMAND CENTER</div>
                <div style="color: #64748b; font-size: 0.9rem;">Stray Dog Detection System (SDDS) v2.0</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
        <div style="display: flex; justify-content: flex-end;">
            <div class="system-badge">● SYSTEM ONLINE</div>
        </div>
    """, unsafe_allow_html=True)

# --- Stats Row ---
c1, c2, c3, c4 = st.columns(4)

def metric_card(title, value, icon, color):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

with c1:
    st.markdown(metric_card("Alerts Today", "12", "⚠️", "red"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Stray Dogs", "42", "🐕", "blue"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Cases Resolved", "8", "✅", "green"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Active Cams", "2", "📹", "slate"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Layout ---
col_main, col_right = st.columns([2, 1])

# --- Live Feed Logic (WebRTC) ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# STUN servers to allow connection through firewalls
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

model_path = "best (1).pt"
try:
    model = YOLO(model_path)
except:
    st.error(f"Model not found at {model_path}. Please check file path.")
    model = None

class DogDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if model:
            # Run inference
            results = model(img, verbose=False, conf=0.3)
            # Annotate
            img = results[0].plot()
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_main:
    st.markdown("<h3 style='color: #0f172a;'>🔴 Live Feed: Jalan SK 6/1</h3>", unsafe_allow_html=True)
    st.write("Click 'START' to use your camera. (Allow permissions)")
    
    webrtc_streamer(
        key="doggos",
        video_transformer_factory=DogDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- Right Sidebar Stack ---
with col_right:
    # Priority Actions
    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
        <h3 style="margin-top:0; color: #ef4444; display: flex; align-items: center; gap: 0.5rem;">
            ⚠️ High Confidence Alert
        </h3>
        <p style="color: #64748b; font-size: 0.9rem;">
            <strong>Location:</strong> Jalan SK 6/1 (Serdang)<br>
            <strong>Time:</strong> Just Now<br>
            <strong>Count:</strong> 1 Stray Dog Detected
        </p>
        <button style="
            background-color: #ef4444; 
            color: white; 
            border: none; 
            padding: 0.5rem 1rem; 
            border-radius: 6px; 
            font-weight: 700; 
            width: 100%; 
            cursor: pointer;
            margin-top: 0.5rem;
        ">DISPATCH TEAM</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent Logs
    st.markdown("<h3 style='color: #0f172a;'>Recent Logs</h3>", unsafe_allow_html=True)
    
    logs = [
        {"loc": "Pasar Borong", "time": "10 mins ago", "type": "STRAY", "color": "#fee2e2", "text": "#ef4444"},
        {"loc": "Taman Universiti", "time": "25 mins ago", "type": "PET", "color": "#dcfce7", "text": "#166534"},
        {"loc": "Jalan Besar", "time": "1 hour ago", "type": "STRAY", "color": "#fee2e2", "text": "#ef4444"},
    ]
    
    for log in logs:
        st.markdown(f"""
        <div style="
            background-color: white; 
            padding: 1rem; 
            border-bottom: 1px solid #f1f5f9; 
            display: flex; 
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #0f172a;">{log['loc']}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{log['time']}</div>
            </div>
            <div style="
                background-color: {log['color']}; 
                color: {log['text']}; 
                padding: 0.2rem 0.6rem; 
                border-radius: 4px; 
                font-size: 0.7rem; 
                font-weight: 700;
            ">{log['type']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Heatmap (Bottom) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #0f172a;'>Zone Map - Seri Kembangan</h3>", unsafe_allow_html=True)

# Create Map
m = folium.Map(location=[3.032, 101.715], zoom_start=14, tiles="CartoDB dark_matter")

# Add Circles
def add_circle(loc, color, count, r):
    folium.Circle(
        location=loc,
        radius=r,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.4,
        popup=f"Detections: {count}"
    ).add_to(m)

add_circle([3.035, 101.718], '#ef4444', 8, 300) # High
add_circle([3.028, 101.710], '#eab308', 5, 250) # Med
add_circle([3.030, 101.722], '#ef4444', 3, 200) 
add_circle([3.040, 101.705], '#06b6d4', 2, 150) # Low

st_folium(m, width="100%", height=400, returned_objects=[])
