import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streetlight_controller import StreetLightController

# Streamlit app configuration
st.set_page_config(
    page_title="Smart Street Light Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize controller
@st.cache_resource
def load_controller():
    return StreetLightController("best.pt")

controller = load_controller()

# App title and description
st.title("Multi-Object Detection for Optimized Street Light Energy Consumption")
st.markdown("""
This system dynamically controls street lights based on real-time object detection to optimize energy usage.
""")

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Real-Time Monitoring")
    video_placeholder = st.empty()
    
    # Control panel
    st.subheader("Control Panel")
    start_button = st.button("Start Simulation")
    stop_button = st.button("Stop Simulation")
    
    if start_button:
        st.session_state.run_simulation = True
    if stop_button:
        st.session_state.run_simulation = False

with col2:
    st.header("Street Light Status")
    status_placeholder = st.empty()
    
    st.header("Energy Savings")
    savings_placeholder = st.empty()
    
    st.header("Traffic Statistics")
    stats_placeholder = st.empty()

# Main processing loop
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

if st.session_state.run_simulation:
    # For demo purposes, we'll use a sample video
    cap = cv2.VideoCapture("traffic_sample.mp4")
    
    while st.session_state.run_simulation and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detections = controller.detect_objects(frame)
        controller.control_lights(detections)
        annotated_frame = controller.visualize(frame, detections)
        
        # Update Streamlit displays
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        # Update status panel
        status_html = "<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;'>"
        for lamp_id, status in controller.lamp_status.items():
            status_text = "ON" if status else "OFF"
            color = "green" if status else "red"
            status_html += f"<p>Street Light {lamp_id}: <span style='color:{color};font-weight:bold;'>{status_text}</span></p>"
        status_html += "</div>"
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        
        # Calculate and display energy savings (example calculation)
        active_lamps = sum(controller.lamp_status.values())
        total_lamps = len(controller.lamp_status)
        savings = ((total_lamps - active_lamps) / total_lamps) * 100
        savings_placeholder.metric("Energy Savings", f"{savings:.1f}%")
        
        # Display traffic statistics
        vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck'])
        pedestrian_count = sum(1 for d in detections if d['class'] == 'person')
        stats_placeholder.write(f"""
        - Vehicles detected: {vehicle_count}
        - Pedestrians detected: {pedestrian_count}
        - Total objects: {len(detections)}
        """)
        
        time.sleep(0.1)  # Control frame rate
    
    cap.release()
else:
    video_placeholder.info("Click 'Start Simulation' to begin")