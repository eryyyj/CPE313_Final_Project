import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from streetlight_controller import StreetLightController

# Streamlit app configuration
st.set_page_config(
    page_title="Smart Street Light Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load object detection controller
@st.cache_resource
def load_controller():
    return StreetLightController("rtdetrweights.pt")

controller = load_controller()

# App title and description
st.title("Multi-Object Detection for Optimized Street Light Energy Consumption")
st.markdown("""
This system dynamically controls street lights based on real-time object detection to optimize energy usage.
""")

# Upload video
uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

# Layout
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()

with col2:
    st.subheader("Control Panel")
    start_button = st.button("Start Simulation")
    stop_button = st.button("Stop Simulation")

    st.header("Street Light Status")
    status_placeholder = st.empty()

    st.header("Energy Savings")
    savings_placeholder = st.empty()

    st.header("Traffic Statistics")
    stats_placeholder = st.empty()

    st.header("Time-Series Graph")
    graph_placeholder = st.empty()

# State initialization
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

if start_button:
    st.session_state.run_simulation = True
    st.session_state.stats_df = pd.DataFrame(columns=['timestamp', 'vehicles', 'pedestrians'])

if stop_button:
    st.session_state.run_simulation = False

# Process uploaded video
if st.session_state.run_simulation and uploaded_video is not None:
    t0 = time.time()
    cap = cv2.VideoCapture(uploaded_video.name)
    
    stats = []

    while cap.isOpened() and st.session_state.run_simulation:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time() - t0

        detections = controller.detect_objects(frame)
        controller.control_lights(detections)
        annotated_frame = controller.visualize(frame, detections)

        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        # Street light status
        status_html = "<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;'>"
        for lamp_id, status in controller.lamp_status.items():
            status_text = "ON" if status else "OFF"
            color = "green" if status else "red"
            status_html += f"<p>Street Light {lamp_id}: <span style='color:{color};font-weight:bold;'>{status_text}</span></p>"
        status_html += "</div>"
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # Energy savings
        active_lamps = sum(controller.lamp_status.values())
        total_lamps = len(controller.lamp_status)
        savings = ((total_lamps - active_lamps) / total_lamps) * 100
        savings_placeholder.metric("Energy Savings", f"{savings:.1f}%")

        # Object counting
        vehicle_count = sum(1 for d in detections if d['class'] in ['car', 'truck'])
        pedestrian_count = sum(1 for d in detections if d['class'] == 'person'])

        stats_placeholder.write(f"""
        - Vehicles detected: {vehicle_count}
        - Pedestrians detected: {pedestrian_count}
        - Total objects: {len(detections)}
        """)

        # Log to DataFrame
        st.session_state.stats_df = pd.concat([
            st.session_state.stats_df,
            pd.DataFrame.from_dict([{
                "timestamp": round(timestamp, 2),
                "vehicles": vehicle_count,
                "pedestrians": pedestrian_count
            }])
        ], ignore_index=True)

        # Show graph
        fig, ax = plt.subplots()
        st.session_state.stats_df.plot(x='timestamp', y=['vehicles', 'pedestrians'], ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        ax.set_title("Time-Series of Detected Vehicles and Pedestrians")
        graph_placeholder.pyplot(fig)

        time.sleep(0.05)

    cap.release()

    # Save time-series CSV
    st.session_state.stats_df.to_csv("detection_stats.csv", index=False)
    st.success("Simulation completed. Time-series data saved to `detection_stats.csv`.")

elif not uploaded_video:
    video_placeholder.info("Upload a video file to begin the simulation.")
elif not st.session_state.run_simulation:
    video_placeholder.info("Click 'Start Simulation' to begin.")
