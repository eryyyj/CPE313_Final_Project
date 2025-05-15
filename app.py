import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Define your vehicle and pedestrian categories here
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}
PEDESTRIAN_CLASSES = {"person"}

def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    model = YOLO(os.path.join(os.getcwd(), 'best.pt'))
    model.eval()
    return model

def detect_objects_in_video(video_path, model, labels, conf_thres=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_series_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
        results = model(frame)[0]

        vehicle_count = 0
        pedestrian_count = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= conf_thres:
                cls = int(box.cls[0])
                label = labels[cls] if cls < len(labels) else str(cls)
                if label in VEHICLE_CLASSES:
                    vehicle_count += 1
                elif label in PEDESTRIAN_CLASSES:
                    pedestrian_count += 1

        time_series_data.append({
            'time_sec': round(timestamp, 2),
            'vehicles': vehicle_count,
            'pedestrians': pedestrian_count
        })

    cap.release()
    return pd.DataFrame(time_series_data)

# --- Streamlit UI ---
st.title("YOLO Video Vehicle and Pedestrian Detection App")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)
    model = load_model()
    labels = load_labels()

    if st.button("Start Detection"):
        with st.spinner("Detecting objects..."):
            df = detect_objects_in_video(tfile.name, model, labels)

        st.success("Detection complete!")
        st.write("Time-Series Detection Data (Vehicles & Pedestrians):")
        st.dataframe(df)

        # Time-series graph
        st.subheader("Detection Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(df['time_sec'], df['vehicles'], label='Vehicles', marker='o')
        plt.plot(df['time_sec'], df['pedestrians'], label='Pedestrians', marker='x')
        plt.xlabel("Time (s)")
        plt.ylabel("Count")
        plt.title("Object Count Over Time")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
