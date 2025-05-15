import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define object categories
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}
PEDESTRIAN_CLASSES = {"person"}

# Load label names from external file
def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    model = YOLO(os.path.join(os.getcwd(), 'best.pt'))
    model.eval()
    return model

# Detection and video output
def detect_objects_and_generate_video(video_path, model, labels, conf_thres=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    time_series_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Time in seconds
        results = model(frame)[0]

        vehicle_count = 0
        pedestrian_count = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= conf_thres:
                cls = int(box.cls[0])
                label = labels[cls] if cls < len(labels) else str(cls)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = (0, 255, 0) if label in PEDESTRIAN_CLASSES else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label in VEHICLE_CLASSES:
                    vehicle_count += 1
                elif label in PEDESTRIAN_CLASSES:
                    pedestrian_count += 1

        time_series_data.append({
            'time_sec': round(timestamp, 2),
            'vehicles': vehicle_count,
            'pedestrians': pedestrian_count
        })

        out_vid.write(frame)

    cap.release()
    out_vid.release()

    df = pd.DataFrame(time_series_data)
    return df, output_path

# Streamlit UI
st.title("YOLO Video Detection App: Vehicles and Pedestrians")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)
    model = load_model()
    labels = load_labels()

    if st.button("Start Detection"):
        with st.spinner("Detecting... please wait"):
            df, output_video_path = detect_objects_and_generate_video(tfile.name, model, labels)

        st.success("Detection complete!")
        st.write("Time-Series Data (Vehicles and Pedestrians):")
        st.dataframe(df)

        # Plot time-series
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

        # Show and download result video
        st.subheader("Detection Output Video")
        with open(output_video_path, 'rb') as vid_file:
            st.video(vid_file.read())
            st.download_button("Download Output Video", vid_file, file_name="detection_output.mp4")
