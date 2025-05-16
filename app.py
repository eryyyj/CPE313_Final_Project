import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics.trackers.byte_tracker import BYTETracker

# Define object categories
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}
PEDESTRIAN_CLASSES = {"person"}

# Load label names from external file
def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    model = YOLO(os.path.join(os.getcwd(), 'rtdetr_weights.pt'))
    model.eval()
    return model

# Detection, tracking, and video output
def detect_and_track(video_path, model, labels, conf_thres=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = BYTETracker()

    time_series_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(frame)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= conf_thres:
                cls = int(box.cls[0])
                label = labels[cls] if cls < len(labels) else str(cls)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': conf,
                    'class_id': cls,
                    'label': label
                })

        tracks = tracker.update(detections)

        vehicle_ids = set()
        pedestrian_ids = set()

        for track in tracks:
            x1, y1, w, h = track['bbox']
            track_id = track['track_id']
            label = labels[track['class_id']] if track['class_id'] < len(labels) else str(track['class_id'])

            color = (0, 255, 0) if label in PEDESTRIAN_CLASSES else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label in VEHICLE_CLASSES:
                vehicle_ids.add(track_id)
            elif label in PEDESTRIAN_CLASSES:
                pedestrian_ids.add(track_id)

        time_series_data.append({
            'timestamp': current_time,
            'vehicles': len(vehicle_ids),
            'pedestrians': len(pedestrian_ids)
        })

        out_vid.write(frame)

    cap.release()
    out_vid.release()

    df = pd.DataFrame(time_series_data)
    return df, output_path

# Streamlit UI
st.title("YOLOv8 + ByteTrack: Vehicle and Pedestrian Tracking")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)
    model = load_model()
    labels = load_labels()

    if st.button("Start Detection and Tracking"):
        with st.spinner("Processing... please wait"):
            df, output_video_path = detect_and_track(tfile.name, model, labels)

        st.success("Processing complete!")
        st.write("Time-Series Data (Vehicles and Pedestrians):")
        st.dataframe(df)

        # Plot time-series
        st.subheader("Detection Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['vehicles'], label='Vehicles', marker='o')
        plt.plot(df['timestamp'], df['pedestrians'], label='Pedestrians', marker='x')
        plt.xticks(rotation=45)
        plt.xlabel("Timestamp")
        plt.ylabel("Count")
        plt.title("Object Count Over Time")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(plt)

        # Show and download result video
        st.subheader("Detection Output Video")
        with open(output_video_path, 'rb') as vid_file:
            video_bytes = vid_file.read()
            st.video(video_bytes)
            st.download_button("Download Output Video", video_bytes, file_name="detection_output.mp4")
