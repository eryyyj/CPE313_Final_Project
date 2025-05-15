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

def load_labels(path='labels.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    model = YOLO(os.path.join(os.getcwd(), 'rtdetr_weights.pt'))
    model.eval()
    return model

def detect_objects_in_video(video_path, model, labels, conf_thres=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    time_series_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_count = defaultdict(int)

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= conf_thres:
                cls = int(box.cls[0])
                label = labels[cls] if cls < len(labels) else str(cls)
                frame_count[label] += 1

        time_series_data.append({'frame': frame_idx, **frame_count})
        frame_idx += 1

    cap.release()
    return pd.DataFrame(time_series_data).fillna(0)

# --- Streamlit Interface ---
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
        st.write("Time-Series Object Count Data:")
        st.dataframe(df)

        # Plotting
        st.subheader("Detection Over Time")
        plt.figure(figsize=(10, 5))
        for label in df.columns:
            if label != 'frame':
                plt.plot(df['frame'], df[label], label=label)

        plt.xlabel("Frame")
        plt.ylabel("Object Count")
        plt.title("Object Detection Over Time")
        plt.legend()
        st.pyplot(plt)
