import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import RTDETR
import torch
import os

st.set_page_config(page_title="RT-DETR Tracker", layout="wide")
st.title("Pedestrian & Vehicle Tracker with RT-DETR + ByteTrack")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    with st.spinner("Processing video..."):

        # code for saving the vid in temp file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

        # code for creating temp output path
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_video_path = temp_output.name

        # loading the model
        model = RTDETR("rtdetr_weights.pt")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # setting the vid properties
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # code for the video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # container for counted objects
        frame_data = []
        unique_vehicles = set()
        unique_pedestrians = set()

        results = model.track(
            input_video_path,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.5
        )

        for frame_idx, result in enumerate(results):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for track_id, class_id in zip(track_ids, class_ids):
                    class_name = model.names[class_id]
                    if class_name in ["car", "truck"]:
                        unique_vehicles.add(track_id)
                    elif class_name == "person":
                        unique_pedestrians.add(track_id)

            # saving the count
            frame_data.append({
                "timestamp": timestamp,
                "frame": frame_idx,
                "vehicles": len(unique_vehicles),
                "pedestrians": len(unique_pedestrians)
            })

            
            frame = result.plot()

            # code for displaying the live count
            cv2.putText(frame, f"Vehicles: {len(unique_vehicles)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pedestrians: {len(unique_pedestrians)}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)

        out.release()

        df = pd.DataFrame(frame_data)

        st.success("Processing is complete!")

        # plot of counts
        st.subheader("Object Count Over Time")
        fig, ax = plt.subplots()
        ax.plot(df["frame"], df["vehicles"], label="Vehicles", color='green')
        ax.plot(df["frame"], df["pedestrians"], label="Pedestrians", color='red')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        # code for getting the output video with annotations
        st.subheader("Download Output Video")
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
            st.download_button(
                label="Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )

        # code for downloading the csv file
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Count CSV", csv_data, "object_counts.csv", "text/csv")
