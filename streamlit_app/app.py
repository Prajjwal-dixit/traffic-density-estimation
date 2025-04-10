import streamlit as st
import os

from CV_Method1 import run_traditional_model1
from CV_Method2 import run_traditional_model2
from YOLO_Model import run_yolo_model
from Faster_RCNN_Model import run_faster_model
from DETR_Model import run_detr_model

# --- Page Configuration ---
st.set_page_config(page_title="Traffic Density Estimation", layout="wide")

st.title("🚦 Traffic Density Estimation App")
st.markdown("Upload a traffic video and choose a model to analyze vehicle density.")

# --- Sidebar Selection ---
model_choice = st.sidebar.selectbox(
    "Select Model",
    (
        "Traditional CV - Background Subtraction",
        "Traditional CV - Optical Flow",
        "YOLO (Deep Learning)",
        "Faster R-CNN (Deep Learning)",
        "DETR (Deep Learning)"
    )
)

uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4", "avi", "mov"])

# --- Model Trigger ---
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Video uploaded successfully.")
    st.video(temp_video_path)

    if st.button("🚀 Run Model"):
        with st.spinner("Processing video... Please wait."):
            try:
                if model_choice == "Traditional CV - Background Subtraction":
                    output_path = run_traditional_model1(temp_video_path)
                elif model_choice == "Traditional CV - Optical Flow":
                    output_path = run_traditional_model2(temp_video_path)
                elif model_choice == "YOLO (Deep Learning)":
                    output_path = run_yolo_model(temp_video_path, model_path="yolov8n.pt")
                elif model_choice == "Faster R-CNN (Deep Learning)":
                    output_path = run_faster_model(temp_video_path)
                elif model_choice == "DETR (Deep Learning)":
                    output_path = run_detr_model(temp_video_path)
                else:
                    st.error("Unknown model selected.")
                    output_path = None

                if output_path:
                    st.success("✅ Processing complete!")
                    st.video(output_path)

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")

else:
    st.info("Upload a video file to begin.")
