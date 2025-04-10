# yolo_model.py
import torch
import cv2
import os
from ultralytics import YOLO

def get_density_label(count):
    if count < 5:
        return "Low"
    elif count < 10:
        return "Medium"
    else:
        return "High"

def draw_count_and_density(frame, vehicle_count, x=10, y=30):
    density_label = get_density_label(vehicle_count)
    text = f"Vehicles: {vehicle_count} | Density: {density_label}"
    color = (0, 255, 0) if density_label == "Low" else (0, 255, 255) if density_label == "Medium" else (0, 0, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def run_yolo_model(input_video_path, model_path="yolov8n.pt"):
    # Load YOLO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = "output_yolo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        vehicle_count = len(results[0].boxes)
        draw_count_and_density(annotated, vehicle_count, x=50, y=50)

        out.write(annotated)

    cap.release()
    out.release()
    return output_path
