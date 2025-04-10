import torch
import torchvision
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
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model_path = "/Users/rajvijayvargiya/Downloads/yolov8n.pt"
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit(1)

model = YOLO(model_path)

input_video = "/Users/rajvijayvargiya/Downloads/Vehicle_Detection_Image_Dataset/sample_video.mp4"
output_video = "/Users/rajvijayvargiya/Downloads/yolo_video_output.mp4"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated = results[0].plot()

    # Count the total detections (vehicles)
    vehicle_count = len(results[0].boxes)

    # Use the previously-defined helper function to draw info
    draw_count_and_density(annotated, vehicle_count, x=50, y=50)

    out.write(annotated)
    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()