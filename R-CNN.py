import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

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

# Use CPU to avoid the MPS error on Apple M2
device = torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
transform = T.Compose([T.ToTensor()])

def faster_process_frame(frame, threshold=0.5):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        preds = model([transform(img).to(device)])[0]
    boxes = preds['boxes'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    idxs = scores >= threshold
    return boxes[idxs], scores[idxs]

def draw_boxes_faster(frame, boxes, scores):
    count = len(boxes)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    draw_count_and_density(frame, count, x=10, y=30)
    return frame

def run_faster_video(input_video, output_video, threshold=0.5):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video.")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores = faster_process_frame(frame, threshold)
        frame = draw_boxes_faster(frame, boxes, scores)
        out.write(frame)
        cv2.imshow("Faster R-CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run the video processing
run_faster_video(
    "/Users/rajvijayvargiya/Downloads/Vehicle_Detection_Image_Dataset/sample_video.mp4",
    "/Users/rajvijayvargiya/Downloads/fasterrcnn_video_output.mp4"
)