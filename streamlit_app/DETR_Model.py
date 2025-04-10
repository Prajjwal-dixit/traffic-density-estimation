import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).eval().to(device)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

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

def box_cxcywh_to_xyxy(box):
    x_c, y_c, w, h = box
    x1, y1 = x_c - w/2, y_c - h/2
    x2, y2 = x_c + w/2, y_c + h/2
    return [x1, y1, x2, y2]

def process_frame_detr(frame, threshold=0.7):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = detr(t)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    boxes = outputs['pred_boxes'][0, keep].cpu()
    scores = probas[keep].max(-1).values.cpu().numpy()
    w, h = img.size
    scaled_boxes = []
    for box in boxes:
        bx = box_cxcywh_to_xyxy(box)
        bx[0] *= w; bx[1] *= h; bx[2] *= w; bx[3] *= h
        scaled_boxes.append(bx)
    return np.array(scaled_boxes), scores

def draw_boxes_detr(frame, boxes, scores, threshold=0.7):
    count = 0
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        if score >= threshold:
            count += 1
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    draw_count_and_density(frame, count)
    return frame

def run_detr_model(input_path, threshold=0.7):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output_detr.mp4"
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores = process_frame_detr(frame, threshold)
        frame = draw_boxes_detr(frame, boxes, scores, threshold)
        out_vid.write(frame)

    cap.release()
    out_vid.release()
    return output_path
