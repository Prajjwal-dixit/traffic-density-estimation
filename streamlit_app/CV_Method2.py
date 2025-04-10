import cv2
import numpy as np

def count_vehicles_from_mask(mask, output_frame=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            vehicle_count += 1
            if output_frame is not None:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return vehicle_count

def estimate_density(mask):
    white_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    return round(white_pixels / total_pixels, 2)

def run_traditional_model2(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        raise Exception("Error reading video.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = "output_traditional2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = np.uint8(mag > 2.0) * 255

        vehicle_count = count_vehicles_from_mask(motion_mask, frame)
        density = estimate_density(motion_mask)

        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        density_level = "LOW" if density < 0.05 else "MEDIUM" if density < 0.1 else "HIGH"
        color = (0, 255, 0) if density_level == "LOW" else (0, 255, 255) if density_level == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, f'Density: {density_level}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        prev_gray = gray

    cap.release()
    out.release()
    return out_path
