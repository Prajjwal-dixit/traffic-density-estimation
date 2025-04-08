import cv2
import numpy as np
import math

# --- Parameters ---
MIN_CONTOUR_AREA = 800         # Minimum contour area to be considered a vehicle
PIXEL_TO_KMPH = 0.1            # Conversion factor: adjust based on calibration
DENSITY_THRESHOLDS = {         # Traffic density thresholds
    "LOW": 5,                
    "MEDIUM": 10             
}
VIDEO_SOURCE = "video.mp4"  

# Non-Maximum Suppression (NMS) parameter
IOU_THRESHOLD_NMS = 0.3

# Persistent detections parameters
MAX_MISSED_FRAMES = 10         # How many frames to keep a detection when not updated

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# --- Initialize Background Subtractor ---
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# List to hold persistent detections:
# Each entry is a dictionary: {'box': (x, y, w, h), 'age': int }
persistent_detections = []

prev_centroids = []  # For speed estimation
prev_frame = None    # Used for frame differencing

# --- Helper Functions ---

def compute_iou(boxA, boxB):
    """ Compute Intersection over Union (IoU) between two boxes.
        Boxes are in (x, y, w, h) format. """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def non_max_suppression_fast(boxes, overlap_thresh):
    """ Applies non-maximum suppression (NMS) to overlapping boxes. """
    if len(boxes) == 0:
        return []

    # Convert to array format: [x, y, x+w, y+h]
    boxes_arr = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=float)
    pick = []

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs) - 1]
        for pos in range(len(idxs)-1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[i]
            if overlap > overlap_thresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    
    return [boxes[i] for i in pick]

def update_persistent_detections(current_boxes):
    """
    Update the persistent detections list with current detection boxes.
    For each existing persistent detection, check if a current box overlaps (IoU > threshold).
    If yes, update the box and reset its age.
    If not, increment its age. New unmatched detections are added.
    Remove detections older than MAX_MISSED_FRAMES.
    """
    global persistent_detections
    updated = []

    # First, try to match each persistent detection with current boxes
    matched_current = [False] * len(current_boxes)
    for detection in persistent_detections:
        best_iou = 0
        best_idx = -1
        for idx, curr_box in enumerate(current_boxes):
            iou = compute_iou(detection['box'], curr_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou > 0.3:  # IoU threshold for matching
            # Update detection with new box and reset age
            detection['box'] = current_boxes[best_idx]
            detection['age'] = 0
            matched_current[best_idx] = True
            updated.append(detection)
        else:
            # Increase age because it was not updated in this frame
            detection['age'] += 1
            if detection['age'] <= MAX_MISSED_FRAMES:
                updated.append(detection)

    # Add new detections that did not match any persistent detection
    for idx, flag in enumerate(matched_current):
        if not flag:
            updated.append({'box': current_boxes[idx], 'age': 0})
            
    persistent_detections = updated

# --- Main processing loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    
    # --- Background Subtraction ---
    fg_mask = bg_subtractor.apply(frame, learningRate=0.005)
    
    # Frame differencing (to capture subtle changes in stagnant vehicles)
    if prev_frame is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame, gray_prev)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_or(fg_mask, diff_mask)
    prev_frame = frame.copy()
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # --- Find Contours ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        current_boxes.append((x, y, w, h))
    
    # --- Apply Non-Maximum Suppression on current detections ---
    current_boxes = non_max_suppression_fast(current_boxes, IOU_THRESHOLD_NMS)
    
    # --- Update Persistent Detections ---
    update_persistent_detections(current_boxes)
    
    # The vehicle count for this frame is taken from persistent detections
    final_boxes = [d['box'] for d in persistent_detections]
    
    current_centroids = []
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = x + w/2, y + h/2
        current_centroids.append((cx, cy))
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    # --- Vehicle Count & Traffic Density ---
    vehicle_count = len(final_boxes)
    if vehicle_count < DENSITY_THRESHOLDS["LOW"]:
        density = "LOW"
    elif vehicle_count < DENSITY_THRESHOLDS["MEDIUM"]:
        density = "MEDIUM"
    else:
        density = "HIGH"
    
    # --- Speed Estimation (Simple average displacement) ---
    speeds = []
    if prev_centroids and current_centroids:
        for cx, cy in current_centroids:
            closest_dist = np.inf
            for pcx, pcy in prev_centroids:
                dist = math.hypot(cx - pcx, cy - pcy)
                if dist < closest_dist:
                    closest_dist = dist
            speeds.append(closest_dist)
    avg_displacement = np.mean(speeds) if speeds else 0
    avg_speed = avg_displacement * PIXEL_TO_KMPH

    # --- Overlay Text ---
    overlay_texts = [
        f"Detected vehicles: {vehicle_count}",
        f"Traffic density: {density}",
        f"Avg speed: {avg_speed:.1f} km/h"
    ]
    for i, text in enumerate(overlay_texts):
        cv2.putText(frame, text, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # --- Display ---
    cv2.imshow("Traffic Analysis", frame)
    
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    prev_centroids = current_centroids.copy()

cap.release()
cv2.destroyAllWindows()
