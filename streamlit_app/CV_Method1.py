import cv2
import numpy as np

def run_traditional_model1(video_path):
    MIN_CONTOUR_AREA = 800
    IOU_THRESHOLD_NMS = 0.3
    MAX_MISSED_FRAMES = 10

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    persistent_detections = []
    prev_frame = None

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def non_max_suppression_fast(boxes, overlap_thresh):
        if len(boxes) == 0:
            return []
        boxes_arr = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=float)
        pick = []
        x1, y1 = boxes_arr[:, 0], boxes_arr[:, 1]
        x2, y2 = boxes_arr[:, 2], boxes_arr[:, 3]
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
                w_ = max(0, xx2 - xx1 + 1)
                h_ = max(0, yy2 - yy1 + 1)
                overlap = float(w_ * h_) / area[i]
                if overlap > overlap_thresh:
                    suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        return [boxes[i] for i in pick]

    def update_persistent_detections(current_boxes):
        nonlocal persistent_detections
        updated = []
        matched_current = [False] * len(current_boxes)
        for detection in persistent_detections:
            best_iou = 0
            best_idx = -1
            for idx, curr_box in enumerate(current_boxes):
                iou = compute_iou(detection['box'], curr_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou > 0.3:
                detection['box'] = current_boxes[best_idx]
                detection['age'] = 0
                matched_current[best_idx] = True
                updated.append(detection)
            else:
                detection['age'] += 1
                if detection['age'] <= MAX_MISSED_FRAMES:
                    updated.append(detection)
        for idx, flag in enumerate(matched_current):
            if not flag:
                updated.append({'box': current_boxes[idx], 'age': 0})
        persistent_detections = updated

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = "output_traditional1.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 600))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 600))
        fg_mask = bg_subtractor.apply(frame, learningRate=0.005)

        if prev_frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_frame, gray_prev)
            _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, diff_mask)
        prev_frame = frame.copy()

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w_, h_ = cv2.boundingRect(cnt)
            current_boxes.append((x, y, w_, h_))

        current_boxes = non_max_suppression_fast(current_boxes, IOU_THRESHOLD_NMS)
        update_persistent_detections(current_boxes)

        final_boxes = [d['box'] for d in persistent_detections]
        for (x, y, w_, h_) in final_boxes:
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)

        vehicle_area = sum([w_ * h_ for (_, _, w_, h_) in final_boxes])
        road_area = frame.shape[0] * frame.shape[1]
        density_ratio = vehicle_area / road_area

        if density_ratio < 0.2:
            density = "LOW"
            color = (0, 255, 0)
        elif density_ratio < 0.5:
            density = "MEDIUM"
            color = (0, 255, 255)
        else:
            density = "HIGH"
            color = (0, 0, 255)

        cv2.putText(frame, f"Detected vehicles: {len(final_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Traffic density: {density}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path
