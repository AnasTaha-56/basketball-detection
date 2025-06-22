# This is best worj^king script curently

from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np

# === Config ===
MODEL_PATH = "./models/newnew.pt"
VIDEO_PATH = "./shots/makeMakeMiss480.mp4"

# Thresholds
HOOP_CONF_THRESHOLD = 0.6
BALL_CONF_THRESHOLD = 0.4

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# === State Variables ===
hoop_locked = False
locked_hoop_box = None
scoring_line_y_one = None
scoring_line_y_two = None
scoring_line_x_one_range = None
scoring_line_x_two_range = None

ball_centers = deque(maxlen=30)
ball_boxes = deque(maxlen=30)
shot_state = "waiting"
frames_since_last_ball = 0
max_no_ball_frames = 20

make_count = 0
miss_count = 0
total_shots = 0

ball_above_hoop = False
passed_line_one = False

# === Helpers ===
def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def check_line_cross(prev_y, curr_y, line_y):
    return prev_y < line_y and curr_y >= line_y

def check_in_range_and_width(cx, box, x_range, hoop_width):
    x1, y1, x2, y2 = box
    ball_width = x2 - x1
    in_range = x_range[0] <= cx <= x_range[1]
    width_ok = 0.4 * hoop_width <= ball_width <= 0.9 * hoop_width
    return in_range and width_ok

def is_likely_ball(ball_box, frame, min_area=100, max_aspect_ratio=2.0, min_circularity=0.1):
    x1, y1, x2, y2 = map(int, ball_box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < min_area or perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if circularity >= min_circularity and aspect_ratio <= max_aspect_ratio:
            return True
    return False

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        if shot_state in ["in_progress", "passed_line_one"] and ball_above_hoop:
            miss_count += 1
            total_shots += 1
        break

    results = model(frame, stream=True)
    ball_detected = False
    current_ball_cx = None
    current_ball_cy = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            label = f"{class_name} {conf:.2f}"

            if class_name.lower() == "hoop" and conf >= HOOP_CONF_THRESHOLD and not hoop_locked:
                locked_hoop_box = (x1, y1, x2, y2)
                hoop_locked = True
                hoop_cx, hoop_cy = get_center(x1, y1, x2, y2)
                hoop_width = x2 - x1
                scoring_line_y_one = y1 + int(hoop_width * 0.3)
                scoring_line_y_two = y2 + int(hoop_width * 0.3)
                scoring_line_x_one_range = (int(hoop_cx - hoop_width * 0.5), int(hoop_cx + hoop_width * 0.5))
                scoring_line_x_two_range = (int(hoop_cx - hoop_width), int(hoop_cx + hoop_width))

            elif class_name.lower() == "basketball" and conf >= BALL_CONF_THRESHOLD:
                if is_likely_ball((x1, y1, x2, y2), frame):
                    ball_detected = True
                    cx, cy = get_center(x1, y1, x2, y2)
                    current_ball_cx, current_ball_cy = cx, cy
                    ball_centers.append((cx, cy))
                    ball_boxes.append((x1, y1, x2, y2))
                    frames_since_last_ball = 0

                    # Start shot if waiting and ball is above hoop
                    if shot_state == "waiting" and cy < scoring_line_y_one:
                        shot_state = "in_progress"
                        ball_above_hoop = False
                        passed_line_one = False

                    # Track if ball goes above hoop during shot
                    if shot_state in ["in_progress", "passed_line_one"] and cy < scoring_line_y_one:
                        ball_above_hoop = True

                    # --- LINE ONE CHECK ---
                    if shot_state == "in_progress" and len(ball_centers) > 1:
                        prev_cx, prev_cy = ball_centers[-2]
                        prev_box = ball_boxes[-2]
                        if check_line_cross(prev_cy, cy, scoring_line_y_one):
                            if check_in_range_and_width(cx, (x1, y1, x2, y2), scoring_line_x_one_range, hoop_width):
                                passed_line_one = True
                                shot_state = "passed_line_one"
                            else:
                                # Crossed line one but not valid: miss
                                miss_count += 1
                                total_shots += 1
                                shot_state = "resetting"
                                ball_above_hoop = False
                                passed_line_one = False
                                continue

                    # --- LINE TWO CHECK ---
                    if shot_state == "passed_line_one" and len(ball_centers) > 1:
                        prev_cx, prev_cy = ball_centers[-2]
                        prev_box = ball_boxes[-2]
                        if check_line_cross(prev_cy, cy, scoring_line_y_two):
                            if check_in_range_and_width(cx, (x1, y1, x2, y2), scoring_line_x_two_range, hoop_width):
                                make_count += 1
                                total_shots += 1
                                shot_state = "resetting"
                                ball_above_hoop = False
                                passed_line_one = False
                                continue
                            else:
                                # Crossed line two but not valid: miss
                                miss_count += 1
                                total_shots += 1
                                shot_state = "resetting"
                                ball_above_hoop = False
                                passed_line_one = False
                                continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If ball is detected, check for miss (ball comes back below line one after going above, but not scored or valid line one)
    if ball_detected and shot_state == "in_progress" and ball_above_hoop and current_ball_cy is not None:
        if current_ball_cy >= scoring_line_y_one and not passed_line_one:
            miss_count += 1
            total_shots += 1
            shot_state = "resetting"
            ball_above_hoop = False
            passed_line_one = False

    # If ball is detected, check for miss (ball comes back below line two after valid line one, but not valid line two)
    if ball_detected and shot_state == "passed_line_one" and ball_above_hoop and current_ball_cy is not None:
        if current_ball_cy >= scoring_line_y_two:
            miss_count += 1
            total_shots += 1
            shot_state = "resetting"
            ball_above_hoop = False
            passed_line_one = False

    if not ball_detected:
        frames_since_last_ball += 1

    if shot_state == "resetting" and frames_since_last_ball > max_no_ball_frames:
        ball_centers.clear()
        ball_boxes.clear()
        shot_state = "waiting"
        frames_since_last_ball = 0
        ball_above_hoop = False
        passed_line_one = False

    if hoop_locked and locked_hoop_box:
        x1, y1, x2, y2 = locked_hoop_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        x1min, x1max = scoring_line_x_one_range
        x2min, x2max = scoring_line_x_two_range
        cv2.line(frame, (x1min, scoring_line_y_one), (x1max, scoring_line_y_one), (0, 0, 255), 2)
        cv2.line(frame, (x2min, scoring_line_y_two), (x2max, scoring_line_y_two), (0, 0, 255), 2)

    # Display frame
    info = f"Makes: {make_count}  Misses: {miss_count}  Total: {total_shots}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("YOLO Basketball Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal Summary:")
print(f"Total shots: {total_shots}")
print(f"Makes: {make_count}")
print(f"Misses: {miss_count}")