from ultralytics import YOLO
import cv2
from collections import deque

# === Config ===
MODEL_PATH = "./models/newnew.pt"
VIDEO_PATH = "./shots/makeMissMake720.mp4"

# Confidence thresholds
HOOP_CONF_THRESHOLD = 0.6
BALL_CONF_THRESHOLD = 0.5

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open the video file
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

ball_centers = deque(maxlen=15)
make_count = 0
shot_state = "waiting"
frames_since_last_ball = 0
max_no_ball_frames = 15

# === Helpers ===
def get_center(x1, y1, x2, y2):
    """Return the (x, y) center of a bounding box."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def check_scoring(ball_positions, line_y1, line_y2, x1_min, x1_max, x2_min, x2_max, shot_state):
    """
    Check if ball passes both lines from above to below,
    returning updated shot_state and score flag.
    """
    if len(ball_positions) < 2:
        return shot_state, False

    for i in range(1, len(ball_positions)):
        prev_x, prev_y = ball_positions[i - 1]
        curr_x, curr_y = ball_positions[i]

        # Crossed line one?
        if shot_state == "in_progress" and x1_min <= curr_x <= x1_max:
            if prev_y < line_y1 and curr_y >= line_y1:
                shot_state = "crossed_line_one"

        # Crossed line two?
        if shot_state == "crossed_line_one" and x2_min <= curr_x <= x2_max:
            if prev_y < line_y2 and curr_y >= line_y2:
                shot_state = "scored"
                return shot_state, True

    return shot_state, False

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    results = model(frame, stream=True)

    ball_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            label = f"{class_name} {conf:.2f}"

            # === HOOP DETECTION ===
            if class_name.lower() == "hoop" and conf >= HOOP_CONF_THRESHOLD and not hoop_locked:
                print(" Hoop detected and locked!")
                locked_hoop_box = (x1, y1, x2, y2)
                hoop_locked = True

                hoop_cx, hoop_cy = get_center(x1, y1, x2, y2)
                hoop_width = x2 - x1

                # Define lines for scoring detection
                scoring_line_y_one = y1 + int(hoop_width * 0.3)
                scoring_line_y_two = y2 + int(hoop_width * 0.3)
                scoring_line_x_one_range = (
                    int(hoop_cx - hoop_width * 0.5), int(hoop_cx + hoop_width * 0.5)
                )
                scoring_line_x_two_range = (
                    int(hoop_cx - hoop_width), int(hoop_cx + hoop_width)
                )

            # === BALL DETECTION ===
            elif class_name.lower() == "basketball" and conf >= BALL_CONF_THRESHOLD:
                ball_detected = True
                cx, cy = get_center(x1, y1, x2, y2)
                ball_centers.append((cx, cy))

                # Reset ball disappearance tracker
                frames_since_last_ball = 0

                # Shot state update
                if shot_state == "waiting" and cy < scoring_line_y_one:
                    shot_state = "in_progress"

                # Check scoring
                if hoop_locked and scoring_line_y_two and scoring_line_y_one and scoring_line_x_one_range and scoring_line_x_two_range:
                    shot_state, scored = check_scoring(
                        ball_centers,
                        scoring_line_y_one,
                        scoring_line_y_two,
                        *scoring_line_x_one_range,
                        *scoring_line_x_two_range,
                        shot_state
                    )

                    if scored:
                        print(" !!!!! Basket made!")
                        make_count += 1
                        shot_state = "resetting"

                # Draw ball box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # === Ball disappearance tracking ===
    if not ball_detected:
        frames_since_last_ball += 1
        if frames_since_last_ball > max_no_ball_frames and shot_state == "resetting":
            # Reset everything
            ball_centers.clear()
            shot_state = "waiting"
            frames_since_last_ball = 0

    # === Draw HOOP + LINES ===
    if hoop_locked and locked_hoop_box:
        x1, y1, x2, y2 = locked_hoop_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if scoring_line_y_one and scoring_line_y_two and scoring_line_x_one_range and scoring_line_x_two_range:
            x1min, x1max = scoring_line_x_one_range
            x2min, x2max = scoring_line_x_two_range

            cv2.line(frame, (x1min, scoring_line_y_one), (x1max, scoring_line_y_one), (0, 0, 255), 2)
            cv2.line(frame, (x2min, scoring_line_y_two), (x2max, scoring_line_y_two), (0, 0, 255), 2)

    # === Show Frame ===
    cv2.imshow("YOLO Basket Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final Result
print(f"\nTotal baskets made: {make_count}")
if make_count > 0:
    print("\n Final result: Basket(s) made!")
else:
    print("\n Final result: No basket detected.")
