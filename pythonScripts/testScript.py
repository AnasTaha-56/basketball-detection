from ultralytics import YOLO
import cv2

# === Config ===
MODEL_PATH = "./models/newnew.pt"
VIDEO_PATH = "./shots/goodMake480.mp4"

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

# Track the first hoop
hoop_locked = False
locked_hoop_box = None
scoring_line_y_one = None
scoring_line_y_two = None
scoring_line_x_one_range = None
scoring_line_x_two_range = None

# Store previous basketball centers
ball_centers = []
max_ball_history = 10  # Number of frames to keep history

line_one_crossed = False
line_two_crossed = False
score_registered = False

frame_counter = 0

shot_state = "not_detected"

make_count = 0

# === Helpers ===
def get_center(x1, y1, x2, y2):
    """Return the (x, y) center of a bounding box."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_score(positions, line_y_one, line_y_two, line_x_min, line_x_max, line_x_one_min, line_x_one_max):
    """Check if ball crosses the scoring line from above to below inside X range."""
    if len(positions) < 2:
        return False
    
    global line_one_crossed, line_two_crossed, frame_counter, shot_state

    for i in range(1, len(positions)):
        prev_x, prev_y = positions[i - 1]
        curr_x, curr_y = positions[i]

        if curr_y > line_y_one:
            shot_state = "in_progress"

        if line_x_one_min <= curr_x <= line_x_one_max and not line_one_crossed:
            # Check crossing from above to below
            if prev_y < line_y_one and curr_y >= line_y_one:
                line_one_crossed = True
                frame_counter += 1
                shot_state = "crossing"

        # Check horizontal bounds
        if line_x_min <= curr_x <= line_x_max:
            # Check crossing from above to below
            if prev_y < line_y_two and curr_y >= line_y_two:
                line_two_crossed = True
                shot_state = "made"
                if line_one_crossed and line_two_crossed and frame_counter <= 10:
                    return True
                
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            label = f"{class_name} {conf:.2f}"

            # Hoop logic
            if class_name.lower() == "hoop":
                if not hoop_locked and conf >= HOOP_CONF_THRESHOLD:
                    print(" Hoop detected and locked!")
                    locked_hoop_box = (x1, y1, x2, y2)
                    hoop_locked = True

                    # Define scoring line below hoop
                    hoop_cx, hoop_cy = get_center(x1, y1, x2, y2)
                    hoop_width = x2 - x1
                    scoring_line_y_one = y1 + int(hoop_width * 0.3)
                    scoring_line_y_two = y2 + int(hoop_width * 0.3)  # Line below the hoop
                    scoring_line_x_one_range = (int(hoop_cx - hoop_width * .5),  int(hoop_cx + hoop_width * .5))
                    scoring_line_x_two_range = (hoop_cx - hoop_width, hoop_cx + hoop_width)

            # Ball logic
            elif class_name.lower() == "basketball" and conf >= BALL_CONF_THRESHOLD:
                print("Basketball detected!")
                cx, cy = get_center(x1, y1, x2, y2)
                ball_centers.append((cx, cy))
                if len(ball_centers) > max_ball_history:
                    ball_centers.pop(0)

                # Check for score
                if hoop_locked and scoring_line_y_two and scoring_line_x_two_range and not score_registered:
                    if is_score(ball_centers, scoring_line_y_one,  scoring_line_y_two, *scoring_line_x_two_range, *scoring_line_x_one_range):
                        print("!!!!!! Basket made!")
                        score_registered = True  # Prevent duplicate detections
                        make_count += 1

                        # Reset state
                        line_one_crossed = False
                        line_two_crossed = False
                        frame_counter = 0
                        score_registered = False
                        shot_state = "still"

                # Draw ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw locked hoop and scoring line
    if hoop_locked and locked_hoop_box:
        x1, y1, x2, y2 = locked_hoop_box
        cx, cy = get_center(x1, y1, x2, y2)
        hoop_width = x2 - x1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if scoring_line_y_two and scoring_line_x_two_range and scoring_line_x_one_range and scoring_line_y_one:
            x_min, x_max = scoring_line_x_two_range
            x1min, x1max = scoring_line_x_one_range
            cv2.line(frame, (x_min, scoring_line_y_two), (x_max, scoring_line_y_two), (0, 0, 255), 2)
            cv2.line(frame, (x1min, scoring_line_y_one), (x1max, scoring_line_y_one), (0, 0, 255), 2)

    # Show current frame
    cv2.imshow("YOLO Basket Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()


print(f"\nTotal baskets made: {make_count}")

# Final result
if make_count > 0:
    print("\n Final result: Basket was made!")
else:
    print("\n Final result: No basket detected.")
