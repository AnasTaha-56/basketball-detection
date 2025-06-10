from ultralytics import YOLO
import cv2

# === Config ===
MODEL_PATH = "./models/newnew.pt"
VIDEO_PATH = "./shots/goodMake480.mp4"

# Confidence thresholds
HOOP_CONF_THRESHOLD = 0.6
BALL_CONF_THRESHOLD = 0.6

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# To store first detected hoop
hoop_locked = False
locked_hoop_box = None


prev_ball_positions = []

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

            # Detect and lock first hoop
            if class_name.lower() == "hoop":
                if not hoop_locked and conf >= HOOP_CONF_THRESHOLD:
                    print("Hoop detected and locked!")
                    locked_hoop_box = (x1, y1, x2, y2)
                    hoop_locked = True

                # Only draw if this is the locked hoop
                if hoop_locked and (x1, y1, x2, y2) == locked_hoop_box:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Detect basketballs with confidence threshold
            elif class_name.lower() == "basketball" and conf >= BALL_CONF_THRESHOLD:
                print("Basketball detected!")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the current frame
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
