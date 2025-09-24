import cv2
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Initialize YOLOv8 ---
model = YOLO("yolov8n.pt")

# --- Initialize DeepSORT Tracker ---
tracker = DeepSort(max_age=30)  # Adjust max_age for tracking persistence

# --- Open Camera ---
cap = cv2.VideoCapture(0)

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed
    frame_resized = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame_resized)[0]

    detections = []
    text_events = []

    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls = int(det.cls[0])
        label = results.names[cls]

        # DeepSORT expects [x1, y1, x2, y2, confidence]
        detections.append(([x1, y1, x2, y2], conf, label))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame_resized)

    # Draw boxes & IDs
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        obj_label = track.get_det_class()  # e.g., 'person', 'cup'

        color = (0, 255, 0) if obj_label == "person" else (255, 0, 0)
        cv2.rectangle(frame_resized, (l, t), (r, b), color, 2)
        cv2.putText(frame_resized, f"{obj_label} {track_id}", (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        text_events.append(f"{obj_label} {track_id}")

    # Optional: Display text summary on frame
    summary = ", ".join(text_events)
    cv2.putText(frame_resized, summary, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Real-Time Multi-Person & Object Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
