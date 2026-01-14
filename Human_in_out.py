import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ================= CONFIG =================
CAMERA_ID = 0
CONF_THRESHOLD = 0.4

# -------- DOOR REGION --------
DOOR_X1 = 280     # left side of door
DOOR_X2 = 360     # right side of door

FRAME_WIDTH = 640

# ================= LOAD MODELS =================
person_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# ================= CAMERA =================
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera error")
    exit()

# ================= VARIABLES =================
entry_count = 0
exit_count = 0

person_state = {}     # LEFT, IN_DOOR, RIGHT
counted_state = {}   # prevents double counting

frame_count = 0
cached_detections = []

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    h, w, _ = frame.shape
    scale = FRAME_WIDTH / w
    frame_small = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))

    # -------- DETECTION --------
    if frame_count % 2 == 0:
        cached_detections = []
        results = person_model(frame_small, conf=CONF_THRESHOLD, verbose=False)[0]

        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                cached_detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "person"))

    tracks = tracker.update_tracks(cached_detections, frame=frame)
    active_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        active_ids.add(tid)

        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # -------- DETERMINE POSITION --------
        if x2 < DOOR_X1:
            current_pos = "LEFT"
        elif x1 > DOOR_X2:
            current_pos = "RIGHT"
        else:
            current_pos = "IN_DOOR"

        # -------- INIT --------
        if tid not in person_state:
            person_state[tid] = current_pos
            counted_state[tid] = False
        else:
            prev_pos = person_state[tid]

            # -------- ENTRY --------
            if prev_pos == "IN_DOOR" and current_pos == "RIGHT" and not counted_state[tid]:
                entry_count += 1
                counted_state[tid] = True

            # -------- EXIT --------
            elif prev_pos == "IN_DOOR" and current_pos == "LEFT" and not counted_state[tid]:
                exit_count += 1
                counted_state[tid] = True

            # -------- RESET AFTER FULL PASS --------
            if current_pos in ["LEFT", "RIGHT"]:
                counted_state[tid] = False

            person_state[tid] = current_pos

        # -------- DRAW --------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid} [{current_pos}]",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # -------- CLEANUP --------
    for tid in list(person_state.keys()):
        if tid not in active_ids:
            person_state.pop(tid, None)
            counted_state.pop(tid, None)

    # -------- DRAW DOOR --------
    cv2.rectangle(frame, (DOOR_X1, 0), (DOOR_X2, h), (255, 0, 0), 2)
    cv2.putText(frame, "DOOR", (DOOR_X1 + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # -------- UI --------
    cv2.putText(frame,
                f"IN: {entry_count}  OUT: {exit_count}  INSIDE: {entry_count-exit_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("Door-based People Counter", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
