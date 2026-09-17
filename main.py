import cv2
import time
from collections import Counter
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Give macOS camera time to initialize
time.sleep(2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    counts = Counter()

    for result in results:
        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            class_id = int(box.cls[0])
            name = model.names[class_id]

            counts[name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Object counts
    y = 35

    cv2.putText(
        frame,
        "OBJECT COUNTER",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    y += 35

    for name, count in counts.items():

        cv2.putText(
            frame,
            f"{name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y += 30

    cv2.imshow("YOLO Object Counter", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
