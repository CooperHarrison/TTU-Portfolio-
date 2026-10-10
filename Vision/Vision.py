# Webcam object detection and tracking:
# - Runs YOLOv8 object detection on the live webcam feed and uses Deep SORT to
#   keep IDs attached to detected objects across frames.
# - Draws a colored box, object label, and track ID for each confirmed object.
# - Run locally with a webcam using: py Vision/Vision.py
# - Press Q in the video window to quit. YOLO weights may download on first run;
#   detection is configured to use the CPU.
# - Requires: py -m pip install opencv-python ultralytics deep-sort-realtime

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

def main():
  
    model = YOLO("yolov8s.pt")

   
    tracker = DeepSort(
        max_age=40,
        n_init=5,
        max_cosine_distance=0.3,
        nms_max_overlap=1.0,
        embedder_gpu=False,
        half=False
    )

  
    class_colors = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       
        results = model(frame, verbose=False, device="cpu", imgsz=640)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            
            if conf < 0.45:
                continue

            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, label))

       
        tracks = tracker.update_tracks(detections, frame=frame)

       
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = [int(v) for v in track.to_ltrb()]
            obj_name = track.get_det_class()

            
            if obj_name not in class_colors:
                class_colors[obj_name] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )

            color = class_colors[obj_name]

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(
                frame,
                f"{obj_name} ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Improved YOLOv8 Object Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
