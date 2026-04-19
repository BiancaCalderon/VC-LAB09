import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
SOURCE = "geek_video.mp4"  # swap to 0 for webcam

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_detection(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    label_y = max(y1, th + 4)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 2, label_y + baseline - 2), color, -1)
    cv2.putText(frame, text, (x1 + 1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    model = YOLO(MODEL_PATH)
    class_names = model.names

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Error: cannot open source {SOURCE}")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                color = COLORS[cls_id % len(COLORS)]

                draw_detection(frame, x1, y1, x2, y2, label, conf, color)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Pokedex — Real-time Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
