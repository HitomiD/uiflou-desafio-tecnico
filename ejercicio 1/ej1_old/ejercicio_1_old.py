import cv2
import mediapipe as mp
import torch
from ultralytics.models.yolo import YOLO
import os

def draw_box_in_frame(frame, box):
     # coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    bottom_left_corner = (x1, y1)
    upper_right_corner = (x2, y2)

    rectangle_color = (0, 255, 0)
    line_width = 2

    cv2.rectangle(frame, bottom_left_corner, upper_right_corner, rectangle_color, line_width)

    #Label drawing
    label_confidence = box.conf[0]
    id = box.id[0]
    label = f"Person {id} ({label_confidence:.2f})"
    label_coordinates = (x1, y1 - 10)

    cv2.putText(frame,
                label,
                label_coordinates,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                rectangle_color, line_width
    )



# 1. Load YOLO model (use yolov8n.pt for speed, or yolov8s/yolov8m for accuracy)
yolo_model = YOLO("yolo11n.pt")  # or "yolov8s.pt", "yolov11n.pt", etc

if(torch.cuda.is_available()):
    yolo_model.to('cuda:0') 

print("Current Working Directory (CWD):", os.getcwd())

# 2. Open input video
video_capture = cv2.VideoCapture("../media/bricks.mp4")

# Video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width   = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height   = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. Output video writer
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

tracker_yaml = "./tracker_configurations/botsort_bricks.yaml"


while True:
    success, frame = video_capture.read()
    if not success:
        break

    
    results = yolo_model.track(
         frame,
         conf=0.45, 
         persist=True, 
         tracker=tracker_yaml
    )

    if results[0].boxes.id is None:
            out.write(frame)
            continue

    # 5. Loop through detections
    for box in results[0].boxes:
        class_id = int(box.cls[0])  # class ID
        if class_id != 0:           # class 0 = "person"
            continue

        draw_box_in_frame(box)

    out.write(frame)

# Cleanup
video_capture.release()
out.release()
print("Done! Saved as output_detected.mp4")
