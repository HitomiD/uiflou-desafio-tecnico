import cv2
from config import DrawingConfig
import numpy as np
# ============================================================
# Drawing tools
# ============================================================


def draw_bounding_box(frame, box, drawing_config: DrawingConfig):
    """
    Draw a bounding box and label on a frame for a tracked object.
    """
    (x1, y1, x2, y2) = map(int, box.xyxy[0])
    track_id = int(box.id[0]) if box.id is not None else -1
    confidence = float(box.conf[0])

    label = f"Person {track_id} ({confidence:.2f})"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        drawing_config.RECTANGLE_COLOR,
        drawing_config.RECTANGLE_THICKNESS
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - drawing_config.LABEL_OFFSET),
        drawing_config.LABEL_FONT,
        drawing_config.LABEL_SCALE,
        drawing_config.RECTANGLE_COLOR,
        drawing_config.LABEL_THICKNESS
    )

def draw_mediapipe_pose(coords, frame, lm_radius, lm_color, lm_thickness):
    """
    Draw Mediapipe keypoints on the frame.
    coords: array of tuples (x, y)
    """
    for (x, y) in coords:
        cv2.circle(frame, (x, y), lm_radius, lm_color, lm_thickness)


def draw_yolo_pose(coords, frame, lm_radius, lm_color, lm_thickness):
    """
    Draw YOLO-Pose keypoints on the frame.
    coords: numpy array of shape (num_keypoints, 2)
    """
    # move to CPU and convert to NumPy
    #(if CUDA is not enabled then this only converts to numpy)
    coords = coords.detach().cpu().numpy()  
    
    for point in coords:
        x, y = map(int, point)  # convert floats to ints
        cv2.circle(frame, (x, y), lm_radius, lm_color, lm_thickness)