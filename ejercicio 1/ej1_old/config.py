import cv2
# ============================================================
# Configuration
# ============================================================

class ModelConfig:
    MODEL_PATH = "yolo11n.pt"
    TRACKER_CONFIG = "./tracker_configurations/botsort_bricks.yaml"

    PERSON_CLASS_ID = 0
    CONFIDENCE_THRESHOLD = 0.45

class PoseModelConfig:
    MODEL_PATH = ".models/yolo11n-pose.pt"

class MediapipeConfig:
    MODEL_PATH = ".models/pose_landmarker_lite.task"
    MODEL_COMPLEXITY = 1
    MIN_DETECTION_CONF = 0.5
    MIN_TRACKING_CONF = 0.5
    EXTRA_PADDING = 5  #Amount of extra px to select outside the detection. Gives background context.


class VideoConfig:
    INPUT_VIDEO_PATH = "../media/bricks.mp4"
    OUTPUT_VIDEO_PATH = "output_detected.mp4"


class DrawingConfig:
    RECTANGLE_COLOR = (0, 255, 0)
    RECTANGLE_THICKNESS = 2
    LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE = 0.6
    LABEL_THICKNESS = 2
    LABEL_OFFSET = 10

    LANDMARK_RADIUS = 3
    LANDMARK_COLOR = (0 , 0, 255)
    LANDMARK_THICKNESS = -1


