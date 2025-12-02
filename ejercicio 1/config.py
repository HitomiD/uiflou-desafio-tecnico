import cv2
# ============================================================
# Configuration
# ============================================================

class ModelConfig:
    MODEL_PATH = "./models/yolo11n-pose.pt"
    TRACKER_CONFIG = "./tracker_configurations/botsort_bricks.yaml"
    PERSON_CLASS_ID = 0
    CONFIDENCE_THRESHOLD = 0.45


class VideoConfig:
    INPUT_FILE_PATH = "../media/bricks.mp4"
    OUTPUT_FILE_PATH = "./output/video_output.mp4"
    FOUR_CC = "mp4v"



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


