import cv2
# ============================================================
# Configuration
# ============================================================

class InferenceConfig:
    MODEL_PATH = "./models/yolov8n-pose.pt"
    TRACKER_CONFIG = "./tracker_configurations/botsort_bricks.yaml"
    PERSON_CLASS_ID = 0
    CONFIDENCE_THRESHOLD = 0.45
    FRAME_QUEUE_TIMEOUT = 0.01  # If no frame arrives in this time, indicate empty queue.
    FRAME_SKIPPING_INDEX = 2    # "Skip a frame every N frame"


class InputVideoConfig:
    INPUT_PATH = "rtsp://192.168.1.161:1945"
    OUTPUT_FILE_PATH = "./output/video_output.mp4"
    MAX_INPUT_FPS = 20  # If the input FPS is bigger, it gets lowered to this value.
    FRAME_QUEUE_MAXSIZE = 10    # A small buffer size is recommended for memory safety and low latency.
    MAX_RECONNECT_ATTEMPTS = 5

class DataExportConfig:
    OUTPUT_JSONL_PATH = "./output/results.jsonl"
    SEGMENT_OUTPUT_DIR = "./output/video" # Where the result video segments will be saved.
    MAX_FRAMES_PER_SEGMENT = 150    # e.g., 5 seconds at 30 FPS
    FOUR_CC = "mp4v"    


class MetricsConfig:
    FRAMES_FOR_AVERAGE = 15 #Amount of frames to use in average frame processing time indicator.


