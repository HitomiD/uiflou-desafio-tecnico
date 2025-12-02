import cv2
from config import InputVideoConfig
# ============================================================
# Video Processing
# ============================================================

def open_video(video_path) -> cv2.VideoCapture:
    """
    Open a video file safely.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return capture


def get_stream_metadata(url: str):
    """
    Gets stream metadata using a temporary capture object.
    """
    tmp_capture = open_video(url)
    if not tmp_capture.isOpened():
        raise ConnectionError(f"Could not connect to stream: {url}")

    fps = int(tmp_capture.get(cv2.CAP_PROP_FPS))
    width = int(tmp_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(tmp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_capture.release()

    return fps, width, height

