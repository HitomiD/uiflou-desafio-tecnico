import cv2
# ============================================================
# Video Processing
# ============================================================

def open_video(video_path) -> cv2.VideoCapture:
    """Open a video file safely."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return capture

def create_video_writer(output_path, fps: float, width: int, height: int):
    """Create a video writer for output video."""
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))