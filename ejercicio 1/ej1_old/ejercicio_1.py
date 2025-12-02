import cv2
import torch
from ultralytics.models.yolo import YOLO
import os
from config import DrawingConfig, ModelConfig, VideoConfig, PoseModelConfig, MediapipeConfig
from modules import video_utils, inference_utils, drawing_utils
import mediapipe as mp


# ============================================================
# Application Entry Point
# ============================================================

def run():
    print("Current Directory:", os.getcwd())
  
    print("Loading YOLO-Pose model...")
    
    pose_model = inference_utils.load_pose_model(PoseModelConfig.MODEL_PATH)

    print("Loading input video...")

    capture = video_utils.open_video(VideoConfig.INPUT_VIDEO_PATH)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = video_utils.create_video_writer(VideoConfig.OUTPUT_VIDEO_PATH, fps, width, height)

    print("Processing video...")

    while True:
        success, frame = capture.read()
        if not success:
            break

        detections = inference_utils.track_objects(
            pose_model,
            frame,
            ModelConfig.CONFIDENCE_THRESHOLD,
            ModelConfig.TRACKER_CONFIG
        )

        if detections.id is None:
            continue  # No detections

        results = pose_model(frame)

        for i, person in enumerate(results[0].keypoints):
            # person.xy = keypoints for person i
            # results[0].boxes[i] = bounding box for person i
            drawing_utils.draw_yolo_pose(
                person.xy, 
                frame, 
                DrawingConfig.LANDMARK_RADIUS,
                DrawingConfig.LANDMARK_COLOR,
                DrawingConfig.LANDMARK_THICKNESS
            )

            drawing_utils.draw_bounding_box(
                frame, 
                results[0].boxes[i], 
                DrawingConfig)

        writer.write(frame)

    capture.release()
    writer.release()

    print(f"Processing completed. Output saved to {VideoConfig.OUTPUT_VIDEO_PATH}")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    run()
