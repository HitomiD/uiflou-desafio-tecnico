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
    print("Loading YOLO detection model...")

    model = inference_utils.load_detection_model(ModelConfig.MODEL_PATH)

    print("Loading Mediapipe model...")
    mp_pose = mp.solutions.pose


    pose_estimator = mp_pose.Pose(
        static_image_mode=True,        # since each ROI is independent
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=MediapipeConfig.MIN_DETECTION_CONF,
        min_tracking_confidence=MediapipeConfig.MIN_TRACKING_CONF,
    )

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
            model,
            frame,
            ModelConfig.CONFIDENCE_THRESHOLD,
            ModelConfig.TRACKER_CONFIG
        )

        if detections.id is None:
            continue  # No detections

        for box in detections:
            if int(box.cls[0]) == ModelConfig.PERSON_CLASS_ID:
                
                pose_coordinates = inference_utils.estimate_pose(
                    pose_estimator, 
                    frame, 
                    box, 
                    MediapipeConfig.EXTRA_PADDING
                )
                if pose_coordinates != None:
                    drawing_utils.draw_mediapipe_pose(
                        pose_coordinates, 
                        frame, 
                        DrawingConfig.LANDMARK_RADIUS, 
                        DrawingConfig.LANDMARK_COLOR, 
                        DrawingConfig.LANDMARK_THICKNESS
                )

                drawing_utils.draw_bounding_box(frame, box, DrawingConfig)

        writer.write(frame)

    capture.release()
    writer.release()

    print(f"Processing completed. Output saved to {VideoConfig.OUTPUT_VIDEO_PATH}")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    run()
