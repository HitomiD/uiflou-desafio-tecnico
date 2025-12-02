from ultralytics import YOLO
import mediapipe as mp
import cv2
import torch



def track_objects(model, frame, confidence, tracker_config):
    """
    Run YOLO tracking on a single frame.
    """
    results = model.track(
        frame,
        conf=confidence,
        persist=True,
        tracker=tracker_config
    )

    detections = results[0].boxes

    return detections


def estimate_pose(pose_estimator, frame, box, padding):
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    region_of_interest = frame[y1:y2, x1:x2]
    
    if region_of_interest.size == 0:
        return None
    
    #RGB conversion required by Mediapipe
    roi_rgb = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB)

    results = pose_estimator.process(roi_rgb)
    if not results.pose_landmarks:
        return None

    roi_height, roi_width = region_of_interest.shape[:2]
    full_coords = []

    for landmark in results.pose_landmarks.landmark:
         
        # Translation from normalized output (0.0-1.0) to pixel
        landmark_x = int(landmark.x * roi_width)
        landmark_y = int(landmark.y * roi_height)

        # Convert ROI coordinates -> global coordinates
        full_x = landmark_x + x1
        full_y = landmark_y + y1

        full_coords.append((full_x, full_y))

    return full_coords