from ultralytics import YOLO
from config import ModelConfig, VideoConfig
from modules import inference_utils, video_utils
import cv2
import os
import time, collections
import torch


print("Current Directory:", os.getcwd())


# ============================================================
# Model preparation
# ============================================================

print("Loading model...")

print(f"Model path: {ModelConfig.MODEL_PATH}")
model = YOLO(ModelConfig.MODEL_PATH)

if torch.cuda.is_available():
    print("CUDA available: Switching to GPU...")
    model.to("cuda")


# ============================================================
# Video preparation
# ============================================================

print("Loading input video...")

capture = video_utils.open_video(VideoConfig.INPUT_FILE_PATH)

# video properties
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
FOURCC = cv2.VideoWriter_fourcc(*VideoConfig.FOUR_CC)

writer = cv2.VideoWriter(
    VideoConfig.OUTPUT_FILE_PATH, 
    FOURCC, 
    fps, 
    (width, height))

print("\n--- Video Properties ---")
print(f"FPS: {fps}")
print(f"Resolution: {width}*{height}")
print("---------------------------\n")


# ============================================================
# Frame processing loop
# ============================================================

print("Processing video...\n")

start_time = time.time()
total_frames = 0
last_results = None
timestamps = collections.deque()
ROLLING_WINDOW = 2.0  # seconds

while True:
    success, frame = capture.read()
    if not success:
        break

    run_inference = (total_frames % 2) != 0 # True for frames 1, 3, 5, etc.

    if run_inference:
        results = model.track(
            frame,
            persist=True,
            tracker=ModelConfig.TRACKER_CONFIG,
            conf=ModelConfig.CONFIDENCE_THRESHOLD,
            verbose=False
        )
        last_results = results
    elif last_results is not None:
        # 3. For skipped frames (0, 2, 4, etc.), reuse the last results
        results = last_results
    else:
        # Skip drawing if no results have been generated yet (only for frame 0)
        total_frames += 1
        continue


    #annotated_frame = results[0].plot()

    writer.write(frame)
    
    total_frames += 1
    
    #Performance report

    now = time.time()
    timestamps.append(now)

    # Remove timestamps older than the rolling window
    while timestamps and (now - timestamps[0]) > ROLLING_WINDOW:
        timestamps.popleft()

    rolling_fps = len(timestamps) / ROLLING_WINDOW

    print(f"Real-time FPS: {rolling_fps:.2f}", end="\r")  # updates live


capture.release()
writer.release()


# ============================================================
# Metrics calculation
# ============================================================
end_time = time.time()
total_duration = end_time - start_time


# Calculate Processing FPS (P-FPS)
if total_duration > 0 and total_frames > 0:
    processing_fps = total_frames / total_duration
else:
    processing_fps = 0

print(f"Processing completed. Output saved to {VideoConfig.OUTPUT_FILE_PATH}")
print("--- Performance Metrics ---")
print(f"Total Frames Processed: {int(total_frames)}")
print(f"Total Processing Time: {total_duration:.2f} seconds")
print(f"Average Processing FPS (P-FPS): {processing_fps:.2f} FPS")
print("---------------------------\n")

# ============================================================
# Run script
# ============================================================

#if __name__ == "__main__":
#    run()

