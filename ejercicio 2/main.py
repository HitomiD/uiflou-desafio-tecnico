from config import InferenceConfig, InputVideoConfig, MetricsConfig, DataExportConfig
from modules import video_utils, producer, inference_utils, fps_tracker, data_exporter, video_writer_manager

import queue
import os

import cv2


def main():

    print("Current Directory:", os.getcwd())


    # ============================================================
    # Model preparation
    # ============================================================

    print("Loading model...")

    model = inference_utils.load(InferenceConfig.MODEL_PATH)


    # ============================================================
    # Video Preparation
    # ============================================================

    print("Loading input video...")

    fps, width, height = video_utils.get_stream_metadata(InputVideoConfig.INPUT_PATH)

    if fps > InputVideoConfig.MAX_INPUT_FPS:
        fps = InputVideoConfig.MAX_INPUT_FPS

    # Use a generic base name for the segments (without extension)
    base_output_name = os.path.splitext(os.path.basename(DataExportConfig.OUTPUT_JSONL_PATH))[0]
    
    # Initialize the VideoWriterManager
    writer_manager = video_writer_manager.VideoWriterManager(
        output_dir=DataExportConfig.SEGMENT_OUTPUT_DIR,
        base_filename=base_output_name,
        fps=fps, 
        width=width, 
        height=height,
        fourcc=DataExportConfig.FOUR_CC
    )

    print("\n--- Video Properties ---")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}*{height}")
    print("---------------------------\n")


    # ============================================================
    # Producer-Consumer Setup
    # ============================================================

    frame_queue = queue.Queue(maxsize=InputVideoConfig.FRAME_QUEUE_MAXSIZE) 

    producer_thread = producer.RTSP_Producer(
        rtsp_url=InputVideoConfig.INPUT_PATH,
        frame_queue=frame_queue,
        max_reconnect_attempts=InputVideoConfig.MAX_RECONNECT_ATTEMPTS
    )

    print("Starting RTSP Producer Thread...")
    producer_thread.start() # Start the thread


    # ============================================================
    # Frame processing loop
    # ============================================================

    print("Preparing for video processing...")

    # Frame skipping helpers
    total_frames = 0
    last_results = None

    # Segment tracking variable
    frames_in_current_segment = 0

    # Data exporter object
    exporter = data_exporter.FrameDataExporter(DataExportConfig.OUTPUT_JSONL_PATH)

    performance_tracker = fps_tracker.FPSTracker(MetricsConfig.FRAMES_FOR_AVERAGE)

    print("Stream processing started.")

    while producer_thread.is_alive() or not frame_queue.empty():
        
        # Try to get a frame from the queue. Blocks for a short timeout.
        try:
            frame = frame_queue.get(timeout=InferenceConfig.FRAME_QUEUE_TIMEOUT) 
        except queue.Empty:
            # If the queue is empty, check if the producer has failed/stopped
            if not producer_thread.is_alive():
                break # Producer has stopped and queue is drained. Exit gracefully.
            continue # Queue temporarily empty, try again in the next loop iteration

        run_inference = (total_frames % InferenceConfig.FRAME_SKIPPING_INDEX) == 0

        if run_inference:
            results = model.track(
                frame,
                persist=True,
                tracker=InferenceConfig.TRACKER_CONFIG,
                conf=InferenceConfig.CONFIDENCE_THRESHOLD,
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

        annotated_frame = results[0].plot()

        # Get the current writer from the manager
        current_writer = writer_manager.get_writer()
        current_writer.write(annotated_frame)
        
        # Increment segment frame counter
        frames_in_current_segment += 1

        total_frames += 1

        # Check for segment break and trigger writer close/open
        if frames_in_current_segment >= DataExportConfig.MAX_FRAMES_PER_SEGMENT:
            writer_manager.release_writer()
            frames_in_current_segment = 0 # Reset counter

        #Show output in a live window
        cv2.imshow("Live Output", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Break the loop if "q" is pressed within output window.
            break

        # Export detection + pose data
        exporter.export_frame(total_frames, results, model.names)

        # Update FPS report
        fps_avg_capacity = performance_tracker.update()
        print(f"Real-time FPS: {fps_avg_capacity:.2f}", end="\r")

    # ============================================================
    # Cleanup
    # ============================================================

    producer_thread.stop()
    producer_thread.join()
    
    writer_manager.release_writer()

    print("\n---------------------------")

    print(f"Processing completed. Output saved to {DataExportConfig.SEGMENT_OUTPUT_DIR}")
    print("---------------------------\n")

    
# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()

