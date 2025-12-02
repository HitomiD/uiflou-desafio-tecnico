import cv2
import os

"""
The VideoWriterManager handles the creation and release of each new writer needed per segment in the main loop.
"""

class VideoWriterManager:
    """
    Manages creation and closing of segmented video writers.
    """
    def __init__(self, output_dir, base_filename, fps, width, height, fourcc):
        self.output_dir = output_dir
        self.base_filename = base_filename
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.current_writer = None
        self.segment_count = 0
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def get_writer(self):
        """
        Returns the current writer, creating a new one if none exists.
        """
        if self.current_writer is None:
            self.segment_count += 1
            segment_filename = f"{self.base_filename}_segment_{self.segment_count:04d}.mp4"
            output_path = os.path.join(self.output_dir, segment_filename)
            
            # The original video_utils.create_video_writer logic
            fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.current_writer = cv2.VideoWriter(output_path, fourcc_code, self.fps, (self.width, self.height))
            print(f"--> Opened new segment: {segment_filename}")
        return self.current_writer

    def release_writer(self):
        """
        Releases the current writer and sets it to None.
        """
        if self.current_writer:
            self.current_writer.release()
            self.current_writer = None
            print(f"<-- Closed segment {self.segment_count:04d}")