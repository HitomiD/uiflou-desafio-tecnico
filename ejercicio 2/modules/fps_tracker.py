import time
import collections

class FPSTracker:
    def __init__(self, max_frames=30):
        """
        Tracks the FPS over the last `max_frames` frames.
        """
        self.frame_times = collections.deque(maxlen=max_frames)
        self.last_time = time.time()

    def update(self):
        """
        Call this once per frame. Returns the current FPS.
        """
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        if not self.frame_times:
            return 0.0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1 / avg_frame_time
