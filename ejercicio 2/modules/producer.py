import threading
import queue
import cv2
import time

class RTSP_Producer(threading.Thread):
    def __init__(self, rtsp_url, frame_queue, max_reconnect_attempts=5):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.MAX_RECONNECT_ATTEMPTS = max_reconnect_attempts
        self.stop_event = threading.Event() # Used to signal the thread to stop
        self.capture = None
        
    def run(self):
        reconnect_attempts = 0
        while not self.stop_event.is_set():
            if self.capture is None or not self.capture.isOpened():
                print(f"Producer: Attempting to connect to {self.rtsp_url}...")
                self.capture = cv2.VideoCapture(self.rtsp_url)
                time.sleep(1) # Give time for the connection to establish
                reconnect_attempts += 1
                
                if reconnect_attempts > self.MAX_RECONNECT_ATTEMPTS:
                    print("Producer: Max reconnect attempts reached. Stopping.")
                    break
                    
                continue # Restart loop to check if connection succeeded
            
            # Connection is open, try to read a frame
            success, frame = self.capture.read()
            
            if success and frame is not None:
                # Put the frame in the queue (non-blocking)
                try:
                    # Using queue.put_nowait() or a small timeout to avoid blocking
                    self.frame_queue.put(frame, timeout=0.1) 
                    reconnect_attempts = 0 # Reset counter on success
                except queue.Full:
                    # If the queue is full, the consumer is too slow. Drop the frame.
                    print("Producer: Warning! Queue full, dropping frame.")
            else:
                # Read failed (stream lost, buffer empty, etc.)
                self.capture.release() # Release to force a proper reconnect
                print("Producer: Read failed. Releasing capture for reconnect.")
                time.sleep(0.5) # Wait before trying to reconnect

        if self.capture is not None:
            self.capture.release()
        print("Producer: Exiting frame capture loop.")

    def stop(self):
        self.stop_event.set()