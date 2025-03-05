# motion.py
import cv2
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

class MotionDetection:
    def __init__(self, queue_len=10, max_workers=1, buffer=12, min_area=5000):
        """
        Initialize the MotionDetection class.
        
        Args:
            queue_len (int): Maximum length of frame queue
            max_workers (int): Maximum number of worker threads
            buffer (int): Number of consecutive motion frames needed to trigger detection
            min_area (int): Minimum contour area to consider as motion
        """
        self.frameQueue = deque(maxlen=queue_len)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.buffer = buffer
        self.motion_check_count = 0
        self.current_motion_status = False
        self.min_contour_area = min_area
        logging.info("MotionDetection initialized with "
                    f"buffer={buffer}, min_area={min_area}")
    
    def motion_detect(self):
        """
        Detect motion by analyzing differences across the last 5 frames in the deque.
        """
        try:
            if len(self.frameQueue) < 12:
                # Not enough frames to perform multi-frame analysis
                return

            # Convert all frames in the deque to grayscale and apply Gaussian blur
            gray_frames = [
                cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                for frame in self.frameQueue
            ]

            # Compute cumulative difference
            cumulative_diff = np.zeros_like(gray_frames[0], dtype=np.float32)
            for i in range(len(gray_frames) - 1):
                diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
                cumulative_diff += diff.astype(np.float32)

            # Normalize and threshold cumulative difference
            cumulative_diff = np.clip(cumulative_diff, 0, 255).astype(np.uint8)
            _, thresh = cv2.threshold(cumulative_diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Check for significant motion
            significant_motion = False
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    significant_motion = True
                    break

            # Update motion status using buffer
            if significant_motion:
                self.motion_check_count += 1
                if self.motion_check_count > self.buffer:
                    if not self.current_motion_status:
                        logging.info("Motion detected")
                    self.current_motion_status = True
            else:
                if self.current_motion_status:
                    logging.info("Motion stopped")
                self.current_motion_status = False
                self.motion_check_count = 0

        except Exception as e:
            logging.error(f"Motion detection failed: {e}")
            raise

    
    def motionUpdate(self, frame):
        """
        Update motion detection with a new frame.

        Args:
            frame (np.ndarray): New frame to process
        """
        if frame is None:
            raise ValueError("Invalid frame provided")

        self.frameQueue.append(frame.copy())

        if len(self.frameQueue) >= 12:
            # Perform motion detection once enough frames are collected
            self.motion_detect()


