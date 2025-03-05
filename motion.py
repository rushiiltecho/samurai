import cv2
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import sys

# Append the current script directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class MotionDetection:
    def __init__(self, queue_len=10, max_workers=1, buffer=12):
        """
        Initialize the MotionDetection class.
        Parameters:
        queue_len (int): Maximum length of the frame queue.
        max_workers (int): Maximum number of worker threads.
        buffer (int): Number of frames to wait before setting motion status to false
        """
        self.motionValue = 1000
        self.frameQueue = deque(maxlen=queue_len)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.threshold = 0.95
        self.buffer = buffer
        self.no_motion_count = 0  # Counter for frames without motion
        self.prev_motion_status = False
        self.current_motion_status = False
        self.motion_check_count = 0

    def motion_detect(self, frame1, frame2, log=True):
        """
        Detect motion by comparing two frames using Structural Similarity Index (SSIM).
        Parameters:
        frame1 (np.ndarray): First frame.
        frame2 (np.ndarray): Second frame.
        log (bool): Whether to log the SSIM index.
        """
        try:
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame1 = cv2.GaussianBlur(gray_frame1, (21, 21), 0)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.GaussianBlur(gray_frame2, (21, 21), 0)
            
            diff = cv2.absdiff(gray_frame1, gray_frame2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                self.no_motion_count += 1
                if self.no_motion_count >= self.buffer:
                    self.current_motion_status = False
                    self.motion_check_count = 0
            else:
                self.no_motion_count = 0
                self.motion_check_count += 1
                if self.motion_check_count > 3:
                    self.current_motion_status = True

            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Ignore small contours to reduce noise
                    continue

        except Exception as e:
            print(f"Motion detection failed due to: {e}")
            raise e

    def motionUpdate(self, frame, mask):
        """
        Update the motion detection with a new frame.
        Parameters:
        frame (np.ndarray): New frame to update motion detection.
        mask: Mask to apply to the frame
        """
        self.prev_motion_status = self.current_motion_status
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        self.frameQueue.append(frame)
        
        if len(self.frameQueue) > 1:
            self.motion_detect(self.frameQueue.popleft(), self.frameQueue.popleft(), False)