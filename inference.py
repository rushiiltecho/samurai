import cv2
import threading
import logging
import os
import uuid
import numpy as np
from collections import deque
from motion import MotionDetection
from samurai import process_video

# Global Declarations
FRAME_QUEUE = deque(maxlen=10)
STOP_SIGNAL = threading.Event()  # Signal to stop the frame reader thread
SAM2_PROCESSING = False  # Flag to indicate SAM2 processing

motion = MotionDetection()

# Helper Functions
def display_text(frame, text, position=(10, 30), color=(0, 255, 0)):
    """Helper function to display text on a frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def decode_fourcc(value):
    """Decode the FourCC codec value."""
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def configure_camera(cap, width=1280, height=720, fps=90, codec="MJPG"):
    """Configure the camera with resolution, FPS, and codec."""
    if not cap or not cap.isOpened():
        return None

    fourcc = cv2.VideoWriter_fourcc(*codec)
    old_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))

    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"Codec changed from {old_fourcc} to {decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))}")
    else:
        print(f"Error: Could not change codec from {old_fourcc}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print(f"Camera configured with FPS: {cap.get(cv2.CAP_PROP_FPS)}, "
          f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
          f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    return cap

def process_motion_session_with_sam2(video_path, bbox_points, model_path, output_folder):
    """Process recorded video with SAM2."""
    x_coords, y_coords = zip(*bbox_points)
    x1, y1 = min(x_coords), min(y_coords)
    w = max(x_coords) - x1
    h = max(y_coords) - y1

    output_path = os.path.join(output_folder, f"{uuid.uuid4()}.mp4")
    motion_type = process_video(
        video_path=video_path,
        coords=(x1, y1, w, h),
        model_path=model_path,
        save_video=True,
        output_path=output_path
    )
    return motion_type

def select_points(frame, num_points=4, context_text=""):
    """Standalone function to select points on a frame."""
    points = []
    temp_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            draw_frame(temp_frame, points, context_text)

    def draw_frame(frame, points, text):
        """Helper function to draw current state."""
        annotated = frame.copy()
        display_text(annotated, text)
        for idx, (px, py) in enumerate(points):
            cv2.circle(annotated, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"P{idx + 1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Select Points", annotated)

    cv2.imshow("Select Points", temp_frame)
    cv2.setMouseCallback("Select Points", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(points) == num_points:
            cv2.destroyWindow("Select Points")
            break
        elif key == ord('r'):
            points = []
        elif key == ord('q'):
            cv2.destroyWindow("Select Points")
            return None

    return points

def main(camera_index=0, width=1280, height=720, fps=90, codec="MJPG"):
    global SAM2_PROCESSING
    motion_session_started = False
    motion_video_writer = None
    motion_detector = MotionDetection()

    # Initialize Camera
    cap = cv2.VideoCapture(camera_index)
    cap = configure_camera(cap, width, height, fps, codec)
    if not cap or not cap.isOpened():
        logging.error("Camera not initialized.")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read initial frame.")
        return

    # Step 1: ROI Selection
    roi_points = select_points(frame, num_points=4, context_text="Select ROI")
    # roi_points = [(433, 322), (249, 656), (1261, 556), (976, 289)]
    print(roi_points)
    if not roi_points:
        logging.error("ROI selection canceled.")
        return
    
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

    # Step 2: BBOX Selection
    bbox_points = select_points(frame, num_points=4, context_text="Select Object to Track")
    print(bbox_points)
    if not bbox_points:
        logging.error("BBOX selection canceled.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply ROI mask and detect motion
            # masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            motion_detector.motionUpdate(frame, roi_mask)

            # Handle motion states
            if motion_detector.current_motion_status:
                print("Motion Detected")
                if not motion_session_started:
                    if not os.path.exists("videos"):
                        os.makedirs("videos")
                    motion_video_path = os.path.join("videos", f"{uuid.uuid4()}.mp4")
                    motion_video_writer = cv2.VideoWriter(
                        motion_video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    )
                    motion_session_started = True
                    logging.info(f"Started motion session: {motion_video_path}")

                if motion_video_writer:
                    motion_video_writer.write(frame)

            elif motion_session_started:
                print("Motion Stopped")
                if motion_video_writer:
                    motion_video_writer.release()
                    logging.info(f"Stopped motion session: {motion_video_path}")
                motion_session_started = False

                # Update display to SAM2 Processing
                display_text(frame, "SAM2 Processing...", position=(10, 50), color=(0, 255, 255))
                cv2.imshow("Live Feed", frame)
                cv2.waitKey(200)

                # Process the recorded video with SAM2
                motion_type = process_motion_session_with_sam2(
                    motion_video_path, bbox_points, "./sam2/checkpoints/sam2.1_hiera_large.pt", "sam2_results"
                )

                # Update display with SAM2 result
                display_text(frame, f"SAM2 Result: {motion_type}", position=(10, 90), color=(0, 255, 0))
                cv2.imshow("Live Feed", frame)
                cv2.waitKey(3000)

            # Display current state
            status = "SAM2 Processing..." if SAM2_PROCESSING else "Live Feed"
            display_text(frame, status)
            cv2.imshow("Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        STOP_SIGNAL.set()
        if motion_video_writer:
            motion_video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(camera_index=6, width=1280, height=720, fps=90, codec="MJPG")