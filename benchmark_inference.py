import datetime
import gc
import queue
import sys
import cv2
import threading
import logging
import os
import uuid
import numpy as np
from collections import deque

import torch
from gdino_vertical_stack import process_single_image
from motion import MotionDetection
from samurai import determine_model_cfg, load_prompt, process_realtime_frame, process_video, process_frame
import time
sys.path.append("./sam2")

from sam2.build_sam import build_sam2_object_tracker

# Global Declarations
FRAME_QUEUE = deque(maxlen=20)
STOP_SIGNAL = threading.Event()  # Signal to stop the frame reader thread
SAM2_PROCESSING = False  # Flag to indicate SAM2 processing
motion = MotionDetection()


# Helper Functions
def display_text(frame, text, position=(10, 30), color=(0, 255, 0)):
    """Helper function to display text on a frame."""
    # now = time.time()
    # while time.time() - now < 3:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def display_misc_text(frame, text, position=(10, 30), color=(0, 255, 0)):
    """Helper function to display text on a frame."""
    now = time.time()
    while time.time() - now < 3:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def decode_fourcc(value):
    """Decode the FourCC codec value."""
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def create_fullscreen_window(window_name):
    """Create a fullscreen window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Get the screen size
    # screen = cv2.getWindowByName(window_name)
    cv2.resizeWindow(window_name, 2560, 1440)  # Set to full HD size
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
def show_frame(window_name, frame):
    """Display frame in fullscreen."""
    height, width = frame.shape[:2]
    # Calculate scaling factors
    window_width = 2560  # Adjust to your screen width
    window_height = 1440  # Adjust to your screen height
    
    scale_width = window_width / width
    scale_height = window_height / height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow(window_name, resized_frame)
    
def draw_enlarged_bbox(frame, bbox_points, scale=1.0):
    """Draw an enlarged bounding box based on the selected points."""
    if not bbox_points or len(bbox_points) != 4:
        return frame
    
    # Convert points to numpy array
    points = np.array(bbox_points)
    
    # Find center of the bounding box
    center = np.mean(points, axis=0)
    
    # Calculate enlarged points by moving them away from center
    enlarged_points = []
    for point in points:
        # Vector from center to point
        vector = point - center
        # Scale the vector and add back to center
        enlarged_point = center + (vector * scale)
        enlarged_points.append(tuple(map(int, enlarged_point)))
    
    # Draw the enlarged bounding box
    frame_copy = frame.copy()
    for i in range(4):
        # Draw lines between points
        cv2.line(frame_copy, enlarged_points[i], enlarged_points[(i + 1) % 4], (0, 255, 0), 2)
        # Draw points
        cv2.circle(frame_copy, enlarged_points[i], 5, (0, 0, 255), -1)
    
    return frame_copy

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

def select_points(frame, num_points=4, context_text="", scale=1.2):
    """Standalone function to select points on a frame."""
    points = []
    temp_frame = frame.copy()
    window_name = "Select Points"
    
    # Get original frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Create fullscreen window
    create_fullscreen_window(window_name)
    cv2.resizeWindow(window_name, 1920, 1080)  # Adjust to your screen resolution

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            # Convert display coordinates back to original frame coordinates
            frame_x = int((x * frame_width) / 1920)  # Scale back to original frame width
            frame_y = int((y * frame_height) / 1080)  # Scale back to original frame height
            points.append((frame_x, frame_y))
            draw_frame(temp_frame, points, context_text)

    def draw_frame(frame, points, text):
        """Helper function to draw current state."""
        annotated = frame.copy()
        display_text(annotated, text)
        
        # Draw points as they're being selected
        for idx, (px, py) in enumerate(points):
            cv2.circle(annotated, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"P{idx + 1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # If all points are selected, draw the enlarged bounding box
        if len(points) == num_points:
            annotated = draw_enlarged_bbox(annotated, points, scale)
            
        # Resize the frame to fill the window
        display_frame = cv2.resize(annotated, (1920, 1080))
        cv2.imshow(window_name, display_frame)

    # Show initial frame
    display_frame = cv2.resize(temp_frame, (1920, 1080))
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(points) == num_points:
            cv2.destroyWindow(window_name)
            break
        elif key == ord('r'):
            points = []
            cv2.imshow(window_name, display_frame)
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

    return points

from IPython.display import clear_output, display
from PIL import Image
class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 ):
        
        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cuda')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )
        
        return mask

    def add_frame(self, frame, mask):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        mask = self.resize_mask(mask=mask)
        mask =  mask.cpu()
        mask = (mask > 0.0).numpy()
        
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]
                
        rgb_frame = Image.fromarray(frame)
        clear_output(wait=True)
        # display(rgb_frame)
        cv2_frame = cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)
        return cv2_frame


some_paths = []
def cleanup(self):
    """Clean up resources"""
    del self.state
    torch.cuda.empty_cache()
    gc.collect()

    cv2.resize("Live Feed", (1280, 720))
    


PATHS = {
    "normal": "sam2_results/90FPS_full_res_normal.mp4",
    "edge": "sam2_results/90FPS_full_res_edege.mp4",
}

def main_without_thread_customization(camera_index=0, width=1280, height=720, fps=90, codec="MJPG", bbox_scale=1.35, req_fps=90):
    global SAM2_PROCESSING
    motion_session_started = False
    motion_video_writer = None
    motion_detector = MotionDetection()
    is_fullscreen = True

    # --------------------------
    # New: Ask for desired processing dimensions and FPS
    try:
        desired_width = int(input("Enter desired processing width: "))
        desired_height = int(input("Enter desired processing height: "))
        desired_processing_fps = int(input("Enter desired processing FPS: "))
    except Exception as e:
        print("Invalid input, using default parameters.")
        desired_width, desired_height, desired_processing_fps = width, height, req_fps
    # --------------------------
    
    #  ======================================================================
    NUM_OBJECTS = 1
    SAM_CHECKPOINT_FILEPATH = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    SAM_CONFIG_FILEPATH = determine_model_cfg(SAM_CHECKPOINT_FILEPATH)
    DEVICE = "cuda"
    sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                    config_file=SAM_CONFIG_FILEPATH,
                                    ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                    device=DEVICE,
                                    verbose=False
                                    )
    #  ======================================================================
    # For real-time visualization, use desired processing resolution
    visualizer = Visualizer(video_width=desired_width,
                            video_height=desired_height
                           )
    # Initialize Camera
    cap = cv2.VideoCapture(camera_index)
    if isinstance(camera_index, int):
        cap = configure_camera(cap, width, height, fps, codec)
    if not cap or not cap.isOpened():
        logging.error("Camera not initialized.")
        return

    # Determine the recorded fps from the capture device, fallback to fps parameter
    recorded_fps = cap.get(cv2.CAP_PROP_FPS)
    if recorded_fps <= 0:
        recorded_fps = fps

    # Calculate frame interval based on recorded fps & desired processing fps.
    # For example, if recorded_fps=90 and desired_processing_fps=45 then sample every 2nd frame.
    frame_interval = max(1, int(recorded_fps / desired_processing_fps))
    
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read initial frame.")
        return

    # Create temporary directory for initialization
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(f"{temp_dir}/state_space", exist_ok=True)
    cv2.imwrite(os.path.join(temp_dir, "0.jpg"), frame)
    
    # Create fullscreen window for live feed
    window_name = "Live Feed"
    create_fullscreen_window(window_name)

    # Hardcoded ROI points
    roi_points = [(477, 311), (1120, 310), (1277, 625), (298, 630)]
    if not roi_points:
        logging.error("ROI selection canceled.")
        return

    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

    # Bounding box selection
    bbox_points = None
    os.makedirs('temp_frames/gdino_inference', exist_ok=True)
    cv2.imwrite(os.path.join("temp_frames/gdino_inference", "__1.jpg"), frame)
    bbox_points = process_single_image('temp_frames/gdino_inference/__1.jpg','temp_frames/gdino_inference/__1_infered.jpg', "medical paper, box")
           
    # while bbox_points is None:
    #     bbox_points = select_points(frame, num_points=4, context_text="Select Object to Track", scale=bbox_scale)
    
    x_coords, y_coords = zip(*bbox_points)
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    sam_box = np.array([[x1, y1], [x2, y2]])
    
    print(f"Initial BBOX Points: {sam_box}, TYPE: {type(sam_box)}")
    #=====================================================================
    
    try:
        frame_idx = 0
        
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply ROI mask and detect motion
                motion_detector.motionUpdate(frame, roi_mask)

                # Draw the enlarged bounding box on the live feed
                copy_frame = frame.copy()
                frame = draw_enlarged_bbox(frame, bbox_points, scale=bbox_scale)

                if motion_detector.current_motion_status:
                    # Process only every 'frame_interval' frame for inference
                    if frame_idx % frame_interval == 0:
                        FRAME_QUEUE.append(copy_frame)
                    if frame_idx == 0:
                        sam_out = sam.track_new_object(frame, box=sam_box)
                    if len(FRAME_QUEUE) >= FRAME_QUEUE.maxlen and frame_idx > 0:
                        pass
                    sam_out = sam.track_all_objects(copy_frame)
                    # Resize frame for visualization based on desired processing dimensions.
                    frame = visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])

                    if not motion_session_started:
                        if not os.path.exists("videos"):
                            os.makedirs("videos")
                        motion_video_path = os.path.join("videos", f"{datetime.datetime.now().strftime('%d%m-%Y_%H%M%S')}.mp4")
                        out_video_path = os.path.join("videos", f"{datetime.datetime.now().strftime('%d%m-%Y_%H%M%S')}_out.mp4")
                        # Use recorded_fps so that the output video matches the source capture rate
                        motion_video_writer = cv2.VideoWriter(
                            motion_video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            recorded_fps,
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        )
                        out_video_writer = cv2.VideoWriter(
                            out_video_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            recorded_fps,
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            # (desired_width, desired_width)
                        )
                        motion_session_started = True
                        logging.info(f"Started motion session: {motion_video_path}")
                    
                    if motion_video_writer:
                        motion_video_writer.write(copy_frame)
                        out_video_writer.write(frame)
                    frame_idx += 1
                    # ============================================================
                    # ================= TO PROCESS IT WITH VIDEO =================
                    # ============================================================
                    global some_paths
                    some_paths = [motion_video_path, bbox_points]
                    # ============================================================ 
                
                elif motion_session_started:
                    print("Motion Stopped")
                    if motion_video_writer:
                        motion_video_writer.release()
                        logging.info(f"Stopped motion session: {motion_video_path}")
                        frame_idx = 0
                        FRAME_QUEUE.clear()
                        os.makedirs('temp_frames/gdino_inference', exist_ok=True)
                        cv2.imwrite(os.path.join("temp_frames/gdino_inference", "__1.jpg"), frame)
                        bbox_points = process_single_image('temp_frames/gdino_inference/__1.jpg','temp_frames/gdino_inference/__1_infered.jpg', "medical paper, box")
           
                    motion_session_started = False
                    if motion_video_writer:
                        del motion_video_writer
                    display_text(frame, "SAM2 Processing...", position=(10, 50), color=(0, 255, 255))
                    show_frame(window_name, frame)
                    cv2.waitKey(200)

                    s_t = time.time()
                    print("Inference time: ", time.time() - s_t)
                    show_frame(window_name, frame)

                status = "SAM2 Processing..." if SAM2_PROCESSING else "Live Feed"
                display_text(frame, status)
                show_frame(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    
    finally:
        def clean_dirs(temp_dir):    
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}. Reason: {e}")
        clean_dirs(f"{temp_dir}/state_space")
        clean_dirs(temp_dir)
        STOP_SIGNAL.set()
        cap.release()
        cv2.destroyAllWindows()


import threading
import queue
import time

# Shared variables for tracking
tracking_queue = queue.Queue(maxsize=5)  # Small queue size to prevent lag
latest_output = None  # Shared variable to hold the latest tracking results
tracking_lock = threading.Lock()  # Lock to ensure thread-safe updates to `latest_output`

class TrackingThread(threading.Thread):
    def __init__(self, sam, visualizer):
        super().__init__()
        self.sam = sam
        self.visualizer = visualizer
        self.stop_event = threading.Event()

    def run(self):
        global latest_output
        while not self.stop_event.is_set():
            try:
                # Get a frame from the queue
                frame = tracking_queue.get(timeout=1)  # Wait for 1 second for a frame
                if frame is None:
                    break

                # Perform object tracking
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    sam_out = self.sam.track_all_objects(frame)

                # Update the latest output
                with tracking_lock:
                    latest_output = self.visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])

            except queue.Empty:
                continue  # No frame in the queue, continue waiting
            except Exception as e:
                logging.error(f"Error in tracking thread: {e}")
                break

    def stop(self):
        self.stop_event.set()


def main_customization(camera_index=0, width=1280, height=720, fps=90, codec="MJPG", bbox_scale=1.35, req_fps=90):
    global SAM2_PROCESSING, latest_output
    motion_session_started = False
    motion_video_writer = None
    motion_detector = MotionDetection()
    is_fullscreen = True

    # Initialize SAM2 and visualizer
    NUM_OBJECTS = 1
    SAM_CHECKPOINT_FILEPATH = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    SAM_CONFIG_FILEPATH = determine_model_cfg(SAM_CHECKPOINT_FILEPATH)
    DEVICE = "cuda"
    sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                    config_file=SAM_CONFIG_FILEPATH,
                                    ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                    device=DEVICE,
                                    verbose=False)
    visualizer = Visualizer(video_width=width, video_height=height)

    # Start the tracking thread
    tracking_thread = TrackingThread(sam, visualizer)
    tracking_thread.start()

    # Initialize Camera
    cap = cv2.VideoCapture(camera_index)
    if isinstance(camera_index, int):
        cap = configure_camera(cap, width, height, fps, codec)
    if not cap or not cap.isOpened():
        logging.error("Camera not initialized.")
        return

    recorded_fps = cap.get(cv2.CAP_PROP_FPS)
    if recorded_fps <= 0:
        recorded_fps = fps

    frame_interval = max(1, int(recorded_fps / req_fps))

    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read initial frame.")
        return

    # Create fullscreen window for live feed
    window_name = "Live Feed"
    create_fullscreen_window(window_name)

    # Hardcoded ROI points
    roi_points = [(477, 311), (1120, 310), (1277, 625), (298, 630)]
    if not roi_points:
        logging.error("ROI selection canceled.")
        return

    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

    # Bounding box selection
    # bbox_points = None
    # while bbox_points is None:
    #     bbox_points = select_points(frame, num_points=4, context_text="Select Object to Track", scale=bbox_scale)
    
    cv2.imwrite(os.path.join("temp_frames/gdino_inference", "__1.jpg"), frame)
    bbox_points = process_single_image('temp_frames/gdino_inference/__1.jpg','temp_frames/gdino_inference/__1_infered.jpg', "medical paper, box")
           
    x_coords, y_coords = zip(*bbox_points)
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    sam_box = np.array([[x1, y1], [x2, y2]])

    print(f"Initial BBOX Points: {sam_box}, TYPE: {type(sam_box)}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply ROI mask and detect motion
            motion_detector.motionUpdate(frame, roi_mask)

            # Draw the enlarged bounding box on the live feed
            copy_frame = frame.copy()
            frame = draw_enlarged_bbox(frame, bbox_points, scale=bbox_scale)

            if motion_detector.current_motion_status:
                # Process only every 'frame_interval' frame for inference
                if frame_idx % frame_interval == 0:
                    if not tracking_queue.full():
                        tracking_queue.put(copy_frame)  # Add frame to the tracking queue

                if frame_idx == 0:
                    sam_out = sam.track_new_object(frame, box=sam_box)

                # Update the frame with the latest tracking results
                with tracking_lock:
                    if latest_output is not None:
                        frame = latest_output

                if not motion_session_started:
                    if not os.path.exists("videos"):
                        os.makedirs("videos")
                    motion_video_path = os.path.join("videos", f"{datetime.datetime.now().strftime('%d%m-%Y_%H%M%S')}.mp4")
                    out_video_path = os.path.join("videos", f"{datetime.datetime.now().strftime('%d%m-%Y_%H%M%S')}_out.mp4")
                    motion_video_writer = cv2.VideoWriter(
                        motion_video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        recorded_fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    )
                    out_video_writer = cv2.VideoWriter(
                        out_video_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        recorded_fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    )
                    motion_session_started = True
                    logging.info(f"Started motion session: {motion_video_path}")

                if motion_video_writer:
                    motion_video_writer.write(copy_frame)
                    out_video_writer.write(frame)
                frame_idx += 1

            elif motion_session_started:
                print("Motion Stopped")
                if motion_video_writer:
                    motion_video_writer.release()
                    logging.info(f"Stopped motion session: {motion_video_path}")
                    frame_idx = 0
                    os.makedirs("temp_frames/gdino_inference", exist_ok=True)
                    cv2.imwrite(os.path.join("temp_frames/gdino_inference", "__1.jpg"), frame)
                    process_single_image('temp_frames/gdino_inference/__1.jpg','temp_frames/gdino_inference/__1_infered.jpg', "medical paper, box")
                motion_session_started = False

            # Display the frame
            show_frame(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    finally:
        # Stop the tracking thread
        tracking_thread.stop()
        tracking_thread.join()

        # Clean up resources
        if motion_video_writer:
            motion_video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     camera_index = PATHS["normal"]
#     main_customization(camera_index=2, width=1280, height=720, fps=120, codec="MJPG", bbox_scale=1.2)



if __name__ == "__main__":
    camera_index = PATHS["normal"]
    main_without_thread_customization(camera_index=0, width=1280, height=720, fps=120, codec="MJPG", bbox_scale=1.2)
    # main_customization(camera_index=2, width=1280, height=720, fps=90, codec="MJPG", bbox_scale=1.2)
    # ============================================================
    # ================= TO PROCESS IT WITH VIDEO =================
    # ============================================================
    # motion_video_path, bbox_points = some_paths
    # motion_type = process_motion_session_with_sam2(
    #                 motion_video_path, bbox_points, "./sam2/checkpoints/sam2.1_hiera_large.pt", "sam2_results"
    #)
