import cv2
import threading
import logging
import os
import uuid
import numpy as np
from motion import MotionDetection
from samurai import process_video
from threading import Thread
from queue import Queue

STOP_SIGNAL = threading.Event() 
SAM2_PROCESSING = False

motion = MotionDetection()

def display_text(frame, text, position=(10, 30), color=(0, 255, 0)):
    """Helper function to display text on a frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def decode_fourcc(value):
    """Decode the FourCC codec value."""
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def create_fullscreen_window(window_name):
    """Create a fullscreen window."""
    pass

def show_frame(window_name, frame):
    """Display frame in a consistent window size."""
    height, width = frame.shape[:2]
    scale_width = 1280 / width
    scale_height = 720 / height
    scale = min(scale_width, scale_height)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow(window_name, resized_frame)

def draw_quadrilateral(frame, points, label=None, enlarge=False, enlarge_factor=1.2):
    """
    Draw a quadrilateral on the frame using the provided points.
    Optionally enlarge the quadrilateral and add a label above it.
    
    Args:
        frame: Input frame to draw the quadrilateral.
        points: List of 4 (x, y) coordinates defining the quadrilateral.
        label: Optional label to display above the quadrilateral.
        enlarge: Boolean flag to determine whether to enlarge the quadrilateral.
        enlarge_factor: Factor by which to enlarge the quadrilateral.
    """
    if len(points) != 4:
        return frame

    points = np.array(points, dtype=np.float32)
    
    if enlarge:
        centroid = np.mean(points, axis=0) 
        points = (points - centroid) * enlarge_factor + centroid
    
    points = points.astype(np.int32)

    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    if label:
        text_position = tuple(points[0] + np.array([0, -10]))
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

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

def process_video_instance(video_path, coords, model_path, save_video, output_path, result_queue, index):
    """
    Wrapper to run `process_video` and push results with their index to a queue.
    """
    try:
        print(model_path)
        result = process_video(video_path, coords, model_path, save_video, output_path)
        result_queue.put((index, result))  # Store the result with its index
    except Exception as e:
        result_queue.put((index, f"Error: {str(e)}"))  # Store the error with its index

def process_motion_session_with_sam2(video_path, bbox_points, model_path, output_folder):
    """Process recorded video with SAM2."""
    
    print(bbox_points)
    
    xywhs = []
    for bbox_point in bbox_points:
        x_coords, y_coords = zip(*bbox_point)
        x1, y1 = min(x_coords), min(y_coords)
        w = max(x_coords) - x1
        h = max(y_coords) - y1
        xywhs.append((x1, y1, w, h))
        
    print("xywhs:", xywhs)
    
    result_queue = Queue()
    threads = []
    
    uid = uuid.uuid4()
    for i, coords in enumerate(xywhs):
        output_path = os.path.join(output_folder, f"{uid}_{i+1}.mp4")
        
        thread = Thread(target=process_video_instance, args=(
            video_path, coords, model_path, True, output_path, result_queue, i
        ))
        
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Collect results and sort them by index
    results = [None] * len(xywhs)  # Preallocate list to match the xywhs indices
    while not result_queue.empty():
        index, result = result_queue.get()
        results[index] = result  # Store result at the correct index
        
    return results

def select_points(frame, context_text="", num_objects=2):
    """Select points for multiple quadrilaterals."""
    all_points = []
    temp_frame = frame.copy()
    window_name = "Select Points"
    frame_height, frame_width = frame.shape[:2]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(all_points) < num_objects * 4:
            scaled_x = int(x * (frame_width / 1280))  # Scale to original frame dimensions
            scaled_y = int(y * (frame_height / 720))
            all_points.append((scaled_x, scaled_y))
            draw_frame(temp_frame, all_points, context_text)

    def draw_frame(frame, points, text):
        annotated = frame.copy()
        display_text(annotated, text)

        current_points = []
        for i, point in enumerate(points):
            cv2.circle(annotated, point, 5, (0, 0, 255), -1)
            current_points.append(point)

            if (i + 1) % 4 == 0:  # Complete a quadrilateral
                annotated = draw_quadrilateral(annotated, current_points)
                current_points = []

        display_frame = cv2.resize(annotated, (1280, 720))
        cv2.imshow(window_name, display_frame)

    display_frame = cv2.resize(temp_frame, (1280, 720))
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(all_points) == num_objects * 4:
            cv2.destroyWindow(window_name)
            break
        elif key == ord('r'):
            all_points = []
            cv2.imshow(window_name, display_frame)
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

    grouped_points = [all_points[i * 4:(i + 1) * 4] for i in range(num_objects)]
    return grouped_points

def main(camera_index=0, width=1280, height=720, fps=90, codec="MJPG"):
    global SAM2_PROCESSING
    motion_session_started = False
    motion_video_writer = None
    is_fullscreen = True

    cap = cv2.VideoCapture(camera_index)
    cap = configure_camera(cap, width, height, fps, codec)
    if not cap or not cap.isOpened():
        logging.error("Camera not initialized.")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to read initial frame.")
        return

    window_name = "Live Feed"
    create_fullscreen_window(window_name)
    
    # Hardcoded ROI points
    roi_points = [(477, 311), (1120, 310), (1277, 625), (298, 630)]
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], 255)

    bbox_points = None
    while bbox_points is None:
        bbox_points = select_points(frame, context_text="Select 2 Compartments to Track")
    print(f"Selected Quadrilaterals: {bbox_points}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            copy_frame = frame.copy()

            for idx, quad in enumerate(bbox_points):
                label = f"Compartment {idx + 1}"
                frame = draw_quadrilateral(frame, quad, label=label, enlarge=True)

            motion.motionUpdate(frame, roi_mask)

            if motion.current_motion_status:
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
                    motion_video_writer.write(copy_frame)

            elif motion_session_started:
                print("Motion Stopped")
                if motion_video_writer:
                    motion_video_writer.release()
                    logging.info(f"Stopped motion session: {motion_video_path}")
                motion_session_started = False

                display_text(frame, "SAM2 Processing...", position=(10, 50), color=(0, 255, 255))
                show_frame(window_name, frame)
                cv2.waitKey(200)

                motion_types = process_motion_session_with_sam2(
                    motion_video_path, bbox_points, "./sam2/checkpoints/sam2.1_hiera_large.pt", "sam2_results"
                )

                # Update display with SAM2 result
                display_text(frame, f"SAM2 Result: Compartment 1: {motion_types[0]} | Compartment 2: {motion_types[1]}", position=(10, 90), color=(0, 255, 0))
                show_frame(window_name, frame)
                cv2.waitKey(3000)
                
                print(motion_types)

            display_text(frame, "Live Feed")
            show_frame(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                bbox_points = select_points(copy_frame, context_text="Re-select Objects to Track")
                if bbox_points:
                    print(f"Updated Quadrilaterals: {bbox_points}")

    finally:
        STOP_SIGNAL.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(camera_index="grab_20241219_164428.mp4", width=1280, height=720, fps=90, codec="MJPG")
