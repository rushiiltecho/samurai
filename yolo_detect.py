import cv2
from ultralytics import YOLO

def process_video_frames(video_path: str, model_path: str, num_frames: int = 10):
    """
    Process specified number of frames from a video using YOLOv8 OBB model.
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the YOLO model weights
        num_frames (int): Number of frames to process (default: 10)
    """
    # Load the YOLOv8 model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video after {frame_count} frames")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        print(f"\nFrame {frame_count + 1}:")
        
        # Check if any detections in this frame
        if len(results) > 0:
            # Get the boxes
            boxes = results[0].boxes
            
            # Process each detection
            if boxes:
                for i, box in enumerate(boxes):
                    # Convert box to xywh format
                    if box.xywh is not None:  # Check if detection exists
                        x, y, w, h = box.xywh[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        print(f"Detection {i + 1}:")
                        print(f"  Class: {model.names[int(cls)]}")
                        print(f"  Confidence: {conf:.2f}")
                        print(f"  Box (x, y, w, h): {x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}")
                break
            else:
                print("No boxes in this frame")
        else:
            print("No detections in this frame")
        
        frame_count += 1

    # Release resources
    cap.release()
    print(f"\nProcessed {frame_count} frames")

if __name__ == "__main__":
    # Example usage
    video_path = "grab/guidezilla_20241210_201517.mp4"
    model_path = "best_rack_obb.pt"  # Use your OBB model path
    
    process_video_frames(
        video_path=video_path,
        model_path=model_path,
        num_frames=10
    )