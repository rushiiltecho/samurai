from collections import deque
import sys
import cv2
import torch

from samurai import determine_model_cfg

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

class VideoProcessor:
    def __init__(self, model_cfg, checkpoint, max_frames=30):
        self.frame_buffer = deque(maxlen=max_frames)
        self.state = None
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
        
    def add_frame(self, frame):
        self.frame_buffer.append(frame)
        
        
    def init_tracking(self, initial_bbox):
        """Initialize tracking with first frame and bbox"""
        if len(self.frame_buffer) == 0:
            return
            
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize state with current frame buffer
            self.state = self.predictor.init_state(
                self.frame_buffer,
                offload_video_to_cpu=True
            )
            
            # Add initial bbox for tracking
            bbox, track_label = initial_bbox, 0
            _, _, masks = self.predictor.add_new_points_or_box(
                self.state, 
                box=bbox,
                frame_idx=0,
                obj_id=0
            )
            
    def process_frames(self):
        """Process frames and return masks"""
        if self.state is None:
            return None
            
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            results = []
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(self.state):
                mask = masks[0].cpu().numpy() # Assuming single object tracking
                results.append((frame_idx, mask))
            return results
        
        
        
        

# Initialize video capture
video_path = "videos/d5f67104-544e-4e4e-8a2d-77d0b4e6cc13.mp4"
cap = cv2.VideoCapture(2)  # or your video source
model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_config = determine_model_cfg(model_path)
processor = VideoProcessor(max_frames=30, model_cfg=model_config, checkpoint=model_path)

# Get initial bbox (could be from user input or detector)
initial_bbox = [(378, 701), (641, 517), (528, 367), (325, 444)] # coordinates in x1,y1,x2,y2 format

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Add frame to buffer
    processor.add_frame(frame)
    
    # Initialize tracking if not started
    if processor.state is None:
        processor.init_tracking(initial_bbox)
        continue
        
    # Process frames and get masks
    results = processor.process_frames()
    if results:
        for frame_idx, mask in results:
            # Visualize or process the mask
            mask_overlay = frame.copy()
            mask_overlay[mask > 0.0] = [0, 0, 255]  # Red overlay for mask
            frame = cv2.addWeighted(frame, 0.8, mask_overlay, 0.2, 0)
            
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 