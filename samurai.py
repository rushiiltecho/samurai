from collections import deque
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def load_prompt(coords):
    x, y, w, h = coords
    prompts = {}
    prompts[0] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    """Determine the model configuration based on model path."""
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    """Prepare video input path."""
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def classify_motion(bbox_sequence, frame_height, frame_width, threshold=0.9):
    """
    Classify whether a motion is a "grab" or invalid based on tracking data.
    A grab is defined as the object being taken away from the stack downward and out of frame.
    """
    if len(bbox_sequence) < 2:
        return "invalid"

    first_bbox = bbox_sequence[0]
    last_bbox = bbox_sequence[-1]
    
    first_center_y = first_bbox[1] + first_bbox[3]/2
    last_center_y = last_bbox[1] + last_bbox[3]/2
    
    moving_down = last_center_y > first_center_y
    print("last_center_y", last_center_y, "first_center_y", first_center_y)
    
    margin = frame_height * threshold
    out_of_frame = last_bbox[1] + last_bbox[3] >= margin
    
    bbox_areas = [(box[2] * box[3]) for box in bbox_sequence]
    
    print()
    
    smooth_motion = all(
        abs(bbox_areas[i] - bbox_areas[i-1]) / bbox_areas[i-1] < 0.5 
        for i in range(1, len(bbox_areas))
    )
    
    print("moving_down:", moving_down, "out_of_frame:", out_of_frame, "smooth_motion:", smooth_motion)
    
    if moving_down and out_of_frame:
        print("grab")
        return "grab"
    
    print("invalid")
    return "invalid"



# ===================================================================================
# ===================================================================================
color = [(255, 0, 0)]
def process_realtime_frame(frame, predictor, state, frame_idx, prompts):
    """
    Process a single frame in real-time
    """
    bbox_sequence = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if frame_idx == 0:
            # Initialize tracking with first frame
            bbox, track_label = prompts[0]
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    if obj_id == 0:  # Track the first object
                        bbox_sequence.append(bbox)
                    
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                # if save_video:
                #     img = loaded_frames[frame_idx].copy()
                #     for obj_id, mask in mask_to_vis.items():
                #         mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
                #         mask_img[mask] = color[(obj_id + 1) % len(color)]
                #         img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                #     for obj_id, bbox in bbox_to_vis.items():
                #         cv2.rectangle(img, (bbox[0], bbox[1]), 
                #                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                #                     color[obj_id % len(color)], 2)

                #     out.write(img)
                
        # Process masks and draw visualization
        img = frame.copy()
        if masks is not None:
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                
                # Draw mask overlay
                mask_img = np.zeros_like(img)
                mask_img[mask] = color[obj_id % len(color)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                # Draw bounding box
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), 
                                color[obj_id % len(color)], 2)
    
        # Classify the motion
        print(frame.shape)
        frame_height, frame_width= frame.shape[:2]
        motion_type = classify_motion(bbox_sequence, frame_height, frame_width)

        return img
    
# ===================================================================================
def process_frame(frame_queue, model_path, coords):
    pass
# ===================================================================================
def __process_realtime_frame(frame, predictor, state, frame_idx, prompts):
    """Process a single frame in real-time"""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if frame_idx == 0:
            # Initialize tracking with first frame
            bbox, track_label = prompts[0]
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        else:
            # Propagate tracking to current frame
            object_ids, masks = predictor.propagate_step(frame, frame_idx)

        # Process masks and draw visualization
        img = frame.copy()
        if masks is not None:
            for obj_id, mask in enumerate(masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                
                # Draw mask overlay
                mask_img = np.zeros_like(img)
                mask_img[mask] = color[obj_id % len(color)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                # Draw bounding box
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), 
                                color[obj_id % len(color)], 2)

        return img
# ===================================================================================
# ===================================================================================



def process_video(video_path, coords, model_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt", 
                 save_video=False, output_path="demo.mp4"):
    """
    Main function to process video and classify motion.
    
    Args:
        video_path (str): Path to input video or frames directory
        coords (str): Path to bounding box text file
        model_path (str): Path to model checkpoint
        save_video (bool): Whether to save visualization video
        output_path (str): Path for output video if save_video is True
        
    Returns:
        str: Motion classification ("grab" or "invalid")
    """
    color = [(255, 0, 0)]
    
    # Initialize model and predictor
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(video_path)
    prompts = load_prompt(coords)

    # Get frame dimensions
    if isinstance(frames_or_path, str):
        if frames_or_path.endswith('.mp4'):
            cap = cv2.VideoCapture(frames_or_path)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            print("SAM2 frame rate: ", frame_rate)
            cap.release()
        else:
            first_frame = cv2.imread(os.path.join(frames_or_path, sorted(os.listdir(frames_or_path))[0]))
            frame_height, frame_width = first_frame.shape[:2]
            frame_rate = 30

    # Initialize video writer if saving output
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
        
        if isinstance(frames_or_path, str) and frames_or_path.endswith('.mp4'):
            cap = cv2.VideoCapture(frames_or_path)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
        else:
            frames = sorted([osp.join(frames_or_path, f) for f in os.listdir(frames_or_path) 
                           if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]

    # Track object and classify motion
    bbox_sequence = []
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    if obj_id == 0:  # Track the first object
                        bbox_sequence.append(bbox)
                    
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

            if save_video:
                img = loaded_frames[frame_idx].copy()
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), 
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                color[obj_id % len(color)], 2)

                out.write(img)

    if save_video:
        out.release()

    # Classify the motion
    motion_type = classify_motion(bbox_sequence, frame_height, frame_width)

    # Cleanup
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    
    return motion_type

#  =================================================================================================
#  =================================================================================================



# class FrameProcessor:
#     def __init__(self, coords, model_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt", device="cuda:0"):
#         """
#         Initialize the frame processor with model and tracking state
        
#         Args:
#             coords: Initial bounding box coordinates (x, y, w, h)
#             model_path: Path to the model checkpoint
#             device: Device to run inference on
#         """
#         self.color = [(255, 0, 0)]
#         self.bbox_sequence = []
#         self.frame_count = 0
        
#         # Initialize model
#         model_cfg = determine_model_cfg(model_path)
#         self.predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
        
#         # Initialize prompts
#         self.prompts = self.load_prompt(coords)
        
#         # Initialize state
#         self.state = None
#         self.initialized = False

#     def load_prompt(self, coords):
#         x, y, w, h = coords
#         prompts = {}
#         prompts[0] = ((x, y, x + w, y + h), 0)
#         return prompts

#     def initialize_state(self, first_frame):
#         """Initialize the tracking state with the first frame"""
#         self.frame_height, self.frame_width = first_frame.shape[:2]
        
#         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#             # Initialize state with first frame
#             self.state = self.predictor.init_state_with_frame(first_frame)
            
#             # Add initial bounding box
#             bbox, track_label = self.prompts[0]
#             _, _, masks = self.predictor.add_new_points_or_box(
#                 self.state, box=bbox, frame_idx=0, obj_id=0
#             )
            
#             self.initialized = True
#             return self.process_masks(masks, first_frame.copy())

#     def process_frame(self, frame):
#         """
#         Process a single frame and return visualization results
        
#         Args:
#             frame: Input frame as numpy array (H,W,C)
            
#         Returns:
#             dict containing:
#                 - annotated_frame: Frame with visualizations
#                 - masks: Dictionary of masks for each tracked object
#                 - bboxes: Dictionary of bounding boxes for each tracked object
#                 - motion: Motion classification (only on last frame)
#         """
#         if not self.initialized:
#             return self.initialize_state(frame)

#         self.frame_count += 1
        
#         with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#             # Propagate tracking to current frame
#             frame_idx, object_ids, masks = next(self.predictor.propagate_with_frame(
#                 self.state, frame
#             ))
            
#             return self.process_masks(masks, frame.copy())

#     def process_masks(self, masks, frame):
#         """Process masks and create visualizations"""
#         mask_to_vis = {}
#         bbox_to_vis = {}
        
#         # Process each mask
#         for obj_id, mask in enumerate(masks):
#             mask = mask[0].cpu().numpy()
#             mask = mask > 0.0
#             non_zero_indices = np.argwhere(mask)
            
#             if len(non_zero_indices) > 0:
#                 y_min, x_min = non_zero_indices.min(axis=0)
#                 y_max, x_max = non_zero_indices.max(axis=0)
#                 bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
#                 if obj_id == 0:  # Track the first object
#                     self.bbox_sequence.append(bbox)
                
#                 bbox_to_vis[obj_id] = bbox
#                 mask_to_vis[obj_id] = mask

#         # Create visualization
#         for obj_id, mask in mask_to_vis.items():
#             mask_img = np.zeros_like(frame)
#             mask_img[mask] = self.color[(obj_id + 1) % len(self.color)]
#             frame = cv2.addWeighted(frame, 1, mask_img, 0.2, 0)

#         for obj_id, bbox in bbox_to_vis.items():
#             cv2.rectangle(frame, (bbox[0], bbox[1]),
#                         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
#                         self.color[obj_id % len(self.color)], 2)

#         # Only classify motion on last frame
#         motion = None
#         if self.frame_count > 1:
#             motion = classify_motion(self.bbox_sequence, 
#                                   self.frame_height, 
#                                   self.frame_width)

#         return {
#             'annotated_frame': frame,
#             'masks': mask_to_vis,
#             'bboxes': bbox_to_vis,
#             'motion': motion
#         }

#     def cleanup(self):
#         """Clean up resources"""
#         del self.predictor, self.state
#         gc.collect()
#         torch.clear_autocast_cache()
#         torch.cuda.empty_cache()
        
        

class SAM2Processor:
    def __init__(self, model_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt"):
        """Initialize SAM2 processor with model and tracking state"""
        self.color = [(255, 0, 0)]
        self.model_cfg = self.determine_model_cfg(model_path)
        self.predictor = build_sam2_video_predictor(self.model_cfg, model_path, device="cuda:0")
        self.state = None
        self.first_frame = True
        self.bbox_sequence = []

    def determine_model_cfg(self, model_path):
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            raise ValueError("Unknown model size in path!")

    def initialize_tracking(self, frame, bbox):
        """Initialize tracking with first frame and bounding box"""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.state = self.predictor.init_state_with_frame(frame)
            _, _, masks = self.predictor.add_new_points_or_box(self.state, box=bbox, frame_idx=0, obj_id=0)
            self.first_frame = False
            return self.process_masks(masks, frame.copy())

    def process_frame(self, frame):
        """Process a single frame after initialization"""
        if self.first_frame:
            raise ValueError("Must initialize tracking first with initialize_tracking()")
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            frame_idx, object_ids, masks = next(self.predictor.propagate_with_frame(self.state, frame))
            return self.process_masks(masks, frame.copy())

    def process_masks(self, masks, frame):
        """Process masks and create visualizations"""
        height, width = frame.shape[:2]
        mask = masks[0][0].cpu().numpy()
        mask = mask > 0.0
        
        # Get bounding box from mask
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0)
            y_max, x_max = non_zero_indices.max(axis=0)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        self.bbox_sequence.append(bbox)
        
        # Create visualization
        mask_img = np.zeros((height, width, 3), np.uint8)
        mask_img[mask] = self.color[0]
        frame = cv2.addWeighted(frame, 1, mask_img, 0.2, 0)
        cv2.rectangle(frame, (bbox[0], bbox[1]), 
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                     self.color[0], 2)
        
        return frame, bbox, mask

    def classify_motion(self, frame_height, frame_width, threshold=0.9):
        """Classify the motion based on tracking data"""
        if len(self.bbox_sequence) < 2:
            return "invalid"

        first_bbox = self.bbox_sequence[0]
        last_bbox = self.bbox_sequence[-1]
        
        first_center_y = first_bbox[1] + first_bbox[3]/2
        last_center_y = last_bbox[1] + last_bbox[3]/2
        
        moving_down = last_center_y > first_center_y
        
        margin = frame_height * threshold
        out_of_frame = last_bbox[1] + last_bbox[3] >= margin
        
        bbox_areas = [(box[2] * box[3]) for box in self.bbox_sequence]
        smooth_motion = all(
            abs(bbox_areas[i] - bbox_areas[i-1]) / bbox_areas[i-1] < 0.5 
            for i in range(1, len(bbox_areas))
        )
        
        if moving_down and out_of_frame:
            return "grab"
        
        return "invalid"

    def cleanup(self):
        """Clean up resources"""
        del self.predictor, self.state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()