import os

import cv2

import numpy as np

import torch

import supervision as sv

from groundingdino.util.inference import load_model, load_image, predict

from groundingdino.util import box_ops



# Paths

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"

WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

#TEXT_PROMPT = "medical product, paper"

BOX_THRESHOLD = 0.35

TEXT_THRESHOLD = 0.25



def get_dynamic_pq_points(image_path, percentage=8):

    """

    Calculate PQ points dynamically based on the image size and a percentage.

    """

    image = cv2.imread(image_path)

    if image is None:

        print("Failed to load the image. Check the path.")

        return None



    # Get image dimensions

    height, width, _ = image.shape

    print(f"Image size: {width}x{height}")



    # Calculate PQ box coordinates based on percentage

    # width_pixels = int((width * percentage) / 100)

    # height_pixels = int((height * percentage) / 100)

    

    #Calculate PQ box with min dim

    min_dimension = min(width, height)

    inset = int((min_dimension * percentage) / 100)

    # PQ box coordinates

    # p1, q1 = width_pixels, height_pixels  # Top left

    # p3, q3 = width - width_pixels, height - height_pixels  # Bottom right

    

    #PQ box with min_dim

    p1, q1 = inset, inset

    p3, q3 = width - inset, height - inset

    # Define PQ box points

    pq_points = [(p1, q1), (p3, q1), (p3, q3), (p1, q3)]

    return pq_points



def convert_xyxy_to_corners(xyxy):

    """

    Convert detection from [x1, y1, x2, y2] format to four corners format.

    """

    x1, y1, x2, y2 = xyxy

    top_left = (int(x1), int(y1))

    top_right = (int(x2), int(y1))

    bottom_right = (int(x2), int(y2))

    bottom_left = (int(x1), int(y2))

    return [top_left, top_right, bottom_right, bottom_left]





def is_bbox_entirely_outside_pq(bbox_corners, pq_points):

    """Check if a bounding box is entirely outside the PQ box."""

    pq_x_coords = [point[0] for point in pq_points]

    pq_y_coords = [point[1] for point in pq_points]



    # PQ box boundaries

    x_min_pq, x_max_pq = min(pq_x_coords), max(pq_x_coords)

    y_min_pq, y_max_pq = min(pq_y_coords), max(pq_y_coords)



    # Check if all bbox points are outside the PQ region

    return all(x < x_min_pq or x > x_max_pq or y < y_min_pq or y > y_max_pq for x, y in bbox_corners)





def draw_polygon(image, points, color=(0, 255, 0), thickness=2):

    """Draw a polygon defined by points on the image."""

    points_np = np.array(points, np.int32).reshape((-1, 1, 2))

    cv2.polylines(image, [points_np], isClosed=True, color=color, thickness=thickness)





def get_bbox_area(bbox_corners):

    """Calculate the area of the bounding box from its corners."""

    x1, y1 = bbox_corners[0]  # Top-left corner

    x2, y2 = bbox_corners[2]  # Bottom-right corner

    return abs(x2 - x1) * abs(y2 - y1)



def __process_single_image(input_image_path, output_image_path, text_prompt):

    # Get dynamic PQ points based on image size

    pq_points = get_dynamic_pq_points(input_image_path)



    # Load image

    image_source, image = load_image(input_image_path)

    

    print("Loading model...")

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)



    # Make predictions

    boxes, logits, phrases = predict(

        model=model,

        image=image,

        caption=text_prompt,

        box_threshold=BOX_THRESHOLD,

        text_threshold=TEXT_THRESHOLD

    )



    # Get image dimensions

    H, W, _ = image_source.shape



    # Transform normalized boxes from xywh to unnormalized xyxy

    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=boxes.device)



    # Convert bbox to numpy array

    boxes_np = boxes_xyxy.cpu().numpy() if torch.is_tensor(boxes_xyxy) else boxes_xyxy



    # Ensure single bounding box is treated as a 2D array

    if boxes_np.ndim == 1:

        boxes_np = boxes_np.reshape(1, -1)



    # Convert boxes to corner format

    bboxes_corners = [convert_xyxy_to_corners(bbox) for bbox in boxes_np]



    valid_boxes = []

    valid_phrases = []



    # Filter valid boxes based on PQ check

    for i, corners in enumerate(bboxes_corners):

        if not is_bbox_entirely_outside_pq(corners, pq_points):

            valid_boxes.append(corners)

            valid_phrases.append(phrases[i])

            print(f"Valid corners: {corners}, Phrase: {phrases[i]}")



    # If there are valid boxes, find the largest one

    if valid_boxes:

        largest_bbox = max(valid_boxes, key=lambda bbox: get_bbox_area(bbox))

        largest_phrase = valid_phrases[valid_boxes.index(largest_bbox)]



        # Draw PQ boundary

        annotated_frame = image_source.copy()



        # Draw only the largest bounding box

        draw_polygon(annotated_frame, largest_bbox, color=(0, 255, 0), thickness=2)



        # Annotate the phrase near the top-left corner of the bounding box

        top_left = largest_bbox[0]

        cv2.putText(annotated_frame, largest_phrase, (top_left[0], top_left[1] - 10), 

                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)



        # Save the annotated image

        cv2.imwrite(output_image_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        print(f"Annotated image with largest bbox saved at {output_image_path}")

        return True  # Indicates detections found

    else:

        print("No valid bounding boxes found inside the PQ region.")

        return False  # Indicates no detections



    # Clear GPU memory

    if torch.cuda.is_available():

        torch.cuda.empty_cache()



def process_single_image(input_image_path, output_image_path, text_prompt):

    # Get dynamic PQ points based on image size

    pq_points = get_dynamic_pq_points(input_image_path)



    # Load image

    image_source, image = load_image(input_image_path)

    

    print("Loading model...")

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)



    # Make predictions

    boxes, logits, phrases = predict(

        model=model,

        image=image,

        caption=text_prompt,

        box_threshold=BOX_THRESHOLD,

        text_threshold=TEXT_THRESHOLD

    )



    # Get image dimensions

    H, W, _ = image_source.shape



    # Transform normalized boxes from xywh to unnormalized xyxy

    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=boxes.device)



    # Convert bbox to numpy array

    boxes_np = boxes_xyxy.cpu().numpy() if torch.is_tensor(boxes_xyxy) else boxes_xyxy



    # Ensure single bounding box is treated as a 2D array

    if boxes_np.ndim == 1:

        boxes_np = boxes_np.reshape(1, -1)



    # Convert boxes to corner format

    bboxes_corners = [convert_xyxy_to_corners(bbox) for bbox in boxes_np]



    valid_boxes = []

    valid_phrases = []



    # Filter valid boxes based on PQ check

    for i, corners in enumerate(bboxes_corners):

        if not is_bbox_entirely_outside_pq(corners, pq_points):

            valid_boxes.append(corners)

            valid_phrases.append(phrases[i])

            print(f"Valid corners: {corners}, Phrase: {phrases[i]}")



    # If there are valid boxes, find the largest one

    if valid_boxes:

        largest_bbox = max(valid_boxes, key=lambda bbox: get_bbox_area(bbox))

        largest_phrase = valid_phrases[valid_boxes.index(largest_bbox)]


        # Draw PQ boundary

        annotated_frame = image_source.copy()



        # Draw only the largest bounding box

        draw_polygon(annotated_frame, largest_bbox, color=(0, 255, 0), thickness=2)



        # Annotate the phrase near the top-left corner of the bounding box

        top_left = largest_bbox[0]

        cv2.putText(annotated_frame, largest_phrase, (top_left[0], top_left[1] - 10), 

                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)



        # Save the annotated image

        cv2.imwrite(output_image_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        print(f"Annotated image with largest bbox saved at {output_image_path}")

        # Clear GPU memory
        if torch.cuda.is_available():

            torch.cuda.empty_cache()
        return largest_bbox
  
  # Indicates detections found

    else:

        print("No valid bounding boxes found inside the PQ region.")

        return []  # Indicates no detections






def process_images_in_folder(input_folder, output_folder, text_prompt, no_detection_list_file):

    # Create the output folder if it doesn't exist

    if not os.path.exists(output_folder):

        os.makedirs(output_folder)



    # Get list of image files in the input folder

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]



    total_images = len(image_files)

    images_with_detections = 0

    images_without_detections = 0

    no_detection_images = []



    # Process each image in the folder

    for image_file in image_files:

        input_image_path = os.path.join(input_folder, image_file)

        output_image_path = os.path.join(output_folder, f"annotated_{image_file}")



        print(f"Processing image: {input_image_path}")

        has_detections = process_single_image(input_image_path, output_image_path, text_prompt)



        if has_detections:

            images_with_detections += 1

        else:

            images_without_detections += 1

            no_detection_images.append(image_file)



    # Print summary

    print(f"Total images processed: {total_images}")

    print(f"Images with detections: {images_with_detections}")

    print(f"Images with no detections: {images_without_detections}")



    # Save names of images with no detections to a text file

    with open(no_detection_list_file, "w") as file:

        for image_name in no_detection_images:

            file.write(f"{image_name}\n")

    print(f"Images with no detections saved to {no_detection_list_file}")



if __name__ == "__main__":

    # Example usage

    INPUT_FOLDER = "/home/jupyter/EfficientNet/GroundingDINO/vertical_benchmark_images"

    OUTPUT_FOLDER = "/home/jupyter/EfficientNet/GroundingDINO/vertical_test_us"

    NO_DETECTION_LIST_FILE = "/home/jupyter/no_detection_images.txt"

    #Vertical stack

    #TEXT_PROMPT = "medical product with paper label on top, paper"

    TEXT_PROMPT = "medical product, paper"

 



    process_images_in_folder(INPUT_FOLDER, OUTPUT_FOLDER, TEXT_PROMPT, NO_DETECTION_LIST_FILE)