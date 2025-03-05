import pandas as pd
import json
import os
import cv2
import uuid
import requests
from tqdm import tqdm
from google.cloud import storage
from optical_flow import lucas_kanade_polygon_tracking

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './faceopen_key.json'  # Replace with your credentials file
storage_client = storage.Client()
BUCKET_NAME = 'auto-ai_resources_fo'
bucket = storage_client.bucket(BUCKET_NAME)

PROGRESS_FILE = 'progress.json'

def load_progress():
    """Load the progress from the checkpoint file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_rows):
    """Save the progress to the checkpoint file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_rows), f)

def upload_image_autoai(model_id, filepath, image_annotations, label, prediction, tag, csv=" "):
    """Upload image to RLEF model backlog with annotations."""
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    img_ext = os.path.basename(filepath).split(".")
    filename = str(uuid.uuid1()) + "." + img_ext[1]

    payload = {
        'status': "backlog",
        'csv': csv,
        'model': model_id,
        'label': label,
        'tag': tag,
        'confidence_score': 100,
        'prediction': prediction,
        'imageAnnotations': image_annotations
    }

    files = [('resource', (filename, open(filepath, 'rb'), 'image/jpeg'))]
    headers = {}
    response = requests.request('POST', url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        print(200)
        os.remove(filepath)
    else:
        print(f'Error while uploading to {csv} AutoAI')
        print(response.text)

def process_video_and_coords(row, download_folder):
    """Process a single row to download video and extract coordinates."""
    file_path = row.get('csv', '')
    file_path = file_path.split('>')[1]
    image_annotation = row.get('imageAnnotations', '{}')
    result = {'video_path': None, 'polygon_coords': None}

    # try:
        # Process video download
    if file_path:
        blob_path = '/'.join(file_path.split('/')[3:])
        blob = bucket.blob(blob_path)

        # Generate unique filename to avoid conflicts
        filename = f"{uuid.uuid4()}.mp4"
        filepath = f"{download_folder}/{filename}"

        # Create download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)

        # Download the file
        blob.download_to_filename(filepath)
        result['video_path'] = filepath
        print(f"Successfully downloaded: {filepath}")

    # Process image annotation
    if image_annotation:
        image_annotation = image_annotation.replace("'", "\"")
        objects = json.loads(image_annotation)

        for obj in objects:
            px = [vertex['x'] for vertex in obj['vertices']]
            py = [vertex['y'] for vertex in obj['vertices']]
            
            segment = [(vertex['x'], vertex['y']) for vertex in obj['vertices']]
            result['segment'] = segment

            result['polygon_coords'] = {'x': px, 'y': py}

    # except Exception as e:
    #     print(f"Error processing row {row.name}: {str(e)}")

    return result

# def process_motion_session_with_sam2(video_path, x_coords, y_coords, model_path, output_folder):
#     """Process recorded video with SAM2."""
#     x1, y1 = min(x_coords), min(y_coords)
#     w = max(x_coords) - x1
#     h = max(y_coords) - y1

#     output_path = os.path.join(output_folder, f"{uuid.uuid4()}.mp4")

#     motion_type = process_video(
#         video_path=video_path,
#         coords=(x1, y1, w, h),
#         model_path=model_path,
#         save_video=False,
#         output_path=output_path
#     )
#     return motion_type

def process_motion_session_with_optical_flow(video_path, x_coords, y_coords):
    """Process recorded video with optical fl;ow."""
    
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    x3, y3 = max(x_coords), min(y_coords)
    x4, y4 = min(x_coords), max(y_coords)
    
    segment = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
    
    print(segment, type(segment), type(segment[0]), type(segment[0][0]))   
    
    
        
    motion_type = lucas_kanade_polygon_tracking(video_path, segment)
    return motion_type


def main():
    # Configuration
    download_folder = 'downloaded_videos'
    csv_file = './dataSetCollection_all_data_45_resources.csv'  # Uploaded file path

    # Clear and create download directory
    os.makedirs(download_folder, exist_ok=True)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} rows to process")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return

    # Load progress
    # processed_rows = load_progress()

    # Process rows sequentially
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # if idx in processed_rows:
        #     continue  # Skip already processed rows

        result = process_video_and_coords(row, download_folder)

        if not result['video_path']:
            print(f"Skipping row {idx} due to missing video path.")
            continue

        filename = result['video_path']

        # try:
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"Error: Could not open video for row {idx}.")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame for row {idx}.")
            continue

        img_path = f"{download_folder}/{filename.split('/')[-1].split('.')[0]}.png"
        cv2.imwrite(img_path, frame)
        cap.release()

        # model_path = './sam2/checkpoints/sam2.1_hiera_large.pt'
        
        # Add to progress
        # processed_rows.add(idx)
        # save_progress(processed_rows)
        
        prediction = process_motion_session_with_optical_flow(video_path=filename, x_coords=result['polygon_coords']['x'], y_coords=result['polygon_coords']['y'])
        
        upload_image_autoai(
            model_id="675c2e1f58b4b98f0ea91aff",
            filepath=img_path,
            image_annotations=None,
            label=row["label"],
            prediction=prediction,
            tag="opt_flow",
            csv="sam2"
        )

        # except Exception as e:
        #     print(f"Error processing row {idx}: {str(e)}")

main()
