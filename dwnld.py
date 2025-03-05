import pandas as pd
import json
from google.cloud import storage
import os
import cv2
import uuid
import requests
from samurai import process_video
from tqdm import tqdm

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './faceopen_key.json'  # Replace with your credentials file
storage_client = storage.Client()
BUCKET_NAME = 'auto-ai_resources_fo'
bucket = storage_client.bucket(BUCKET_NAME)

def upload_image_autoai(model_id, filepath, image_annotations, label, prediction, tag, csv=" "):
    """
    Upload image to RLEF model backlog with annotations.
    
    Args:
        filepath (str): path to image file being uploaded.
        csv (str): csv data for rlef.
        image_annotations (str): image annotations in rlef format.
        label (str): label of the rlef resource.
        tag (str): tag of the rlef resoirce.

    Returns:
        None
    """

    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    img_ext = os.path.basename(filepath).split(".")
    filename = str(uuid.uuid1()) + "." + img_ext[1]

    payload = {'status': "backlog",
               'csv': csv,
               'model': model_id, #item_of_interest
               'label': label,
               'tag': tag,
               'confidence_score': 100,
               'prediction': prediction,
               'imageAnnotations': image_annotations}

    files = [('resource', (filename, open(filepath, 'rb'), 'image/jpeg'))]
    headers = {}
    response = requests.request(
        'POST', url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        print(200)
        os.remove(filepath)
    else:
        print(f'Error while uploading to {csv} AutoAI')
        print(response.text)
    
    return

def process_video_and_coords(row, download_folder):
    """
    Process a single row to download video and extract coordinates
    
    Args:
        row (pd.Series): A single row from the DataFrame
        download_folder (str): Local folder to save downloaded videos
    
    Returns:
        dict: Contains video file path and polygon coordinates
    """
    file_path = row.get('csv', '')
    file_path = file_path.split('>')[1]
    image_annotation = row.get('imageAnnotations', '{}')
    result = {'video_path': None, 'polygon_coords': None}
    
    try:
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
            try:
                image_annotation = image_annotation.replace("\'", "\"")
            except:
                print("passed")
                pass
            
            objects = json.loads(str(image_annotation))
            
            print(objects)
            
            for object in objects:
                px = [vertex['x'] for vertex in object['vertices']]
                py = [vertex['y'] for vertex in object['vertices']]
                
                result['polygon_coords'] = {'x': px, 'y': py}
                
                
        
    except Exception as e:
        print(f"Error processing row {row.name}: {str(e)}")
    
    return result

def process_motion_session_with_sam2(video_path, x_coords, y_coords, model_path, output_folder):
    """Process recorded video with SAM2."""

    x1, y1 = min(x_coords), min(y_coords)
    w = max(x_coords) - x1
    h = max(y_coords) - y1

    output_path = os.path.join(output_folder, f"{uuid.uuid4()}.mp4")
    
    motion_type = process_video(
        video_path=video_path,
        coords=(x1, y1, w, h),
        model_path=model_path,
        save_video=False,
        output_path=output_path
    )
    return motion_type

def main():
    # Configuration
    download_folder = 'downloaded_videos'
    csv_file = './dataSetCollection_all_data_45_resources.csv'  # Uploaded file path
    
    # Clear and create download directory
    os.system(f"rm -rf {download_folder}")
    os.system(f"mkdir {download_folder}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} rows to process")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    
    # Process rows sequentially
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_video_and_coords(row, download_folder)
        # print(result)
        
        filename = result['video_path']
        
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return False
        
        img_path = f"downloaded_videos/{filename.split('/')[1].split('.')[0]}.png"
        cv2.imwrite(img_path, frame)
        cap.release()
        
        model_path='./sam2/checkpoints/sam2.1_hiera_large.pt'
        
        try:
            prediction = process_motion_session_with_sam2(
            video_path=filename,
            x_coords=result['polygon_coords']['x'],
            y_coords=result['polygon_coords']['y'],
            model_path=model_path,
            output_folder='./sam2_results'
        )
        
        except Exception as e:
            print(f"Error processing row {row.name}: {str(e)}")
            prediction = "NA"
            continue
        
        print(prediction)
        
        upload_image_autoai("675a659ae783d0f3a55b809c", img_path, None, row["label"], prediction, tag = model_path, csv="sam2")
          
main()