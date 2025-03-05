STREAMING_FRAME_SIZE = (640, 480)
import cv2
from collections import deque
import numpy as np
import os 
from concurrent.futures import ThreadPoolExecutor
from utils import maskHelper, polygonHelper
# from ocr_utils.textDecoder import TextDecoder
import os 
from threading import Lock 
from utils.camera_id import fetch_device_id




def create_runtime_folder(directory = 'runtimeLog'):
    #### Creating Runtime Log for saving images 
    if not os.path.exists(directory):
        os.mkdir(directory)

    sub_folder_list = ['angle_correction', 'crop', 'DIS', 'ner', 'ocr', 'onboarding', 'session', 'yolo_results', 'claude_inference']
    for folder in sub_folder_list:
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

HEADERS = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBOYW1lIjoiUHJvZHVjdCBPbmJvYXJkaW5nIiwidXNlckVtYWlsIjoiZGl2eWFuc2gua3VtYXJAdGVjaG9sdXRpb24uY29tIiwidXNlcklkIjoiNjQ5MThiNTE2NDkzYTk2NTI5ODM3MzgwIiwic2NvcGVPZlRva2VuIjp7InByb2plY3RJZCI6IjY1MGFmNzk1ZWJlNzBmMWY0Zjc5YmYxZSIsInNjb3BlcyI6eyJwcm9qZWN0Ijp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJyZXNvdXJjZSI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwibW9kZWwiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sImRhdGFTZXRDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJtb2RlbENvbGxlY3Rpb24iOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sInRlc3RDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJjb3BpbG90Ijp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJjb3BpbG90UmVzb3VyY2UiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sIm1vZGVsR3JvdXAiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX19fSwiaWF0IjoxNzExNjQxMzMyfQ.D88X0Aix3IlDTWduT0Yq_BSHTHvGPczuUm_iKC0ygKs'
}

BASE_DIR = os.getcwd()
log_dir = os.path.join(BASE_DIR, "runtimeLog")

threads_executor = ThreadPoolExecutor(max_workers = 30)

# textDecoder = TextDecoder(ocr_model_type="google-ocr")
polyHelper = polygonHelper.PolygonHelper()
maskClass = maskHelper.MaskImage()

# Define constants and configuration parameters



SESSION_SCANNED_DETAILS = {}
MAIN_FRAME_QUEUE = deque(maxlen=500)
LOG = False
FONT = cv2.FONT_HERSHEY_SIMPLEX
ORG = (50, 50)
FONT_SCALE = 1
COLOR = (255, 125, 100)
THICKNESS = 2
QUEUE_ADD_INTERVAL = 3
CALIBRATION = True
CALIBRATION_TIME = 10  # seconds
FPS = 30
CALIBRATION_FRAME_COUNT = CALIBRATION_TIME * FPS
NUM_POINTS = 30
SIM_THRESH = 80  # Similarity threshold for motion detection
RACK_REFERENCE = "61c03eef-840c-49df-ba20-0952553a4953"
MOTION_THRESH = 0.85
EMPTY_FILL = False
LOW_RES_SHAPE = (480, 640)
HIGH_RES_SHAPE = (2160, 3840)
AF_LOW_RES = (720, 1280)
PLANOGRAM_REF_NO = ""
CURRENT_REF_NO = ""
BLUR_THRESHOLD = 600 # Minimum threshold 
EMPTY_WORKSTATION = 180
RESCAN_BUFFER_TIME = 50 #FPS * time
RECORD_FLAG = False 
PIPELINE_LOG = False
USER_IMAGE_DIR = f'runtimeLog/user_images'

SMALLEST_AREA_VALUE = 10000

MIN_FOCUS_VALUE = 60
MAX_FOCUS_VALUE = 100
YOLO_THRESHOLD = 0.3
RUNTIME_IMAGE = None 

GCP_CREDS_PATH = "service_account_keys/onm_service_key.json"
GCS_BUCKET_NAME = "onm-backend-asmlab"
GCS_GEN_AI_KEY = "AIzaSyCx9hv0h6tl1pSbBVTHk-DGDh0kkjIuGO0"

REF_IMG_PATH = "runtimeLog/reference_image_2.jpg"
RUNTIME_IMG_PATH = "runtimeLog/action_image.jpg"
HIGH_RES_PATH = "runtimeLog/high_res_image.jpg"
OCR_CROP_IMG_PATH = "runtimeLog/ocr"
REF_MASK_PATH = "runtimeLog/ref_mask.jpg"
TEXT_LOCALIZATION_YOLO_PATH = "weights/text_localization_3.pt"
DIS_MODEL_PATH = "workstation_model.pth"

CURRENT_FRAME_PATH = "runtimeLog/current_frame.jpg"
YOLO_ANNOT_IMG_PATH = "runtimeLog/annotated_image.jpg"
VIDEOS_DIR_PATH = "runtimeLog/videos"
SEGMENT_PATH = "runtimeLog/segment.jpg"
TIME_PROFILE_PATH = "runtimeLog/time_profiling.csv"

OCR_PROCESS_LOCK = Lock() 
RESCAN_BUFFER = 60 #FPS * seconds 

#RLEF FLAGS
RLEF_URL = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"
RLEF_REQUEST_ID = "1234234567"
SEND_RLEF = True
# RLEF MODEL IDS 
RLEF_ANGLE_YOLO_MODEL_ID = "66e0480214080fc8ee185e47"
RLEF_VLLM_MODEL_ID = "671a2253b90bc2781276e301"
RLEF_DIS_MODEL_ID = "66e0486c14080fc8ee187f80"
RLEF_TEXT_DETECTION_MODEL_ID = "66e0483c14080fc8ee187117"
RLEF_IMAGE_MODEL_ID = "66e0420d14080fc8ee167873" # 66e0420d14080fc8ee167873
RLEF_ANGLE_GOOGLE_MODEL_ID = "671a21e9b90bc2781276ba6b"
RLEF_NER_MODEL_ID = "671a223db90bc2781276d95b"
RLEF_OCR_MODEL_ID = "66e0489214080fc8ee188b2c"
TEXT_DETECTION_TRAIN_MODEL_ID = "66965ed548169ca1bf32ef13"

## RLEF COPILOT IDS 
TEXT_DETECTION_COPILOT_ID = "66ec1026574ab0089accdc99"

# Dummy database entry for tracking item information

FOCUS_CALIBRATION_RANGE = [40,90]
EXPOSURE_CALIBRATION_RANGE = [200,300]

INDIA_SETUP = False

##ROI Decision
FOCUS_VAL = 64
EXPOSURE = 178
CAMERA_ID = fetch_device_id(device_type= '48MP')
USER_CAMERA_ID = fetch_device_id(device_type= '16MP')
print(f'CAMERA ID IS {CAMERA_ID}')
print(f'User image capture id is {USER_CAMERA_ID}')

if CAMERA_ID is None:
    raise BufferError('NO CAMERA CONNECTED')

# NER SORTING
NER_SORTING = False

TOPIC_ID = "product-onboarding-workstation"
PROJECT_ID = "proj-qsight-asmlab"
SUBSCRIPTION_ID = "product-onboarding-workstation-sub"

ALGO = "None"

ROI_POLYGON = [(66,11), (18,276), (149,297), (136, 423), (638,453), (628,7)]
HEIGHT, WIDTH = 480, 640 
mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

polygon = np.array([ROI_POLYGON], dtype=np.int32)

# Fill the polygon with white (255) on the mask
ROI_MASK = cv2.fillPoly(mask, polygon, 255)
# cv2.imwrite('ROI_MASK.png', ROI_MASK)

## NETWORK THRESHOLDS
UPLOAD_SPEED_THRESHOLD_MBPS=30
DOWNLOAD_SPEED_THRESHOLD_MBPS=30

US_TAG = "Live-data-US"
def initialize_system():
    """
    Initializes the motion detection and tracking system with predefined configurations.
    """
    print("Initializing system with the following configurations:")
    print(f"FPS: {FPS}")
    print(f"Calibration Time: {CALIBRATION_TIME} seconds")
    print(f"Similarity Threshold: {SIM_THRESH}")
    print(f"Motion Threshold: {MOTION_THRESH}")

if __name__ == "__main__":
    initialize_system()
    create_runtime_folder()
    # Additional setup and motion detection logic can be implemented here