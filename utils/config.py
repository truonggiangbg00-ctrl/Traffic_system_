"""
config.py: Centralized configuration for Real-time Traffic Monitoring System
"""

import numpy as np
from pathlib import Path

import sys

# ============================================================================
# 1. PROJECT ROOT & PATHS (TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG ĐÓNG GÓI)
# ============================================================================
if getattr(sys, 'frozen', False):
    # Nếu chạy từ file .exe đã đóng gói, gốc dự án là thư mục chứa file .exe đó
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    # Nếu đang code/dev bình thường bằng python main.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEIGHTS_DIR = PROJECT_ROOT / "weights"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
OUTPUT_DIR = PROJECT_ROOT / "output"
ROI_CONFIG_FILE = OUTPUT_DIR / "roi_presets.json"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================================
# 2. MODEL PATHS & HARDWARE
# ============================================================================
USE_OPTIMIZED_MODEL = True  

BASELINE_MODEL_PATH = str(WEIGHTS_DIR / "yolo_basic.pt")
OPTIMIZED_MODEL_PATH = str(WEIGHTS_DIR / "yolo_optimized.pt")
OPTIMIZED_MODEL_TYPE = "pt"

DEVICE = "cuda"

# ============================================================================
# 3. VIDEO SOURCE & PROPERTIES
# ============================================================================
DEFAULT_VIDEO_SOURCE = ""  # Nguồn dự phòng (rỗng hoặc 0 cho Webcam)

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MIN_FPS = 15

# ============================================================================
# 4. CLASSES & DETECTION PARAMETERS
# ============================================================================
# Tên lớp phải đồng nhất với file data.yaml nhưng viết thường để so khớp logic
CLASS_NAMES = {
    0: "bike",
    1: "bus",
    2: "car",
    3: "motorbike",
    4: "truck"
}

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MIN_DETECTION_FRAMES = 3
VIOLATION_TIME_THRESHOLD = 3.0  # Phải đi trong làn cấm đủ n giây (VD: 3.0s) mới tính là vi phạm
VIOLATION_GRACE_TIME = 0.5      # Dung sai 0.5s chống nhiễu khi Bbox bị giật hoặc mất dấu tạm thời

# ============================================================================
# 5. LANE CONFIGURATION (Dữ liệu dự phòng - Chỉ dùng nếu không vẽ ROI trong GUI)
# ============================================================================
LANE_POLYGONS = {
    "LANE_1": np.array([[441, 450], [706, 452], [561, 1074], [2, 1064]], dtype=np.int32),
    "LANE_2": np.array([[706, 453], [921, 448], [1005, 1076], [558, 1072]], dtype=np.int32),
    "LANE_3": np.array([[921, 447], [1138, 441], [1485, 1077], [1006, 1072]], dtype=np.int32),
    "LANE_4": np.array([[1143, 438], [1374, 441], [1917, 927], [1918, 1068], [1497, 1074]], dtype=np.int32),
}

# [QUAN TRỌNG]: Mọi giá trị phải viết thường để khớp với logic .lower()
LANE_RESTRICTIONS = {
    "LANE_1": ["motorbike", "bike"],
    "LANE_2": ["motorbike", "car"],
    "LANE_3": ["car", "bus", "motorbike"],
    "LANE_4": ["car", "truck"],
}

# ============================================================================
# 6. LOGGING, VISUALIZATION & EVIDENCE
# ============================================================================
DRAW_ROI_POLYGONS = True
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = str(OUTPUT_DIR / "traffic_monitoring_output.mp4")

SAVE_VIOLATION_IMAGES = True
VIOLATION_IMAGE_FORMAT = "jpg"

LOG_FILE = str(OUTPUT_DIR / "traffic_violations.csv")
LOG_COLUMNS = [
    "frame_id", "timestamp", "vehicle_id", "class_name", 
    "violation_type", "confidence", "bbox_bottom_center", "violation_lane"
]

# ============================================================================
# 7. DATASET CONFIGURATION
# ============================================================================
DATASET_NAME = "Data_4"  
DATASET_DIR = PROJECT_ROOT / DATASET_NAME
DATA_YAML_PATH = DATASET_DIR / "data.yaml"