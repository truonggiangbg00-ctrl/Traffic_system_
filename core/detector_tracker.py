"""
detector_tracker.py: YOLO object detection and BoT-SORT tracking
Handles both PyTorch (.pt) and optimized (TensorRT/ONNX) models
Applies adaptive preprocessing for low-light conditions
"""

import cv2
import numpy as np
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

from .datatypes import FrameState, TrackedVehicle, ModelConfig


class DetectorTracker:
    """
    YOLO-based vehicle detection with BoT-SORT tracking
    Supports both baseline PyTorch and optimized models
    Includes adaptive preprocessing for challenging lighting conditions
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize detector and tracker"""
        self.model_config = model_config
        self.model = None
        self.clahe = None
        
        # Thiết lập logger chuyên nghiệp thay vì dùng print()
        self.logger = logging.getLogger("DetectorTracker")
        
        self._load_model(model_config)
        self._init_preprocessing()
        
        self.logger.info(f"✓ DetectorTracker initialized ({model_config.model_type.upper()}) on {model_config.device.upper()}")
    
    def _load_model(self, config: ModelConfig):
        """Dynamically load model based on type"""
        model_path = Path(config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {config.model_path}")
        
        self.logger.info(f"Loading model from: {config.model_path}...")
        
        try:
            # YOLO tự động nhận diện định dạng (.pt, .onnx, .engine)
            self.model = YOLO(str(model_path), task='detect')
            
            # Ép cấu hình TensorRT (.engine) bắt buộc chạy trên phần cứng NVIDIA
            if config.model_type.lower() == 'engine' and config.device != 'cuda':
                self.logger.warning("⚠️ TensorRT (.engine) requires CUDA. Automatically overriding device to 'cuda'.")
                config.device = 'cuda'
                
            # Hàm .to(device) thường chỉ an toàn khi gọi trên mô hình PyTorch gốc (.pt)
            if model_path.suffix == '.pt':
                self.model.to(config.device)
                
            self.logger.info("✓ Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"✗ Error loading model {config.model_path}: {str(e)}")
            raise RuntimeError(f"Error loading model {config.model_path}: {str(e)}")
    
    def _init_preprocessing(self):
        """Initialize preprocessing for low-light conditions"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Giúp làm rõ biển số và hình dáng xe trong điều kiện trời chạng vạng, ngược sáng
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess_frame(self, frame: np.ndarray, use_clahe: bool = False) -> np.ndarray:
        """Preprocess frame for better detection in challenging conditions"""
        if not use_clahe or self.clahe is None:
            return frame
        
        # Chuyển sang không gian màu LAB. 
        # L (Lightness): Kênh độ sáng. 
        # A, B: Các kênh màu sắc.
        # Ta chỉ áp dụng cân bằng sáng lên kênh L để bảo toàn màu sơn thực tế của xe.
        lab = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        lab[0] = self.clahe.apply(lab[0])
        
        return cv2.cvtColor(cv2.merge(lab), cv2.COLOR_LAB2BGR)
    
    def process(self, frame_state: FrameState, 
                use_preprocessing: bool = False) -> FrameState:
        """Run detection and tracking on frame"""
        frame = frame_state.original_frame
        
        if use_preprocessing:
            frame = self.preprocess_frame(frame, use_clahe=True)
        
        # Reset danh sách xe ở mỗi frame mới
        frame_state.vehicles = []
        
        try:
            # Chuẩn hóa tham số device cho Ultralytics (tránh lỗi khi dùng CUDA)
            infer_device = 0 if self.model_config.device == "cuda" else self.model_config.device

            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
                conf=float(self.model_config.confidence_threshold),
                iou=float(self.model_config.iou_threshold),
                device=infer_device
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_names = result.names
                    
                    # [FIX AN TOÀN]: Nếu ở vài frame đầu Tracker chưa kịp cấp ID thì tạo ID tạm thời thay vì bỏ qua toàn bộ xe
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy()
                    else:
                        track_ids = np.arange(1, len(boxes) + 1)
                    
                    for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                        vehicle = TrackedVehicle(
                            track_id=int(track_id),
                            bbox=tuple(map(int, box)),
                            cls_id=int(cls_id),
                            cls_name=class_names[int(cls_id)].lower(),
                            conf=float(conf)
                        )
                        frame_state.vehicles.append(vehicle)
                        
        except Exception as e:
            print(f"❌ [DetectorTracker] Lỗi nhận diện tại frame {frame_state.frame_id}: {e}")
        
        return frame_state
    
    def get_model_info(self) -> dict:
        """Get information about loaded model"""
        if self.model:
            return {
                "model_type": self.model_config.model_type,
                "model_path": self.model_config.model_path,
                "device": self.model_config.device,
                "confidence_threshold": self.model_config.confidence_threshold,
                "iou_threshold": self.model_config.iou_threshold
            }
        return {}