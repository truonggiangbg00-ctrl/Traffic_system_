"""
logger.py: Asynchronous I/O for violation evidence and logging
Uses threading to prevent blocking the main video processing loop
"""

import csv
import threading
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from queue import Queue, Empty
from .config import EVIDENCE_DIR, LOG_FILE, LOG_COLUMNS, SAVE_VIOLATION_IMAGES

class ViolationLogger:
    """
    Thread-safe logger for violation evidence and statistics
    Saves violation images asynchronously without blocking main loop
    """
    
    def __init__(self):
        self.evidence_queue = Queue()
        self.log_queue = Queue()
        self._running = True 
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        self._init_log_file()
        print(f"✓ Logger initialized (Evidence dir: {EVIDENCE_DIR})")
    
    def _init_log_file(self):
        evidence_path = Path(EVIDENCE_DIR)
        evidence_path.mkdir(parents=True, exist_ok=True)
        
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            with open(log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)
    
    def log_violation(self, frame_id: int, vehicle_id: int, class_name: str, 
                     violation_type: str, confidence: float, bbox: tuple, 
                     violation_lane: str = ""):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'frame_id': frame_id, 'timestamp': timestamp, 'vehicle_id': vehicle_id,
            'class_name': class_name, 'violation_type': violation_type,
            'confidence': f"{confidence:.4f}", 'bbox': str(bbox), 'violation_lane': violation_lane
        }
        self.log_queue.put(log_entry)
    
    def save_violation_image(self, frame: np.ndarray, vehicle_id: int, 
                            bbox: tuple, violation_type: str, frame_id: int):
        if not SAVE_VIOLATION_IMAGES:
            return
            
        evidence_item = {
            # Bắt buộc copy frame để tránh luồng I/O thay đổi ảnh của luồng AI chính
            'frame': frame.copy(), 
            'vehicle_id': vehicle_id,
            'bbox': bbox,
            'violation_type': violation_type,
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat()
        }
        self.evidence_queue.put(evidence_item)
    
    def _worker(self):
        while self._running or not self.log_queue.empty() or not self.evidence_queue.empty():
            while not self.log_queue.empty():
                try:
                    log_entry = self.log_queue.get_nowait()
                    self._write_log(log_entry)
                    self.log_queue.task_done()
                except Empty: break
                except Exception as e: print(f"❌ Error in log queue: {e}")
            
            while not self.evidence_queue.empty():
                try:
                    evidence_item = self.evidence_queue.get_nowait()
                    self._save_image(evidence_item)
                    self.evidence_queue.task_done()
                except Empty: break
                except Exception as e: print(f"❌ Error in evidence queue: {e}")
            
            threading.Event().wait(0.01)
            
    def _write_log(self, log_entry: dict):
        try:
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_entry.get('frame_id', ''), log_entry.get('timestamp', ''),
                    log_entry.get('vehicle_id', ''), log_entry.get('class_name', ''), 
                    log_entry.get('violation_type', ''), log_entry.get('confidence', ''),
                    log_entry.get('bbox', ''), log_entry.get('violation_lane', '')
                ])
        except Exception as e:
            print(f"❌ Error writing log: {e}")
    
    def _save_image(self, evidence_item: dict):
        """
        [ĐÃ CẬP NHẬT] - Giữ nguyên Full-frame, chỉ vẽ thêm Bounding Box đỏ 
        và nhãn vi phạm để làm bằng chứng thay vì cắt ảnh.
        """
        try:
            full_frame = evidence_item['frame']
            vehicle_id = evidence_item['vehicle_id']
            bbox = evidence_item['bbox']
            violation_type = evidence_item['violation_type']
            frame_id = evidence_item['frame_id']
            timestamp = evidence_item['timestamp']
            
            # Ép kiểu tọa độ về số nguyên
            x1, y1, x2, y2 = map(int, bbox)
            
            # 1. Vẽ Bounding Box màu đỏ (BGR: 0, 0, 255) có độ dày 3px
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # 2. Tạo nhãn text cảnh báo
            label = f"VIOLATION: ID#{vehicle_id} ({violation_type})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Vẽ nền đen trong suốt mờ mờ cho Text để dễ đọc
            text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            text_bg_p1 = (max(0, x1), max(0, y1 - text_size[1] - 10))
            text_bg_p2 = (x1 + text_size[0], max(0, y1))
            
            cv2.rectangle(full_frame, text_bg_p1, text_bg_p2, (0, 0, 0), -1)
            
            # 3. Ghi Text lên viền trên của Bounding Box
            cv2.putText(full_frame, label, (max(0, x1), max(0, y1 - 5)), 
                        font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Tạo tên file và đường dẫn lưu
            timestamp_str = datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"violation_vid{vehicle_id}_frm{frame_id}_{violation_type}_{timestamp_str}.jpg"
            filepath = Path(EVIDENCE_DIR) / filename
            
            # [FIX UNICODE WINDOWS]: Dùng imencode + tofile thay cho cv2.imwrite để không lỗi dấu tiếng Việt
            is_success, im_buf_arr = cv2.imencode(".jpg", full_frame)
            if is_success:
                im_buf_arr.tofile(str(filepath))
        
        except Exception as e:
            print(f"❌ Error saving violation image: {e}")
            
    def stop(self):
        self._running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0) 
            print("✓ Logger shutdown complete.")

    def get_statistics(self) -> dict:
        return {
            'log_queue_size': self.log_queue.qsize(),
            'evidence_queue_size': self.evidence_queue.qsize(),
            'log_file': str(LOG_FILE),
            'evidence_dir': str(EVIDENCE_DIR)
        }