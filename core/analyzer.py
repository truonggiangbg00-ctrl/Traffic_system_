"""
analyzer.py: Traffic violation detection using spatial and temporal analysis
"""

import cv2
import numpy as np
from .datatypes import FrameState, TrackedVehicle, Violation
from utils.config import VIOLATION_TIME_THRESHOLD, VIOLATION_GRACE_TIME


class TrafficAnalyzer:
    def __init__(self, lane_polygons: dict, lane_restrictions: dict, 
                 video_fps: float = 30.0, time_threshold: float = VIOLATION_TIME_THRESHOLD):
        self.lane_polygons = {}
        if lane_polygons:
            for name, pts in lane_polygons.items():
                self.lane_polygons[name] = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        self.lane_restrictions = {}
        if lane_restrictions:
            for lane, allowed_classes in lane_restrictions.items():
                self.lane_restrictions[lane] = [cls.lower() for cls in allowed_classes]

        self.set_video_fps(video_fps)
        self.time_threshold = float(time_threshold)
        
        self.violation_history = set()
        self.violation_candidates = {}
        
        # [FIX]: Tự quản lý bộ đếm frame nội bộ để không phụ thuộc vào bên ngoài
        self.frame_count = 0

    def set_video_fps(self, fps: float):
        """Cập nhật FPS thực tế từ nguồn video (Giới hạn an toàn từ 10 - 120 FPS)"""
        if fps and 10.0 <= float(fps) <= 120.0:
            self.video_fps = float(fps)
        else:
            self.video_fps = 30.0

    def process(self, frame_state: FrameState) -> FrameState:
        # [FIX]: Tăng frame_count mỗi lần xử lý 1 khung hình
        self.frame_count += 1
        current_frame = self.frame_count
        grace_frames = max(1, int(VIOLATION_GRACE_TIME * self.video_fps))

        if hasattr(frame_state, 'vehicles') and frame_state.vehicles:
            for vehicle in frame_state.vehicles:
                vehicle_id = vehicle.track_id
                cls_name_lower = vehicle.cls_name.strip().lower()

                x1, y1, x2, y2 = vehicle.bbox
                check_y = int(y2 - (y2 - y1) * 0.2)
                check_point = (int((x1 + x2) / 2), check_y)
                vehicle.check_point = check_point

                current_lane = None
                for lane_name, polygon in self.lane_polygons.items():
                    if cv2.pointPolygonTest(polygon, check_point, False) >= 0:
                        current_lane = lane_name
                        break
                
                vehicle.current_lane = current_lane

                if current_lane and current_lane in self.lane_restrictions:
                    allowed_classes = self.lane_restrictions[current_lane]
                    vehicle.allowed_classes = allowed_classes

                    if cls_name_lower not in allowed_classes:
                        # Nếu xe đã bị chốt phạt trước đó -> Giữ nguyên khung Đỏ
                        if vehicle_id in self.violation_history:
                            vehicle.is_violating = True
                            vehicle.is_warning = False
                            vehicle.violation_lane = current_lane
                            vehicle.violation_type = "Đi Sai Làn"
                            vehicle.violation_timer = self.time_threshold
                            continue

                        cand = self.violation_candidates.get(vehicle_id)
                        if cand is None or cand["lane"] != current_lane:
                            cand = {
                                "lane": current_lane,
                                "frames": 1,
                                "last_viol_frame": current_frame
                            }
                            self.violation_candidates[vehicle_id] = cand
                        else:
                            frame_gap = max(1, current_frame - cand["last_viol_frame"])
                            if frame_gap <= grace_frames:
                                cand["frames"] += frame_gap
                            else:
                                cand["frames"] = 1
                            cand["last_viol_frame"] = current_frame

                        elapsed_sec = cand["frames"] / self.video_fps
                        vehicle.violation_timer = elapsed_sec

                        if elapsed_sec >= self.time_threshold:
                            vehicle.is_violating = True
                            vehicle.is_warning = False
                            vehicle.violation_lane = current_lane
                            vehicle.violation_type = "Đi Sai Làn"

                            if vehicle_id not in self.violation_history:
                                viol = Violation(
                                    track_id=vehicle_id,
                                    cls_name=vehicle.cls_name,
                                    violation_type="Đi Sai Làn",
                                    conf=vehicle.conf,
                                    bbox=vehicle.bbox,
                                    violation_lane=current_lane
                                )
                                frame_state.violations.append(viol)
                                self.violation_history.add(vehicle_id)
                                self.violation_candidates.pop(vehicle_id, None)
                        else:
                            vehicle.is_warning = True
                            vehicle.violation_lane = current_lane
                    else:
                        if vehicle_id in self.violation_candidates:
                            if current_frame - self.violation_candidates[vehicle_id]["last_viol_frame"] > grace_frames:
                                del self.violation_candidates[vehicle_id]

        if current_frame % 300 == 0:
            self._cleanup_candidates(current_frame, max_idle_frames=int(10 * self.video_fps))

        return frame_state

    def _cleanup_candidates(self, current_frame: int, max_idle_frames: int):
        expired_ids = [
            tid for tid, data in self.violation_candidates.items()
            if current_frame - data["last_viol_frame"] > max_idle_frames
        ]
        for tid in expired_ids:
            del self.violation_candidates[tid]

    def get_violation_count(self):
        return len(self.violation_history)