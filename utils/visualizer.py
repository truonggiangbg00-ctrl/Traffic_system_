"""
visualizer.py: Advanced CV visualization with Motion Trajectories, Alpha ROI Overlay, and HUD
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from utils.config import VIOLATION_TIME_THRESHOLD


class Visualizer:
    def __init__(self, draw_roi=True, draw_trails=True):
        self.draw_roi = draw_roi
        self.draw_trails = draw_trails
        # Lưu lịch sử 25 tọa độ gần nhất của mỗi track_id để vẽ đuôi quỹ đạo
        self.track_trails = defaultdict(lambda: deque(maxlen=25))

    def draw(self, frame_state, lane_polygons, time_threshold=VIOLATION_TIME_THRESHOLD):
        img = frame_state.original_frame.copy()
        current_ids = set()
        violating_lanes = set()

        # Kiểm tra trước xem làn nào đang có xe vi phạm hoặc cảnh báo
        if hasattr(frame_state, 'vehicles'):
            for v in frame_state.vehicles:
                current_ids.add(v.track_id)
                if getattr(v, 'is_violating', False):
                    violating_lanes.add(getattr(v, 'current_lane', None))

        # 1. VẼ LÀN ĐƯỜNG (ROI) BÁN TRONG SUỐT
        if self.draw_roi and lane_polygons:
            overlay = img.copy()
            for lane_name, polygon in lane_polygons.items():
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                
                # Nếu làn đang có xe vi phạm -> Phủ màu Đỏ nhạt, ngược lại phủ màu Xanh lam nhẹ
                fill_color = (0, 0, 180) if lane_name in violating_lanes else (180, 120, 0)
                border_color = (0, 0, 255) if lane_name in violating_lanes else (0, 165, 255)
                
                cv2.fillPoly(overlay, [pts], fill_color)
                cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
                
                # Nhãn tên làn đường
                lx, ly = pts[0][0]
                cv2.rectangle(img, (lx, ly - 24), (lx + 95, ly), border_color, -1)
                cv2.putText(img, lane_name, (lx + 6, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Hòa trộn lớp phủ màu với độ trong suốt 18%
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # 2. VẼ QUỸ ĐẠO DI CHUYỂN & BOUNDING BOX PHƯƠNG TIỆN
        if hasattr(frame_state, 'vehicles'):
            for v in frame_state.vehicles:
                x1, y1, x2, y2 = v.bbox
                detected_lane = getattr(v, 'current_lane', None)
                allowed = getattr(v, 'allowed_classes', [])

                # Xác định màu sắc theo 3 cấp độ: Hợp lệ (Xanh) -> Cảnh báo đếm giây (Cam) -> Phạt nguội (Đỏ)
                color = (0, 255, 0)
                if detected_lane:
                    label = f"#{v.track_id} {v.cls_name} | {detected_lane}"
                else:
                    label = f"#{v.track_id} {v.cls_name}"

                if getattr(v, 'is_warning', False):
                    color = (0, 165, 255)  # Cam
                    timer = getattr(v, 'violation_timer', 0.0)
                    label = f"CANH BAO #{v.track_id}: {v.cls_name} ({timer:.1f}s/{time_threshold:.1f}s)"
                elif getattr(v, 'is_violating', False):
                    color = (0, 0, 255)    # Đỏ
                    label = f"SAI LAN #{v.track_id}: {v.cls_name} | {detected_lane}"

                # Cập nhật và vẽ vệt quỹ đạo di chuyển (Track Tail)
                pt = getattr(v, 'check_point', v.bottom_center)
                self.track_trails[v.track_id].append(pt)

                if self.draw_trails and len(self.track_trails[v.track_id]) > 1:
                    trail_pts = list(self.track_trails[v.track_id])
                    for i in range(1, len(trail_pts)):
                        thickness = max(1, int(np.sqrt(float(i + 1)) * 1.2))
                        cv2.line(img, trail_pts[i - 1], trail_pts[i], color, thickness, cv2.LINE_AA)

                # Vẽ Bounding Box & Nhãn
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + w + 8, y1), color, -1)
                cv2.putText(img, label, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, pt, 4, (0, 255, 255), -1)

        # Dọn rác quỹ đạo của các xe đã rời khỏi khung hình
        if frame_state.frame_id % 150 == 0:
            expired = [tid for tid in self.track_trails if tid not in current_ids]
            for tid in expired:
                del self.track_trails[tid]

        # 3. VẼ THANH HUD THỐNG KÊ TRÊN GÓC TRÁI VIDEO (Cho file MP4 xuất ra)
        hud_overlay = img.copy()
        cv2.rectangle(hud_overlay, (15, 15), (340, 110), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.65, img, 0.35, 0, img)

        total_veh = sum(frame_state.counts.values()) if frame_state.counts else 0
        cv2.putText(img, f"FPS: {frame_state.fps:.1f}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"TONG XE QUA TRAM: {total_veh}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(img, f"XE DANG TRONG HINH: {len(current_ids)}", (30, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        frame_state.processed_frame = img
        return img