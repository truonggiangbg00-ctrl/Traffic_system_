"""
roi_drawer.py: Interactive ROI drawing tool with Dynamic Lane Restrictions and UI Overlay
"""

import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk

class ROIDrawer:
    # [CẬP NHẬT HÀM __init__]: Nhận thêm existing_polygons và existing_restrictions
    def __init__(self, video_source, existing_polygons=None, existing_restrictions=None):
        if str(video_source).isdigit():
            self.video_source = int(video_source)
            print(f"🎬 Loading Webcam ID: {self.video_source}")
        else:
            self.video_source = str(video_source)
            if not Path(self.video_source).exists():
                raise FileNotFoundError(f"Video not found: {self.video_source}")
            print(f"🎬 Loading video: {self.video_source}")
            
        cap = cv2.VideoCapture(self.video_source)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Cannot read first frame. Camera might be locked.")
            
        self.base_frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]
        
        self.current_polygon = []
        self.all_polygons = {}
        self.all_restrictions = {}
        
        # [MỚI]: Nạp sẵn các làn đường đã lưu trước đó (nếu có)
        if existing_polygons and existing_restrictions:
            for k, v in existing_polygons.items():
                self.all_polygons[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)
            self.all_restrictions = dict(existing_restrictions)
            self.lane_counter = len(self.all_polygons) + 1
        else:
            self.lane_counter = 1
            
        self.mouse_pos = (0, 0)
        self.available_classes = ["Bike", "Bus", "Car", "Motorbike", "Truck"]
        
        self.window_name = "ROI Drawer - Draw Lane Boundaries"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        display_w = min(self.frame_w, 1280)
        display_h = int(self.frame_h * (display_w / self.frame_w))
        cv2.resizeWindow(self.window_name, display_w, display_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _get_lane_restrictions(self, lane_name):
        import tkinter as tk
        
        # Dùng Toplevel thay vì Tk() để gắn ké vào GUI chính
        try:
            root = tk.Toplevel() 
        except tk.TclError:
            root = tk.Tk()

        root.title(f"Cấu hình {lane_name}")
        root.geometry("300x250")
        root.attributes('-topmost', True)

        tk.Label(root, text=f"Chọn xe ĐƯỢC PHÉP đi vào {lane_name}:", font=("Arial", 10, "bold")).pack(pady=10)

        vars_dict = {}
        for cls in self.available_classes:
            var = tk.BooleanVar(root, value=True) 
            cb = tk.Checkbutton(root, text=cls.upper(), variable=var, font=("Arial", 10))
            cb.pack(anchor='w', padx=70)
            vars_dict[cls] = var

        selected_classes = []
        def on_save():
            for cls, var in vars_dict.items():
                if var.get():
                    selected_classes.append(cls)
            root.destroy()

        tk.Button(root, text="💾 Lưu Cấu Hình", command=on_save, bg="#28a745", fg="white", font=("Arial", 10, "bold")).pack(pady=15)
        
        root.wait_window()
        return selected_classes if selected_classes else self.available_classes

    def _mouse_callback(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_polygon:
                self.current_polygon.pop()

    def _draw_frame(self) -> np.ndarray:
        display = self.base_frame.copy()
        
        # ==========================================
        # [MỚI]: VẼ BẢNG HƯỚNG DẪN BÁN TRONG SUỐT
        # ==========================================
        overlay = display.copy()
        
        # Tọa độ khung nền đen (x1, y1, x2, y2)
        # Trong hàm _draw_frame(self):
        cv2.rectangle(overlay, (15, 15), (450, 230), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        guide_text = [
            "--- HUONG DAN VE LAN DUONG (ROI) ---",
            "[*] CHUOT TRAI: Cham de tao diem",
            "[*] CHUOT PHAI: Xoa diem vua tao",
            "[*] PHIM 'N'  : Luu lan & Chon xe",
            "[*] PHIM 'C'  : Xoa net ve nhap",
            "[*] PHIM 'R'  : Xoa TAT CA lan da luu",
            "[*] PHIM 'Q'  : Hoan tat & Thoat"
        ]
        
        y_offset = 45
        for line in guide_text:
            # Dòng tiêu đề màu Vàng, các dòng khác màu Trắng
            color = (0, 255, 255) if "HUONG DAN" in line else (255, 255, 255)
            font_scale = 0.6 if "HUONG DAN" in line else 0.55
            thickness = 2 if "HUONG DAN" in line else 1
            
            cv2.putText(display, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += 28
        # ==========================================

        # Vẽ các làn đường đã lưu
        for lane_name, polygon in self.all_polygons.items():
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            
            text_pos = tuple(polygon[0])
            cv2.rectangle(display, (text_pos[0]-5, text_pos[1]-25), (text_pos[0]+100, text_pos[1]+5), (0,0,0), -1)
            cv2.putText(display, lane_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Vẽ nét đứt nháp và con trỏ chuột
        if self.current_polygon:
            for i, pt in enumerate(self.current_polygon):
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display, tuple(self.current_polygon[i-1]), tuple(pt), (0, 0, 255), 2, cv2.LINE_AA)
            
            if len(self.current_polygon) > 2:
                cv2.line(display, tuple(self.current_polygon[-1]), tuple(self.current_polygon[0]), (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(display, tuple(self.current_polygon[-1]), self.mouse_pos, (255, 255, 255), 1, cv2.LINE_AA)
        
        return display

    def run(self) -> tuple:
        print("="*70 + "\n🎯 INTERACTIVE ROI DRAWER\n" + "="*70)
        try:
            while True:
                display_frame = self._draw_frame()
                cv2.imshow(self.window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('n'), ord('N')):
                    if len(self.current_polygon) >= 3:
                        lane_name = f"LANE_{self.lane_counter}"
                        
                        restrictions = self._get_lane_restrictions(lane_name)
                        
                        self.all_polygons[lane_name] = self.current_polygon.copy()
                        self.all_restrictions[lane_name] = restrictions
                        
                        print(f"✅ Đã lưu {lane_name}. Phép đi: {restrictions}")
                        self.current_polygon.clear()
                        self.lane_counter += 1
                    else:
                        print("❌ Cần ít nhất 3 điểm!")
                # Trong vòng lặp while True của hàm run(self):
                elif key in (ord('c'), ord('C')):
                    self.current_polygon.clear()
                elif key in (ord('r'), ord('R')): # [MỚI]: Xóa toàn bộ làn cũ để vẽ lại từ LANE_1
                    self.all_polygons.clear()
                    self.all_restrictions.clear()
                    self.current_polygon.clear()
                    self.lane_counter = 1
                    print("🗑️ Đã xóa toàn bộ các làn đường để vẽ lại từ đầu.")
                elif key in (ord('q'), ord('Q'), 27):
                    break
        finally:
            cv2.destroyAllWindows()
            
        return self.all_polygons, self.all_restrictions