"""
main.py: Main orchestrator for Real-time Traffic Monitoring System
"""

import cv2
import numpy as np
import time
import sys
import argparse
import queue

# Core logic
from core.datatypes import FrameState, ModelConfig
from core.detector_tracker import DetectorTracker
from core.analyzer import TrafficAnalyzer
from core.counter import TrafficCounter

# Utilities
from utils.config import (
    DEFAULT_VIDEO_SOURCE, BASELINE_MODEL_PATH, OPTIMIZED_MODEL_PATH, DEVICE,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, LANE_POLYGONS, LANE_RESTRICTIONS,
    DRAW_ROI_POLYGONS, SAVE_OUTPUT_VIDEO, OUTPUT_VIDEO_PATH, MIN_FPS
)
from utils.visualizer import Visualizer
from utils.logger import ViolationLogger

class TrafficMonitoringSystem:
    """Main orchestrator for real-time traffic monitoring pipeline"""
    
    # [NÂNG CẤP]: Nhận thêm custom_polygons và custom_restrictions từ GUI
    def __init__(self, video_path=None, model_type='pt', use_optimized=False, custom_polygons=None, custom_restrictions=None):
        print("\n" + "="*80)
        print("🚗 REAL-TIME HIGHWAY TRAFFIC MONITORING SYSTEM")
        print("="*80)
        
        self.model_type = model_type
        self.use_optimized = use_optimized
        self.video_path = video_path if video_path is not None else DEFAULT_VIDEO_SOURCE
        
        # Nếu có ROI vẽ bằng tay từ GUI thì dùng, không thì dùng mặc định trong config.py
        self.current_polygons = custom_polygons if custom_polygons is not None else LANE_POLYGONS
        self.current_restrictions = custom_restrictions if custom_restrictions is not None else LANE_RESTRICTIONS
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        self.use_clahe = False
        
        try:
            self._init_components()
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            raise

    def _init_components(self):
        model_path = OPTIMIZED_MODEL_PATH if self.use_optimized else BASELINE_MODEL_PATH
        print(f"[1/5] Loading detection model ({self.model_type})...")
        config = ModelConfig(
            model_path=model_path, model_type=self.model_type,
            confidence_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD, device=DEVICE
        )
        self.detector = DetectorTracker(config)
        
        print("[2/5] Initializing analyzers & counters...")
        # Truyền ROI động vào Analyzer
        self.analyzer = TrafficAnalyzer(self.current_polygons, self.current_restrictions)
        self.counter = TrafficCounter()
        
        print("[3/5] Initializing I/O utilities...")
        self.visualizer = Visualizer(draw_roi=DRAW_ROI_POLYGONS)
        self.logger = ViolationLogger()
        
        print(f"[4/5] Opening video source: {self.video_path}")
        if str(self.video_path).isdigit():
            vid_src = int(self.video_path)  
        else:
            vid_src = str(self.video_path)  
            
        self.cap = cv2.VideoCapture(vid_src)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.analyzer.set_video_fps(self.video_fps)
        
        print("[5/5] Setup video writer...")
        self.video_writer = None
        if SAVE_OUTPUT_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO_PATH), fourcc, self.video_fps, (self.video_width, self.video_height)
            )

    def process_frame(self, frame: np.ndarray) -> FrameState:
        self.frame_count += 1
        frame_state = FrameState(frame_id=self.frame_count, original_frame=frame.copy(), timestamp=time.time())
        
        # [CẬP NHẬT]: Truyền cờ self.use_clahe vào DetectorTracker
        frame_state = self.detector.process(frame_state, use_preprocessing=self.use_clahe)       
        frame_state = self.analyzer.process(frame_state)       
        frame_state = self.counter.process(frame_state)        
        
        for viol in frame_state.violations:
            self.logger.log_violation(
                frame_state.frame_id, viol.track_id, viol.cls_name, 
                viol.violation_type, viol.conf, viol.bbox, viol.violation_lane
            )
            self.logger.save_violation_image(
                frame, viol.track_id, viol.bbox, viol.violation_type, frame_state.frame_id
            )
            
        # [CẬP NHẬT]: Đồng bộ số giây time_threshold sang Visualizer để hiển thị đúng trên nhãn Cam
        frame_state.processed_frame = self.visualizer.draw(
            frame_state, self.current_polygons, time_threshold=self.analyzer.time_threshold
        )
        frame_state.fps = self.fps_history[-1] if self.fps_history else 0
        return frame_state

    def run(self, display: bool = True):
        print("\n" + "="*80)
        print("▶ Monitoring Started (Controls: Q=Quit, P=Pause)")
        print("="*80)
        
        paused = False
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("\n✓ Video stream ended.")
                        break
                    
                    frame_start = time.time()
                    frame_state = self.process_frame(frame)
                    process_time = time.time() - frame_start
                    
                    current_fps = 1.0 / process_time if process_time > 0 else 0
                    self.fps_history.append(current_fps)
                    
                    if display:
                        cv2.imshow("Traffic Monitoring", frame_state.processed_frame)
                    if self.video_writer:
                        self.video_writer.write(frame_state.processed_frame)
                        
                    self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('p'): paused = not paused
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n" + "="*80)
        print("SHUTTING DOWN SYSTEM...")
        if hasattr(self, 'cap'): self.cap.release()
        if hasattr(self, 'video_writer') and self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'logger'): self.logger.stop()
        
        total_time = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        total_veh = self.counter.get_total() if hasattr(self, 'counter') else 0
        total_viol = self.analyzer.get_violation_count() if hasattr(self, 'analyzer') else 0
        
        print(f"\n📊 FINAL REPORT:")
        print(f"  Frames Processed : {self.frame_count}")
        print(f"  Total Time       : {total_time:.1f}s")
        print(f"  Average FPS      : {avg_fps:.1f} (Target Min: {MIN_FPS})")
        print(f"  Total Vehicles   : {total_veh}")
        print(f"  Total Violations : {total_viol}")
        print("="*80)

# ============================================================================
# LUỒNG AI KẾT NỐI VỚI GUI (NHẬN THÊM ROI TỪ GUI)
# ============================================================================
def ai_engine_worker(video_source, custom_polygons, custom_restrictions, frame_queue, stats_queue, command_queue, is_running_func, use_optimized=True):
    system = None
    try:
        m_type = OPTIMIZED_MODEL_TYPE if use_optimized else 'pt'
        system = TrafficMonitoringSystem(
            video_path=video_source, model_type=m_type, use_optimized=use_optimized,
            custom_polygons=custom_polygons, custom_restrictions=custom_restrictions
        )
    except Exception as e:
        stats_queue.put({'action': 'engine_stopped', 'error': str(e)})
        return

    paused = False
    total_frames = int(system.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video_file = total_frames > 0 and not str(video_source).isdigit()

    try:
        while system.cap.isOpened() and is_running_func():
            # Xử lý mọi lệnh điều khiển thời gian thực từ GUI
            while not command_queue.empty():
                try:
                    cmd = command_queue.get_nowait()
                    action = cmd.get("action")
                    val = cmd.get("value")
                    
                    if action == "set_confidence" and hasattr(system, 'detector'):
                        system.detector.model_config.confidence_threshold = float(val)
                    elif action == "set_violation_time" and hasattr(system, 'analyzer'):
                        system.analyzer.time_threshold = float(val)
                    elif action == "set_clahe":
                        system.use_clahe = bool(val)
                    elif action == "set_draw_roi" and hasattr(system, 'visualizer'):
                        system.visualizer.draw_roi = bool(val)
                    elif action == "set_draw_trails" and hasattr(system, 'visualizer'):
                        system.visualizer.draw_trails = bool(val)
                    elif action == "set_pause":
                        paused = bool(val)
                except queue.Empty:
                    break

            # Nếu đang tạm dừng (Pause) -> Ngủ nhẹ 50ms giữ nguyên trạng thái
            if paused:
                time.sleep(0.05)
                continue

            ret, frame = system.cap.read()
            if not ret: 
                break
            
            frame_start = time.time()
            frame_state = system.process_frame(frame)
            
            process_time = time.time() - frame_start
            current_fps = 1.0 / process_time if process_time > 0 else 0
            system.fps_history.append(current_fps)
            
            # Lưu frame vào file video đầu ra (nếu bật SAVE_OUTPUT_VIDEO)
            if system.video_writer and frame_state.processed_frame is not None:
                system.video_writer.write(frame_state.processed_frame)
            
            total_cars = system.counter.get_total() if hasattr(system, 'counter') else 0
            class_counts = system.counter.get_counts() if hasattr(system, 'counter') else {}
            violations_count = system.analyzer.get_violation_count() if hasattr(system, 'analyzer') else 0
            
            # Tính toán thanh tiến trình thời gian cho file Video
            progress_ratio = 0.0
            progress_text = "LIVE STREAM"
            if is_video_file and total_frames > 0:
                progress_ratio = min(1.0, system.frame_count / total_frames)
                cur_sec = int(system.frame_count / system.video_fps)
                tot_sec = int(total_frames / system.video_fps)
                progress_text = f"{cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d} ({int(progress_ratio * 100)}%)"

            live_viol_data = []
            for viol in frame_state.violations:
                live_viol_data.append({
                    "time": time.strftime("%H:%M:%S"),
                    "id": viol.track_id,
                    "class": viol.cls_name.upper(),
                    "type": viol.violation_type,
                    "lane": viol.violation_lane,
                    "conf": f"{viol.conf:.2f}",
                    "bbox": viol.bbox
                })

            if not frame_queue.full():
                frame_queue.put(frame_state.processed_frame)
                
            if not stats_queue.full():
                stats_queue.put({
                    'action': 'processing',
                    'fps': current_fps,
                    'total_vehicles': total_cars,
                    'active_vehicles': len(frame_state.vehicles),
                    'class_counts': class_counts,
                    'violations': violations_count,
                    'live_viol_data': live_viol_data,
                    'progress_ratio': progress_ratio,
                    'progress_text': progress_text
                })
                
    except Exception as e:
        stats_queue.put({'action': 'engine_stopped', 'error': f"Lỗi runtime: {str(e)}"})
    finally:
        if system:
            system.shutdown()
        stats_queue.put({'action': 'engine_stopped'})

if __name__ == "__main__":
    from ui.dashboard import TrafficDashboard
    
    from utils.config import (
        DEFAULT_VIDEO_SOURCE, BASELINE_MODEL_PATH, OPTIMIZED_MODEL_PATH, OPTIMIZED_MODEL_TYPE, DEVICE,
        CONFIDENCE_THRESHOLD, IOU_THRESHOLD, LANE_POLYGONS, LANE_RESTRICTIONS,
        DRAW_ROI_POLYGONS, SAVE_OUTPUT_VIDEO, OUTPUT_VIDEO_PATH, MIN_FPS
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help="Chạy chế độ Terminal (Không mở GUI)")
    parser.add_argument('--model', choices=['pt', 'engine'], default='pt')
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--video', type=str, default=None, help="Đường dẫn video tùy chọn")
    args = parser.parse_args()
    
    if args.cli:
        system = TrafficMonitoringSystem(video_path=args.video, model_type=args.model, use_optimized=args.optimized)
        system.run(display=True)
    else:
        app = TrafficDashboard(ai_engine_callback=ai_engine_worker)
        app.mainloop()