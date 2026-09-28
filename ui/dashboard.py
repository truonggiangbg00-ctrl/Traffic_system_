"""
dashboard.py: Giao diện người dùng phong cách Modern SaaS / CRM Light Dashboard
Tích hợp Sidebar Navigation, KPI Metric Cards, Dual-View Phạt Nguội, Biểu đồ & Lưu ROI JSON.
"""

import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
import threading
import queue
import time
import os
import sys
import json
import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections

from utils.config import OUTPUT_DIR, EVIDENCE_DIR

# Tự động tương thích hằng số với utils/config.py
try:
    from utils.config import ROI_CONFIG_FILE
except ImportError:
    ROI_CONFIG_FILE = Path(OUTPUT_DIR) / "roi_presets.json"

try:
    from utils.config import VIOLATION_TIME_THRESHOLD
except ImportError:
    VIOLATION_TIME_THRESHOLD = 3.0

# ============================================================================
# BẢNG MÀU & FONT CHỮ CHUẨN MODERN SAAS CRM (LIGHT MODE)
# ============================================================================
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

SIDEBAR_BG = "#2563EB"          # Xanh Royal Blue đặc trưng của Sidebar
SIDEBAR_ACTIVE = "#3B82F6"      # Màu nút Sidebar khi được chọn
SIDEBAR_HOVER = "#1D4ED8"       # Màu nút Sidebar khi di chuột
MAIN_BG = "#F4F7FE"             # Nền xám xanh nhạt hiện đại của vùng làm việc
CARD_BG = "#FFFFFF"             # Nền trắng tinh của các thẻ Card
CARD_BORDER = "#E2E8F0"         # Viền mảnh quanh các thẻ Card
TEXT_PRIMARY = "#0F172A"        # Chữ chính màu đen than đậm
TEXT_MUTED = "#64748B"          # Chữ phụ màu xám trung tính

ACCENT_BLUE = "#2563EB"
ACCENT_GREEN = "#10B981"
ACCENT_ORANGE = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#8B5CF6"

FONT_MAIN = ("Segoe UI", 13)
FONT_BOLD = ("Segoe UI", 13, "bold")
FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_CARD_HEAD = ("Segoe UI", 14, "bold")
FONT_KPI_VAL = ("Segoe UI", 24, "bold")
FONT_KPI_SUB = ("Segoe UI", 11)


class TrafficDashboard(ctk.CTk):
    def __init__(self, ai_engine_callback):
        super().__init__()
        
        self.title("🚦 TrafficAI CRM - Hệ thống Giám sát Giao thông & Phạt Nguội Thông minh")
        self.geometry("1520x900")
        self.minsize(1320, 780)
        self.configure(fg_color=MAIN_BG)
        
        self.start_ai_engine = ai_engine_callback
        
        # Hàng đợi giao tiếp đa luồng
        self.frame_queue = queue.Queue(maxsize=5)
        self.stats_queue = queue.Queue(maxsize=5)
        self.command_queue = queue.Queue()
        
        # Trạng thái vận hành
        self.is_running = False
        self.is_paused = False
        self.video_source = None
        
        # Bộ nhớ vi phạm & ROI
        self.recorded_violation_ids = set()
        self.violation_records = {}
        self.selected_viol_id = None
        
        self.custom_polygons = None
        self.custom_restrictions = None
        self.current_bgr_frame = None
        
        # Dữ liệu lịch sử biểu đồ
        self.history_time = collections.deque(maxlen=60)
        self.history_vehicles = collections.deque(maxlen=60)
        self.start_app_time = time.time()
        
        self._setup_layout()
        self.switch_page("dashboard")

    # ========================================================================
    # 1. KIẾN TRÚC BỐ CỤC TỔNG THỂ (SIDEBAR + MAIN WORKSPACE)
    # ========================================================================
    def _setup_layout(self):
        self.grid_columnconfigure(0, weight=0)  # Cột 0: Sidebar cố định 230px
        self.grid_columnconfigure(1, weight=1)  # Cột 1: Không gian chính co giãn
        self.grid_rowconfigure(0, weight=1)

        # --- 1.1. SIDEBAR BÊN TRÁI ---
        self.sidebar = ctk.CTkFrame(self, width=235, corner_radius=0, fg_color=SIDEBAR_BG)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo thương hiệu
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=20, pady=(25, 30))
        
        logo_circle = ctk.CTkLabel(
            brand_frame, text="🚦", font=("Segoe UI", 22), 
            width=40, height=40, fg_color="#FFFFFF", corner_radius=20
        )
        logo_circle.pack(side="left")
        ctk.CTkLabel(
            brand_frame, text="TrafficCRM", font=("Segoe UI", 20, "bold"), text_color="#FFFFFF"
        ).pack(side="left", padx=12)

        # Các nút điều hướng trang (Navigation Menu)
        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "📊   Dashboard Giám sát"),
            ("violations", "📸   Hồ sơ Phạt Nguội"),
            ("analytics", "📈   Báo cáo & Biểu đồ"),
            ("settings", "⚙️   Cấu hình Hệ thống"),
        ]
        for key, text in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=text, anchor="w", height=44, corner_radius=8,
                font=("Segoe UI", 14, "bold"), fg_color="transparent", text_color="#FFFFFF",
                hover_color=SIDEBAR_HOVER, command=lambda k=key: self.switch_page(k)
            )
            btn.pack(fill="x", padx=14, pady=4)
            self.nav_buttons[key] = btn

        # Nút tiện ích dưới đáy Sidebar
        bottom_sidebar = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_sidebar.pack(side="bottom", fill="x", padx=14, pady=20)

        ctk.CTkButton(
            bottom_sidebar, text="📁   Mở Thư mục Xuất", anchor="w", height=40, corner_radius=8,
            font=FONT_MAIN, fg_color="#1E40AF", hover_color="#1E3A8A", text_color="#FFFFFF",
            command=self.open_output_folder
        ).pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            bottom_sidebar, text="🛑   Tắt Hệ Thống", anchor="w", height=40, corner_radius=8,
            font=FONT_BOLD, fg_color="#DC2626", hover_color="#B91C1C", text_color="#FFFFFF",
            command=self.hard_exit
        ).pack(fill="x")

        # --- 1.2. KHÔNG GIAN LÀM VIỆC CHÍNH (BÊN PHẢI) ---
        self.main_area = ctk.CTkFrame(self, fg_color=MAIN_BG, corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_rowconfigure(2, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        self._build_top_navbar()
        self._build_kpi_row()

        # Khung chứa các trang (Pages Container)
        self.pages_container = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.pages_container.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 18))
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for page_name in ["dashboard", "violations", "analytics", "settings"]:
            frame = ctk.CTkFrame(self.pages_container, fg_color="transparent")
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[page_name] = frame

        self._build_page_dashboard(self.pages["dashboard"])
        self._build_page_violations(self.pages["violations"])
        self._build_page_analytics(self.pages["analytics"])
        self._build_page_settings(self.pages["settings"])

    # ========================================================================
    # 2. THANH TOP NAVBAR & LƯỚI NÚT ĐIỀU KHIỂN (2 HÀNG x 3 CỘT ĐỒNG BỘ)
    # ========================================================================
    def _build_top_navbar(self):
        # Tăng chiều cao Header lên 98px để chứa vừa đẹp 2 hàng nút
        top_frame = ctk.CTkFrame(
            self.main_area, fg_color=CARD_BG, height=98, 
            corner_radius=0, border_width=1, border_color=CARD_BORDER
        )
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.grid_propagate(False)

        # --- BÊN TRÁI: TIÊU ĐỀ TRANG & TRẠNG THÁI NGUỒN ---
        left_box = ctk.CTkFrame(top_frame, fg_color="transparent")
        left_box.pack(side="left", fill="y", padx=24, pady=12)

        title_row = ctk.CTkFrame(left_box, fg_color="transparent")
        title_row.pack(anchor="w")

        self.lbl_page_title = ctk.CTkLabel(title_row, text="Dashboard Giám Sát", font=FONT_TITLE, text_color=TEXT_PRIMARY)
        self.lbl_page_title.pack(side="left", padx=(0, 14))

        self.status_badge = ctk.CTkLabel(
            title_row, text="🟢 Hệ thống sẵn sàng", font=("Segoe UI", 12, "bold"),
            fg_color="#DCFCE7", text_color="#15803D", corner_radius=6, padx=12, pady=4
        )
        self.status_badge.pack(side="left")

        info_row = ctk.CTkFrame(left_box, fg_color="transparent")
        info_row.pack(anchor="w", pady=(8, 0))

        self.lbl_source = ctk.CTkLabel(info_row, text="Nguồn: Chưa chọn", font=("Segoe UI", 12), text_color=TEXT_MUTED)
        self.lbl_source.pack(side="left", padx=(0, 10))

        self.lbl_roi_status = ctk.CTkLabel(info_row, text="• ROI: Mặc định", font=("Segoe UI", 12, "italic"), text_color=TEXT_MUTED)
        self.lbl_roi_status.pack(side="left", padx=(0, 8))

        # Nút nhỏ xóa nhanh cấu hình ROI đặt gọn cạnh trạng thái ROI
        self.btn_clear_roi = ctk.CTkButton(
            info_row, text="↺ Reset ROI", command=self.reset_roi_preset,
            width=76, height=24, corner_radius=6,
            fg_color="#FEF2F2", hover_color="#FEE2E2", text_color=ACCENT_RED,
            border_width=1, border_color="#FECACA", font=("Segoe UI", 11, "bold")
        )
        self.btn_clear_roi.pack(side="left")

        # --- BÊN PHẢI: LƯỚI NÚT ĐIỀU KHIỂN (MỖI DÒNG 3 NÚT ĐỒNG BỘ 100%) ---
        right_box = ctk.CTkFrame(top_frame, fg_color="transparent")
        right_box.pack(side="right", padx=24, pady=10)

        # Ép 3 cột có độ rộng bằng nhau tuyệt đối bằng tham số uniform
        for col in range(3):
            right_box.grid_columnconfigure(col, weight=1, uniform="toolbar_btn")

        # Chuẩn hóa Form chung cho cả 6 nút
        BTN_STYLE = {
            "width": 138,
            "height": 34,
            "corner_radius": 8,
            "border_width": 1,
            "font": ("Segoe UI", 12, "bold"),
        }

        # --- DÒNG 1: 3 NÚT THIẾT LẬP ĐẦU VÀO ---
        self.btn_select_file = ctk.CTkButton(
            right_box, text="📂 Chọn Video", command=self.select_file,
            fg_color="#F8FAFC", hover_color="#E2E8F0", text_color=TEXT_PRIMARY,
            border_color="#CBD5E1", **BTN_STYLE
        )
        self.btn_select_file.grid(row=0, column=0, padx=4, pady=3, sticky="ew")

        self.btn_select_cam = ctk.CTkButton(
            right_box, text="📷 Mở Webcam", command=self.select_webcam,
            fg_color="#F8FAFC", hover_color="#E2E8F0", text_color=TEXT_PRIMARY,
            border_color="#CBD5E1", **BTN_STYLE
        )
        self.btn_select_cam.grid(row=0, column=1, padx=4, pady=3, sticky="ew")

        self.btn_draw_roi = ctk.CTkButton(
            right_box, text="📐 Vẽ Làn ROI", command=self.launch_roi_drawer,
            fg_color="#F8FAFC", hover_color="#E2E8F0", text_color=TEXT_PRIMARY,
            border_color="#CBD5E1", **BTN_STYLE
        )
        self.btn_draw_roi.grid(row=0, column=2, padx=4, pady=3, sticky="ew")

        # --- DÒNG 2: 3 NÚT ĐIỀU KHIỂN LUỒNG AI ---
        self.btn_start = ctk.CTkButton(
            right_box, text="▶ Khởi Động AI", command=self.start_processing,
            fg_color=ACCENT_BLUE, hover_color=SIDEBAR_HOVER, text_color="#FFFFFF",
            border_color=ACCENT_BLUE, **BTN_STYLE
        )
        self.btn_start.grid(row=1, column=0, padx=4, pady=3, sticky="ew")

        self.btn_pause = ctk.CTkButton(
            right_box, text="⏸ Tạm Dừng", command=self.toggle_pause_processing,
            fg_color="#F8FAFC", hover_color="#E2E8F0", text_color=TEXT_PRIMARY,
            border_color="#CBD5E1", state="disabled", **BTN_STYLE
        )
        self.btn_pause.grid(row=1, column=1, padx=4, pady=3, sticky="ew")

        self.btn_stop = ctk.CTkButton(
            right_box, text="⏹ Kết Thúc", command=self.stop_processing,
            fg_color="#F8FAFC", hover_color="#FEE2E2", text_color=ACCENT_RED,
            border_color="#CBD5E1", state="disabled", **BTN_STYLE
        )
        self.btn_stop.grid(row=1, column=2, padx=4, pady=3, sticky="ew")

    # ========================================================================
    # 3. HÀNG 6 THẺ KPI (METRIC CARDS) GIỐNG MẪU CRM
    # ========================================================================
    def _build_kpi_row(self):
        kpi_row = ctk.CTkFrame(self.main_area, fg_color="transparent")
        kpi_row.grid(row=1, column=0, sticky="ew", padx=24, pady=(16, 14))
        for i in range(6):
            kpi_row.grid_columnconfigure(i, weight=1, uniform="kpi")

        self.lbl_fps, self.bar_fps = self._create_kpi_card(
            kpi_row, 0, "Tốc độ Xử lý (FPS)", "0.0", "Mục tiêu >= 15 FPS", ACCENT_BLUE
        )
        self.lbl_total_vehicles, self.bar_total = self._create_kpi_card(
            kpi_row, 1, "Tổng Lưu Lượng", "0", "Phương tiện qua trạm", ACCENT_GREEN
        )
        self.lbl_violations, self.bar_viol = self._create_kpi_card(
            kpi_row, 2, "Vi Phạm Sai Làn", "0", "Đã ghi nhận phạt nguội", ACCENT_RED
        )
        self.lbl_four_wheel, self.bar_four_wheel = self._create_kpi_card(
            kpi_row, 3, "Ô tô / Tải / Bus", "0", "Nhóm xe cơ giới lớn", ACCENT_ORANGE
        )
        self.lbl_two_wheel, self.bar_two_wheel = self._create_kpi_card(
            kpi_row, 4, "Xe máy / Xe đạp", "0", "Nhóm xe 2 bánh", ACCENT_PURPLE
        )
        self.lbl_kpi_threshold, self.bar_threshold = self._create_kpi_card(
            kpi_row, 5, "Ngưỡng Bắt Lỗi", f"{VIOLATION_TIME_THRESHOLD:.1f}s", "Thời gian đi trong làn cấm", "#0EA5E9"
        )
        self.bar_threshold.set(min(1.0, VIOLATION_TIME_THRESHOLD / 10.0))

    def _create_kpi_card(self, parent, col, title, default_val, subtitle, color):
        card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_BORDER, height=112)
        card.grid(row=0, column=col, padx=6, sticky="nsew")
        card.grid_propagate(False)

        ctk.CTkLabel(card, text=title, font=("Segoe UI", 12, "bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(12, 0))

        val_row = ctk.CTkFrame(card, fg_color="transparent")
        val_row.pack(fill="x", padx=16, pady=(2, 4))

        val_lbl = ctk.CTkLabel(val_row, text=default_val, font=FONT_KPI_VAL, text_color=TEXT_PRIMARY)
        val_lbl.pack(side="left")

        ctk.CTkLabel(val_row, text=subtitle, font=FONT_KPI_SUB, text_color=color).pack(side="right", pady=(6, 0))

        bar = ctk.CTkProgressBar(card, height=6, progress_color=color, fg_color="#F1F5F9", corner_radius=3)
        bar.set(0.15)
        bar.pack(fill="x", padx=16, pady=(4, 10))

        return val_lbl, bar

    # ========================================================================
    # 4. TRANG 1: DASHBOARD GIÁM SÁT TRUNG TÂM (VIDEO + LIVE PHẠT NGUỘI)
    # ========================================================================
    def _build_page_dashboard(self, page):
        page.grid_columnconfigure(0, weight=6)  # Cột trái: Luồng Video AI (60%)
        page.grid_columnconfigure(1, weight=4)  # Cột phải: Bằng chứng & Bảng vi phạm (40%)
        page.grid_rowconfigure(0, weight=1)

        # --- CARD TRÁI: LUỒNG CAMERA / VIDEO AI ---
        video_card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        video_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        video_card.grid_rowconfigure(1, weight=1)
        video_card.grid_columnconfigure(0, weight=1)

        vid_header = ctk.CTkFrame(video_card, fg_color="transparent")
        vid_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        ctk.CTkLabel(vid_header, text="🎥 Luồng Giám Sát Thời Gian Thực (Live AI Stream)", font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY).pack(side="left")
        self.lbl_video_time = ctk.CTkLabel(vid_header, text="00:00 / 00:00", font=("Segoe UI", 12, "bold"), text_color=ACCENT_BLUE)
        self.lbl_video_time.pack(side="right")

        # Khung chứa hình ảnh video
        self.video_frame = ctk.CTkFrame(video_card, fg_color="#0F172A", corner_radius=10)
        self.video_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_frame, text="Vui lòng chọn File Video hoặc Webcam trên thanh công cụ để bắt đầu...",
            font=("Segoe UI", 15), text_color="#94A3B8"
        )
        self.video_label.grid(row=0, column=0)

        # Thanh tiến trình thời lượng video
        self.video_progress_bar = ctk.CTkProgressBar(video_card, height=8, progress_color=ACCENT_BLUE, fg_color="#E2E8F0")
        self.video_progress_bar.set(0)
        self.video_progress_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

        # --- CARD PHẢI: BẰNG CHỨNG KÉP & NHẬT KÝ PHẠT NGUỘI TRỰC TIẾP ---
        right_card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        right_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        right_card.grid_rowconfigure(2, weight=1)
        right_card.grid_columnconfigure(0, weight=1)

        # Header Card Phải
        ev_header = ctk.CTkFrame(right_card, fg_color="transparent")
        ev_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
        ctk.CTkLabel(ev_header, text="📸 Bằng Chứng & Nhật Ký Phạt Nguội", font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY).pack(side="left")

        self.btn_zoom_evidence = ctk.CTkButton(
            ev_header, text="🔍 Phóng to ảnh", width=105, height=28,
            font=("Segoe UI", 12, "bold"), fg_color="#EFF6FF", hover_color="#DBEAFE", text_color=ACCENT_BLUE,
            command=self._open_zoom_popup, state="disabled"
        )
        self.btn_zoom_evidence.pack(side="right")

        # Khung Preview Kép (Toàn cảnh + Cận cảnh xe)
        preview_card = ctk.CTkFrame(right_card, fg_color="#F8FAFC", corner_radius=10, height=195, border_width=1, border_color="#E2E8F0")
        preview_card.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        preview_card.grid_propagate(False)
        preview_card.grid_columnconfigure(0, weight=3)
        preview_card.grid_columnconfigure(1, weight=2)
        preview_card.grid_rowconfigure(0, weight=1)

        full_box = ctk.CTkFrame(preview_card, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#E2E8F0")
        full_box.grid(row=0, column=0, padx=(8, 4), pady=8, sticky="nsew")
        ctk.CTkLabel(full_box, text="Toàn cảnh làn đường", font=("Segoe UI", 11, "bold"), text_color=TEXT_MUTED).pack(pady=(4, 0))
        self.lbl_snapshot = ctk.CTkLabel(full_box, text="[ Chưa có vi phạm ]", font=("Segoe UI", 11), text_color="#94A3B8")
        self.lbl_snapshot.pack(expand=True, padx=4, pady=4)

        crop_box = ctk.CTkFrame(preview_card, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color="#E2E8F0")
        crop_box.grid(row=0, column=1, padx=(4, 8), pady=8, sticky="nsew")
        ctk.CTkLabel(crop_box, text="Cận cảnh xe", font=("Segoe UI", 11, "bold"), text_color=TEXT_MUTED).pack(pady=(4, 0))
        self.lbl_crop_snapshot = ctk.CTkLabel(crop_box, text="[ Trống ]", font=("Segoe UI", 11), text_color="#94A3B8")
        self.lbl_crop_snapshot.pack(expand=True, padx=4, pady=4)

        # Bảng Treeview Nhật ký vi phạm (Style Sáng chuẩn CRM)
        table_container = ctk.CTkFrame(right_card, fg_color="transparent")
        table_container.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 14))
        table_container.grid_rowconfigure(1, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        self.lbl_evidence_info = ctk.CTkLabel(
            table_container, text="💡 Bấm vào từng dòng bên dưới để xem lại ảnh, nhấp đúp để phóng to.",
            font=("Segoe UI", 11, "italic"), text_color=TEXT_MUTED, anchor="w"
        )
        self.lbl_evidence_info.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview", background="#FFFFFF", foreground=TEXT_PRIMARY,
            rowheight=34, fieldbackground="#FFFFFF", borderwidth=0, font=("Segoe UI", 12)
        )
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", "#1E40AF")])
        style.configure(
            "Treeview.Heading", background="#F1F5F9", foreground="#334155",
            relief="flat", font=("Segoe UI", 11, "bold")
        )
        style.map("Treeview.Heading", background=[("active", "#E2E8F0")])

        tree_frame = ctk.CTkFrame(table_container, fg_color="#FFFFFF", corner_radius=8, border_width=1, border_color=CARD_BORDER)
        tree_frame.grid(row=1, column=0, sticky="nsew")

        columns = ("time", "id", "class", "lane", "conf")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("time", text="THỜI GIAN")
        self.tree.heading("id", text="ID XE")
        self.tree.heading("class", text="LOẠI XE")
        self.tree.heading("lane", text="VỊ TRÍ")
        self.tree.heading("conf", text="ĐỘ TIN CẬY")

        self.tree.column("time", width=80, anchor="center")
        self.tree.column("id", width=55, anchor="center")
        self.tree.column("class", width=80, anchor="center")
        self.tree.column("lane", width=70, anchor="center")
        self.tree.column("conf", width=75, anchor="center")

        self.tree.bind("<<TreeviewSelect>>", self._on_violation_select)
        self.tree.bind("<Double-1>", lambda e: self._open_zoom_popup())

        scrollbar = ctk.CTkScrollbar(tree_frame, orientation="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side="right", fill="y", padx=4, pady=4)

    # ========================================================================
    # 5. TRANG 2: HỒ SƠ PHẠT NGUỘI PHÓNG TO (TRA CỨU CHI TIẾT)
    # ========================================================================
    def _build_page_violations(self, page):
        card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        card.pack(fill="both", expand=True)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(18, 10))
        ctk.CTkLabel(header, text="📸 Trình Xem Bằng Chứng Phạt Nguội Độ Phân Giải Cao", font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkButton(
            header, text="📁 Mở Thư Mục Ảnh Evidence", command=lambda: os.startfile(str(EVIDENCE_DIR)) if sys.platform == "win32" else None,
            fg_color=ACCENT_BLUE, text_color="#FFFFFF", height=34
        ).pack(side="right")

        self.lbl_large_violation_view = ctk.CTkLabel(
            card, text="Khi có xe vi phạm, ảnh phóng to sắc nét sẽ hiển thị tại đây.\nBạn cũng có thể bấm vào bất kỳ xe nào ở bảng bên trang Dashboard để xem.",
            font=("Segoe UI", 14), text_color=TEXT_MUTED
        )
        self.lbl_large_violation_view.pack(expand=True, fill="both", padx=24, pady=16)

    # ========================================================================
    # 6. TRANG 3: BÁO CÁO & BIỂU ĐỒ LƯU LƯỢNG (TÍCH HỢP CẢ 3 PHƯƠNG ÁN)
    # ========================================================================
    def _build_page_analytics(self, page):
        page.grid_columnconfigure(0, weight=4)  # Cột Trái: Thống kê 5 loại xe (40%)
        page.grid_columnconfigure(1, weight=6)  # Cột Phải: Đa biểu đồ tương tác (60%)
        page.grid_rowconfigure(0, weight=1)

        # --- CARD TRÁI: PHÂN BỔ LƯU LƯỢNG 5 LOẠI PHƯƠNG TIỆN ---
        left_card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        left_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")

        ctk.CTkLabel(
            left_card, text="📊 Phân Bổ Lưu Lượng Theo Phương Tiện", 
            font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=22, pady=(18, 12))

        vehicle_meta = [
            ("car", "🚗 Ô tô con (Car)", ACCENT_BLUE),
            ("motorbike", "🛵 Xe máy (Motorbike)", ACCENT_GREEN),
            ("bus", "🚌 Xe buýt (Bus)", ACCENT_ORANGE),
            ("truck", "🚚 Xe tải (Truck)", ACCENT_RED),
            ("bike", "🚲 Xe đạp (Bike)", ACCENT_PURPLE),
        ]

        self.class_bars = {}
        self.class_labels = {}

        for cls_key, display_name, color in vehicle_meta:
            row = ctk.CTkFrame(left_card, fg_color="#F8FAFC", corner_radius=10, border_width=1, border_color="#E2E8F0")
            row.pack(fill="x", padx=20, pady=7)

            top_row = ctk.CTkFrame(row, fg_color="transparent")
            top_row.pack(fill="x", padx=14, pady=(10, 4))

            ctk.CTkLabel(top_row, text=display_name, font=FONT_BOLD, text_color=TEXT_PRIMARY).pack(side="left")
            val_lbl = ctk.CTkLabel(top_row, text="0 xe (0%)", font=FONT_BOLD, text_color=color)
            val_lbl.pack(side="right")

            bar = ctk.CTkProgressBar(row, height=10, progress_color=color, fg_color="#E2E8F0", corner_radius=5)
            bar.set(0)
            bar.pack(fill="x", padx=14, pady=(2, 12))

            self.class_bars[cls_key] = bar
            self.class_labels[cls_key] = val_lbl

        # --- CARD PHẢI: TRUNG TÂM BIỂU ĐỒ ĐA PHƯƠNG ÁN ---
        right_card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        right_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        right_card.grid_rowconfigure(1, weight=1)
        right_card.grid_columnconfigure(0, weight=1)

        # Header + Thanh chọn chế độ xem biểu đồ
        chart_header = ctk.CTkFrame(right_card, fg_color="transparent")
        chart_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))

        ctk.CTkLabel(
            chart_header, text="📈 Phân Tích Chuyên Sâu", 
            font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY
        ).pack(side="left")

        self.chart_mode_var = ctk.StringVar(value="⊞ Tất Cả (3 Phương Án)")
        self.chart_Selector = ctk.CTkSegmentedButton(
            chart_header,
            values=["⊞ Tất Cả (3 Phương Án)", "🎯 Tỷ Lệ & Làn", "🚗 Theo Loại Xe", "📈 Mật Độ Live"],
            variable=self.chart_mode_var,
            command=self._on_chart_mode_change,
            font=("Segoe UI", 12, "bold"),
            selected_color=ACCENT_BLUE,
            selected_hover_color=SIDEBAR_HOVER,
            unselected_color="#F1F5F9",
            unselected_hover_color="#E2E8F0",
            text_color=TEXT_PRIMARY
        )
        self.chart_Selector.pack(side="right")

        # Khung vẽ Matplotlib
        self.chart_frame = ctk.CTkFrame(right_card, fg_color=CARD_BG, corner_radius=10)
        self.chart_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.fig = plt.figure(figsize=(6.5, 4.8), dpi=100)
        self.fig.patch.set_facecolor(CARD_BG)

        # Bộ nhớ dữ liệu cho cả 3 phương án
        self.lane_violation_counts = collections.defaultdict(int)
        self.class_violation_counts = collections.defaultdict(int)
        self.density_time_history = collections.deque(maxlen=60)
        self.density_active_history = collections.deque(maxlen=60)
        self.density_viol_history = collections.deque(maxlen=60)
        self.last_chart_sample_time = 0.0
        self.latest_total_veh = 0
        self.latest_viol_cnt = 0

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._redraw_all_analytics_charts()

    # ========================================================================
    # 7. TRANG 4: CẤU HÌNH HỆ THỐNG & THAM SỐ AI
    # ========================================================================
    def _build_page_settings(self, page):
        card = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color=CARD_BORDER)
        card.pack(fill="both", expand=True)

        ctk.CTkLabel(card, text="⚙️ Tinh Chỉnh Mô Hình AI & Luật Bắt Lỗi Giao Thông", font=FONT_CARD_HEAD, text_color=TEXT_PRIMARY).pack(anchor="w", padx=28, pady=(22, 16))

        # 1. Ngưỡng Confidence
        box1 = ctk.CTkFrame(card, fg_color="#F8FAFC", corner_radius=10, border_width=1, border_color=CARD_BORDER)
        box1.pack(fill="x", padx=28, pady=8)
        ctk.CTkLabel(box1, text="Ngưỡng tin cậy nhận diện YOLO (Confidence Threshold):", font=FONT_BOLD, text_color=TEXT_PRIMARY).pack(anchor="w", padx=18, pady=(12, 4))
        
        s1_row = ctk.CTkFrame(box1, fg_color="transparent")
        s1_row.pack(fill="x", padx=18, pady=(0, 14))
        self.conf_slider = ctk.CTkSlider(s1_row, from_=0.15, to=0.90, number_of_steps=15, command=self.update_conf, progress_color=ACCENT_BLUE)
        self.conf_slider.set(0.50)
        self.conf_slider.pack(side="left", fill="x", expand=True, padx=(0, 16))
        self.lbl_conf_val = ctk.CTkLabel(s1_row, text="50%", font=("Segoe UI", 16, "bold"), text_color=ACCENT_BLUE, width=50)
        self.lbl_conf_val.pack(side="right")

        # 2. Thời gian đi trong làn cấm (n giây)
        box2 = ctk.CTkFrame(card, fg_color="#F8FAFC", corner_radius=10, border_width=1, border_color=CARD_BORDER)
        box2.pack(fill="x", padx=28, pady=8)
        ctk.CTkLabel(box2, text="Thời gian di chuyển trong làn cấm để xác nhận vi phạm (n giây):", font=FONT_BOLD, text_color=TEXT_PRIMARY).pack(anchor="w", padx=18, pady=(12, 4))
        
        s2_row = ctk.CTkFrame(box2, fg_color="transparent")
        s2_row.pack(fill="x", padx=18, pady=(0, 14))
        self.viol_time_slider = ctk.CTkSlider(s2_row, from_=0.5, to=10.0, number_of_steps=19, command=self.update_violation_time, progress_color=ACCENT_ORANGE)
        self.viol_time_slider.set(VIOLATION_TIME_THRESHOLD)
        self.viol_time_slider.pack(side="left", fill="x", expand=True, padx=(0, 16))
        self.lbl_viol_time_val = ctk.CTkLabel(s2_row, text=f"{VIOLATION_TIME_THRESHOLD:.1f}s", font=("Segoe UI", 16, "bold"), text_color=ACCENT_ORANGE, width=50)
        self.lbl_viol_time_val.pack(side="right")

        # 3. Các công tắc xử lý hình ảnh
        box3 = ctk.CTkFrame(card, fg_color="#F8FAFC", corner_radius=10, border_width=1, border_color=CARD_BORDER)
        box3.pack(fill="x", padx=28, pady=8)

        self.switch_clahe = ctk.CTkSwitch(box3, text="🌙 Bật cân bằng sáng thích ứng ban đêm / ngược sáng (CLAHE)", font=FONT_MAIN, text_color=TEXT_PRIMARY, command=self.toggle_clahe)
        self.switch_clahe.pack(anchor="w", padx=18, pady=(14, 8))

        self.switch_draw_roi = ctk.CTkSwitch(box3, text="📐 Hiển thị vùng đa giác làn đường (ROI Overlay) trên Video", font=FONT_MAIN, text_color=TEXT_PRIMARY, command=self.toggle_draw_roi)
        self.switch_draw_roi.select()
        self.switch_draw_roi.pack(anchor="w", padx=18, pady=8)

        self.switch_optimized = ctk.CTkSwitch(box3, text="⚡ Sử dụng mô hình YOLOv8 đã tối ưu hóa (yolo_optimized.pt)", font=FONT_MAIN, text_color=TEXT_PRIMARY)
        self.switch_optimized.select()
        self.switch_optimized.pack(anchor="w", padx=18, pady=(8, 14))

    # ========================================================================
    # 8. ĐIỀU HƯỚNG SIDEBAR & CẬP NHẬT THAM SỐ
    # ========================================================================
    def switch_page(self, page_key: str):
        titles = {
            "dashboard": "Dashboard Giám Sát",
            "violations": "Hồ Sơ Phạt Nguội",
            "analytics": "Báo Cáo & Biểu Đồ",
            "settings": "Cấu Hình Hệ Thống"
        }
        self.lbl_page_title.configure(text=titles.get(page_key, "Dashboard"))

        for k, btn in self.nav_buttons.items():
            btn.configure(fg_color=SIDEBAR_ACTIVE if k == page_key else "transparent")

        self.pages[page_key].tkraise()

    def update_conf(self, value):
        self.lbl_conf_val.configure(text=f"{int(value * 100)}%")
        self.command_queue.put({"action": "set_confidence", "value": float(value)})

    def update_violation_time(self, value):
        val = float(value)
        self.lbl_viol_time_val.configure(text=f"{val:.1f}s")
        self.lbl_kpi_threshold.configure(text=f"{val:.1f}s")
        self.bar_threshold.set(min(1.0, val / 10.0))
        self.command_queue.put({"action": "set_violation_time", "value": val})

    def toggle_clahe(self):
        self.command_queue.put({"action": "set_clahe", "value": bool(self.switch_clahe.get())})

    def toggle_draw_roi(self):
        self.command_queue.put({"action": "set_draw_roi", "value": bool(self.switch_draw_roi.get())})

    # ========================================================================
    # 9. QUẢN LÝ LƯU / NẠP CẤU HÌNH ROI (JSON PRESETS)
    # ========================================================================
    def _get_source_key(self) -> str:
        if self.video_source is None:
            return ""
        if str(self.video_source).isdigit():
            return f"WEBCAM_{self.video_source}"
        return Path(str(self.video_source)).name

    def _save_roi_preset(self):
        source_key = self._get_source_key()
        if not source_key or not self.custom_polygons:
            return
        try:
            presets = {}
            if Path(ROI_CONFIG_FILE).exists():
                with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                    presets = json.load(f)

            serializable_polys = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.custom_polygons.items()
            }
            presets[source_key] = {
                "polygons": serializable_polys,
                "restrictions": self.custom_restrictions
            }
            with open(ROI_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Không thể lưu ROI preset: {e}")

    def _load_roi_preset(self) -> bool:
        source_key = self._get_source_key()
        if not source_key or not Path(ROI_CONFIG_FILE).exists():
            self.custom_polygons = None
            self.custom_restrictions = None
            self.lbl_roi_status.configure(text="• ROI: Mặc định", text_color=TEXT_MUTED)
            return False
        try:
            with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                presets = json.load(f)
            if source_key in presets:
                data = presets[source_key]
                self.custom_polygons = {k: np.array(v, dtype=np.int32) for k, v in data["polygons"].items()}
                self.custom_restrictions = data["restrictions"]
                self.lbl_roi_status.configure(text=f"• ROI: Đã nạp ({len(self.custom_polygons)} làn)", text_color=ACCENT_GREEN)
                return True
        except Exception:
            pass
        self.custom_polygons = None
        self.custom_restrictions = None
        self.lbl_roi_status.configure(text="• ROI: Mặc định", text_color=TEXT_MUTED)
        return False

    def reset_roi_preset(self):
        source_key = self._get_source_key()
        if not source_key:
            return
        if messagebox.askyesno("Xóa cấu hình làn", f"Bạn có muốn xóa các làn đường đã lưu của '{source_key}'?"):
            self.custom_polygons = None
            self.custom_restrictions = None
            if Path(ROI_CONFIG_FILE).exists():
                try:
                    with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                        presets = json.load(f)
                    if source_key in presets:
                        del presets[source_key]
                        with open(ROI_CONFIG_FILE, "w", encoding="utf-8") as f:
                            json.dump(presets, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            self.lbl_roi_status.configure(text="• ROI: Đã reset về mặc định", text_color=ACCENT_ORANGE)

    # ========================================================================
    # 10. XỬ LÝ ẢNH BẰNG CHỨNG PHẠT NGUỘI (TOÀN CẢNH + CẬN CẢNH + ZOOM)
    # ========================================================================
    def _record_and_display_snapshot(self, viol_data: dict):
        if self.current_bgr_frame is None:
            return

        viol_id = viol_data['id']
        rgb_full = cv2.cvtColor(self.current_bgr_frame, cv2.COLOR_BGR2RGB)
        full_pil = Image.fromarray(rgb_full)

        crop_pil = full_pil
        bbox = viol_data.get('bbox')
        if bbox and len(bbox) == 4:
            img_h, img_w = rgb_full.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            pad = 15
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(img_w, x2 + pad), min(img_h, y2 + pad)
            if cx2 > cx1 and cy2 > cy1:
                crop_pil = Image.fromarray(rgb_full[cy1:cy2, cx1:cx2])

        self.violation_records[viol_id] = {
            "id": viol_id,
            "time": viol_data.get("time", ""),
            "class": viol_data.get("class", ""),
            "lane": viol_data.get("lane", ""),
            "conf": viol_data.get("conf", ""),
            "type": viol_data.get("type", "Đi Sai Làn"),
            "full_pil": full_pil,
            "crop_pil": crop_pil
        }
        self._display_violation_preview(viol_id)

    def _display_violation_preview(self, viol_id: int):
        record = self.violation_records.get(viol_id)
        if not record:
            return

        self.selected_viol_id = viol_id
        self.btn_zoom_evidence.configure(state="normal")

        # 1. Ảnh toàn cảnh ở Dashboard
        full_pil = record["full_pil"]
        fw, fh = full_pil.size
        target_fh = 135
        target_fw = max(1, int(fw * (target_fh / fh)))
        full_ctk = ctk.CTkImage(light_image=full_pil, dark_image=full_pil, size=(target_fw, target_fh))
        self.lbl_snapshot.configure(image=full_ctk, text="")

        # 2. Ảnh cận cảnh xe
        crop_pil = record["crop_pil"]
        cw, ch = crop_pil.size
        scale = min(135 / max(1, cw), 135 / max(1, ch))
        crop_ctk = ctk.CTkImage(light_image=crop_pil, dark_image=crop_pil, size=(max(1, int(cw * scale)), max(1, int(ch * scale))))
        self.lbl_crop_snapshot.configure(image=crop_ctk, text="")

        # 3. Cập nhật cả ảnh lớn bên trang "Hồ sơ Phạt Nguội"
        lg_scale = min(880 / fw, 520 / fh)
        lg_ctk = ctk.CTkImage(light_image=full_pil, dark_image=full_pil, size=(int(fw * lg_scale), int(fh * lg_scale)))
        self.lbl_large_violation_view.configure(image=lg_ctk, text="")

        self.lbl_evidence_info.configure(
            text=f"🚨 Xe ID #{viol_id} ({record['class']}) | {record['type']} tại {record['lane']} | Lúc: {record['time']}",
            text_color=ACCENT_RED
        )

    def _on_violation_select(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        item_values = self.tree.item(selected_items[0], "values")
        if item_values and len(item_values) >= 2:
            self._display_violation_preview(int(item_values[1]))

    def _open_zoom_popup(self):
        if self.selected_viol_id is None or self.selected_viol_id not in self.violation_records:
            return
        rec = self.violation_records[self.selected_viol_id]
        popup = ctk.CTkToplevel(self)
        popup.title(f"🔍 Chi tiết Phạt Nguội - Xe ID #{rec['id']} ({rec['class']})")
        popup.geometry("960x620")
        popup.configure(fg_color=CARD_BG)
        popup.attributes("-topmost", True)

        ctk.CTkLabel(
            popup,
            text=f"BẰNG CHỨNG PHẠT NGUỘI | ID: #{rec['id']} | Loại: {rec['class']} | Làn: {rec['lane']} | Lúc: {rec['time']}",
            font=("Segoe UI", 15, "bold"), text_color=ACCENT_RED
        ).pack(pady=(14, 8))

        full_pil = rec["full_pil"]
        fw, fh = full_pil.size
        scale = min(920 / fw, 530 / fh)
        large_ctk = ctk.CTkImage(light_image=full_pil, dark_image=full_pil, size=(int(fw * scale), int(fh * scale)))
        ctk.CTkLabel(popup, image=large_ctk, text="").pack(expand=True, padx=15, pady=10)

    # ========================================================================
    # 11. VẬN HÀNH HỆ THỐNG (CHỌN NGUỒN, VẼ ROI, START / PAUSE / STOP)
    # ========================================================================
    def select_file(self):
        file_path = filedialog.askopenfilename(title="Chọn Video Giao Thông", filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.video_source = file_path
            self.lbl_source.configure(text=f"Nguồn: {Path(file_path).name}")
            self.status_badge.configure(text="📂 Đã nạp Video", fg_color="#DBEAFE", text_color="#1D4ED8")
            self._load_roi_preset()

    def select_webcam(self):
        self.video_source = 0
        self.lbl_source.configure(text="Nguồn: Webcam Live")
        self.status_badge.configure(text="📷 Sẵn sàng Webcam", fg_color="#DBEAFE", text_color="#1D4ED8")
        self._load_roi_preset()

    def launch_roi_drawer(self):
        if hasattr(self, 'ai_thread') and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        if self.video_source is None:
            messagebox.showwarning("Chưa chọn nguồn", "Vui lòng chọn File Video hoặc Webcam trước khi vẽ làn đường!")
            return
        if self.is_running:
            messagebox.showwarning("Đang chạy", "Vui lòng bấm '⏹ Kết thúc' trước khi vẽ lại làn đường!")
            return
        try:
            from tools.roi_drawer import ROIDrawer
            self.status_badge.configure(text="🖍️ Đang vẽ ROI...", fg_color="#FEF3C7", text_color="#B45309")
            self.update()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            try:
                drawer = ROIDrawer(self.video_source, self.custom_polygons, self.custom_restrictions)
            except TypeError:
                drawer = ROIDrawer(self.video_source)

            polygons, restrictions = drawer.run()
            if polygons:
                self.custom_polygons = {k: np.array(v, dtype=np.int32) for k, v in polygons.items()}
                self.custom_restrictions = restrictions
                self._save_roi_preset()
                self.lbl_roi_status.configure(text=f"• ROI: Đã lưu ({len(self.custom_polygons)} làn)", text_color=ACCENT_GREEN)
                if messagebox.askyesno("Hoàn tất", "Đã lưu cấu hình ROI mới!\nBạn có muốn KHỞI ĐỘNG AI ngay không?"):
                    self.start_processing()
            else:
                self.status_badge.configure(text="🟢 Hệ thống sẵn sàng", fg_color="#DCFCE7", text_color="#15803D")
        except Exception as e:
            messagebox.showerror("Lỗi công cụ ROI", f"Chi tiết: {e}")

    def start_processing(self):
        if self.video_source is None or self.video_source == "":
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn File Video hoặc Webcam trước khi khởi động!")
            return

        # [FIX 1]: Đợi luồng AI của lần chạy trước đóng hoàn toàn rồi mới tạo luồng mới
        if hasattr(self, 'ai_thread') and self.ai_thread is not None and self.ai_thread.is_alive():
            self.is_running = False
            self.status_badge.configure(text="⏳ Đang dọn luồng cũ...", fg_color="#FEF3C7", text_color="#B45309")
            self.update()
            self.ai_thread.join(timeout=3.0)

        self.is_running = True
        self.is_paused = False
        self.switch_page("dashboard")

        self.btn_start.configure(state="disabled")
        self.btn_select_file.configure(state="disabled")
        self.btn_select_cam.configure(state="disabled")
        self.btn_draw_roi.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸ Tạm Dừng", fg_color="#F8FAFC", text_color=TEXT_PRIMARY)
        self.btn_stop.configure(state="normal")

        self.status_badge.configure(text="⏳ Đang nạp AI...", fg_color="#FEF3C7", text_color="#B45309")
        self.video_label.configure(text="⏳ Đang khởi tạo mô hình và mở lại luồng Video...")
        self.update()

        # [FIX 2]: Dọn sạch sẽ 100% các tín hiệu 'engine_stopped' còn sót từ lần chạy trước
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        while not self.stats_queue.empty():
            try: self.stats_queue.get_nowait()
            except queue.Empty: break
        while not self.command_queue.empty():
            try: self.command_queue.get_nowait()
            except queue.Empty: break

        # Reset toàn bộ số liệu & biểu đồ về 0 cho lượt chạy mới
        self.history_time.clear()
        self.history_vehicles.clear()
        self.start_app_time = time.time()
        self.video_progress_bar.set(0)
        self.lbl_video_time.configure(text="00:00 / 00:00")

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.recorded_violation_ids.clear()
        self.violation_records.clear()
        self.selected_viol_id = None
        self.btn_zoom_evidence.configure(state="disabled")

        for cls_key in self.class_bars:
            self.class_bars[cls_key].set(0)
            self.class_labels[cls_key].configure(text="0 xe (0%)")

        from utils.config import LANE_POLYGONS, LANE_RESTRICTIONS
        polygons_to_send = self.custom_polygons if self.custom_polygons else LANE_POLYGONS
        restrictions_to_send = self.custom_restrictions if self.custom_restrictions else LANE_RESTRICTIONS

        use_opt = bool(self.switch_optimized.get())

        self.ai_thread = threading.Thread(
            target=self.start_ai_engine,
            args=(
                self.video_source, polygons_to_send, restrictions_to_send,
                self.frame_queue, self.stats_queue, self.command_queue,
                lambda: self.is_running, use_opt
            )
        )
        self.ai_thread.daemon = True
        self.ai_thread.start()

        # Đồng bộ cấu hình từ trang Settings xuống luồng AI
        self.command_queue.put({"action": "set_confidence", "value": float(self.conf_slider.get())})
        self.command_queue.put({"action": "set_violation_time", "value": float(self.viol_time_slider.get())})
        self.command_queue.put({"action": "set_clahe", "value": bool(self.switch_clahe.get())})
        self.command_queue.put({"action": "set_draw_roi", "value": bool(self.switch_draw_roi.get())})

        self.update_gui_loop()

        self.lane_violation_counts.clear()
        self.class_violation_counts.clear()
        self.density_time_history.clear()
        self.density_active_history.clear()
        self.density_viol_history.clear()
        self.last_chart_sample_time = 0.0
        self.latest_total_veh = 0
        self.latest_viol_cnt = 0
        self._redraw_all_analytics_charts()
        self.canvas.draw_idle()

    def toggle_pause_processing(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.command_queue.put({"action": "set_pause", "value": self.is_paused})
        if self.is_paused:
            self.btn_pause.configure(text="▶ Tiếp Tục", fg_color="#EFF6FF", text_color=ACCENT_BLUE, border_color="#93C5FD")
            self.status_badge.configure(text="⏸ Đang tạm dừng", fg_color="#FEF3C7", text_color="#B45309")
        else:
            self.btn_pause.configure(text="⏸ Tạm Dừng", fg_color="#F8FAFC", text_color=TEXT_PRIMARY, border_color="#CBD5E1")
            self.status_badge.configure(text="🟢 Đang giám sát", fg_color="#DCFCE7", text_color="#15803D")

    def stop_processing(self):
        self.is_running = False
        self.is_paused = False
        self.btn_start.configure(state="normal")
        self.btn_select_file.configure(state="normal")
        self.btn_select_cam.configure(state="normal")
        self.btn_draw_roi.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="⏸ Tạm Dừng", fg_color="#F8FAFC", text_color=TEXT_PRIMARY)
        self.btn_stop.configure(state="disabled")

        # [FIX 3]: Tạo một ảnh nền tối trống thay vì truyền image=None hoặc image="" (tránh sập CTkLabel)
        empty_pil = Image.new("RGB", (640, 360), color=(15, 23, 42))
        self.current_ctk_image = ctk.CTkImage(light_image=empty_pil, dark_image=empty_pil, size=(640, 360))
        self.video_label.configure(
            image=self.current_ctk_image,
            text="⏹ Đã kết thúc phiên giám sát.\nBấm '▶ Khởi Động AI' nếu muốn phát lại từ đầu.",
            compound="center"
        )
        self.status_badge.configure(text="🔴 Đã dừng", fg_color="#FEE2E2", text_color="#B91C1C")

    # ========================================================================
    # 12. VÒNG LẶP CẬP NHẬT GIAO DIỆN CHÍNH (30MS)
    # ========================================================================
    def update_gui_loop(self):
        if not self.is_running:
            return

        # 1. Hiển thị Frame Video chuẩn tỷ lệ 16:9
        try:
            frame = self.frame_queue.get_nowait()
            self.current_bgr_frame = frame.copy()

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)

            img_h, img_w = frame.shape[:2]
            container_w = max(100, self.video_frame.winfo_width() - 12)
            container_h = max(100, self.video_frame.winfo_height() - 12)

            scale = min(container_w / img_w, container_h / img_h)
            display_w = max(1, int(img_w * scale))
            display_h = max(1, int(img_h * scale))

            # [FIX 4]: Lưu tham chiếu vào self.current_ctk_image để tránh bị Garbage Collector xóa ảnh
            self.current_ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(display_w, display_h))
            self.video_label.configure(image=self.current_ctk_image, text="")
            if not self.is_paused:
                self.status_badge.configure(text="🟢 Đang giám sát Live", fg_color="#DCFCE7", text_color="#15803D")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Lỗi hiển thị Frame: {e}")

        # 2. Cập nhật 6 Thẻ KPI, Bảng Phạt Nguội & Biểu đồ
        try:
            stats = self.stats_queue.get_nowait()
            if stats.get('action') == 'engine_stopped':
                self.stop_processing()
                if "error" in stats:
                    messagebox.showerror("Lỗi Luồng AI", f"Luồng AI bị dừng đột ngột:\n{stats['error']}")
                else:
                    self.status_badge.configure(text="✓ Hoàn tất Video", fg_color="#DBEAFE", text_color="#1D4ED8")
                return

            fps_val = stats.get('fps', 0.0)
            current_total = stats.get('total_vehicles', 0)
            viol_count = stats.get('violations', 0)
            class_counts = stats.get('class_counts', {})

            four_wheel_cnt = class_counts.get('car', 0) + class_counts.get('truck', 0) + class_counts.get('bus', 0)
            two_wheel_cnt = class_counts.get('motorbike', 0) + class_counts.get('bike', 0)

            self.lbl_fps.configure(text=f"{fps_val:.1f}")
            self.bar_fps.set(min(1.0, fps_val / 45.0))

            self.lbl_total_vehicles.configure(text=str(current_total))
            self.bar_total.set(min(1.0, current_total / 100.0))

            self.lbl_violations.configure(text=str(viol_count))
            self.bar_viol.set(min(1.0, viol_count / 20.0))

            self.lbl_four_wheel.configure(text=str(four_wheel_cnt))
            self.bar_four_wheel.set((four_wheel_cnt / current_total) if current_total > 0 else 0.0)

            self.lbl_two_wheel.configure(text=str(two_wheel_cnt))
            self.bar_two_wheel.set((two_wheel_cnt / current_total) if current_total > 0 else 0.0)

            if 'progress_ratio' in stats:
                self.video_progress_bar.set(stats['progress_ratio'])
            if 'progress_text' in stats:
                self.lbl_video_time.configure(text=stats['progress_text'])

            for cls_key, bar in self.class_bars.items():
                cnt = class_counts.get(cls_key, 0)
                ratio = (cnt / current_total) if current_total > 0 else 0.0
                bar.set(ratio)
                self.class_labels[cls_key].configure(text=f"{cnt} xe ({int(ratio * 100)}%)")

            # Ghi nhận vi phạm mới vào Bảng, bộ đếm Làn (PA1) và bộ đếm Loại xe (PA2)
            if 'live_viol_data' in stats:
                for v in stats['live_viol_data']:
                    if v['id'] not in self.recorded_violation_ids:
                        conf_pct = f"{float(v['conf']) * 100:.0f}%" if str(v.get('conf', '')).replace('.', '', 1).isdigit() else v.get('conf', '')
                        self.tree.insert("", "0", values=(v['time'], v['id'], v['class'], v['lane'], conf_pct))
                        self.recorded_violation_ids.add(v['id'])
                        
                        self.lane_violation_counts[v.get('lane', 'KHÁC')] += 1
                        self.class_violation_counts[str(v.get('class', 'CAR')).upper()] += 1
                        self._record_and_display_snapshot(v)

            # Cập nhật dữ liệu cho cả 3 phương án biểu đồ (Lấy mẫu chuẩn mỗi 0.8 giây)
            self.latest_total_veh = current_total
            self.latest_viol_cnt = viol_count
            current_time = time.time() - self.start_app_time

            if current_time - self.last_chart_sample_time >= 0.8:
                self.last_chart_sample_time = current_time
                active_veh = stats.get('active_vehicles', 0)
                
                self.density_time_history.append(round(current_time, 1))
                self.density_active_history.append(active_veh)
                self.density_viol_history.append(viol_count)

                self._redraw_all_analytics_charts()
                self.canvas.draw_idle()

            if len(self.history_time) % 5 == 0:
                self.line.set_data(self.history_time, self.history_vehicles)
                self.ax.set_xlim(max(0, current_time - 60), current_time + 5)
                self.ax.set_ylim(0, max(self.history_vehicles) + 5 if self.history_vehicles else 10)
                self.canvas.draw_idle()

        except queue.Empty:
            pass

        self.after(30, self.update_gui_loop)

    def open_output_folder(self):
        try:
            folder_path = str(OUTPUT_DIR)
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở thư mục: {e}")

    def hard_exit(self):
        if messagebox.askyesno("Xác nhận thoát", "Bạn có chắc chắn muốn tắt hoàn toàn hệ thống giám sát?"):
            self.is_running = False
            self.quit()
            self.destroy()
            sys.exit(0)

    def _on_chart_mode_change(self, selected_mode: str):
        """Vẽ lại ngay lập tức khi người dùng bấm chuyển đổi chế độ biểu đồ"""
        self._redraw_all_analytics_charts()
        self.canvas.draw_idle()

    def _style_axis(self, ax, title: str):
        """Chuẩn hóa giao diện Sáng (CRM Light Style) cho từng trục biểu đồ"""
        ax.set_facecolor(CARD_BG)
        ax.set_title(title, fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold", pad=8)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')

    def _draw_donut_compliance(self, ax):
        """[PHƯƠNG ÁN 1A]: Biểu đồ Donut Tỷ lệ Tuân thủ vs Vi phạm"""
        ax.set_facecolor(CARD_BG)
        total_veh = self.latest_total_veh
        viol_cnt = self.latest_viol_cnt
        compliant_cnt = max(0, total_veh - viol_cnt)
        viol_rate = (viol_cnt / total_veh * 100) if total_veh > 0 else 0.0

        if total_veh == 0:
            ax.pie([1], colors=["#E2E8F0"], startangle=90, wedgeprops={'width': 0.36, 'edgecolor': 'white'})
            ax.text(0, 0, "0%\nVi phạm", ha='center', va='center', fontsize=9.5, color=TEXT_MUTED, fontweight='bold')
        else:
            ax.pie(
                [compliant_cnt, viol_cnt], colors=[ACCENT_GREEN, ACCENT_RED],
                startangle=90, counterclock=False,
                wedgeprops={'width': 0.36, 'edgecolor': 'white', 'linewidth': 2}
            )
            center_color = ACCENT_RED if viol_cnt > 0 else ACCENT_GREEN
            ax.text(0, 0.08, f"{viol_rate:.1f}%", ha='center', va='center', fontsize=13, fontweight='bold', color=center_color)
            ax.text(0, -0.22, "Sai làn", ha='center', va='center', fontsize=8, color=TEXT_MUTED)

        ax.set_title(f"Tuân thủ: {compliant_cnt}  |  Vi phạm: {viol_cnt}", fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold", pad=4)

    def _draw_bar_lanes(self, ax):
        """[PHƯƠNG ÁN 1B]: Biểu đồ Cột Điểm nóng vi phạm theo Làn đường"""
        self._style_axis(ax, "Điểm nóng Vi phạm theo Làn đường")
        if self.lane_violation_counts:
            lanes = sorted(self.lane_violation_counts.keys())
            counts = [self.lane_violation_counts[k] for k in lanes]
        else:
            lanes = ["LANE_1", "LANE_2", "LANE_3", "LANE_4"]
            counts = [0, 0, 0, 0]

        bars = ax.bar(lanes, counts, color=ACCENT_BLUE, width=0.45)
        if max(counts) > 0:
            bars[counts.index(max(counts))].set_color(ACCENT_RED)

        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(c),
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=TEXT_PRIMARY)
        ax.set_ylim(0, max(counts) + 3 if counts else 5)

    def _draw_bar_classes(self, ax):
        """[PHƯƠNG ÁN 2]: Biểu đồ Cột ngang Vi phạm theo Loại phương tiện"""
        self._style_axis(ax, "Số ca Vi phạm theo Loại phương tiện")
        classes = ["CAR", "MOTORBIKE", "BUS", "TRUCK", "BIKE"]
        labels = ["Ô tô", "Xe máy", "Xe buýt", "Xe tải", "Xe đạp"]
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_PURPLE]
        counts = [self.class_violation_counts.get(c, 0) for c in classes]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, counts, color=colors, height=0.52)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold", color=TEXT_PRIMARY)
        ax.invert_yaxis()  # Đưa Ô tô, Xe máy lên đầu
        ax.set_xlim(0, max(counts) + 3 if counts else 5)

        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, str(c),
                        va='center', ha='left', fontsize=8.5, fontweight='bold', color=TEXT_PRIMARY)

    def _draw_area_density(self, ax):
        """[PHƯƠNG ÁN 3]: Biểu đồ Miền (Area Chart) Mật độ xe thực tế & Vi phạm giống mẫu CRM"""
        self._style_axis(ax, "Mật độ Xe trên đường & Tổng Vi phạm (Live)")
        t_list = list(self.density_time_history) if self.density_time_history else [0]
        act_list = list(self.density_active_history) if self.density_active_history else [0]
        viol_list = list(self.density_viol_history) if self.density_viol_history else [0]

        ax.plot(t_list, act_list, color=ACCENT_BLUE, linewidth=2, label="Xe đang trong hình")
        ax.fill_between(t_list, act_list, color=ACCENT_BLUE, alpha=0.18)

        ax.plot(t_list, viol_list, color=ACCENT_RED, linewidth=2, linestyle="--", label="Tổng ca vi phạm")
        ax.fill_between(t_list, viol_list, color=ACCENT_RED, alpha=0.15)

        max_y = max(max(act_list, default=0), max(viol_list, default=0))
        ax.set_xlim(max(0, t_list[-1] - 60), max(10, t_list[-1] + 2))
        ax.set_ylim(0, max_y + 4 if max_y > 0 else 10)
        ax.legend(loc="upper left", frameon=False, fontsize=8)

    def _redraw_all_analytics_charts(self):
        """Tự động chia bố cục Matplotlib theo chế độ người dùng chọn"""
        self.fig.clear()
        mode = self.chart_mode_var.get()

        if mode == "🎯 Tỷ Lệ & Làn":
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            self._draw_donut_compliance(ax1)
            self._draw_bar_lanes(ax2)

        elif mode == "🚗 Theo Loại Xe":
            ax = self.fig.add_subplot(1, 1, 1)
            self._draw_bar_classes(ax)

        elif mode == "📈 Mật Độ Live":
            ax = self.fig.add_subplot(1, 1, 1)
            self._draw_area_density(ax)

        else:  # "⊞ Tất Cả (3 Phương Án)" -> Lưới 2x2 hiển thị trọn bộ
            ax1 = self.fig.add_subplot(2, 2, 1)
            ax2 = self.fig.add_subplot(2, 2, 2)
            ax3 = self.fig.add_subplot(2, 2, 3)
            ax4 = self.fig.add_subplot(2, 2, 4)
            self._draw_donut_compliance(ax1)
            self._draw_bar_lanes(ax2)
            self._draw_bar_classes(ax3)
            self._draw_area_density(ax4)

        self.fig.tight_layout(pad=1.6)