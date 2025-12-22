"""
VISCA Camera Video Tracking Application - PyQt6 Version
Modern UI with smooth MediaPipe + PID based tracking

Uses MediaPipe for face/pose detection and PID controller for smooth camera movements.
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QSlider, QGroupBox, QGridLayout, QStatusBar, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import json
import os
import sys
from visca_over_ip import Camera
from camera_manager import CameraManager
from smooth_tracker import SmoothTracker, MEDIAPIPE_AVAILABLE


class VideoThread(QThread):
    """Background thread for video capture to avoid blocking the UI"""
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False
        self.cap = None
    
    def run(self):
        """Main thread loop for capturing video frames"""
        self.running = True
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url,cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Failed to open video stream: {self.rtsp_url}")
                return
            
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    # Retry connection
                    time.sleep(0.5)
                    
        except Exception as e:
            self.error_occurred.emit(f"Video thread error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def stop(self):
        """Gracefully stop the video thread"""
        self.running = False
        self.wait()


class VideoTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISCA Camera Video Tracking")
        self.setGeometry(100, 100, 1200, 800)
        
        # Camera control variables
        self.camera = None
        self.connected = False
        self.tracking_preview = False
        self.tracking_follow = False
        self.tracker = None  # SmoothTracker (MediaPipe + PID)
        self.invert_camera = False
        
        # Video streaming variables
        self.video_thread = None
        self.video_running = False
        self.last_frame = None
        self.detected_faces = []
        self.tracking_target = None

        
        # Performance optimization variables
        self.detection_scale_factor = 0.5
        self.detection_interval = 0.3
        self.last_detection_time = 0
        
        # Movement control variables
        self.move_speed = 7.0
        
        # UI state variables
        self.show_performance = False
        self.show_detector_info = False
        self.show_face_grid = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Camera configuration
        self.cameras = []
        self.load_cameras()
        
        # Create the UI
        self.setup_ui()
    
    def load_cameras(self):
        """Load camera configurations from JSON file"""
        try:
            # Updated path to config folder
            json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.cameras = data.get('cameras', [])
            else:
                self.cameras = [{
                    "name": "Default Camera",
                    "ip": "10.1.2.8",
                    "control_port": 52381,
                    "stream_url": "rtsp://10.1.2.8/video2",
                    "inverted": False
                }]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load camera configurations: {str(e)}")
            self.cameras = []
    
    def setup_ui(self):
        """Setup the main UI with 3-column layout"""
        # Set initial window size
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # LEFT PANEL - Camera Connection & Manual Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 0)  # Fixed width
        
        # CENTER - Video Display
        video_widget = self.create_video_panel()
        main_layout.addWidget(video_widget, 1)  # Stretch to fill
        
        # RIGHT PANEL - Tracking Controls & Settings  
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 0)  # Fixed width
        
        # Status bar at the bottom
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_video_panel(self):
        """Create the video display panel"""
        group = QGroupBox("Video Stream")
        layout = QVBoxLayout()
        
        # Video label (where frames will be displayed)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #666;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No video stream")
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label)
        group.setLayout(layout)
        return group
    
    def create_left_panel(self):
        """Create left panel with camera connection and manual controls"""
        panel = QWidget()
        panel.setMaximumWidth(280)
        panel.setMinimumWidth(280)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Camera connection at top
        layout.addWidget(self.create_camera_connection())
        
        # Manual controls below
        layout.addWidget(self.create_manual_controls())
        
        # Push everything to the top
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right panel with tracking controls and settings"""
        panel = QWidget()
        panel.setMaximumWidth(280)
        panel.setMinimumWidth(280)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tracking controls at top
        layout.addWidget(self.create_tracking_controls())
        
        # UI settings below
        layout.addWidget(self.create_ui_settings())
        
        # Push everything to the top
        layout.addStretch()
        
        return panel
    
    def create_camera_connection(self):
        """Create camera connection panel"""
        group = QGroupBox("Camera Connection")
        layout = QVBoxLayout()
        
        # Camera selection row
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        camera_names = [cam["name"] for cam in self.cameras]
        self.camera_combo.addItems(camera_names)
        self.camera_combo.currentTextChanged.connect(self.update_camera_fields)
        camera_row.addWidget(self.camera_combo, 2)
        
        manage_btn = QPushButton("Manage")
        manage_btn.clicked.connect(self.open_camera_manager)
        camera_row.addWidget(manage_btn, 1)
        
        self.invert_check = QCheckBox("Inverted")
        self.invert_check.stateChanged.connect(self.toggle_camera_inversion)
        camera_row.addWidget(self.invert_check)
        
        layout.addLayout(camera_row)
        
        # Address and port
        address_row = QHBoxLayout()
        address_row.addWidget(QLabel("Address:"))
        self.camera_ip_edit = QLineEdit()
        address_row.addWidget(self.camera_ip_edit, 3)
        
        address_row.addWidget(QLabel("Port:"))
        self.camera_port_edit = QLineEdit()
        self.camera_port_edit.setMaximumWidth(60)
        address_row.addWidget(self.camera_port_edit, 1)
        layout.addLayout(address_row)
        
        # Connect Both button (connects camera and stream together)
        self.connect_both_btn = QPushButton("▶ Connect Both")
        self.connect_both_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.connect_both_btn.clicked.connect(self.toggle_both_connections)
        layout.addWidget(self.connect_both_btn)
        
        # Individual connect buttons row
        individual_row = QHBoxLayout()
        
        # Camera connect button with status
        self.camera_connect_btn = QPushButton("Connect Camera")
        self.camera_connect_btn.clicked.connect(self.toggle_camera_connection)
        individual_row.addWidget(self.camera_connect_btn)
        
        self.camera_status_label = QLabel("●")
        self.camera_status_label.setStyleSheet("color: red; font-size: 16px;")
        individual_row.addWidget(self.camera_status_label)
        
        # Stream connect button with status  
        self.stream_connect_btn = QPushButton("Connect Stream")
        self.stream_connect_btn.clicked.connect(self.toggle_stream_connection)
        individual_row.addWidget(self.stream_connect_btn)
        
        self.stream_status_label = QLabel("●")
        self.stream_status_label.setStyleSheet("color: red; font-size: 16px;")
        individual_row.addWidget(self.stream_status_label)
        
        layout.addLayout(individual_row)
        
        # Stream URL
        layout.addWidget(QLabel("RTSP URL:"))
        self.rtsp_url_edit = QLineEdit()
        layout.addWidget(self.rtsp_url_edit)

        
        group.setLayout(layout)
        
        # Load first camera if available
        if self.cameras:
            self.camera_combo.setCurrentIndex(0)
            self.update_camera_fields()
        
        return group
    
    def create_manual_controls(self):
        """Create manual camera control panel"""
        group = QGroupBox("Manual Camera Controls")
        layout = QHBoxLayout()
        
        # Directional buttons (3x3 grid)
        dir_widget = QWidget()
        dir_grid = QGridLayout(dir_widget)
        dir_grid.setSpacing(2)
        
        # Create directional buttons
        btn_ul = QPushButton("↖")
        btn_u = QPushButton("↑")
        btn_ur = QPushButton("↗")
        btn_l = QPushButton("←")
        btn_stop = QPushButton("■")
        btn_r = QPushButton("→")
        btn_dl = QPushButton("↙")
        btn_d = QPushButton("↓")
        btn_dr = QPushButton("↘")
        
        # Set fixed size for buttons
        for btn in [btn_ul, btn_u, btn_ur, btn_l, btn_stop, btn_r, btn_dl, btn_d, btn_dr]:
            btn.setMaximumSize(35, 35)
        
        # Connect press/release events
        btn_ul.pressed.connect(self.move_left_up)
        btn_ul.released.connect(self.move_stop)
        btn_u.pressed.connect(self.move_up)
        btn_u.released.connect(self.move_stop)
        btn_ur.pressed.connect(self.move_right_up)
        btn_ur.released.connect(self.move_stop)
        btn_l.pressed.connect(self.move_left)
        btn_l.released.connect(self.move_stop)
        btn_stop.clicked.connect(self.move_stop)
        btn_r.pressed.connect(self.move_right)
        btn_r.released.connect(self.move_stop)
        btn_dl.pressed.connect(self.move_left_down)
        btn_dl.released.connect(self.move_stop)
        btn_d.pressed.connect(self.move_down)
        btn_d.released.connect(self.move_stop)
        btn_dr.pressed.connect(self.move_right_down)
        btn_dr.released.connect(self.move_stop)
        
        # Add to grid
        dir_grid.addWidget(btn_ul, 0, 0)
        dir_grid.addWidget(btn_u, 0, 1)
        dir_grid.addWidget(btn_ur, 0, 2)
        dir_grid.addWidget(btn_l, 1, 0)
        dir_grid.addWidget(btn_stop, 1, 1)
        dir_grid.addWidget(btn_r, 1, 2)
        dir_grid.addWidget(btn_dl, 2, 0)
        dir_grid.addWidget(btn_d, 2, 1)
        dir_grid.addWidget(btn_dr, 2, 2)
        
        layout.addWidget(dir_widget)
        
        # Zoom and home buttons
        zoom_widget = QWidget()
        zoom_layout = QVBoxLayout(zoom_widget)
        
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.pressed.connect(self.zoom_in)
        zoom_in_btn.released.connect(self.zoom_stop)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.pressed.connect(self.zoom_out)
        zoom_out_btn.released.connect(self.zoom_stop)
        zoom_layout.addWidget(zoom_out_btn)
        
        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self.move_home)
        zoom_layout.addWidget(home_btn)
        
        layout.addWidget(zoom_widget)
        
        group.setLayout(layout)
        return group
    
    def create_tracking_controls(self):
        """Create tracking control panel"""
        group = QGroupBox("Tracking Controls")
        layout = QVBoxLayout()
        
        # Preview and follow checkboxes
        self.preview_check = QCheckBox("Preview Tracking")
        self.preview_check.stateChanged.connect(self.toggle_tracking_preview)
        layout.addWidget(self.preview_check)
        
        self.follow_check = QCheckBox("Start Tracking Camera Control")
        self.follow_check.stateChanged.connect(self.toggle_tracking_follow)
        layout.addWidget(self.follow_check)
        
        # Center face button
        center_btn = QPushButton("Center Face")
        center_btn.clicked.connect(self.center_face)
        layout.addWidget(center_btn)
        
        # Camera speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Camera Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(70)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("7.0")
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
        
        # Tracking options
        options_group = QGroupBox("--- Tracking Options ---")
        options_layout = QVBoxLayout()
        
        # MediaPipe status indicator
        mediapipe_status = QLabel("✓ MediaPipe + PID Tracking Active" if MEDIAPIPE_AVAILABLE else "⚠ MediaPipe not available")
        mediapipe_status.setStyleSheet("color: green; font-weight: bold;" if MEDIAPIPE_AVAILABLE else "color: red;")
        options_layout.addWidget(mediapipe_status)
        
        # Body tracking checkbox
        self.body_check = QCheckBox("Use Body Tracking")
        self.body_check.setChecked(True)
        self.body_check.setToolTip("Fall back to body detection when face not visible")
        self.body_check.stateChanged.connect(self.toggle_body_fallback)
        options_layout.addWidget(self.body_check)
        
        # Require face & body
        self.require_face_body_check = QCheckBox("Require Face & Body")
        self.require_face_body_check.setToolTip("Only move camera when both face and body are detected")
        self.require_face_body_check.stateChanged.connect(self.toggle_require_face_body)
        options_layout.addWidget(self.require_face_body_check)
        
        # Tracking persistence
        self.persistence_check = QCheckBox("Tracking Persistence")
        self.persistence_check.setChecked(True)
        self.persistence_check.setToolTip("Remember last position when target is temporarily lost")
        self.persistence_check.stateChanged.connect(self.toggle_persistence)
        options_layout.addWidget(self.persistence_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Detection interval slider
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Detection Interval:"))
        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setMinimum(1)  # 0.1 second
        self.interval_slider.setMaximum(10)  # 1.0 second
        self.interval_slider.setValue(3)  # 0.3 second default
        self.interval_slider.valueChanged.connect(self.update_interval)
        interval_layout.addWidget(self.interval_slider)
        self.interval_label = QLabel("0.3 Sec")
        interval_layout.addWidget(self.interval_label)
        layout.addLayout(interval_layout)
        
        group.setLayout(layout)
        return group

    
    def create_ui_settings(self):
        """Create UI settings panel"""
        group = QGroupBox("UI Settings")
        layout = QVBoxLayout()
        
        self.performance_check = QCheckBox("Show Performance Info")
        self.performance_check.stateChanged.connect(self.toggle_performance_display)
        layout.addWidget(self.performance_check)
        
        self.detector_info_check = QCheckBox("Show on screen detector info")
        self.detector_info_check.stateChanged.connect(self.toggle_detector_info)
        layout.addWidget(self.detector_info_check)
        
        self.face_grid_check = QCheckBox("Show face square grid")
        self.face_grid_check.stateChanged.connect(self.toggle_face_grid)
        layout.addWidget(self.face_grid_check)
        
        group.setLayout(layout)
        return group
    
    # ============ Camera Connection Methods ============
    
    def update_camera_fields(self):
        """Update connection fields based on selected camera"""
        selected_camera = self.camera_combo.currentText()
        
        for camera in self.cameras:
            if camera["name"] == selected_camera:
                self.camera_ip_edit.setText(camera["ip"])
                self.camera_port_edit.setText(str(camera["control_port"]))
                self.rtsp_url_edit.setText(camera["stream_url"])
                
                self.invert_camera = camera.get("inverted", False)
                self.invert_check.setChecked(self.invert_camera)
                break
        
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Selected camera: {selected_camera}")
    
    def toggle_both_connections(self):
        """Toggle both camera and stream connections"""
        # If either is disconnected, try to connect both
        if not self.connected or not self.video_running:
            # Connect Camera if not connected
            if not self.connected:
                self.toggle_camera_connection()
            
            # Connect Stream if not connected
            if not self.video_running:
                self.toggle_stream_connection()
                
            # Update button text based on result
            if self.connected and self.video_running:
                self.connect_both_btn.setText("⏹ Disconnect Both")
                self.connect_both_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        
        # If both are connected (or at least valid enough to disconnect), disconnect both
        else:
            if self.connected:
                self.toggle_camera_connection()
            
            if self.video_running:
                self.toggle_stream_connection()
            
            self.connect_both_btn.setText("▶ Connect Both")
            self.connect_both_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")

    def toggle_camera_connection(self):
        """Connect or disconnect camera"""
        if self.connected:
            # Disconnect
            if self.camera:
                self.camera.pantilt(pan_speed=0, tilt_speed=0)
                self.camera.close_connection()
            self.camera = None
            self.connected = False
            # Clean up tracker
            if self.tracker:
                self.tracker.close()
            self.tracker = None
            self.camera_connect_btn.setText("Connect Camera")
            self.camera_status_label.setStyleSheet("color: red; font-size: 16px;")
            self.status_bar.showMessage("Camera disconnected")
            
            # Update connect both button state
            if not self.video_running:
                self.connect_both_btn.setText("▶ Connect Both")
                self.connect_both_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        else:
            # Connect
            ip = self.camera_ip_edit.text()
            port_str = self.camera_port_edit.text()
            
            if not ip or not port_str:
                QMessageBox.warning(self, "Error", "Please enter Camera IP and Port.")
                return
            
            try:
                port = int(port_str)
                self.camera = Camera(ip, port)
                self.connected = True
                
                # Initialize SmoothTracker (MediaPipe + PID)
                self.tracker = SmoothTracker(
                    self.camera,
                    move_speed=self.move_speed,
                    invert_camera=self.invert_camera
                )
                self.tracker.use_body_fallback = self.body_check.isChecked()
                self.tracker.use_persistence = self.persistence_check.isChecked()
                self.tracker.require_face_and_body = self.require_face_body_check.isChecked()
                
                self.camera_connect_btn.setText("Disconnect Camera")
                self.camera_status_label.setStyleSheet("color: green; font-size: 16px;")
                self.status_bar.showMessage(f"Connected to camera at {ip}:{port} - MediaPipe+PID tracking")
                
                # Update connect both button state
                if self.video_running:
                    self.connect_both_btn.setText("⏹ Disconnect Both")
                    self.connect_both_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", 
                                   f"Failed to connect to camera: {str(e)}")
                self.camera = None
                self.connected = False
                self.tracker = None
    


    def toggle_stream_connection(self):
        """Connect or disconnect video stream"""
        if self.video_running:
            self.stop_video_stream()
        else:
            self.start_video_stream()
    
    def start_video_stream(self):
        """Start the video stream"""
        rtsp_url = self.rtsp_url_edit.text()
        if not rtsp_url:
            QMessageBox.warning(self, "Error", "Please enter a valid RTSP URL")
            return
        
        try:
            self.video_thread = VideoThread(rtsp_url)
            self.video_thread.frame_ready.connect(self.process_frame)
            self.video_thread.error_occurred.connect(self.handle_video_error)
            self.video_thread.start()
            
            self.video_running = True
            self.stream_connect_btn.setText("Disconnect Stream")
            self.stream_status_label.setStyleSheet("color: green; font-size: 16px;")
            self.status_bar.showMessage(f"Connected to stream: {rtsp_url}")
            
            # Update connect both button state
            if self.connected:
                self.connect_both_btn.setText("⏹ Disconnect Both")
                self.connect_both_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
            
            # Reset FPS counter
            self.frame_count = 0
            self.fps_start_time = time.time()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start video stream: {str(e)}")
    
    def stop_video_stream(self):
        """Stop the video stream"""
        self.video_running = False
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        self.stream_connect_btn.setText("Connect Stream")
        self.stream_status_label.setStyleSheet("color: red; font-size: 16px;")
        self.status_bar.showMessage("Video stream disconnected")
        
        # Update connect both button state
        if not self.connected:
            self.connect_both_btn.setText("▶ Connect Both")
            self.connect_both_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.video_label.setText("No video stream")
    
    def handle_video_error(self, error_msg):
        """Handle errors from the video thread"""
        QMessageBox.critical(self, "Video Error", error_msg)
        self.stop_video_stream()
    
    # ============ Video Processing Methods ============
    
    def process_frame(self, frame):
        """Process each video frame (called from video thread signal)"""
        if not self.video_running:
            return
        
        # Store the last frame for tracking
        self.last_frame = frame.copy()
        
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_time = time.time()
        
        # Face detection and tracking logic
        if self.tracker and (self.tracking_preview or self.tracking_follow):
            # Perform detection at intervals
            if current_time - self.last_detection_time >= self.detection_interval:
                # MediaPipe handles scaling efficiently, use full resolution
                faces, target = self.tracker.detect_faces(frame)  # Use BGR frame
                self.detected_faces = faces if faces else []
                self.tracking_target = target
                self.last_detection_time = current_time
            
            # Draw tracking visualization if preview is enabled
            if self.tracking_preview:
                frame_rgb = self.draw_tracking_overlay(frame_rgb)
            
            # Track the face if follow is enabled
            if self.tracking_follow and self.tracking_target:
                self.tracker.track_face(self.last_frame)
        
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time >= 1.0:
            self.current_fps = self.frame_count / elapsed_time
            
            # Update status bar with performance info if enabled
            if self.show_performance:
                status_text = f"FPS: {self.current_fps:.1f}"
                if self.tracking_preview or self.tracking_follow:
                    status_text += f" | Tracking: {'ON' if self.tracking_follow else 'Preview'}"
                    status_text += f" | Interval: {self.detection_interval:.1f}s"
                self.status_bar.showMessage(status_text)
            
            self.frame_count = 0
            self.fps_start_time = time.time()
        
        # Display the frame
        self.display_frame(frame_rgb)

    
    def draw_tracking_overlay(self, frame):
        """Draw tracking visualization on the frame"""
        h_frame, w_frame = frame.shape[:2]
        
        # Draw rule of thirds grid if enabled
        if self.show_face_grid:
            third_h1 = h_frame // 3
            third_h2 = 2 * h_frame // 3
            third_w1 = w_frame // 3
            third_w2 = 2 * w_frame // 3
            grid_color = (150, 150, 150)
            cv2.line(frame, (0, third_h1), (w_frame, third_h1), grid_color, 1)
            cv2.line(frame, (0, third_h2), (w_frame, third_h2), grid_color, 1)
            cv2.line(frame, (third_w1, 0), (third_w1, h_frame), grid_color, 1)
            cv2.line(frame, (third_w2, 0), (third_w2, h_frame), grid_color, 1)
        
        # Draw dead zone / center box
        if self.tracker and self.tracker.center_box:
            x, y, w, h = self.tracker.center_box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "Dead Zone", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw detected faces/bodies
        for (x, y, w_f, h_f) in self.detected_faces:
            if self.tracker:
                # Green for face, blue for body
                is_main_face = (self.tracker.detected_face and 
                               (x, y, w_f, h_f) == self.tracker.detected_face)
                is_body = (self.tracker.detected_body and 
                          (x, y, w_f, h_f) == self.tracker.detected_body and
                          not is_main_face)
                
                if is_main_face:
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)  # Green
                elif is_body:
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (255, 0, 0), 2)  # Blue
                else:
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 128, 255), 2)  # Orange
            else:
                cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)  # Default green
        
        # Draw tracking status
        status_y = 30
        if self.tracking_follow and self.tracking_target:
            status_text = "TRACKING"
        else:
            status_text = "PREVIEW"
        cv2.putText(frame, status_text, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_y += 30
        
        # Draw on-screen detector info if enabled
        if self.show_detector_info and self.tracker:
            # Face confidence
            if hasattr(self.tracker, 'face_confidence'):
                conf_text = f"Face Confidence: {self.tracker.face_confidence:.2f}"
                will_move = self.tracker.face_confidence >= 0.3
                conf_color = (0, 255, 0) if will_move else (0, 0, 255)
                cv2.putText(frame, conf_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
                status_y += 25
            
            # Camera lock status
            if hasattr(self.tracker, 'require_face_and_body') and self.tracker.require_face_and_body:
                if self.tracker.detected_face and self.tracker.detected_body:
                    camera_lock_status = "CAMERA UNLOCKED (Face & Body)"
                    lock_color = (0, 255, 0)
                else:
                    camera_lock_status = "CAMERA LOCKED (Need Face & Body)"
                    lock_color = (0, 0, 255)
            elif hasattr(self.tracker, 'face_confidence') and self.tracker.face_confidence >= 0.3:
                camera_lock_status = "CAMERA UNLOCKED"
                lock_color = (0, 255, 0)
            else:
                camera_lock_status = "CAMERA LOCKED"
                lock_color = (0, 0, 255)
            
            cv2.putText(frame, camera_lock_status, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 2)
            status_y += 25
            
            # Show detector type
            cv2.putText(frame, "MediaPipe + PID", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            status_y += 20
            
            # Show PID output if moving
            if hasattr(self.tracker, 'camera_moving') and self.tracker.camera_moving:
                pid_text = f"PID: Pan={self.tracker.last_pan_speed:+d} Tilt={self.tracker.last_tilt_speed:+d}"
                cv2.putText(frame, pid_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
                status_y += 20
        
        return frame


    
    def display_frame(self, frame_rgb):
        """Convert frame to QPixmap and display it"""
        # Get frame dimensions
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # Convert to QImage
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Display
        self.video_label.setPixmap(scaled_pixmap)
    
    # ============ Tracking Control Methods ============
    
    def toggle_tracking_preview(self):
        """Toggle tracking preview overlay"""
        self.tracking_preview = self.preview_check.isChecked()
        
        if self.tracking_preview and not self.video_running:
            QMessageBox.warning(self, "Warning", 
                              "Video stream not connected. Connect stream for tracking preview.")
            self.preview_check.setChecked(False)
            self.tracking_preview = False
            return
        
        status = "enabled" if self.tracking_preview else "disabled"
        self.status_bar.showMessage(f"Face tracking preview {status}")
    
    def toggle_tracking_follow(self):
        """Toggle active tracking that controls the camera"""
        self.tracking_follow = self.follow_check.isChecked()
        
        if not self.tracking_follow and self.connected:
            # Stop camera movement
            self.camera.pantilt(pan_speed=0, tilt_speed=0)
            self.status_bar.showMessage("Face tracking disabled, camera stopped")
            return
        
        if self.tracking_follow:
            if not self.connected:
                QMessageBox.warning(self, "Warning", 
                                  "Camera not connected. Connect camera for tracking follow.")
                self.follow_check.setChecked(False)
                self.tracking_follow = False
                return
            
            if not self.video_running:
                QMessageBox.warning(self, "Warning", 
                                  "Video stream not connected. Connect stream for tracking follow.")
                self.follow_check.setChecked(False)
                self.tracking_follow = False
                return
            
            self.status_bar.showMessage("Face tracking follow enabled")
    
    def center_face(self):
        """Center face to portrait position with continuous feedback loop"""
        # Check prerequisites
        if not self.connected or not self.camera:
            QMessageBox.warning(self, "Warning", "Camera not connected.")
            return
            
        if not self.video_running or self.last_frame is None:
            QMessageBox.warning(self, "Warning", "Video stream not connected.")
            return
        
        # Stop any existing centering
        if hasattr(self, 'centering_timer') and self.centering_timer.isActive():
            self.centering_timer.stop()
        
        # Check if face is detected before starting
        tracker = self.tracker if self.tracker else SmoothTracker(self.camera, move_speed=self.move_speed, invert_camera=self.invert_camera)
        faces, target = tracker.detect_faces(self.last_frame)
        
        if target is None:
            QMessageBox.warning(self, "Warning", "No face detected. Please ensure a face is visible.")
            if tracker != self.tracker:
                tracker.close()
            return
            
        self.status_bar.showMessage("Centering face to portrait position...")
        
        # State for centering loop
        self.centering_start_time = time.time()
        self.centering_max_duration = 5.0  # Max 5 seconds to center
        
        # Reset zero-crossing flags
        self.centering_crossed_x = False
        self.centering_crossed_y = False
        self.initial_error_x = None
        self.initial_error_y = None
        
        # Create temp tracker if needed and store it
        if self.tracker is None:
            self.temp_centering_tracker = tracker  # Reuse the one we created for detection
        else:
            self.temp_centering_tracker = None  # Use self.tracker
            
        # Start the control loop
        self.centering_timer = QTimer()
        self.centering_timer.timeout.connect(self._centering_step)
        self.centering_timer.start(150) # Update every 150ms

    def _centering_step(self):
        """Single step of the centering control loop"""
        # check timeout
        if time.time() - self.centering_start_time > self.centering_max_duration:
            self._stop_centering("Timeout")
            return
            
        # Get tracker
        tracker = self.tracker if self.tracker else self.temp_centering_tracker
        if not tracker:
            self._stop_centering("Tracker error")
            return
            
        # Get frame
        if self.last_frame is None:
            return
            
        # Detect
        faces, target = tracker.detect_faces(self.last_frame)
        
        if target is None:
            # Don't stop immediately if target lost, maybe just blinked
            return
            
        # Calculate error
        h, w = self.last_frame.shape[:2]
        x, y, fw, fh = target
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        
        # Portrait position: center horizontally, 1/3 from top vertically
        target_x = w / 2
        target_y = h / 3  # Portrait position - face should sit in upper third
        
        error_x = (face_center_x - target_x) / (w / 2)
        error_y = (face_center_y - target_y) / (h / 2)
        
        # Initialize initial errors on first successful detection
        if self.initial_error_x is None:
            self.initial_error_x = error_x
            self.initial_error_y = error_y
            
        # Zero-Crossing Check
        # If sign of current error is different from initial error, we crossed 0. Stop that axis.
        if not self.centering_crossed_x and (self.initial_error_x is not None) and (error_x * self.initial_error_x <= 0):
            self.centering_crossed_x = True
            
        if not self.centering_crossed_y and (self.initial_error_y is not None) and (error_y * self.initial_error_y <= 0):
            self.centering_crossed_y = True
            
        # Dead zone (very tight)
        dead_zone = 0.01
        
        # Check if centered or crossed
        x_done = self.centering_crossed_x or abs(error_x) < dead_zone
        y_done = self.centering_crossed_y or abs(error_y) < dead_zone
        
        if x_done and y_done:
            self._stop_centering("Centered!")
            return
        
        # Calculate speeds
        max_speed = int(self.move_speed)
        
        def calc_speed(error, mult):
            val = int(error * max_speed * mult)
            # Ensure min speed if outside deadzone
            if val == 0 and abs(error) > dead_zone:
                return 1 if error > 0 else -1
            return val
            
        pan_speed = 0
        if not x_done:
            # Lower multiplier for smoother approach
            pan_speed = calc_speed(error_x, 1.5)
            pan_speed = max(-max_speed, min(max_speed, pan_speed))
            
        tilt_speed = 0
        if not y_done:
            # Lower multiplier for smoother approach
            tilt_speed = calc_speed(-error_y, 1.0)
            tilt_speed = max(-max_speed, min(max_speed, tilt_speed))
            
            if self.invert_camera:
                tilt_speed = -tilt_speed
        
        # Apply movement
        self.camera.pantilt(pan_speed=pan_speed, tilt_speed=tilt_speed)
        
        status = f"Centering... ({abs(error_x):.2f}, {abs(error_y):.2f})"
        if self.centering_crossed_x: status += " X-Crossed"
        if self.centering_crossed_y: status += " Y-Crossed"
        self.status_bar.showMessage(status)
    
    def _stop_centering(self, reason=""):
        """Stop the centering loop"""
        if hasattr(self, 'centering_timer') and self.centering_timer.isActive():
            self.centering_timer.stop()
            
        if self.connected and self.camera:
            self.camera.pantilt(pan_speed=0, tilt_speed=0)
            
        if hasattr(self, 'temp_centering_tracker') and self.temp_centering_tracker:
            self.temp_centering_tracker.close()
            self.temp_centering_tracker = None
            
        self.status_bar.showMessage(f"Centering complete: {reason}")
        
    def toggle_body_fallback(self):
        """Toggle body detection fallback"""
        checked = self.body_check.isChecked()
        if self.tracker:
            self.tracker.use_body_fallback = checked
        status = "enabled" if checked else "disabled"
        self.status_bar.showMessage(f"Body tracking {status}")
    
    def toggle_require_face_body(self):
        """Toggle requirement for both face and body"""
        checked = self.require_face_body_check.isChecked()
        if self.tracker:
            self.tracker.require_face_and_body = checked
        status = "enabled" if checked else "disabled"
        self.status_bar.showMessage(f"Require Face & Body {status}")
    
    def toggle_persistence(self):
        """Toggle tracking persistence"""
        checked = self.persistence_check.isChecked()
        if self.tracker:
            self.tracker.use_persistence = checked
        status = "enabled" if checked else "disabled"
        self.status_bar.showMessage(f"Tracking persistence {status}")
    
    def update_speed(self):
        """Update camera speed from slider"""
        value = self.speed_slider.value() / 10.0
        self.move_speed = value
        self.speed_label.setText(f"{value:.1f}")
        if self.tracker:
            self.tracker.update_settings(move_speed=self.move_speed)
    
    def update_interval(self):
        """Update detection interval from slider"""
        value = self.interval_slider.value() / 10.0
        self.detection_interval = value
        self.interval_label.setText(f"{value:.1f} Sec")
    
    def toggle_camera_inversion(self):
        """Toggle camera inversion setting"""
        self.invert_camera = self.invert_check.isChecked()
        if self.tracker:
            self.tracker.update_settings(invert_camera=self.invert_camera)
        status = "enabled" if self.invert_camera else "disabled"
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Camera inversion {status}")

    
    def toggle_performance_display(self):
        """Toggle performance info display"""
        self.show_performance = self.performance_check.isChecked()
        status = "enabled" if self.show_performance else "disabled"
        if not self.show_performance:
            self.status_bar.showMessage(f"Performance display {status}")
    
    def toggle_detector_info(self):
        """Toggle on-screen detector info"""
        self.show_detector_info = self.detector_info_check.isChecked()
        status = "enabled" if self.show_detector_info else "disabled"
        self.status_bar.showMessage(f"On-screen detector info {status}")
    
    def toggle_face_grid(self):
        """Toggle face grid overlay"""
        self.show_face_grid = self.face_grid_check.isChecked()
        status = "enabled" if self.show_face_grid else "disabled"
        self.status_bar.showMessage(f"Face square grid {status}")
    
    # ============ Camera Movement Methods ============
    
    def check_connection(self):
        """Check if camera is connected before movement"""
        if not self.connected or not self.camera:
            QMessageBox.warning(self, "Not Connected", "Please connect to a camera first.")
            return False
        return True
    
    def move_left(self):
        if self.check_connection():
            self.camera.pantilt(pan_speed=-int(self.move_speed), tilt_speed=0)
    
    def move_right(self):
        if self.check_connection():
            self.camera.pantilt(pan_speed=int(self.move_speed), tilt_speed=0)
    
    def move_up(self):
        if self.check_connection():
            tilt = -int(self.move_speed) if self.invert_camera else int(self.move_speed)
            self.camera.pantilt(pan_speed=0, tilt_speed=tilt)
    
    def move_down(self):
        if self.check_connection():
            tilt = int(self.move_speed) if self.invert_camera else -int(self.move_speed)
            self.camera.pantilt(pan_speed=0, tilt_speed=tilt)
    
    def move_left_up(self):
        if self.check_connection():
            tilt = -int(self.move_speed) if self.invert_camera else int(self.move_speed)
            self.camera.pantilt(pan_speed=-int(self.move_speed), tilt_speed=tilt)
    
    def move_right_up(self):
        if self.check_connection():
            tilt = -int(self.move_speed) if self.invert_camera else int(self.move_speed)
            self.camera.pantilt(pan_speed=int(self.move_speed), tilt_speed=tilt)
    
    def move_left_down(self):
        if self.check_connection():
            tilt = int(self.move_speed) if self.invert_camera else -int(self.move_speed)
            self.camera.pantilt(pan_speed=-int(self.move_speed), tilt_speed=tilt)
    
    def move_right_down(self):
        if self.check_connection():
            tilt = int(self.move_speed) if self.invert_camera else -int(self.move_speed)
            self.camera.pantilt(pan_speed=int(self.move_speed), tilt_speed=tilt)
    
    def move_stop(self):
        if self.check_connection():
            self.camera.pantilt(pan_speed=0, tilt_speed=0)
    
    def move_home(self):
        if self.check_connection():
            self.camera.pantilt_home()
    
    def zoom_in(self):
        if self.check_connection():
            self.camera.zoom(speed=3)
    
    def zoom_out(self):
        if self.check_connection():
            self.camera.zoom(speed=-3)
    
    def zoom_stop(self):
        if self.check_connection():
            self.camera.zoom(speed=0)
    
    # ============ Camera Manager Methods ============
    
    def open_camera_manager(self):
        """Open camera manager dialog"""
        from camera_manager import CameraManager, CameraEditorDialog
        
        # Reload cameras in case they were changed externally
        self.camera_manager = CameraManager()
        
        dialog = CameraEditorDialog(self, self.camera_manager)
        if dialog.exec():  # Modal dialog
            # Reload cameras in the main app
            self.load_cameras()
            
            # Update the combo box with new camera list
            self.camera_combo.clear()
            camera_names = [cam["name"] for cam in self.cameras]
            self.camera_combo.addItems(camera_names)
            
            # Select first camera if available
            if self.cameras:
                self.camera_combo.setCurrentIndex(0)
                self.update_camera_fields()
            
            self.status_bar.showMessage("Camera list updated")

    
    # ============ Cleanup Methods ============
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop video stream
        if self.video_running:
            self.stop_video_stream()
        
        # Clean up tracker
        if self.tracker:
            self.tracker.close()
            self.tracker = None
        
        # Disconnect camera
        if self.connected and self.camera:
            self.camera.close_connection()
        
        event.accept()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTrackingApp()
    window.show()
    sys.exit(app.exec())
