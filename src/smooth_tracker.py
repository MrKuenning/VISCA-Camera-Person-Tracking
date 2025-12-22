"""
Smooth Tracker Module - MediaPipe + PID Controller

This module provides smooth face/body tracking using:
- MediaPipe Face Detection for accurate, temporally-stable detection
- EMA (Exponential Moving Average) for position smoothing
- PID Controller for smooth camera pan/tilt control

Designed to eliminate jitter and provide smooth PTZ camera tracking.
"""

import cv2
import time
import numpy as np

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Run: pip install mediapipe")

# Try to import simple-pid
try:
    from simple_pid import PID
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False
    print("Warning: simple-pid not installed. Run: pip install simple-pid")


class PositionSmoother:
    """
    Exponential Moving Average (EMA) filter for smoothing detected positions.
    
    This reduces noise from frame-to-frame detection variations,
    providing a more stable position signal for the PID controller.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize the position smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing, higher = more responsive.
                   Default 0.3 provides good balance between smoothness and responsiveness.
        """
        self.alpha = alpha
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothed_w = None
        self.smoothed_h = None
    
    def update(self, x: float, y: float, w: float = None, h: float = None):
        """
        Update the smoothed position with a new detection.
        
        Args:
            x, y: New detected position (center of face/body)
            w, h: Optional width/height of bounding box
            
        Returns:
            Tuple of smoothed (x, y) or (x, y, w, h) if dimensions provided
        """
        if self.smoothed_x is None:
            # First detection - initialize with raw values
            self.smoothed_x = x
            self.smoothed_y = y
            self.smoothed_w = w
            self.smoothed_h = h
        else:
            # Apply EMA smoothing
            self.smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothed_y
            
            if w is not None and h is not None:
                if self.smoothed_w is not None:
                    self.smoothed_w = self.alpha * w + (1 - self.alpha) * self.smoothed_w
                    self.smoothed_h = self.alpha * h + (1 - self.alpha) * self.smoothed_h
                else:
                    self.smoothed_w = w
                    self.smoothed_h = h
        
        if w is not None and h is not None:
            return self.smoothed_x, self.smoothed_y, self.smoothed_w, self.smoothed_h
        return self.smoothed_x, self.smoothed_y
    
    def reset(self):
        """Reset the smoother state (call when target is lost)"""
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothed_w = None
        self.smoothed_h = None
    
    def get_position(self):
        """Get the current smoothed position"""
        return self.smoothed_x, self.smoothed_y


class SimplePIDController:
    """
    Simple PID Controller implementation for camera control.
    
    Used as fallback if simple-pid is not installed.
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.05, kd: float = 0.15,
                 output_limits: tuple = (-10, 10)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain - immediate response to error
            ki: Integral gain - eliminates steady-state error
            kd: Derivative gain - dampens oscillations
            output_limits: (min, max) output clamp values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = 0
        self.integral = 0
        self.last_time = None
        
    def __call__(self, error: float) -> float:
        """
        Calculate PID output for given error.
        
        Args:
            error: Current error (difference from setpoint)
            
        Returns:
            Control output (camera speed)
        """
        current_time = time.time()
        
        if self.last_time is None:
            dt = 0.033  # Assume ~30fps
        else:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)  # Prevent division by zero
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        # Clamp integral to prevent windup
        max_integral = (self.output_limits[1] - self.output_limits[0]) / (2 * self.ki) if self.ki > 0 else 100
        self.integral = max(-max_integral, min(max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        
        # Combined output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Store for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset the PID controller state"""
        self.last_error = 0
        self.integral = 0
        self.last_time = None


class SmoothTracker:
    """
    Main smooth tracking class using MediaPipe detection and PID control.
    
    This replaces the legacy Haar/DLib based tracking with a more robust
    and smooth tracking system.
    """
    
    def __init__(self, camera, move_speed: float = 5, invert_camera: bool = False,
                 pid_kp: float = 0.5, pid_ki: float = 0.05, pid_kd: float = 0.15,
                 ema_alpha: float = 0.3):
        """
        Initialize the smooth tracker.
        
        Args:
            camera: VISCA camera instance for sending commands
            move_speed: Maximum movement speed (scales PID output)
            invert_camera: Whether camera is inverted/upside-down
            pid_kp, pid_ki, pid_kd: PID controller gains
            ema_alpha: EMA smoothing factor (0-1)
        """
        self.camera = camera
        self.move_speed = move_speed
        self.invert_camera = invert_camera
        
        # Detection state
        self.tracking_target = None
        self.detected_face = None
        self.detected_body = None
        self.face_confidence = 0.0
        self.center_box = None
        
        # Tracking mode flags
        self.use_body_fallback = True
        self.use_persistence = True
        self.require_face_and_body = False
        
        # Persistence tracking
        self.target_lost_time = None
        self.max_target_lost_time = 2.0
        self.last_valid_target = None
        
        # Position smoother
        self.position_smoother = PositionSmoother(alpha=ema_alpha)
        
        # PID Controllers for pan and tilt
        # Scale output limits by move_speed
        output_limit = move_speed
        
        if PID_AVAILABLE:
            self.pan_pid = PID(pid_kp, pid_ki, pid_kd, setpoint=0)
            self.pan_pid.output_limits = (-output_limit, output_limit)
            self.pan_pid.sample_time = None  # Update on every call
            
            self.tilt_pid = PID(pid_kp, pid_ki, pid_kd, setpoint=0)
            self.tilt_pid.output_limits = (-output_limit, output_limit)
            self.tilt_pid.sample_time = None
        else:
            self.pan_pid = SimplePIDController(pid_kp, pid_ki, pid_kd, 
                                               output_limits=(-output_limit, output_limit))
            self.tilt_pid = SimplePIDController(pid_kp, pid_ki, pid_kd,
                                                output_limits=(-output_limit, output_limit))
        
        # Movement state
        self.last_pan_speed = 0
        self.last_tilt_speed = 0
        self.camera_moving = False
        
        # Initialize MediaPipe detectors
        self.mp_face_detection = None
        self.mp_pose = None
        self._initialize_mediapipe()
        
        # Fallback OpenCV cascade for body detection
        try:
            self.upper_body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            )
        except Exception:
            self.upper_body_cascade = None
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe detection models"""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        try:
            # Face detection - Model 1 for full range (stage/distance tracking)
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0 = short range, 1 = full range (2+ meters)
                min_detection_confidence=0.5
            )
            
            # Pose detection for body tracking
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0 = lite, 1 = full, 2 = heavy
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            self.mp_face_detection = None
            self.mp_pose = None
    
    def update_settings(self, move_speed: float = None, invert_camera: bool = None,
                       pid_kp: float = None, pid_ki: float = None, pid_kd: float = None,
                       ema_alpha: float = None):
        """
        Update tracker settings dynamically.
        
        Args:
            move_speed: Maximum movement speed
            invert_camera: Camera inversion setting
            pid_kp, pid_ki, pid_kd: PID gains (will reset PID state)
            ema_alpha: EMA smoothing factor (will reset smoother)
        """
        if move_speed is not None:
            self.move_speed = move_speed
            # Update PID output limits
            if PID_AVAILABLE:
                self.pan_pid.output_limits = (-move_speed, move_speed)
                self.tilt_pid.output_limits = (-move_speed, move_speed)
            else:
                self.pan_pid.output_limits = (-move_speed, move_speed)
                self.tilt_pid.output_limits = (-move_speed, move_speed)
        
        if invert_camera is not None:
            self.invert_camera = invert_camera
        
        if any(p is not None for p in [pid_kp, pid_ki, pid_kd]):
            # Recreate PID controllers with new gains
            kp = pid_kp if pid_kp is not None else self.pan_pid.Kp if PID_AVAILABLE else self.pan_pid.kp
            ki = pid_ki if pid_ki is not None else self.pan_pid.Ki if PID_AVAILABLE else self.pan_pid.ki
            kd = pid_kd if pid_kd is not None else self.pan_pid.Kd if PID_AVAILABLE else self.pan_pid.kd
            
            output_limit = self.move_speed
            if PID_AVAILABLE:
                self.pan_pid = PID(kp, ki, kd, setpoint=0)
                self.pan_pid.output_limits = (-output_limit, output_limit)
                self.tilt_pid = PID(kp, ki, kd, setpoint=0)
                self.tilt_pid.output_limits = (-output_limit, output_limit)
            else:
                self.pan_pid = SimplePIDController(kp, ki, kd, 
                                                   output_limits=(-output_limit, output_limit))
                self.tilt_pid = SimplePIDController(kp, ki, kd,
                                                    output_limits=(-output_limit, output_limit))
        
        if ema_alpha is not None:
            self.position_smoother = PositionSmoother(alpha=ema_alpha)
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame using MediaPipe.
        
        Args:
            frame: BGR or RGB frame to process
            
        Returns:
            Tuple of (all_detections, tracking_target)
        """
        faces = []
        bodies = []
        largest_face = None
        largest_body = None
        max_face_area = 0
        
        self.face_confidence = 0.0
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # MediaPipe Face Detection
        if MEDIAPIPE_AVAILABLE and self.mp_face_detection is not None:
            try:
                results = self.mp_face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        fw = int(bbox.width * w)
                        fh = int(bbox.height * h)
                        
                        # Ensure coordinates are valid
                        x = max(0, x)
                        y = max(0, y)
                        fw = min(fw, w - x)
                        fh = min(fh, h - y)
                        
                        if fw > 0 and fh > 0:
                            faces.append((x, y, fw, fh))
                            area = fw * fh
                            
                            if area > max_face_area:
                                max_face_area = area
                                largest_face = (x, y, fw, fh)
                                # MediaPipe provides confidence score
                                self.face_confidence = detection.score[0] if detection.score else 0.8
                
            except Exception as e:
                print(f"MediaPipe face detection error: {e}")
        
        # Body detection using MediaPipe Pose
        if self.use_body_fallback and MEDIAPIPE_AVAILABLE and self.mp_pose is not None:
            try:
                pose_results = self.mp_pose.process(frame_rgb)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    # Calculate body bounding box from pose landmarks
                    # Use shoulder and hip landmarks to estimate body bounds
                    xs = []
                    ys = []
                    
                    # Key landmarks for body bounds
                    body_indices = [
                        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                        mp.solutions.pose.PoseLandmark.LEFT_HIP,
                        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                        mp.solutions.pose.PoseLandmark.NOSE,
                    ]
                    
                    for idx in body_indices:
                        lm = landmarks[idx.value]
                        if lm.visibility > 0.5:
                            xs.append(lm.x * w)
                            ys.append(lm.y * h)
                    
                    if len(xs) >= 3:
                        x_min = int(max(0, min(xs) - 50))
                        y_min = int(max(0, min(ys) - 30))
                        x_max = int(min(w, max(xs) + 50))
                        y_max = int(min(h, max(ys) + 30))
                        
                        bw = x_max - x_min
                        bh = y_max - y_min
                        
                        if bw > 30 and bh > 30:
                            largest_body = (x_min, y_min, bw, bh)
                            bodies.append(largest_body)
                            
            except Exception as e:
                print(f"MediaPipe pose detection error: {e}")
        
        # Fallback to OpenCV for body if MediaPipe pose not available
        if largest_body is None and self.use_body_fallback and self.upper_body_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                upper_bodies = self.upper_body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
                )
                
                max_body_area = 0
                for (x, y, bw, bh) in upper_bodies:
                    area = bw * bh
                    if area > max_body_area:
                        max_body_area = area
                        largest_body = (x, y, bw, bh)
                        bodies.append(largest_body)
            except Exception:
                pass
        
        # Correlation check for face and body
        if largest_face is not None and largest_body is not None:
            if self._check_face_body_correlation(largest_face, largest_body):
                self.face_confidence = min(1.0, self.face_confidence + 0.2)
        
        # Fallback to body if no face found
        if not faces and self.use_body_fallback and largest_body is not None:
            largest_face = largest_body
            faces.append(largest_body)
            self.face_confidence = 0.4
        
        # Handle persistence
        if self.use_persistence:
            if largest_face is not None:
                self.last_valid_target = largest_face
                self.target_lost_time = None
            elif self.last_valid_target is not None:
                if self.target_lost_time is None:
                    self.target_lost_time = time.time()
                elif time.time() - self.target_lost_time < self.max_target_lost_time:
                    largest_face = self.last_valid_target
                    elapsed = time.time() - self.target_lost_time
                    self.face_confidence = max(0.1, 0.6 - (elapsed / self.max_target_lost_time) * 0.5)
                else:
                    self.last_valid_target = None
                    self.target_lost_time = None
                    self.face_confidence = 0.0
                    self.position_smoother.reset()
        
        # Store detections
        self.detected_face = largest_face
        self.detected_body = largest_body
        self.tracking_target = largest_face
        
        # Apply position smoothing to tracking target
        if self.tracking_target is not None:
            x, y, fw, fh = self.tracking_target
            center_x = x + fw / 2
            center_y = y + fh / 2
            
            # Smooth the center position
            smooth_x, smooth_y, smooth_w, smooth_h = self.position_smoother.update(
                center_x, center_y, fw, fh
            )
            
            # Convert back to corner coordinates
            new_x = int(smooth_x - smooth_w / 2)
            new_y = int(smooth_y - smooth_h / 2)
            new_w = int(smooth_w)
            new_h = int(smooth_h)
            
            self.tracking_target = (new_x, new_y, new_w, new_h)
            
            # Also update detected_face with smoothed position for visualization
            if self.detected_face is not None:
                self.detected_face = self.tracking_target
        
        all_detections = faces + [b for b in bodies if b not in faces]
        return all_detections, self.tracking_target
    
    def _check_face_body_correlation(self, face, body):
        """Check if face and body detections belong to the same person"""
        if face is None or body is None:
            return False
        
        fx, fy, fw, fh = face
        bx, by, bw, bh = body
        
        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2
        body_center_x = bx + bw / 2
        body_center_y = by + bh / 2
        
        horizontal_alignment = abs(face_center_x - body_center_x) < bw / 2
        vertical_position = face_center_y < body_center_y
        reasonable_size = fw < bw * 1.5 and fh < bh * 0.6
        
        return horizontal_alignment and vertical_position and reasonable_size
    
    def track_face(self, frame):
        """
        Control the camera to track the detected face using PID control.
        
        Uses hysteresis tracking:
        - Dead zone prevents STARTING movement if face is already close
        - Once movement starts, continue until face reaches actual center
        - This minimizes adjustments while ensuring proper centering
        
        Args:
            frame: Current video frame (used for dimensions)
        """
        if self.tracking_target is None or self.camera is None:
            # No target or camera - stop movement
            if self.camera_moving:
                self.camera.pantilt(pan_speed=0, tilt_speed=0)
                self.camera_moving = False
                self._is_correcting_x = False
                self._is_correcting_y = False
            return
        
        # Check confidence threshold
        if self.face_confidence < 0.3:
            if self.camera_moving:
                self.camera.pantilt(pan_speed=0, tilt_speed=0)
                self.camera_moving = False
                self._is_correcting_x = False
                self._is_correcting_y = False
            return
        
        # Check require_face_and_body setting
        if self.require_face_and_body:
            if self.detected_face is None or self.detected_body is None:
                if self.camera_moving:
                    self.camera.pantilt(pan_speed=0, tilt_speed=0)
                    self.camera_moving = False
                    self._is_correcting_x = False
                    self._is_correcting_y = False
                return
            
            if not self._check_face_body_correlation(self.detected_face, self.detected_body):
                if self.camera_moving:
                    self.camera.pantilt(pan_speed=0, tilt_speed=0)
                    self.camera_moving = False
                    self._is_correcting_x = False
                    self._is_correcting_y = False
                return
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Get face center
        x, y, fw, fh = self.tracking_target
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        
        # Portrait position: center horizontally, 1/3 from top vertically
        target_x = w / 2
        target_y = h / 3
        
        # Calculate normalized error (-1 to 1 range)
        error_x = (face_center_x - target_x) / (w / 2)
        error_y = (face_center_y - target_y) / (h / 2)
        
        # Dead zone for TRIGGERING movement (larger - 1/3 of frame)
        trigger_zone_x = 1/3
        trigger_zone_y = 1/3
        
        # Target zone for STOPPING movement (tight - near actual center)
        target_zone = 0.05  # Stop when within 5% of center
        
        # Store center box coordinates for visualization (shows trigger zone)
        box_left = int(target_x - (trigger_zone_x * w / 2))
        box_right = int(target_x + (trigger_zone_x * w / 2))
        box_top = int(target_y - (trigger_zone_y * h / 2))
        box_bottom = int(target_y + (trigger_zone_y * h / 2))
        self.center_box = (box_left, box_top, box_right - box_left, box_bottom - box_top)
        
        # Initialize correction state if not exists
        if not hasattr(self, '_is_correcting_x'):
            self._is_correcting_x = False
        if not hasattr(self, '_is_correcting_y'):
            self._is_correcting_y = False
        
        # Check if face is in trigger zone (for starting movement)
        in_trigger_zone_x = abs(error_x) < trigger_zone_x
        in_trigger_zone_y = abs(error_y) < trigger_zone_y
        
        # Check if face is at target (for stopping movement)
        at_target_x = abs(error_x) < target_zone
        at_target_y = abs(error_y) < target_zone
        
        # Hysteresis logic for X axis:
        # - Start correcting if face leaves trigger zone
        # - Stop correcting when face reaches target center
        if not self._is_correcting_x and not in_trigger_zone_x:
            self._is_correcting_x = True  # Face left trigger zone, start correcting
        elif self._is_correcting_x and at_target_x:
            self._is_correcting_x = False  # Face reached center, stop correcting
        
        # Hysteresis logic for Y axis
        if not self._is_correcting_y and not in_trigger_zone_y:
            self._is_correcting_y = True  # Face left trigger zone, start correcting
        elif self._is_correcting_y and at_target_y:
            self._is_correcting_y = False  # Face reached center, stop correcting
        
        # If not correcting on either axis, stop camera
        if not self._is_correcting_x and not self._is_correcting_y:
            if self.camera_moving:
                self.camera.pantilt(pan_speed=0, tilt_speed=0)
                self.camera_moving = False
                # Reset PID integrators when stopped
                if hasattr(self.pan_pid, 'reset'):
                    self.pan_pid.reset()
                    self.tilt_pid.reset()
                elif PID_AVAILABLE:
                    self.pan_pid._integral = 0
                    self.tilt_pid._integral = 0
            return
        
        # Calculate PID output for pan and tilt
        # Only apply PID if actively correcting that axis
        
        if not self._is_correcting_x:
            pan_speed = 0
        else:
            # Negate output because simple-pid calculates (setpoint - input)
            pan_speed = -self.pan_pid(error_x)
        
        if not self._is_correcting_y:
            tilt_speed = 0
        else:
            # Negate output and apply direction correction
            tilt_speed = -self.tilt_pid(error_y)
            tilt_speed = -tilt_speed  # Invert for correct direction
            
            # Apply camera inversion
            if self.invert_camera:
                tilt_speed = -tilt_speed

        
        # Round to integers for VISCA
        pan_speed = int(round(pan_speed))
        tilt_speed = int(round(tilt_speed))
        
        # Clamp to valid range
        max_speed = int(self.move_speed)
        pan_speed = max(-max_speed, min(max_speed, pan_speed))
        tilt_speed = max(-max_speed, min(max_speed, tilt_speed))
        
        # Only send command if speeds changed significantly
        if (abs(pan_speed - self.last_pan_speed) > 0 or 
            abs(tilt_speed - self.last_tilt_speed) > 0 or
            not self.camera_moving):
            
            self.camera.pantilt(pan_speed=pan_speed, tilt_speed=tilt_speed)
            self.last_pan_speed = pan_speed
            self.last_tilt_speed = tilt_speed
            self.camera_moving = (pan_speed != 0 or tilt_speed != 0)
    
    def draw_center_box(self, frame):
        """Draw the center thirds box (dead zone) on the frame"""
        if self.center_box is not None:
            x, y, w, h = self.center_box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "Dead Zone", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame
    
    def draw_tracking_info(self, frame):
        """Draw tracking visualization on the frame"""
        # Draw dead zone
        self.draw_center_box(frame)
        
        # Draw detected face
        if self.detected_face is not None:
            x, y, w, h = self.detected_face
            # Green rectangle for face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Confidence text
            conf_text = f"Confidence: {self.face_confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detected body (if different from face)
        if self.detected_body is not None and self.detected_body != self.detected_face:
            x, y, w, h = self.detected_body
            # Blue rectangle for body
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw PID info
        y_offset = 30
        if self.camera_moving:
            pid_text = f"PID: Pan={self.last_pan_speed:+d} Tilt={self.last_tilt_speed:+d}"
            cv2.putText(frame, pid_text, (10, frame.shape[0] - y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def stop_tracking(self):
        """Stop camera movement and reset tracking state"""
        if self.camera is not None and self.camera_moving:
            self.camera.pantilt(pan_speed=0, tilt_speed=0)
            self.camera_moving = False
        
        self.last_pan_speed = 0
        self.last_tilt_speed = 0
        
        # Reset PIDs
        if hasattr(self.pan_pid, 'reset'):
            self.pan_pid.reset()
            self.tilt_pid.reset()
        elif PID_AVAILABLE:
            self.pan_pid._integral = 0
            self.tilt_pid._integral = 0
    
    def close(self):
        """Clean up resources"""
        self.stop_tracking()
        
        if self.mp_face_detection:
            self.mp_face_detection.close()
        if self.mp_pose:
            self.mp_pose.close()
