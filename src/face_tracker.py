import cv2
import time
import numpy as np
from tkinter import messagebox

class FaceTracker:
    def __init__(self, camera, move_speed=5, move_threshold=50, movement_cooldown=0.2, invert_camera=False):
        self.camera = camera
        self.move_speed = move_speed
        self.move_threshold = move_threshold
        self.movement_cooldown = movement_cooldown
        self.invert_camera = invert_camera

        self.tracking_target = None
        self.last_movement_time = 0
        self.last_pan_direction = 0
        self.last_tilt_direction = 0
        self.direction_change_count = 0
        
        # Tracking persistence variables
        self.target_lost_time = None
        self.max_target_lost_time = 2.0  # Maximum time to keep tracking a lost target (seconds)
        self.last_valid_target = None
        self.detected_face = None  # Store the detected face separately
        self.detected_body = None  # Store the detected body separately
        self.center_box = None  # Store the center thirds box coordinates for visualization
        
        # Correlation tracking variables
        self.face_confidence = 0.0  # Confidence level for face detection (0.0-1.0)
        self.min_confidence_to_move = 0.3  # Minimum confidence required to move camera (lowered for better tracking)
        
        # Initialize face detection models
        self.initialize_detectors()
        
        # Tracking mode flags
        self.use_dlib = True  # Use DLib HOG+SVM detector (more accurate but slower)
        self.use_body_fallback = True  # Fall back to body detection when face not found
        self.use_persistence = True  # Continue tracking briefly when target lost
        self.require_face_and_body = False  # Only move camera if both face and body are detected and correlated

    def initialize_detectors(self):
        # Initialize OpenCV Haar cascade as fallback
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load OpenCV detection models: {str(e)}")
            self.face_cascade = None
            self.body_cascade = None
            self.upper_body_cascade = None
        
        # Initialize DLib HOG face detector (more robust than Haar cascade)
        self.dlib_detector = None
        try:
            # Try to import dlib - if not available, we'll fall back to OpenCV
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
        except ImportError:
            self.use_dlib = False
            messagebox.showwarning("Warning", "DLib not available. Using OpenCV for face detection only.")
        except Exception as e:
            self.use_dlib = False
            messagebox.showwarning("Warning", f"Failed to initialize DLib detector: {str(e)}. Using OpenCV only.")

    def update_settings(self, move_speed, move_threshold, movement_cooldown, invert_camera):
        self.move_speed = move_speed
        self.move_threshold = move_threshold
        self.movement_cooldown = movement_cooldown
        self.invert_camera = invert_camera

    def detect_faces(self, frame):
        faces = []
        bodies = []
        largest_face = None
        largest_body = None
        max_face_area = 0
        max_body_area = 0
        
        # Reset detection confidence
        self.face_confidence = 0.0
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try DLib HOG detector first (more robust for different angles)
        if self.use_dlib and self.dlib_detector is not None:
            try:
                # Detect faces using DLib
                dlib_faces = self.dlib_detector(gray, 0)  # 0 = no upsampling for speed
                
                # Convert DLib rectangles to OpenCV format (x,y,w,h)
                for face in dlib_faces:
                    x, y = face.left(), face.top()
                    w, h = face.width(), face.height()
                    faces.append((x, y, w, h))
                    
                    # Track largest face
                    area = w * h
                    if area > max_face_area:
                        max_face_area = area
                        largest_face = (x, y, w, h)
                        
                # If we found faces with DLib, increase confidence
                if faces:
                    self.face_confidence = 0.8  # DLib is more reliable
            except Exception as e:
                # If DLib fails, we'll fall back to OpenCV
                print(f"DLib detection error: {str(e)}")
        
        # Also try OpenCV Haar cascade (can sometimes detect faces DLib misses)
        if self.face_cascade is not None:
            try:
                opencv_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Add OpenCV detected faces
                for (x, y, w, h) in opencv_faces:
                    # Check if this face overlaps with any existing face
                    is_new_face = True
                    for (fx, fy, fw, fh) in faces:
                        # Calculate overlap
                        overlap_x = max(0, min(fx+fw, x+w) - max(fx, x))
                        overlap_y = max(0, min(fy+fh, y+h) - max(fy, y))
                        overlap_area = overlap_x * overlap_y
                        min_area = min(fw*fh, w*h)
                        
                        # If significant overlap, consider it the same face
                        if overlap_area > 0.5 * min_area:
                            is_new_face = False
                            # Increase confidence if both detectors found the same face
                            self.face_confidence = max(self.face_confidence, 0.9)
                            break
                    
                    if is_new_face:
                        faces.append((x, y, w, h))
                        
                        # Track largest face
                        area = w * h
                        if area > max_face_area:
                            max_face_area = area
                            largest_face = (x, y, w, h)
                            # Set moderate confidence for OpenCV-only detections
                            if self.face_confidence < 0.6:
                                self.face_confidence = 0.6
            except Exception as e:
                print(f"OpenCV face detection error: {str(e)}")
        
        # Always perform body detection for correlation
        body_target = self.detect_body(frame)
        if body_target is not None:
            bodies.append(body_target)
            largest_body = body_target
            
            # If we have both face and body detections, check for correlation
            if largest_face is not None:
                # Use the dedicated correlation check function
                if self.check_face_body_correlation(largest_face, largest_body):
                    # Strong correlation between face and body
                    self.face_confidence = 1.0
                else:
                    # Face detected but doesn't correlate with body
                    # Still maintain reasonable confidence but not maximum
                    self.face_confidence = max(self.face_confidence, 0.8)
        
        # Fall back to body detection if no faces found and fallback enabled
        if not faces and self.use_body_fallback and largest_body is not None:
            largest_face = largest_body
            faces.append(largest_body)
            self.face_confidence = 0.4  # Lower confidence for body-only detection
        
        # Handle tracking persistence
        if self.use_persistence:
            if largest_face is not None:
                # We found a target, update the last valid target
                self.last_valid_target = largest_face
                self.target_lost_time = None
            elif self.last_valid_target is not None:
                # No target found, check if we should use the last valid target
                if self.target_lost_time is None:
                    # Start counting time since target was lost
                    self.target_lost_time = time.time()
                elif time.time() - self.target_lost_time < self.max_target_lost_time:
                    # Use the last valid target if within the persistence time window
                    largest_face = self.last_valid_target
                    # Decrease confidence as time passes
                    elapsed = time.time() - self.target_lost_time
                    self.face_confidence = max(0.1, 0.6 - (elapsed / self.max_target_lost_time) * 0.5)
                else:
                    # Target lost for too long, reset tracking
                    self.last_valid_target = None
                    self.target_lost_time = None
                    self.face_confidence = 0.0
        
        # Store detected face and body separately for visualization
        self.detected_face = largest_face
        self.detected_body = largest_body
        
        # Update the current tracking target
        self.tracking_target = largest_face
        
        # Return all detections and the primary target
        all_detections = faces + [b for b in bodies if b not in faces]  # Combine faces and bodies without duplicates
        return all_detections, self.tracking_target
    
    def detect_body(self, frame):
        """Detect bodies when faces aren't visible"""
        if self.body_cascade is None and self.upper_body_cascade is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        largest_body = None
        max_area = 0
        
        # Try upper body detection first (more likely to be visible)
        if self.upper_body_cascade is not None:
            upper_bodies = self.upper_body_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
            
            for (x, y, w, h) in upper_bodies:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_body = (x, y, w, h)
        
        # If no upper body found, try full body detection
        if largest_body is None and self.body_cascade is not None:
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(70, 70))
            
            for (x, y, w, h) in bodies:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_body = (x, y, w, h)
        
        return largest_body
        
    def check_face_body_correlation(self, face, body):
        """Check if a face and body detection are likely to belong to the same person"""
        if face is None or body is None:
            return False
            
        fx, fy, fw, fh = face
        bx, by, bw, bh = body
        
        # Calculate centers
        face_center_x = fx + fw/2
        face_center_y = fy + fh/2
        body_center_x = bx + bw/2
        body_center_y = by + bh/2
        
        # Face should be above the body center and within the body width
        horizontal_alignment = abs(face_center_x - body_center_x) < bw/2
        vertical_position = face_center_y < body_center_y
        
        # Face should be reasonably sized compared to body
        reasonable_size = fw < bw * 1.2 and fh < bh * 0.5
        
        return horizontal_alignment and vertical_position and reasonable_size

    def track_face(self, frame):
        if self.tracking_target is None or self.camera is None:
            return

        # Check if we have enough confidence to move the camera
        if self.face_confidence < self.min_confidence_to_move:
            # Not confident enough about the face detection, don't move
            return
        
        # Check if require_face_and_body is enabled
        if self.require_face_and_body:
            # Both face and body must be detected
            if self.detected_face is None or self.detected_body is None:
                # Missing face or body, don't move camera
                return
            
            # Check if face and body are properly correlated
            if not self.check_face_body_correlation(self.detected_face, self.detected_body):
                # Face and body don't correlate, don't move camera
                return

        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]

        # Get face position
        x, y, w, h = self.tracking_target
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Calculate frame center and target positions
        frame_center_x = frame_w // 2
        target_y = int(frame_h * 1/3)  # Aim for 1/3 down from the top (upper third)

        # Calculate offset from target positions
        offset_x = face_center_x - frame_center_x
        offset_y = face_center_y - target_y

        # Define the imaginary box in the center thirds quadrant
        # Box width/height is 1/3rd of the frame, so threshold is 1/6th from center
        center_threshold_x = frame_w / 6  # 1/3 of frame width (1/6 on each side of center)
        center_threshold_y = frame_h / 6  # 1/3 of frame height (1/6 on each side of target)
        
        # Calculate the box boundaries for visualization (if needed)
        box_left = frame_center_x - center_threshold_x
        box_right = frame_center_x + center_threshold_x
        box_top = target_y - center_threshold_y
        box_bottom = target_y + center_threshold_y
        
        # Store the center box coordinates for potential visualization
        self.center_box = (int(box_left), int(box_top), int(box_right - box_left), int(box_bottom - box_top))

        # Check if face is inside the center thirds box
        is_centered = (abs(offset_x) <= center_threshold_x and abs(offset_y) <= center_threshold_y)

        if is_centered:
            self.camera.pantilt(pan_speed=0, tilt_speed=0)
            self.last_movement_time = time.time()
            return

        current_time = time.time()

        # Dynamic cooldown calculation
        distance_from_center = max(abs(offset_x) / frame_w, abs(offset_y) / frame_h)
        dynamic_cooldown = self.movement_cooldown
        if distance_from_center > 0.3:
            dynamic_cooldown = self.movement_cooldown * 0.5
        elif distance_from_center < 0.1:
            dynamic_cooldown = self.movement_cooldown * 1.5

        if current_time - self.last_movement_time >= dynamic_cooldown:
            pan_speed = 0
            tilt_speed = 0

            h_importance = abs(offset_x) / center_threshold_x if center_threshold_x > 0 else 0
            v_importance = abs(offset_y) / center_threshold_y if center_threshold_y > 0 else 0

            need_h_adjustment = abs(offset_x) > center_threshold_x
            need_v_adjustment = abs(offset_y) > center_threshold_y

            # Horizontal movement (proportional speed)
            if need_h_adjustment:
                h_direction = 1 if offset_x > 0 else -1
                # Speed proportional to offset, capped by move_speed
                # Scale factor determines how quickly max speed is reached (e.g., 3 means max speed at 3*threshold offset)
                scale_factor = 3
                proportional_speed = abs(offset_x) / (center_threshold_x * scale_factor) * self.move_speed
                pan_speed = int(h_direction * min(self.move_speed, max(1, proportional_speed))) # Ensure minimum speed of 1 if moving

            # Vertical movement (proportional speed)
            if need_v_adjustment:
                v_direction = 1 if offset_y > 0 else -1
                # Speed proportional to offset, capped by move_speed
                scale_factor = 3
                proportional_speed = abs(offset_y) / (center_threshold_y * scale_factor) * self.move_speed
                tilt_speed = int(v_direction * min(self.move_speed, max(1, proportional_speed))) # Ensure minimum speed of 1 if moving
                
                # Invert tilt speed for correct direction
                tilt_speed = -tilt_speed
                if self.invert_camera:
                    tilt_speed = -tilt_speed # Double inversion if camera is upside down

            # Adaptive movement strategy
            if need_h_adjustment and need_v_adjustment:
                if h_importance > v_importance * 2:
                    tilt_speed = int(tilt_speed * 0.5)
                elif v_importance > h_importance * 2:
                    pan_speed = int(pan_speed * 0.5)

            # Trajectory correction
            current_pan_direction = 1 if pan_speed > 0 else (-1 if pan_speed < 0 else 0)
            current_tilt_direction = 1 if tilt_speed > 0 else (-1 if tilt_speed < 0 else 0)

            pan_direction_changed = (self.last_pan_direction != 0 and
                                   current_pan_direction != 0 and
                                   current_pan_direction != self.last_pan_direction)

            tilt_direction_changed = (self.last_tilt_direction != 0 and
                                    current_tilt_direction != 0 and
                                    current_tilt_direction != self.last_tilt_direction)

            if pan_direction_changed or tilt_direction_changed:
                self.direction_change_count += 1
                if self.direction_change_count > 2:
                    if pan_direction_changed:
                        pan_speed = int(pan_speed * 0.6)
                    if tilt_direction_changed:
                        tilt_speed = int(tilt_speed * 0.6)
                    if self.direction_change_count > 4:
                        self.direction_change_count = 0
            else:
                if self.direction_change_count > 0 and int(current_time) % 3 == 0:
                    self.direction_change_count -= 1

            # Move camera if needed
            if pan_speed != 0 or tilt_speed != 0:
                if pan_speed != 0 and abs(pan_speed) < 1:
                    pan_speed = 1 if pan_speed > 0 else -1
                if tilt_speed != 0 and abs(tilt_speed) < 1:
                    tilt_speed = 1 if tilt_speed > 0 else -1

                self.camera.pantilt(pan_speed=pan_speed, tilt_speed=tilt_speed)
                self.last_movement_time = current_time
                self.last_pan_direction = current_pan_direction
                self.last_tilt_direction = current_tilt_direction
            else:
                # Explicitly stop if no movement calculated this cycle
                self.camera.pantilt(pan_speed=0, tilt_speed=0)
                self.last_movement_time = current_time
                
    def draw_center_box(self, frame):
        """Draw the center thirds box on the frame"""
        if self.center_box is not None:
            x, y, w, h = self.center_box
            # Draw a semi-transparent yellow rectangle for the center thirds box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)
            # Add slight transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Add label for the center box
            cv2.putText(frame, "Center Zone", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame
            
    def draw_tracking_info(self, frame):
        """Draw tracking information on the frame"""
        # Draw the center thirds box first (so it appears behind other elements)
        self.draw_center_box(frame)
        
        if self.detected_face is not None:
            x, y, w, h = self.detected_face
            # Draw green rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence level
            confidence_text = f"Confidence: {self.face_confidence:.2f}"
            cv2.putText(frame, confidence_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.detected_body is not None and self.detected_body != self.detected_face:
            x, y, w, h = self.detected_body
            # Draw blue rectangle around detected body
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        return frame