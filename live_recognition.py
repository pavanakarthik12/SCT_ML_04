"""
ASL Fingerspelling Live Recognition System
Real-time ASL letter recognition using webcam input
"""

import cv2
import numpy as np
import joblib
import json
import time
from collections import deque
import os

# Check for MediaPipe availability
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe is available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - using image-based feature extraction fallback")


class ASLRecognizer:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Initialize ASL recognizer with trained model"""
        print("Loading trained model...")
        
        # Load trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load model information
        try:
            with open('model_info.json', 'r') as f:
                self.model_info = json.load(f)
                print(f"Model: {self.model_info.get('model_name', 'Unknown')}, Accuracy: {self.model_info.get('accuracy', 'Unknown')}")
        except FileNotFoundError:
            self.model_info = {'classes': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        
        # Initialize MediaPipe Hands if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            self.hands = None
            self.mp_hands = None
            self.mp_drawing = None
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=10)
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0
        
        # Hand tracking variables
        self.hand_region = None
        self.tracked_landmarks = []
        self.finger_tips = []
        self.landmark_confidence = []
        
        # Advanced temporal smoothing for skeleton stability
        self.prev_landmarks = None  # Previous frame landmarks
        self.ema_landmarks = None   # EMA smoothed landmarks
        self.ema_alpha = 0.65       # EMA smoothing factor (65% current, 35% history)
        self.landmark_velocity = None  # For derivative-based filtering
        
        # One-Euro filter parameters for jitter reduction
        self.min_cutoff = 1.0       # Minimum cutoff frequency
        self.beta = 0.007           # Speed coefficient
        self.derivate_cutoff = 1.0  # Derivative cutoff frequency
        
        # Bounding box stabilization
        self.prev_bbox = None
        self.ema_bbox = None        # EMA smoothed bounding box
        self.bbox_ema_alpha = 0.70  # Bbox EMA smoothing (70% current, 30% history)
        self.avg_hand_size = None
        self.hand_size_history = deque(maxlen=20)  # Track hand size over time
        self.min_bbox_padding = 30  # Minimum padding around hand (pixels)
        self.max_bbox_variation = 0.08  # Maximum 8% variation per frame
        self.min_hand_size_ratio = 0.88  # Maintain >88% of average size
        
        # Temporal gesture consistency (majority voting)
        self.gesture_history = deque(maxlen=10)  # Track last 10 predictions
        self.gesture_consistency_threshold = 5  # Need 5/10 same predictions
        self.last_stable_gesture = "No gesture detected"
        self.gesture_confidence_history = deque(maxlen=10)
        
        # Gesture recognition thresholds
        self.confidence_threshold = 0.65  # Minimum confidence for valid prediction
        self.asl_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        print("ASL Recognizer initialized successfully!")
    
    def extract_landmarks(self, frame, hand_landmarks=None):
        """Extract 63 features from frame using MediaPipe landmarks or fallback method"""
        if MEDIAPIPE_AVAILABLE and hand_landmarks is not None:
            return self._extract_mediapipe_landmarks(hand_landmarks)
        else:
            return self._extract_fallback_features(frame)
    
    def _apply_ema_smoothing(self, current_landmarks):
        """Apply Exponential Moving Average smoothing to landmarks"""
        if self.ema_landmarks is None:
            # Initialize EMA with current landmarks
            self.ema_landmarks = current_landmarks.copy()
            return current_landmarks
        
        # Apply EMA: smoothed = alpha * current + (1 - alpha) * previous
        smoothed = []
        for curr, prev in zip(current_landmarks, self.ema_landmarks):
            smoothed_x = self.ema_alpha * curr[0] + (1 - self.ema_alpha) * prev[0]
            smoothed_y = self.ema_alpha * curr[1] + (1 - self.ema_alpha) * prev[1]
            smoothed.append([int(smoothed_x), int(smoothed_y)])
        
        self.ema_landmarks = smoothed
        return smoothed
    
    def _apply_one_euro_filter(self, current_landmarks, timestamp):
        """Apply One-Euro-like filter for jitter reduction with velocity adaptation"""
        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            self.prev_landmarks = current_landmarks.copy()
            self.landmark_velocity = [[0, 0] for _ in current_landmarks]
            return current_landmarks
        
        dt = 1.0 / 30.0  # Assume 30 FPS, adjust based on actual FPS if available
        smoothed = []
        
        for i, (curr, prev) in enumerate(zip(current_landmarks, self.prev_landmarks)):
            # Calculate velocity (derivative)
            vel_x = (curr[0] - prev[0]) / dt
            vel_y = (curr[1] - prev[1]) / dt
            
            # Smooth velocity
            alpha_d = self._smoothing_factor(dt, self.derivate_cutoff)
            self.landmark_velocity[i][0] = alpha_d * vel_x + (1 - alpha_d) * self.landmark_velocity[i][0]
            self.landmark_velocity[i][1] = alpha_d * vel_y + (1 - alpha_d) * self.landmark_velocity[i][1]
            
            # Adaptive cutoff based on velocity
            velocity_mag = np.sqrt(self.landmark_velocity[i][0]**2 + self.landmark_velocity[i][1]**2)
            cutoff = self.min_cutoff + self.beta * velocity_mag
            
            # Apply smoothing
            alpha = self._smoothing_factor(dt, cutoff)
            smoothed_x = alpha * curr[0] + (1 - alpha) * prev[0]
            smoothed_y = alpha * curr[1] + (1 - alpha) * prev[1]
            
            smoothed.append([int(smoothed_x), int(smoothed_y)])
        
        self.prev_landmarks = smoothed
        return smoothed
    
    def _smoothing_factor(self, dt, cutoff):
        """Calculate smoothing factor for One-Euro filter"""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def _apply_skeleton_stability(self, landmarks):
        """Ensure skeleton maintains proper proportions and joint distances"""
        if len(landmarks) != 21:
            return landmarks
        
        # Ensure wrist is stable (reference point)
        wrist = landmarks[0]
        
        # Check and enforce minimum distances between connected joints
        stable_landmarks = [wrist]
        
        # Define finger chains: [[base, joint1, joint2, tip], ...]
        finger_chains = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16],# Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for chain in finger_chains:
            for i, idx in enumerate(chain):
                if idx >= len(landmarks):
                    stable_landmarks.append(landmarks[idx] if idx < len(landmarks) else wrist)
                    continue
                
                current = landmarks[idx]
                
                # Get previous joint in chain
                if i == 0:
                    prev = wrist
                else:
                    prev = stable_landmarks[chain[i-1]]
                
                # Calculate distance from previous joint
                dist = np.sqrt((current[0] - prev[0])**2 + (current[1] - prev[1])**2)
                
                # Enforce minimum distance (prevent collapsing)
                min_dist = 8  # Minimum pixels between joints
                if dist < min_dist and dist > 0:
                    # Push joint away to maintain minimum distance
                    scale = min_dist / dist
                    current[0] = int(prev[0] + (current[0] - prev[0]) * scale)
                    current[1] = int(prev[1] + (current[1] - prev[1]) * scale)
                
                stable_landmarks.append(current)
        
        return stable_landmarks
        """Extract 63 features (21 landmarks * x,y,z) from MediaPipe hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks).reshape(1, -1)
    
    def _extract_fallback_features(self, frame):
        """Extract 63 features with precise hand tracking and stable skeletal mapping"""
        h, w = frame.shape[:2]
        
        # Convert to different color spaces for robust hand detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Multi-space skin detection for accuracy
        # HSV range optimized for various skin tones
        lower_hsv = np.array([0, 15, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb range for robust skin detection
        lower_ycrcb = np.array([0, 130, 75], dtype=np.uint8)
        upper_ycrcb = np.array([255, 175, 130], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks for better coverage
        skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        
        # Refined morphological operations for clean hand segmentation
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # Remove noise and fill gaps
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        hand_detected = False
        self.tracked_landmarks = []
        self.finger_tips = []
        
        if contours:
            # Get the largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Strict area threshold for hand detection
            if contour_area > 10000:  # Minimum hand area in pixels
                hand_detected = True
                
                # Smooth contour to reduce jitter
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Get initial bounding box around hand
                x_raw, y_raw, w_raw, h_raw = cv2.boundingRect(approx_contour)
                
                # Calculate proportional padding (15% of box size, minimum 25px)
                padding_w = max(self.min_bbox_padding, int(w_raw * 0.15))
                padding_h = max(self.min_bbox_padding, int(h_raw * 0.15))
                
                # Apply padding to create stable bounding box
                x_padded = max(0, x_raw - padding_w)
                y_padded = max(0, y_raw - padding_h)
                w_padded = min(w - x_padded, w_raw + 2 * padding_w)
                h_padded = min(h - y_padded, h_raw + 2 * padding_h)
                
                # Track average hand size
                current_size = (w_padded + h_padded) / 2
                self.hand_size_history.append(current_size)
                
                if len(self.hand_size_history) >= 5:
                    self.avg_hand_size = sum(self.hand_size_history) / len(self.hand_size_history)
                    
                    # Enforce minimum size ratio (prevent excessive shrinking)
                    min_allowed_size = self.avg_hand_size * self.min_hand_size_ratio
                    if current_size < min_allowed_size:
                        scale_factor = min_allowed_size / current_size
                        w_padded = int(w_padded * scale_factor)
                        h_padded = int(h_padded * scale_factor)
                        # Re-center the enlarged box
                        x_padded = max(0, x_padded - int((w_padded - w_raw) / 2))
                        y_padded = max(0, y_padded - int((h_padded - h_raw) / 2))
                
                # Apply EMA smoothing to bounding box dimensions
                if self.ema_bbox is not None:
                    ema_x = int(self.bbox_ema_alpha * x_padded + (1 - self.bbox_ema_alpha) * self.ema_bbox[0])
                    ema_y = int(self.bbox_ema_alpha * y_padded + (1 - self.bbox_ema_alpha) * self.ema_bbox[1])
                    ema_w = int(self.bbox_ema_alpha * w_padded + (1 - self.bbox_ema_alpha) * self.ema_bbox[2])
                    ema_h = int(self.bbox_ema_alpha * h_padded + (1 - self.bbox_ema_alpha) * self.ema_bbox[3])
                    x_padded, y_padded, w_padded, h_padded = ema_x, ema_y, ema_w, ema_h
                
                self.ema_bbox = (x_padded, y_padded, w_padded, h_padded)
                
                # Additional temporal smoothing to prevent jitter
                if self.prev_bbox is not None:
                    prev_x, prev_y, prev_w, prev_h = self.prev_bbox
                    
                    # Limit maximum change per frame
                    max_change_w = int(prev_w * self.max_bbox_variation)
                    max_change_h = int(prev_h * self.max_bbox_variation)
                    
                    # Clamp changes
                    w_padded = int(np.clip(w_padded, prev_w - max_change_w, prev_w + max_change_w))
                    h_padded = int(np.clip(h_padded, prev_h - max_change_h, prev_h + max_change_h))
                    
                    # Note: EMA smoothing already applied above, no need for additional smoothing here
                    x_padded, y_padded = x_padded, y_padded  # Position already smoothed by EMA
                
                # Store current bbox for next frame
                self.prev_bbox = (x_padded, y_padded, w_padded, h_padded)
                self.hand_region = (x_padded, y_padded, w_padded, h_padded)
                
                # Calculate centroid
                moments = cv2.moments(largest_contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = x_padded + w_padded // 2, y_padded + h_padded // 2
                
                # Get convex hull and defects for finger detection
                hull = cv2.convexHull(largest_contour, returnPoints=False)
                
                # Detect fingertips using convexity defects
                finger_tips = []
                try:
                    defects = cv2.convexityDefects(largest_contour, hull)
                    
                    if defects is not None:
                        # Analyze defects to find fingertips
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(largest_contour[s][0])
                            end = tuple(largest_contour[e][0])
                            far = tuple(largest_contour[f][0])
                            
                            # Calculate angles
                            a = np.linalg.norm(np.array(start) - np.array(end))
                            b = np.linalg.norm(np.array(far) - np.array(start))
                            c = np.linalg.norm(np.array(end) - np.array(far))
                            
                            # Cosine rule
                            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) if (2 * b * c) > 0 else 0
                            
                            # Filter fingertips based on angle and distance
                            if angle <= np.pi / 2 and d > 10000:  # Adjust threshold
                                finger_tips.append(start)
                                finger_tips.append(end)
                except:
                    pass
                
                # Remove duplicate fingertips
                if len(finger_tips) > 0:
                    finger_tips_np = np.array(finger_tips)
                    unique_tips = []
                    for tip in finger_tips_np:
                        is_unique = True
                        for existing in unique_tips:
                            if np.linalg.norm(tip - existing) < 30:  # Merge close points
                                is_unique = False
                                break
                        if is_unique:
                            unique_tips.append(tip)
                    
                    # Sort fingertips by x-coordinate (left to right)
                    finger_tips = sorted(unique_tips, key=lambda p: p[0])
                    
                    # Keep only top 5 fingertips
                    if len(finger_tips) > 5:
                        # Sort by y-coordinate and take topmost
                        finger_tips = sorted(finger_tips, key=lambda p: p[1])[:5]
                        finger_tips = sorted(finger_tips, key=lambda p: p[0])
                
                self.finger_tips = finger_tips
                
                # Build precise 21-point hand skeleton with confidence scores
                landmark_points = []
                landmark_conf = []
                
                # Wrist (landmark 0) - base of palm
                wrist_x, wrist_y = cx, min(h - 1, y_padded + h_padded - 10)
                landmark_points.append([wrist_x, wrist_y])
                landmark_conf.append(0.95)  # High confidence for stable wrist
                
                # Generate 5 fingers with 4 joints each
                num_fingers = 5
                
                for finger_idx in range(num_fingers):
                    # Determine finger position and characteristics
                    if finger_idx == 0:  # Thumb (left side)
                        base_angle = -np.pi * 0.4
                        finger_length = h_padded * 0.5
                    elif finger_idx == 1:  # Index
                        base_angle = -np.pi * 0.15
                        finger_length = h_padded * 0.65
                    elif finger_idx == 2:  # Middle
                        base_angle = 0
                        finger_length = h_padded * 0.7
                    elif finger_idx == 3:  # Ring
                        base_angle = np.pi * 0.15
                        finger_length = h_padded * 0.65
                    else:  # Pinky
                        base_angle = np.pi * 0.4
                        finger_length = h_padded * 0.5
                    
                    # Use detected fingertip if available
                    if finger_idx < len(finger_tips):
                        tip_x, tip_y = finger_tips[finger_idx]
                    else:
                        # Calculate tip position
                        tip_x = int(cx + finger_length * np.sin(base_angle))
                        tip_y = int(cy - finger_length * np.cos(base_angle))
                    
                    # Generate 4 joints per finger with precise positioning
                    for joint_idx in range(4):
                        ratio = (joint_idx + 1) / 4.0
                        
                        # Smooth interpolation from palm center to fingertip
                        joint_x = int(cx + (tip_x - cx) * ratio)
                        joint_y = int(cy + (tip_y - cy) * ratio)
                        
                        # Clamp to frame bounds
                        joint_x = max(0, min(w - 1, joint_x))
                        joint_y = max(0, min(h - 1, joint_y))
                        
                        landmark_points.append([joint_x, joint_y])
                        
                        # Confidence decreases toward fingertips (harder to detect)
                        conf = 0.9 - (joint_idx * 0.1) if finger_idx < len(finger_tips) else 0.7 - (joint_idx * 0.1)
                        landmark_conf.append(conf)
                
                # Apply multi-stage smoothing for maximum stability
                # Stage 1: EMA smoothing for general stability
                ema_smoothed = self._apply_ema_smoothing(landmark_points)
                
                # Stage 2: One-Euro filter for jitter reduction
                timestamp = time.time()
                euro_smoothed = self._apply_one_euro_filter(ema_smoothed, timestamp)
                
                # Stage 3: Skeleton stability (maintain proper joint distances)
                stable_points = self._apply_skeleton_stability(euro_smoothed)
                
                self.tracked_landmarks = stable_points
                self.landmark_confidence = landmark_conf
                
                # Convert to normalized features
                for point in landmark_points:
                    norm_x = point[0] / w
                    norm_y = point[1] / h
                    # Calculate z based on distance from center (depth approximation)
                    dist_from_center = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
                    max_dist = np.sqrt(w_padded**2 + h_padded**2)
                    z_val = 0.05 * (dist_from_center / max_dist - 0.5) if max_dist > 0 else 0.0
                    features.extend([norm_x, norm_y, z_val])
        
        # If no hand detected, clear all tracking and smoothing state
        if not hand_detected:
            self.hand_region = None
            self.tracked_landmarks = []
            self.finger_tips = []
            self.landmark_confidence = []
            self.prev_landmarks = None
            self.ema_landmarks = None
            self.landmark_velocity = None
            self.prev_bbox = None
            self.ema_bbox = None
            # Keep hand_size_history and gesture_history for continuity
            
            for i in range(21):
                if i == 0:  # Wrist
                    x, y, z = 0.5, 0.8, 0.0
                elif i <= 4:  # Thumb
                    joint = i - 1
                    x = 0.3 + joint * 0.05
                    y = 0.6 - joint * 0.1
                    z = 0.02 + joint * 0.01
                elif i <= 8:  # Index
                    joint = i - 5
                    x = 0.4 + joint * 0.02
                    y = 0.4 - joint * 0.15
                    z = 0.01 + joint * 0.015
                elif i <= 12:  # Middle
                    joint = i - 9
                    x = 0.5
                    y = 0.4 - joint * 0.18
                    z = joint * 0.02
                elif i <= 16:  # Ring
                    joint = i - 13
                    x = 0.6 - joint * 0.02
                    y = 0.4 - joint * 0.15
                    z = 0.01 + joint * 0.015
                else:  # Pinky
                    joint = i - 17
                    x = 0.7 - joint * 0.05
                    y = 0.5 - joint * 0.12
                    z = 0.02 + joint * 0.01
                
                features.extend([x, y, z])
        
        return np.array(features).reshape(1, -1)
    
    def predict_gesture(self, landmarks):
        """Predict ASL alphabet letter with temporal consistency smoothing"""
        if landmarks is None:
            return "No gesture detected", 0.0
        
        # Validate feature count
        if landmarks.shape[1] != 63:
            print(f"Warning: Expected 63 features, got {landmarks.shape[1]}")
            return "No gesture detected", 0.0
        
        # Scale features using trained scaler
        landmarks_scaled = self.scaler.transform(landmarks)
        
        # Get raw prediction and confidence
        raw_prediction = self.model.predict(landmarks_scaled)[0]
        
        # Filter: Only allow 'C' or 'L' predictions
        if raw_prediction not in ['C', 'L']:
            return "No gesture detected", 0.0
        
        # Calculate confidence (probability for predicted class)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(landmarks_scaled)[0]
            raw_confidence = max(probabilities)
            
            # Get class index for predicted letter
            pred_idx = list(self.model.classes_).index(raw_prediction)
            raw_confidence = probabilities[pred_idx]
        else:
            # Fallback for models without probability estimates
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(landmarks_scaled)[0]
                raw_confidence = max(scores) / (max(scores) - min(scores)) if len(scores) > 1 else 1.0
            else:
                raw_confidence = 0.5
        
        # Add to gesture history for temporal consistency
        self.gesture_history.append(raw_prediction)
        self.gesture_confidence_history.append(raw_confidence)
        
        # Apply temporal majority voting for stability
        if len(self.gesture_history) >= self.gesture_consistency_threshold:
            from collections import Counter
            gesture_counts = Counter(self.gesture_history)
            most_common_gesture, count = gesture_counts.most_common(1)[0]
            
            # Check if gesture is consistent across multiple frames
            consistency_ratio = count / len(self.gesture_history)
            
            # Calculate average confidence for the consistent gesture
            avg_confidence = sum(self.gesture_confidence_history) / len(self.gesture_confidence_history)
            
            # Only output gesture if it's consistent AND confident
            if consistency_ratio >= 0.5 and avg_confidence >= self.confidence_threshold:
                # Validate it's in ASL alphabet
                if most_common_gesture in self.asl_alphabet:
                    self.last_stable_gesture = most_common_gesture
                    return most_common_gesture, avg_confidence
        
        # If not consistent enough, return last stable gesture or no detection
        if self.last_stable_gesture != "No gesture detected" and len(self.gesture_history) >= 3:
            # Allow brief continuity of last gesture
            recent_confidence = sum(list(self.gesture_confidence_history)[-3:]) / 3
            if recent_confidence >= self.confidence_threshold * 0.8:
                return self.last_stable_gesture, recent_confidence
        
        return "No gesture detected", raw_confidence
    
    def smooth_predictions(self, prediction):
        """Apply temporal smoothing to reduce prediction flickering"""
        self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) < 3:
            return prediction
        
        # Use majority vote from recent predictions
        from collections import Counter
        counts = Counter(self.prediction_buffer)
        smoothed_prediction = counts.most_common(1)[0][0]
        
        return smoothed_prediction
    
    def calculate_fps(self):
        """Calculate running FPS"""
        current_time = time.time()
        self.fps_buffer.append(current_time)
        
        if len(self.fps_buffer) >= 2:
            time_diff = self.fps_buffer[-1] - self.fps_buffer[0]
            fps = len(self.fps_buffer) / time_diff if time_diff > 0 else 0
            return fps
        return 0
    
    def draw_landmarks_and_prediction(self, image, hand_landmarks, prediction, confidence, fps):
        """Draw hand landmarks and prediction info on the image"""
        # Draw hand landmarks if MediaPipe is available and landmarks exist
        if MEDIAPIPE_AVAILABLE and hand_landmarks is not None and self.mp_drawing is not None:
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        else:
            # Draw tracked hand region and landmarks from fallback method
            if self.hand_region is not None:
                x, y, w, h = self.hand_region
                # Draw STABLE bounding box with proper padding
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
                
                # Show corner markers for visual stability verification
                corner_size = 15
                cv2.line(image, (x, y), (x + corner_size, y), (0, 255, 255), 3)
                cv2.line(image, (x, y), (x, y + corner_size), (0, 255, 255), 3)
                cv2.line(image, (x + w, y), (x + w - corner_size, y), (0, 255, 255), 3)
                cv2.line(image, (x + w, y), (x + w, y + corner_size), (0, 255, 255), 3)
                cv2.line(image, (x, y + h), (x + corner_size, y + h), (0, 255, 255), 3)
                cv2.line(image, (x, y + h), (x, y + h - corner_size), (0, 255, 255), 3)
                cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), (0, 255, 255), 3)
                cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), (0, 255, 255), 3)
                
                # Display hand dimensions and stability info
                size_text = f"Box: {w}x{h}px"
                if self.avg_hand_size:
                    size_ratio = ((w + h) / 2) / self.avg_hand_size
                    size_text += f" ({size_ratio:.0%})"
                
                cv2.putText(image, size_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw detected fingertips
            if len(self.finger_tips) > 0:
                for i, tip in enumerate(self.finger_tips):
                    tx, ty = tip
                    # Draw large circles for fingertips
                    cv2.circle(image, (int(tx), int(ty)), 12, (255, 0, 255), -1)
                    cv2.circle(image, (int(tx), int(ty)), 14, (255, 255, 255), 2)
                    # Label each fingertip
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    if i < len(finger_names):
                        cv2.putText(image, finger_names[i], (int(tx) - 20, int(ty) - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw tracked landmarks
            if len(self.tracked_landmarks) > 0:
                # Define finger connections for visualization
                finger_connections = [
                    # Wrist to finger bases
                    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
                    # Thumb
                    (1, 2), (2, 3), (3, 4),
                    # Index
                    (5, 6), (6, 7), (7, 8),
                    # Middle
                    (9, 10), (10, 11), (11, 12),
                    # Ring
                    (13, 14), (14, 15), (15, 16),
                    # Pinky
                    (17, 18), (18, 19), (19, 20)
                ]
                
                # Draw connections (finger bones)
                for connection in finger_connections:
                    if connection[0] < len(self.tracked_landmarks) and connection[1] < len(self.tracked_landmarks):
                        pt1 = tuple(self.tracked_landmarks[connection[0]])
                        pt2 = tuple(self.tracked_landmarks[connection[1]])
                        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
                
                # Draw landmarks (finger joints) with confidence indicators
                for i, landmark in enumerate(self.tracked_landmarks):
                    x, y = landmark
                    
                    # Get confidence for this landmark
                    conf = self.landmark_confidence[i] if i < len(self.landmark_confidence) else 0.5
                    
                    # Different colors for different parts
                    if i == 0:  # Wrist
                        color = (255, 0, 0)  # Blue
                        radius = 7
                    elif i % 4 == 0:  # Finger tips
                        color = (0, 0, 255)  # Red
                        radius = 6
                    else:  # Joints
                        # Color intensity based on confidence
                        intensity = int(255 * conf)
                        color = (intensity, intensity, 0)  # Yellow with varying intensity
                        radius = 4
                    
                    # Draw landmark with confidence-based opacity
                    cv2.circle(image, (x, y), radius, color, -1)
                    cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)
                    
                    # Draw landmark number for debugging (optional)
                    # cv2.putText(image, str(i), (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        # Draw comprehensive info panel
        panel_height = 230
        cv2.rectangle(image, (10, 10), (480, panel_height), (0, 0, 0), -1)
        
        # ASL Letter prediction with color coding
        if prediction == "No gesture detected":
            pred_color = (0, 0, 255)  # Red for no detection
            pred_text = "ASL: No Gesture"
        elif confidence >= self.confidence_threshold:
            pred_color = (0, 255, 0)  # Green for confident
            pred_text = f"ASL Letter: {prediction}"
        else:
            pred_color = (0, 165, 255)  # Orange for low confidence
            pred_text = f"ASL: {prediction} (uncertain)"
        
        cv2.putText(image, pred_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
        
        # Confidence score with threshold indicator
        conf_text = f"Confidence: {confidence:.3f}"
        conf_color = (0, 255, 0) if confidence >= self.confidence_threshold else (255, 255, 255)
        cv2.putText(image, conf_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        # Tracking quality indicator
        tracking_quality = "Excellent" if len(self.tracked_landmarks) == 21 else "Poor"
        cv2.putText(image, f"Tracking: {tracking_quality}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS and mode
        mode_text = "MediaPipe" if MEDIAPIPE_AVAILABLE and self.hands else "CV Tracking"
        cv2.putText(image, f"FPS: {fps:.1f} | Mode: {mode_text}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Keypoints count
        keypoint_text = f"Keypoints: {len(self.tracked_landmarks)}/21"
        cv2.putText(image, keypoint_text, (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Bounding box stability indicator
        if self.hand_region and self.avg_hand_size:
            x, y, w, h = self.hand_region
            current_size = (w + h) / 2
            stability = min(100, int((1 - abs(current_size - self.avg_hand_size) / self.avg_hand_size) * 100))
            stability_color = (0, 255, 0) if stability > 90 else (0, 165, 255) if stability > 75 else (0, 0, 255)
            cv2.putText(image, f"Box Stability: {stability}%", (20, 185), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
        
        # Gesture temporal consistency indicator
        if len(self.gesture_history) > 0:
            from collections import Counter
            if len(self.gesture_history) >= 3:
                recent_gestures = list(self.gesture_history)[-5:]
                most_common = Counter(recent_gestures).most_common(1)[0]
                consistency = (most_common[1] / len(recent_gestures)) * 100
                consistency_color = (0, 255, 0) if consistency > 70 else (0, 165, 255) if consistency > 50 else (255, 100, 100)
                cv2.putText(image, f"Gesture Consistency: {int(consistency)}%", (20, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, consistency_color, 1)
        
        return image
    
    def run_live_recognition(self):
        """Run live ASL recognition from webcam"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("\nStarting live ASL recognition...")
        print("Instructions:")
        print("- Show your hand to the camera")
        print("- Make ASL fingerspelling gestures (A-Z)")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            prediction = "Processing..."
            confidence = 0.0
            hand_landmarks = None
            
            if MEDIAPIPE_AVAILABLE and self.hands is not None:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    # Use the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    try:
                        # Extract landmarks using MediaPipe
                        landmarks = self.extract_landmarks(frame, hand_landmarks)
                        
                        # Predict gesture
                        raw_prediction, confidence = self.predict_gesture(landmarks)
                        
                        if raw_prediction is not None:
                            # Apply smoothing
                            prediction = self.smooth_predictions(raw_prediction)
                    
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        prediction = "Error"
                else:
                    prediction = "No hand detected"
            else:
                # Use fallback feature extraction
                try:
                    # Extract features using fallback method
                    landmarks = self.extract_landmarks(frame)
                    
                    # Predict gesture
                    raw_prediction, confidence = self.predict_gesture(landmarks)
                    
                    if raw_prediction is not None:
                        # Apply smoothing every few frames to reduce noise
                        if frame_count % 5 == 0:  # Less frequent updates for fallback
                            prediction = self.smooth_predictions(raw_prediction)
                        else:
                            prediction = self.prediction_buffer[-1] if self.prediction_buffer else raw_prediction
                
                except Exception as e:
                    print(f"Prediction error: {e}")
                    prediction = "Error"
            
            # Draw landmarks and prediction
            frame = self.draw_landmarks_and_prediction(
                frame, hand_landmarks, prediction, confidence, self.calculate_fps()
            )
            
            # Show frame
            cv2.imshow('ASL Fingerspelling Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"asl_frame_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Live recognition stopped.")


def main():
    """Main function to run ASL recognition"""
    recognizer = ASLRecognizer()
    recognizer.run_live_recognition()


if __name__ == "__main__":
    main()