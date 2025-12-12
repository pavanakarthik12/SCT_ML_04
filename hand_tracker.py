"""
Standalone Hand Tracking Module with Real ASL Recognition
High-accuracy hand and finger joint tracking with trained ASL classifier
"""

import cv2
import numpy as np
import json
import time
from collections import deque
import os
import joblib

# Check for MediaPipe availability
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available - install with: pip install mediapipe")


class HandTracker:
    """Advanced hand tracking with finger joint detection and real ASL recognition"""
    
    def __init__(self, 
                 max_hands=2,
                 detection_confidence=0.8,
                 tracking_confidence=0.7,
                 enable_smoothing=True,
                 enable_angle_analysis=True,
                 model_path='model.pkl',
                 scaler_path='scaler.pkl'):
        """
        Initialize hand tracker with ASL recognition
        
        Args:
            max_hands: Maximum number of hands to detect (1-2)
            detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
            enable_smoothing: Enable temporal smoothing for stable tracking
            enable_angle_analysis: Enable detailed finger angle calculations
            model_path: Path to trained ASL classifier model
            scaler_path: Path to feature scaler
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.enable_smoothing = enable_smoothing
        self.enable_angle_analysis = enable_angle_analysis
        
        # Load trained ASL model
        self.asl_model = None
        self.asl_scaler = None
        self.asl_enabled = False
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.asl_model = joblib.load(model_path)
                self.asl_scaler = joblib.load(scaler_path)
                self.asl_enabled = True
                print(f"✓ ASL Recognition Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load ASL model: {e}")
                print("Continuing with hand tracking only (no ASL recognition)")
        else:
            print("Warning: ASL model files not found. Hand tracking only.")
            print(f"Expected: {model_path} and {scaler_path}")
            print("Run 'python data_loader_and_training.py' to train a model first.")
        
        # Initialize MediaPipe Hands
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print(f"Hand Tracker initialized - MediaPipe mode (detection: {detection_confidence}, tracking: {tracking_confidence})")
        else:
            raise ImportError("MediaPipe is required for HandTracker. Install with: pip install mediapipe")
        
        # Tracking state
        self.current_landmarks = None
        self.landmark_history = deque(maxlen=10)
        self.hand_bbox = None
        self.hand_present = False
        
        # Smoothing filters
        self.ema_landmarks = None
        self.ema_alpha = 0.75  # 75% current, 25% history
        
        # Finger analysis
        self.finger_states = {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
        self.finger_angles = {}
        
        # ASL Recognition state
        self.current_asl_prediction = "No gesture"
        self.asl_confidence = 0.0
        self.prediction_history = deque(maxlen=5)  # Smooth predictions over 5 frames
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Export data storage
        self.tracking_data = []
        
        # Hand landmark indices (MediaPipe format)
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        self.FINGER_PIPS = [2, 6, 10, 14, 18]  # Proximal interphalangeal joints
        
        # Joint angle indices: [MCP, PIP, DIP] for each finger
        # MCP = Metacarpophalangeal, PIP = Proximal Interphalangeal, DIP = Distal Interphalangeal
        self.THUMB_JOINTS = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]  # CMC, MCP, IP
        self.INDEX_JOINTS = [(0, 5, 6), (5, 6, 7), (6, 7, 8)]  # MCP, PIP, DIP
        self.MIDDLE_JOINTS = [(0, 9, 10), (9, 10, 11), (10, 11, 12)]
        self.RING_JOINTS = [(0, 13, 14), (13, 14, 15), (14, 15, 16)]
        self.PINKY_JOINTS = [(0, 17, 18), (17, 18, 19), (18, 19, 20)]
        
        self.ALL_FINGER_JOINTS = {
            'thumb': self.THUMB_JOINTS,
            'index': self.INDEX_JOINTS,
            'middle': self.MIDDLE_JOINTS,
            'ring': self.RING_JOINTS,
            'pinky': self.PINKY_JOINTS
        }
        
    def process_frame(self, frame):
        """
        Process a frame and detect hand landmarks
        
        Args:
            frame: BGR image from camera/video
            
        Returns:
            results: MediaPipe hand detection results
            landmarks_3d: List of 21 landmarks with (x, y, z) coordinates
            landmarks_2d: List of 21 landmarks with pixel (x, y) coordinates
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        landmarks_3d = []
        landmarks_2d = []
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks
            for landmark in hand_landmarks.landmark:
                # Normalized 3D coordinates
                landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                
                # Pixel coordinates
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                landmarks_2d.append([px, py])
            
            # Apply smoothing if enabled
            if self.enable_smoothing:
                landmarks_2d = self._apply_smoothing(landmarks_2d)
            
            self.current_landmarks = landmarks_2d
            self.landmark_history.append(landmarks_2d)
            self.hand_present = True
            
            # Calculate bounding box
            self.hand_bbox = self._calculate_bbox(landmarks_2d)
            
            # Analyze finger states
            self._analyze_fingers(landmarks_2d)
            
            # Predict ASL gesture if model is available
            if self.asl_enabled and landmarks_3d:
                self._predict_asl_gesture(landmarks_3d)
            
        else:
            self.hand_present = False
            self.current_landmarks = None
            self.current_asl_prediction = "No gesture"
            self.asl_confidence = 0.0
        
        # Update FPS
        self._update_fps()
        
        return results, landmarks_3d, landmarks_2d
    
    def _apply_smoothing(self, landmarks):
        """Apply EMA smoothing to landmarks"""
        if self.ema_landmarks is None or len(self.ema_landmarks) != len(landmarks):
            self.ema_landmarks = landmarks.copy()
            return landmarks
        
        smoothed = []
        for curr, prev in zip(landmarks, self.ema_landmarks):
            x = self.ema_alpha * curr[0] + (1 - self.ema_alpha) * prev[0]
            y = self.ema_alpha * curr[1] + (1 - self.ema_alpha) * prev[1]
            smoothed.append([int(x), int(y)])
        
        self.ema_landmarks = smoothed
        return smoothed
    
    def _calculate_bbox(self, landmarks):
        """Calculate tight bounding box around hand"""
        if not landmarks:
            return None
        
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add padding
        padding = 20
        return (x_min - padding, y_min - padding, 
                x_max - x_min + 2*padding, y_max - y_min + 2*padding)
    
    def _analyze_fingers(self, landmarks):
        """Analyze finger extension states and detailed joint angles (DIP, PIP, MCP)"""
        if len(landmarks) != 21:
            return
        
        # Check which fingers are extended
        wrist = np.array(landmarks[self.WRIST])
        
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        tip_indices = [4, 8, 12, 16, 20]
        pip_indices = [2, 6, 10, 14, 18]
        
        for name, tip_idx, pip_idx in zip(finger_names, tip_indices, pip_indices):
            tip = np.array(landmarks[tip_idx])
            pip = np.array(landmarks[pip_idx])
            
            # Calculate distance from wrist
            tip_dist = np.linalg.norm(tip - wrist)
            pip_dist = np.linalg.norm(pip - wrist)
            
            # Finger is extended if tip is farther from wrist than pip
            self.finger_states[name] = tip_dist > pip_dist
        
        # Calculate detailed joint angles if enabled
        if self.enable_angle_analysis:
            self.finger_angles = {}
            
            for finger_name, joints in self.ALL_FINGER_JOINTS.items():
                joint_angles = []
                joint_names = ['MCP', 'PIP', 'DIP'] if finger_name != 'thumb' else ['CMC', 'MCP', 'IP']
                
                for (p1_idx, p2_idx, p3_idx), joint_name in zip(joints, joint_names):
                    p1 = np.array(landmarks[p1_idx])
                    p2 = np.array(landmarks[p2_idx])
                    p3 = np.array(landmarks[p3_idx])
                    
                    angle = self._calculate_angle(p1, p2, p3)
                    joint_angles.append({
                        'joint': joint_name,
                        'angle': round(angle, 2)
                    })
                
                self.finger_angles[finger_name] = joint_angles
    
    def _predict_asl_gesture(self, landmarks_3d):
        """Predict ASL gesture from 3D landmarks using trained model"""
        if not self.asl_enabled or len(landmarks_3d) != 21:
            return
        
        try:
            # Convert landmarks to feature vector (63 features: 21 landmarks × 3 coords)
            features = []
            for landmark in landmarks_3d:
                features.extend(landmark)  # [x, y, z]
            
            # Reshape for model input
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.asl_scaler.transform(features_array)
            
            # Predict
            prediction = self.asl_model.predict(features_scaled)[0]
            
            # Get confidence if available
            if hasattr(self.asl_model, 'predict_proba'):
                probabilities = self.asl_model.predict_proba(features_scaled)[0]
                pred_idx = list(self.asl_model.classes_).index(prediction)
                confidence = probabilities[pred_idx]
            else:
                confidence = 0.75  # Default confidence for models without probability
            
            # Add to prediction history for smoothing
            self.prediction_history.append((prediction, confidence))
            
            # Use majority voting for stable prediction
            if len(self.prediction_history) >= 3:
                from collections import Counter
                recent_preds = [p[0] for p in self.prediction_history]
                most_common = Counter(recent_preds).most_common(1)[0][0]
                avg_confidence = sum(p[1] for p in self.prediction_history) / len(self.prediction_history)
                
                # Only update if confidence is reasonable
                if avg_confidence > 0.5:
                    self.current_asl_prediction = most_common
                    self.asl_confidence = avg_confidence
            else:
                self.current_asl_prediction = prediction
                self.asl_confidence = confidence
                
        except Exception as e:
            print(f"ASL prediction error: {e}")
            self.current_asl_prediction = "Error"
            self.asl_confidence = 0.0
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (in degrees)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.fps_history) > 0:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def get_finger_count(self):
        """Get number of extended fingers"""
        return sum(self.finger_states.values())
    
    def draw_landmarks(self, frame, landmarks_2d, draw_connections=True, draw_labels=False):
        """
        Draw hand landmarks on frame
        
        Args:
            frame: Image to draw on
            landmarks_2d: List of 21 (x, y) pixel coordinates
            draw_connections: Draw skeleton connections
            draw_labels: Draw landmark numbers
        """
        if not landmarks_2d or len(landmarks_2d) != 21:
            return frame
        
        # Define finger connections
        connections = [
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
        
        # Draw connections
        if draw_connections:
            for start_idx, end_idx in connections:
                start = tuple(landmarks_2d[start_idx])
                end = tuple(landmarks_2d[end_idx])
                cv2.line(frame, start, end, (0, 255, 0), 1)
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks_2d):
            # Color coding
            if i == 0:  # Wrist
                color = (255, 0, 0)
                radius = 4
            elif i in self.FINGER_TIPS:  # Fingertips
                color = (0, 0, 255)
                radius = 4
            else:  # Joints
                color = (255, 255, 0)
                radius = 3
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), 1)
            
            # Draw labels if requested
            if draw_labels:
                cv2.putText(frame, str(i), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame
    
    def draw_info(self, frame, detailed_angles=True):
        """Draw tracking information overlay with optional detailed joint angles"""
        # Info panel background
        panel_height = 250 if detailed_angles and self.enable_angle_analysis else 180
        cv2.rectangle(frame, (10, 10), (450, panel_height), (0, 0, 0), -1)
        
        # ASL Prediction (real prediction from trained model)
        if self.hand_present and self.asl_enabled:
            color = (0, 255, 0) if self.asl_confidence > 0.7 else (0, 165, 255)
            asl_text = f"ASL: {self.current_asl_prediction}"
            cv2.putText(frame, asl_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Confidence
            conf_text = f"Confidence: {self.asl_confidence:.2f}"
            cv2.putText(frame, conf_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        elif self.hand_present and not self.asl_enabled:
            cv2.putText(frame, "ASL: Model not loaded", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Hand detection status
        status = "HAND DETECTED" if self.hand_present else "NO HAND"
        color = (0, 255, 0) if self.hand_present else (0, 0, 255)
        y_offset = 85 if (self.hand_present and self.asl_enabled) else 65
        cv2.putText(frame, status, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS
        fps = self.get_fps()
        y_offset += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Finger count
        if self.hand_present:
            y_offset += 25
            finger_count = self.get_finger_count()
            cv2.putText(frame, f"Fingers Extended: {finger_count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detailed joint angles
            if detailed_angles and self.enable_angle_analysis and self.finger_angles:
                y_offset += 25
                cv2.putText(frame, "Joint Angles (degrees):", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                y_offset += 20
                
                for finger_name, joints in self.finger_angles.items():
                    angle_text = f"{finger_name.capitalize()}: "
                    for joint in joints:
                        angle_text += f"{joint['joint']}={joint['angle']:.0f}° "
                    
                    cv2.putText(frame, angle_text, (25, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    y_offset += 18
            else:
                # Simple finger states
                y_offset += 25
                for name, extended in self.finger_states.items():
                    state = "UP" if extended else "DOWN"
                    color = (0, 255, 0) if extended else (100, 100, 100)
                    cv2.putText(frame, f"{name.capitalize()}: {state}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 20
        
        return frame
    
    def draw_bbox(self, frame):
        """Draw bounding box around hand"""
        if self.hand_bbox:
            x, y, w, h = self.hand_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Corner markers
            corner_len = 15
            cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 255), 3)
            cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y), (x + w - corner_len, y), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y), (x + w, y + corner_len), (0, 255, 255), 3)
            cv2.line(frame, (x, y + h), (x + corner_len, y + h), (0, 255, 255), 3)
            cv2.line(frame, (x, y + h), (x, y + h - corner_len), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), (0, 255, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), (0, 255, 255), 3)
        
        return frame
    
    def save_tracking_data(self, frame_id, landmarks_3d, landmarks_2d):
        """Store tracking data for export"""
        data = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'hand_present': self.hand_present,
            'landmarks_3d': landmarks_3d,
            'landmarks_2d': landmarks_2d,
            'finger_states': self.finger_states.copy(),
            'finger_angles': self.finger_angles.copy(),
            'bbox': self.hand_bbox,
            'fps': self.get_fps()
        }
        self.tracking_data.append(data)
    
    def export_to_json(self, filepath='hand_tracking_data.json'):
        """Export tracking data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        print(f"Tracking data exported to {filepath}")
    
    def export_landmarks_csv(self, filepath='hand_landmarks.csv'):
        """Export landmarks to CSV format (compatible with ML training)"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['frame_id', 'timestamp', 'fps']
            for i in range(21):
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            header.extend(['finger_count', 'thumb', 'index', 'middle', 'ring', 'pinky'])
            
            # Add angle headers if available
            if self.enable_angle_analysis and len(self.tracking_data) > 0:
                if 'finger_angles' in self.tracking_data[0] and self.tracking_data[0]['finger_angles']:
                    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                        header.extend([f'{finger}_MCP', f'{finger}_PIP', f'{finger}_DIP'])
            
            writer.writerow(header)
            
            # Data rows
            for data in self.tracking_data:
                if not data['hand_present']:
                    continue
                
                row = [data['frame_id'], data['timestamp'], data['fps']]
                
                # Landmarks
                for landmark in data['landmarks_3d']:
                    row.extend(landmark)
                
                # Finger states
                row.append(sum(data['finger_states'].values()))
                row.extend([int(v) for v in data['finger_states'].values()])
                
                # Joint angles
                if self.enable_angle_analysis and 'finger_angles' in data and data['finger_angles']:
                    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                        if finger in data['finger_angles']:
                            for joint_data in data['finger_angles'][finger]:
                                row.append(joint_data['angle'])
                
                writer.writerow(row)
        
        print(f"Landmarks exported to {filepath}")
    
    def export_to_onnx(self, filepath='hand_tracking_model.onnx'):
        """
        Export MediaPipe model configuration to ONNX-compatible format
        Note: MediaPipe models are pre-trained, this exports tracking configuration
        """
        config = {
            'model_type': 'MediaPipe_Hand_Landmark',
            'num_landmarks': 21,
            'max_hands': self.max_hands,
            'detection_confidence': self.detection_confidence,
            'tracking_confidence': self.tracking_confidence,
            'smoothing_enabled': self.enable_smoothing,
            'ema_alpha': self.ema_alpha,
            'angle_analysis_enabled': self.enable_angle_analysis,
            'joint_indices': {
                'thumb': self.THUMB_JOINTS,
                'index': self.INDEX_JOINTS,
                'middle': self.MIDDLE_JOINTS,
                'ring': self.RING_JOINTS,
                'pinky': self.PINKY_JOINTS
            },
            'landmark_names': [
                'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
                'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
                'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
                'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
            ]
        }
        
        # Save configuration as JSON (ONNX model export requires separate training)
        config_file = filepath.replace('.onnx', '_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model configuration exported to {config_file}")
        print("Note: MediaPipe uses pre-trained models. For custom ONNX export, train a custom model.")
        print("Current tracking data can be used as training dataset via export_landmarks_csv()")
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        if not self.tracking_data:
            return {}
        
        fps_values = [d['fps'] for d in self.tracking_data]
        
        stats = {
            'total_frames': len(self.tracking_data),
            'avg_fps': sum(fps_values) / len(fps_values),
            'min_fps': min(fps_values),
            'max_fps': max(fps_values),
            'frames_with_hand': sum(1 for d in self.tracking_data if d['hand_present']),
            'detection_rate': sum(1 for d in self.tracking_data if d['hand_present']) / len(self.tracking_data) * 100
        }
        
        return stats
    
    def release(self):
        """Release resources"""
        if self.hands:
            self.hands.close()


def demo_hand_tracking():
    """Demo the hand tracking system with real ASL recognition"""
    print("=" * 60)
    print("ULTIMATE HAND TRACKING + ASL RECOGNITION DEMO")
    print("=" * 60)
    print("Features:")
    print("  ✓ 21-point hand landmark detection")
    print("  ✓ Real ASL gesture recognition (A-Z)")
    print("  ✓ DIP, PIP, MCP joint angle analysis")
    print("  ✓ Real-time FPS monitoring")
    print("  ✓ Export to JSON/CSV/ONNX config")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  e - Export tracking data")
    print("  a - Toggle angle display")
    print("  p - Print performance stats")
    print("=" * 60)
    
    # Initialize tracker with full features + ASL model
    tracker = HandTracker(
        max_hands=1,  # Use 1 hand for ASL recognition
        detection_confidence=0.8,
        tracking_confidence=0.7,
        enable_smoothing=True,
        enable_angle_analysis=True,
        model_path='model.pkl',
        scaler_path='scaler.pkl'
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    frame_id = 0
    show_angles = True
    
    if tracker.asl_enabled:
        print("\n✓ ASL Recognition ENABLED - Show ASL gestures (A-Z)")
    else:
        print("\n⚠ ASL Recognition DISABLED - Hand tracking only")
        print("  Run 'python data_loader_and_training.py' to train ASL model first")
    
    print("\nStarting hand tracking... Show your hand to the camera!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results, landmarks_3d, landmarks_2d = tracker.process_frame(frame)
        
        # Save tracking data
        if tracker.hand_present:
            tracker.save_tracking_data(frame_id, landmarks_3d, landmarks_2d)
        
        # Draw visualizations
        if landmarks_2d:
            frame = tracker.draw_landmarks(frame, landmarks_2d, draw_connections=True)
            frame = tracker.draw_bbox(frame)
        
        frame = tracker.draw_info(frame, detailed_angles=show_angles)
        
        # Display
        cv2.imshow('Ultimate Hand Tracking', frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'hand_frame_{frame_id}.jpg'
            cv2.imwrite(filename, frame)
            print(f"✓ Saved {filename}")
        elif key == ord('e'):
            print("\nExporting tracking data...")
            tracker.export_to_json('tracking_data.json')
            tracker.export_landmarks_csv('landmarks.csv')
            tracker.export_to_onnx('hand_model.onnx')
            print("✓ Export complete!")
        elif key == ord('a'):
            show_angles = not show_angles
            print(f"Angle display: {'ON' if show_angles else 'OFF'}")
        elif key == ord('p'):
            stats = tracker.get_performance_stats()
            print("\n" + "=" * 40)
            print("PERFORMANCE STATISTICS")
            print("=" * 40)
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("=" * 40)
        
        frame_id += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    
    # Final export and statistics
    if len(tracker.tracking_data) > 0:
        print("\n" + "=" * 60)
        print("SESSION COMPLETE")
        print("=" * 60)
        
        tracker.export_to_json('final_tracking_data.json')
        tracker.export_landmarks_csv('final_landmarks.csv')
        tracker.export_to_onnx('final_hand_model.onnx')
        
        stats = tracker.get_performance_stats()
        print(f"\nFrames processed: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Detection rate: {stats['detection_rate']:.1f}%")
        print(f"Exported files:")
        print("  - final_tracking_data.json")
        print("  - final_landmarks.csv")
        print("  - final_hand_model_config.json")
        print("\n✓ All data exported successfully!")
        print("=" * 60)


if __name__ == "__main__":
    demo_hand_tracking()
