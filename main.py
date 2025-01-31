import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os

class FacialHealthAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Create output directory
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Analysis thresholds
        self.REDNESS_THRESHOLD = 150  # Threshold for skin redness
        self.EYE_DROOPINESS_THRESHOLD = 0.25  # Threshold for eye droopiness
        self.PIMPLE_THRESHOLD = 5  # Threshold for number of pimples
        
        # Abnormal thresholds
        self.ABNORMAL_REDNESS = 200  # High redness threshold
        self.ABNORMAL_EYE_DROOPINESS = 0.15  # Severe eye droopiness threshold
        self.ABNORMAL_PIMPLES = 10  # High number of pimples threshold

    def analyze_face(self, frame):
        """Analyze facial features and conditions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, None
            
        landmarks = results.multi_face_landmarks[0]
        
        # Create face mask
        mask = self.create_face_mask(frame, landmarks)
        
        # Analyze different aspects
        skin_analysis = self.analyze_skin(frame, mask)
        eye_analysis = self.analyze_eyes(frame, landmarks)
        
        # Determine if the person is abnormal
        is_abnormal = self.determine_abnormality(skin_analysis, eye_analysis)
        
        # Combine results
        analysis_results = {
            'skin_conditions': skin_analysis,
            'eye_analysis': eye_analysis,
            'is_abnormal': is_abnormal
        }
        
        # Draw findings on frame
        frame = self.draw_analysis_on_frame(frame, landmarks, analysis_results)
        
        return frame, analysis_results
    
    def determine_abnormality(self, skin_analysis, eye_analysis):
        """Determine if the person is abnormal based on thresholds"""
        # Check skin conditions
        high_redness = skin_analysis['redness_level'] > self.ABNORMAL_REDNESS
        high_pimples = skin_analysis['pimples_detected'] > self.ABNORMAL_PIMPLES
        
        # Check eye conditions
        severe_droopiness = eye_analysis['eye_ratio'] < self.ABNORMAL_EYE_DROOPINESS
        
        # If any condition is met, the person is abnormal
        return high_redness or high_pimples or severe_droopiness

    def create_face_mask(self, frame, landmarks):
        """Create a mask for the face region"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) 
                 for landmark in landmarks.landmark]
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask
    
    def analyze_skin(self, frame, mask):
        """Analyze skin conditions"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze skin tone
        skin_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        redness = np.mean(skin_hsv[:, :, 1][mask > 0])
        
        # Detect pimples using blob detection
        pimples = self.detect_pimples(gray, mask)
        
        return {
            'redness_level': float(redness),
            'pimples_detected': len(pimples)
        }
    
    def detect_pimples(self, gray_img, mask):
        """Detect pimples in the face region"""
        # Apply threshold to detect potential pimples
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 2)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply mask
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter pimples by size and shape
        pimples = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 100:
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.7:
                    pimples.append(contour)
        
        return pimples
    
    def analyze_eyes(self, frame, landmarks):
        """Analyze eye-related features"""
        # Get eye landmarks
        left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_eye_ratio(landmarks, left_eye_landmarks)
        right_ear = self.calculate_eye_ratio(landmarks, right_eye_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect eye redness
        eye_redness = self.detect_eye_redness(frame, landmarks, left_eye_landmarks + right_eye_landmarks)
        
        return {
            'eye_ratio': avg_ear,
            'redness': eye_redness
        }
    
    def calculate_eye_ratio(self, landmarks, eye_points):
        """Calculate the eye aspect ratio"""
        points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                          for point in eye_points])
        
        # Calculate vertical and horizontal distances
        vertical_dist = np.mean([
            np.linalg.norm(points[1] - points[5]),
            np.linalg.norm(points[2] - points[4])
        ])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        
        if horizontal_dist == 0:
            return 0
        return vertical_dist / horizontal_dist
    
    def detect_eye_redness(self, frame, landmarks, eye_points):
        """Detect redness in eye regions"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create eye mask
        height, width = frame.shape[:2]
        eye_mask = np.zeros((height, width), dtype=np.uint8)
        
        for point in eye_points:
            x = int(landmarks.landmark[point].x * width)
            y = int(landmarks.landmark[point].y * height)
            cv2.circle(eye_mask, (x, y), 2, 255, -1)
        
        # Analyze redness in eye regions
        eye_region = cv2.bitwise_and(hsv, hsv, mask=eye_mask)
        redness = np.mean(eye_region[:, :, 1][eye_mask > 0])
        
        return float(redness)
    
    def draw_analysis_on_frame(self, frame, landmarks, results):
        """Draw analysis results on the frame"""
        height, width = frame.shape[:2]
        
        # Draw face mesh points
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Add analysis results text
        y_pos = 30
        cv2.putText(frame, "Analysis Results:", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show skin analysis
        y_pos += 30
        skin = results['skin_conditions']
        cv2.putText(frame, f"Pimples: {skin['pimples_detected']}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        y_pos += 25
        cv2.putText(frame, f"Skin redness: {skin['redness_level']:.1f}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Show eye analysis
        y_pos += 25
        eyes = results['eye_analysis']
        cv2.putText(frame, f"Eye droopiness: {eyes['eye_ratio']:.2f}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add abnormality status
        y_pos += 35
        status = "Abnormal" if results['is_abnormal'] else "Normal"
        color = (0, 0, 255) if results['is_abnormal'] else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

def main():
    analyzer = FacialHealthAnalyzer()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        cv2.imshow('Camera Preview (Press C to capture, Q to quit)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Analyze captured frame
            analyzed_frame, results = analyzer.analyze_face(frame.copy())
            
            if results is not None:
                # Show analyzed image
                cv2.imshow('Analysis Results', analyzed_frame)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"analysis_results/analyzed_{timestamp}.jpg", analyzed_frame)
            else:
                print("No face detected in the image")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()