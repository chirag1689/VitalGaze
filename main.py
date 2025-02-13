import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, Label
from PIL import Image, ImageTk
import threading
from queue import Queue

class ResultWindow:
    def __init__(self, image, results):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Analysis Results")
        self.root.geometry("800x900")
        self.root.configure(bg='white')
        
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Resize image while maintaining aspect ratio
        display_size = (600, 450)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image label
        self.image_label = ttk.Label(self.main_frame, image=self.photo)
        self.image_label.pack(pady=20)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Result.TLabel', font=('Helvetica', 12))
        self.style.configure('Warning.TLabel', font=('Helvetica', 12), foreground='red')
        self.style.configure('Normal.TLabel', font=('Helvetica', 12), foreground='green')
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        
        # Results frame
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Add title
        title = ttk.Label(self.results_frame, text="Analysis Results", style='Title.TLabel')
        title.pack(pady=10)
        
        # Add separator
        ttk.Separator(self.results_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Display results
        self._add_result("Pimples Detected", 
                        str(results['skin_conditions']['pimples_detected']),
                        results['skin_conditions']['pimples_detected'] > 5)
        
        self._add_result("Skin Redness", 
                        f"{results['skin_conditions']['redness_level']:.1f}",
                        results['skin_conditions']['redness_level'] > 150)
        
        self._add_result("Eye Droopiness", 
                        f"{results['eye_analysis']['eye_ratio']:.2f}",
                        results['eye_analysis']['eye_ratio'] < 0.25)
        
        self._add_result("Toxicated", 
                        "Yes" if results['is_toxicated'] else "No",
                        results['is_toxicated'])
        
        # Add separator
        ttk.Separator(self.results_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Overall status
        status_frame = ttk.Frame(self.results_frame)
        status_frame.pack(pady=10)
        
        status_label = ttk.Label(status_frame, text="Overall Status: ", 
                               style='Header.TLabel')
        status_label.pack(side=tk.LEFT)
        
        status_style = 'Warning.TLabel' if results['is_abnormal'] else 'Normal.TLabel'
        status_text = "ABNORMAL" if results['is_abnormal'] else "NORMAL"
        status_value = ttk.Label(status_frame, text=status_text, style=status_style)
        status_value.pack(side=tk.LEFT)
        
        # Close button
        close_button = ttk.Button(self.main_frame, text="Close", 
                                command=self.root.destroy)
        close_button.pack(pady=20)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Start the window
        self.root.mainloop()
    
    def _add_result(self, label, value, is_warning):
        """Add a result row to the results frame"""
        frame = ttk.Frame(self.results_frame)
        frame.pack(fill='x', pady=5)
        
        label = ttk.Label(frame, text=f"{label}:", style='Result.TLabel')
        label.pack(side=tk.LEFT, padx=(20, 10))
        
        style = 'Warning.TLabel' if is_warning else 'Result.TLabel'
        value_label = ttk.Label(frame, text=value, style=style)
        value_label.pack(side=tk.LEFT)

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
        self.REDNESS_THRESHOLD = 150
        self.EYE_DROOPINESS_THRESHOLD = 0.25
        self.PIMPLE_THRESHOLD = 5
        
        # Abnormal thresholds
        self.ABNORMAL_REDNESS = 200
        self.ABNORMAL_EYE_DROOPINESS = 0.15
        self.ABNORMAL_PIMPLES = 10
        
        # Toxication thresholds
        self.TOXICATION_REDNESS_THRESHOLD = 180
        self.TOXICATION_EYE_THRESHOLD = 0.2

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
        
        # Determine toxication
        is_toxicated = self.detect_toxication(skin_analysis, eye_analysis)
        
        # Determine if the person is abnormal
        is_abnormal = self.determine_abnormality(skin_analysis, eye_analysis, is_toxicated)
        
        # Combine results
        analysis_results = {
            'skin_conditions': skin_analysis,
            'eye_analysis': eye_analysis,
            'is_toxicated': is_toxicated,
            'is_abnormal': is_abnormal
        }
        
        # Draw findings on frame
        frame = self.draw_analysis_on_frame(frame, landmarks, analysis_results)
        
        return frame, analysis_results
    
    def detect_toxication(self, skin_analysis, eye_analysis):
        """Detect if person shows signs of toxication"""
        high_redness = skin_analysis['redness_level'] > self.TOXICATION_REDNESS_THRESHOLD
        droopy_eyes = eye_analysis['eye_ratio'] < self.TOXICATION_EYE_THRESHOLD
        
        # Consider toxicated if both conditions are met
        return high_redness and droopy_eyes
    
    def determine_abnormality(self, skin_analysis, eye_analysis, is_toxicated):
        """Determine if the person is abnormal based on thresholds"""
        high_redness = skin_analysis['redness_level'] > self.ABNORMAL_REDNESS
        high_pimples = skin_analysis['pimples_detected'] > self.ABNORMAL_PIMPLES
        severe_droopiness = eye_analysis['eye_ratio'] < self.ABNORMAL_EYE_DROOPINESS
        
        return high_redness or high_pimples or severe_droopiness or is_toxicated

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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        skin_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        redness = np.mean(skin_hsv[:, :, 1][mask > 0])
        
        pimples = self.detect_pimples(gray, mask)
        
        return {
            'redness_level': float(redness),
            'pimples_detected': len(pimples)
        }
    
    def detect_pimples(self, gray_img, mask):
        """Detect pimples in the face region"""
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 2)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        
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
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        left_ear = self.calculate_eye_ratio(landmarks, left_eye)
        right_ear = self.calculate_eye_ratio(landmarks, right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        eye_redness = self.detect_eye_redness(frame, landmarks, left_eye + right_eye)
        
        return {
            'eye_ratio': avg_ear,
            'redness': eye_redness
        }
    
    def calculate_eye_ratio(self, landmarks, eye_points):
        """Calculate the eye aspect ratio"""
        points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                          for point in eye_points])
        
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
        height, width = frame.shape[:2]
        eye_mask = np.zeros((height, width), dtype=np.uint8)
        
        for point in eye_points:
            x = int(landmarks.landmark[point].x * width)
            y = int(landmarks.landmark[point].y * height)
            cv2.circle(eye_mask, (x, y), 2, 255, -1)
        
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
        
        return frame

def show_results(image, results):
    """Show results in a new window"""
    ResultWindow(image, results)

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
                # Start result window in a new thread
                threading.Thread(target=show_results, 
                              args=(analyzed_frame, results),
                              daemon=True).start()
                
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