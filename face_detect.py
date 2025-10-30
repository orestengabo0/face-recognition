import cv2
import time
from collections import deque
import numpy as np

# Load multiple cascade classifiers for better detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam with optimized settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Movement tracking variables
position_history = deque(maxlen=5)  # Store last 5 positions for smoothing
time_history = deque(maxlen=5)
prev_center = None
prev_time = None

# Calibration and thresholds
MOVEMENT_THRESHOLD = 8  # Minimum pixels to detect movement
DIRECTION_THRESHOLD = 15  # Minimum displacement for clear direction
MIN_FACE_SIZE = (80, 80)  # Minimum face size to detect
SMOOTHING_FACTOR = 0.7  # For exponential smoothing (0-1)

# Tracking state
smoothed_center = None
frame_count = 0
detection_confidence = 0

def detect_face_with_eyes(frame, gray):
    """Enhanced face detection using multiple methods and eye verification"""
    
    # Method 1: Primary face detection
    faces1 = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Method 2: Alternative cascade for better detection
    faces2 = face_cascade_alt.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )
    
    # Combine detections
    all_faces = list(faces1) + list(faces2)
    
    if len(all_faces) == 0:
        return None, 0
    
    # Score each face based on size and eye detection
    best_face = None
    best_score = 0
    
    for (x, y, w, h) in all_faces:
        score = w * h  # Base score on face size
        
        # Verify with eye detection
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Boost score if eyes detected (2 eyes is ideal)
        if len(eyes) >= 2:
            score *= 1.5
        elif len(eyes) == 1:
            score *= 1.2
        
        # Check face position (prefer centered faces)
        frame_center_x = frame.shape[1] / 2
        face_center_x = x + w / 2
        distance_from_center = abs(face_center_x - frame_center_x)
        center_bonus = max(0, 1 - distance_from_center / frame_center_x)
        score *= (1 + center_bonus * 0.3)
        
        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)
    
    # Calculate confidence (0-100)
    confidence = min(100, int((best_score / 50000) * 100))
    
    return best_face, confidence

def smooth_position(new_center, prev_smoothed):
    """Apply exponential smoothing to reduce jitter"""
    if prev_smoothed is None:
        return new_center
    
    smoothed_x = int(SMOOTHING_FACTOR * new_center[0] + (1 - SMOOTHING_FACTOR) * prev_smoothed[0])
    smoothed_y = int(SMOOTHING_FACTOR * new_center[1] + (1 - SMOOTHING_FACTOR) * prev_smoothed[1])
    
    return (smoothed_x, smoothed_y)

def calculate_movement_direction(positions, times):
    """Calculate movement direction using multiple frames for accuracy"""
    if len(positions) < 2:
        return None, None, 0, 0
    
    # Calculate average displacement
    total_dx = 0
    total_dy = 0
    total_time = 0
    
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dt = times[i] - times[i-1]
        
        total_dx += dx
        total_dy += dy
        total_time += dt
    
    # Average displacement
    avg_dx = total_dx / (len(positions) - 1)
    avg_dy = total_dy / (len(positions) - 1)
    
    # Calculate speed
    distance = (avg_dx**2 + avg_dy**2) ** 0.5
    speed = distance / total_time if total_time > 0 else 0
    
    # Determine primary direction with better logic
    direction = None
    abs_dx = abs(avg_dx)
    abs_dy = abs(avg_dy)
    
    # Check if movement is significant
    if abs_dx > MOVEMENT_THRESHOLD or abs_dy > MOVEMENT_THRESHOLD:
        # Determine if movement is more horizontal or vertical
        if abs_dx > abs_dy * 0.7:  # Primarily horizontal
            if avg_dx > DIRECTION_THRESHOLD:
                direction = "RIGHT"
            elif avg_dx < -DIRECTION_THRESHOLD:
                direction = "LEFT"
        
        if abs_dy > abs_dx * 0.7:  # Primarily vertical
            if avg_dy > DIRECTION_THRESHOLD:
                direction = "DOWN"
            elif avg_dy < -DIRECTION_THRESHOLD:
                direction = "UP"
        
        # Handle diagonal movements
        if abs_dx > DIRECTION_THRESHOLD and abs_dy > DIRECTION_THRESHOLD:
            h_dir = "RIGHT" if avg_dx > 0 else "LEFT"
            v_dir = "DOWN" if avg_dy > 0 else "UP"
            direction = f"{v_dir}-{h_dir}"
    
    return direction, speed, avg_dx, avg_dy

print("Enhanced Face Movement Tracker Started!")
print("Features: Multi-cascade detection, Eye verification, Smoothed tracking")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better detection in varying light
    gray = cv2.equalizeHist(gray)
    
    # Detect face with enhanced method
    face, confidence = detect_face_with_eyes(frame, gray)
    
    current_time = time.time()
    frame_count += 1
    
    if face is not None:
        x, y, w, h = face
        
        # Calculate center
        cx = x + w // 2
        cy = y + h // 2
        
        # Apply smoothing
        smoothed_center = smooth_position((cx, cy), smoothed_center)
        sx, sy = smoothed_center
        
        # Store in history
        position_history.append(smoothed_center)
        time_history.append(current_time)
        
        # Draw face rectangle (color based on confidence)
        color = (0, int(confidence * 2.55), int(255 - confidence * 2.55))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (sx, sy), 3, (255, 255, 255), -1)
        
        # Draw detection confidence
        cv2.putText(frame, f"Confidence: {confidence}%", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate movement
        direction, speed, dx, dy = calculate_movement_direction(
            list(position_history), 
            list(time_history)
        )
        
        # Display movement information
        y_offset = 30
        if direction:
            # Direction and speed
            text = f"Moving: {direction}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_offset += 35
            
            speed_text = f"Speed: {speed:.1f} px/s"
            cv2.putText(frame, speed_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
            
            # Displacement details
            disp_text = f"dx: {dx:.1f}, dy: {dy:.1f}"
            cv2.putText(frame, disp_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw movement arrow
            arrow_start = (sx, sy)
            arrow_end = (int(sx + dx * 3), int(sy + dy * 3))
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
        else:
            cv2.putText(frame, "STABLE", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Display coordinates
        coord_text = f"Position: ({sx}, {sy})"
        cv2.putText(frame, coord_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    else:
        # No face detected - clear history
        position_history.clear()
        time_history.clear()
        smoothed_center = None
        
        cv2.putText(frame, "No Face Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Move closer or adjust lighting", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Display FPS
    if frame_count > 1:
        elapsed = current_time - time_history[0] if len(time_history) > 0 else 0.033
        fps = len(time_history) / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Enhanced Face Movement Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Face Movement Tracker Stopped")