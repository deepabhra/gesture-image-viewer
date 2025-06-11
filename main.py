import cv2
import os
import glob
import time
import numpy as np
from collections import deque
from cvzone.HandTrackingModule import HandDetector

# === Config ===
image_folder = r"E:\college\Sem_VI\Project\gesture-image-viewer\resources"
zoom_step = 0.05
min_zoom = 0.1
max_zoom = 5.0
gesture_cooldown = 0.6
frame_width, frame_height = 800, 540

# === Load Images ===
image_paths = glob.glob(os.path.join(image_folder, '*.*'))
supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in supported_ext]

if not image_paths:
    print("⚠️ No images found in the folder.")
    exit()

# === Init State ===
index = 0
zoom_factor = 0.1
target_zoom = 0.1
detector = HandDetector(detectionCon=0.7, maxHands=2)
gesture_buffer = deque(maxlen=3)
last_gesture_time = time.time()
last_displayed_index = -1
last_zoom_factor = -1

def display_image(img_path, zoom):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return None

    h, w = img.shape[:2]
    new_w, new_h = int(w * zoom), int(h * zoom)

    # Ensure the zoomed image dimensions are valid
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Resize the image based on the zoom factor
    resized = cv2.resize(img, (new_w, new_h))

    # Create a blank canvas
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Calculate offsets to center the image on the canvas
    x_offset = max(0, (frame_width - new_w) // 2)
    y_offset = max(0, (frame_height - new_h) // 2)

    # Calculate the region of the resized image to display
    x_start = max(0, -x_offset)
    y_start = max(0, -y_offset)
    x_end = min(new_w, frame_width - x_offset)
    y_end = min(new_h, frame_height - y_offset)

    # Place the resized image on the canvas
    canvas[y_offset:y_offset + (y_end - y_start), x_offset:x_offset + (x_end - x_start)] = resized[y_start:y_end, x_start:x_end]

    return canvas

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, draw = False)
    left_hand, right_hand = None, None
    current_time = time.time()

    gesture = None  # Reset gesture

    if hands:
        for hand in hands:
            if hand["lmList"][0][0] < frame.shape[1] // 2:
                right_hand = hand
            else:
                left_hand = hand
        
        # Draw connection dots and lines for each hand
        for hand in hands:
            lmList = hand["lmList"]
            # Draw dots
            for lm in lmList:
                cv2.circle(frame, (lm[0], lm[1]), 7, (0, 255, 0), cv2.FILLED)
            # Draw connections (lines)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                (0, 9), (9,10), (10,11), (11,12),    # Middle
                (0,13), (13,14), (14,15), (15,16),   # Ring
                (0,17), (17,18), (18,19), (19,20)    # Pinky
            ]
            for start, end in connections:
                cv2.line(frame, (lmList[start][0], lmList[start][1]), (lmList[end][0], lmList[end][1]), (255, 0, 0), 2)

        # Handle exit gesture (right hand with all fingers down)
        if right_hand and detector.fingersUp(right_hand) == [0, 0, 0, 0, 0]:
            if current_time - last_gesture_time > gesture_cooldown:
                print("👋 Exiting program...")
                break

        # Handle delete gesture (left hand grab - all fingers curled)
        elif left_hand and detector.fingersUp(left_hand) == [0, 0, 0, 0, 0]:
            if current_time - last_gesture_time > gesture_cooldown:
                print("🗑️ Deleted current image...")
                os.remove(image_paths[index])
                del image_paths[index]
                if not image_paths:
                    print("🧼 All images deleted. Exiting.")
                    break
                index %= len(image_paths)
                zoom_factor = 0.1
                last_gesture_time = current_time
                continue

        # Handle zoom gestures
        elif left_hand and right_hand:
            left_dist, _, _ = detector.findDistance(left_hand["lmList"][4][:2], left_hand["lmList"][8][:2], img)
            right_dist, _, _ = detector.findDistance(right_hand["lmList"][4][:2], right_hand["lmList"][8][:2], img)

            if current_time - last_gesture_time > gesture_cooldown:
                if left_dist < 40 and right_dist < 40:
                    target_zoom = max(min_zoom, target_zoom - zoom_step)
                    gesture = "🔍 Zoom Out"
                elif left_dist > 80 and right_dist > 80:
                    target_zoom = min(max_zoom, target_zoom + zoom_step)
                    gesture = "🔍 Zoom In"
                last_gesture_time = current_time

        # Handle image navigation gestures
        elif left_hand and detector.fingersUp(left_hand) == [0, 1, 1, 1, 1]:
            gesture_buffer.append("next")

        elif right_hand and detector.fingersUp(right_hand) == [0, 1, 1, 1, 1]:
            gesture_buffer.append("prev")

        if len(gesture_buffer) == gesture_buffer.maxlen and all(g == "next" for g in gesture_buffer):
            if current_time - last_gesture_time > gesture_cooldown:
                index = (index + 1) % len(image_paths)
                target_zoom = 0.1
                gesture = "➡️ Next Image"
                last_gesture_time = current_time
                gesture_buffer.clear()

        elif len(gesture_buffer) == gesture_buffer.maxlen and all(g == "prev" for g in gesture_buffer):
            if current_time - last_gesture_time > gesture_cooldown:
                index = (index - 1) % len(image_paths)
                target_zoom = 0.1
                gesture = "⬅️ Previous Image"
                last_gesture_time = current_time
                gesture_buffer.clear()

    zoom_factor += (target_zoom - zoom_factor) * 0.2

    # Only print output when a gesture occurs (gesture != None)
    if gesture:
        print(f"Gesture: {gesture}")  # Print gesture in terminal
        cv2.putText(frame, gesture, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display the image with the updated zoom
    image_view = display_image(image_paths[index], zoom_factor)
    if image_view is not None:
        cv2.imshow("Image Viewer", image_view)

    cv2.putText(frame, f"Zoom: {zoom_factor:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Image {index+1}/{len(image_paths)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 Exiting via ESC.")
        break

cap.release()
cv2.destroyAllWindows()