import cv2
import numpy as np
import random

def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def load_colors(classes):
    # Assign specific colors to each class
    class_colors = {}
    for idx, cls in enumerate(classes):
        if cls == "head":
            class_colors[cls] = (145, 255, 0)  # Blue for class 0
        elif cls == "person":
            class_colors[cls] = (255, 213, 0)  # Green for class 1
        else:
            # Assign random color for other classes
            class_colors[cls] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return class_colors

def draw_detection(frame, classes, class_colors, detections, confidence_threshold=0.5, alpha=0.1):
    num_person_detections = 0
    num_head_detections = 0

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Get the coordinates of the bounding box
                center_x, center_y, w, h = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)

                # Calculate the top-left corner of the bounding box
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Set color based on class
                color = class_colors[classes[class_id]]

                # Create a hologram effect by adding transparency to the bounding box
                hologram_frame = frame.copy()
                cv2.rectangle(hologram_frame, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(hologram_frame, alpha, frame, 1 - alpha, 0, frame)

                # Draw the bounding box with transparent edges
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
                cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)

                # Draw the label on the frame
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if classes[class_id] == "person":
                    num_person_detections += 1
                elif classes[class_id] == "head":
                    num_head_detections += 1

    return frame, num_person_detections, num_head_detections

def draw_text(frame, text, position, color):
    alpha = 1  # Adjust the alpha value for transparency
    font_size = 0.6  # Adjust the font size
    thickness = 2  # Adjust the thickness

    overlay = frame.copy()

    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)

    # Add the text to the frame with transparency
    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)