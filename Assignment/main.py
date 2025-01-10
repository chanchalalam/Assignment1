
import cv2
import json
import time
import os
from ultralytics import YOLO
from utils import extract_subobjects, format_json, save_subobject_images

# Initialize model
model = YOLO("yolov8n.pt")

# Path to video
cap = cv2.VideoCapture("videos/people.mp4")

# Directory for output images if not exists
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)


# Benchmarking variables
start_time = time.time()
frame_count = 0

# Results
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device="cpu")
    frame_count += 1

    # Process detections
    for result in results:
        objects, sub_objects = extract_subobjects(result.boxes.xyxy, result.names, frame)
        detection = format_json(objects, sub_objects)
        detections.append(detection)

        # Debugging
        print("Detection:", detection)

        # Save sub-object images with frame number
        save_subobject_images(objects, sub_objects, frame, output_dir, frame_count)

# Calculate FPS
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print(f"FPS: {fps:.2f}")

cap.release()

# Save JSON output
output_path = "output.json"
with open(output_path, "w") as f:
    json.dump(detections, f, indent=4)
print(f"Results saved to {output_path}")


