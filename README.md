# Object and Sub-Object Detection System
<img width="1440" alt="Screenshot 2025-01-10 at 12 33 50 PM" src="https://github.com/user-attachments/assets/18a9ab11-57ef-4909-ae47-8abb25f0a009" />


## Overview

This project is a computer vision system for detecting objects and their associated sub-objects in a hierarchical structure. It processes video streams in real-time, outputs results in JSON format, and allows retrieval of cropped images for detected objects and sub-objects. The system is optimized for CPU inference to achieve real-time performance.

## Features

Hierarchical Detection: Objects and associated sub-objects with unique IDs.

JSON Output: Outputs detection in structured format.

Image Saving: Cropped images of objects and sub-objects.

Real-Time: Optimized for 10–30 FPS on CPU.


## Requirements

Python 3.8+

Required Libraries:

ultralytics,opencv-python,numpy

## How to Run the System

1. Clone the Repository: git clone https://github.com/chanchalalam/Assignment1.git
   
2. Prepare Input Video: Place your input video in the videos/ directory.
 Update the video path in main.py: cap = cv2.VideoCapture("videos/people.mp4")
   
5. Run the System: Execute the main script: python main.py
   
6. View Outputs: JSON results are saved in output.json. Cropped images are saved in the output_images/ directory.



