# Real-Time Object Counter

A simple real-time Computer Vision project that uses a webcam and a pretrained YOLO11 model to detect and count objects.

The application captures frames from the webcam, runs object detection using YOLO11, draws bounding boxes around detected objects, and displays the number of detected objects for each class.

## Demo

<img width="1265" height="696" alt="Screenshot 2026-09-17 at 10 14 30 PM" src="https://github.com/user-attachments/assets/b114dae2-7bb3-4a60-8f07-29638d2b09cc" />


## Features

- Real-time webcam object detection
- YOLO11 pretrained object detection model
- Bounding boxes around detected objects
- Confidence scores for detections
- Object counting by class
- Runs locally using OpenCV
- Supports common COCO objects such as:
  - Person
  - Cell phone
  - Laptop
  - Bottle
  - Cup
  - Backpack
  - Chair
  - etc.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO11
- PyTorch

## How It Works

```text
Webcam
   |
   v
OpenCV
   |
   v
Video Frame
   |
   v
YOLO11
   |
   v
Object Detection
   |
   +------> Bounding Boxes
   |
   +------> Confidence Scores
   |
   +------> Object Classes
   |
   v
Object Counter
   |
   v

