A high-performance real-time crowd analytics and queue management system powered by advanced computer vision techniques to provide precise insights into crowd density and wait times.

## 🚀 Overview
**VisionLab** is designed to transform raw video streams into actionable data. By integrating state-of-the-art YOLO models with robust tracking algorithms, the system provides real-time analytics, including:
* **Precise Crowd Tracking:** Identifying and tracking individual movement patterns.
* **Dwell Time Analysis:** Calculating the average duration each individual spends within a defined area of interest.
* **Dynamic Heatmaps:** Visualizing congestion hotspots to facilitate efficient space management.
* **Interactive HUD:** A real-time dashboard displaying performance metrics and occupancy levels.

## 🛠 Tech Stack
* **Core:** Python
* **Vision Model:** YOLOv11 (Ultralytics)
* **Tracking:** ByteTrack
* **Processing:** OpenCV
* **Data Visualization:** Matplotlib & NumPy
* **Concurrency:** Multi-threading (to ensure smooth real-time processing)

## 📋 Key Features
* **Optimized Performance:** Utilizes multi-threading to minimize latency and maintain high FPS.
* **Intelligent Alerts:** Automated warning system triggered when crowd density exceeds the defined threshold.
* **Professional UI:** Live plotting to visualize density trends and average wait times.
* **Modular Design:** Highly configurable settings for sensitivity, heatmap decay, and threshold levels.

## ⚙️ Getting Started
1. Install the required dependencies:
```
   pip install opencv-python ultralytics matplotlib numpy
```
2. Configure your video source: 
   Update the `VIDEO_PATH` variable in `main.py`  to point to the local path of the video file you wish to process.
   
   > **Note:** Ensure that the `VIDEO_PATH` is correctly set to the absolute path of your local video file to allow the system to load the media successfully.
