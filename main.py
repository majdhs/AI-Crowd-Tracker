import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from queue import Queue
import matplotlib.pyplot as plt

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
VIDEO_PATH = "video path"  # Ensure this path is correct crowd1
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_LIMIT = 0.10  # Optimal for catching dense crowd targets

# Heatmap parameters tuned for instant color transformation (Blue -> Green -> Red)
HEATMAP_DECAY = 0.950   # Extremely slow decay to allow continuous accumulation
BLUR_RADIUS = 35        # Large radius to force overlapping heat signatures in dense clusters
CROWD_THRESHOLD = 8
# ==========================================
# 2. THREADED VIDEO STREAMING ENGINE
# ==========================================
class ThreadedVideoStream:
    def __init__(self, src=VIDEO_PATH):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.queue = Queue(maxsize=3) 
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 3. ADVANCED ANALYTICS & UI CORE ENGINE
# ==========================================
class AdvancedVisionSystem:
    def __init__(self, model_name=MODEL_NAME):
        self.model = YOLO(model_name)
        
        self.track_start_times = {} 
        self.dwell_times = {}       
        self.heatmap_accumulator = None
        
        # Performance & Plotting history
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.time_history = []
        self.crowd_history = []
        self.dwell_history = []
        self.plot_start_time = time.time()

        # Matplotlib interactive plotting setup
        plt.ion()  
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.suptitle("Dynamic Crowd Analytics Dashboard", fontsize=14, fontweight='bold')
        self.setup_plots()

    def setup_plots(self):
        self.ax1.set_title("Live Queue Density Tracking")
        self.ax1.set_ylabel("Detected People")
        self.ax1.grid(True, linestyle='--', alpha=0.6)
        self.line1, = self.ax1.plot([], [], color='crimson', linewidth=2, label="Live Count")
        
        self.ax2.set_title("Average Queue Dwell Time Trend")
        self.ax2.set_xlabel("Elapsed Time (Seconds)")
        self.ax2.set_ylabel("Avg Wait Duration (Sec)")
        self.ax2.grid(True, linestyle='--', alpha=0.6)
        self.line2, = self.ax2.plot([], [], color='darkorange', linewidth=2, label="Avg Dwell")

    def update_analytics(self, frame_shape, boxes, track_ids):
        h, w = frame_shape[:2]
        current_time = time.time()
        
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((h, w), dtype=np.float32)
            
        self.heatmap_accumulator *= HEATMAP_DECAY
        current_frame_ids = set(track_ids)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            box_w, box_h = x2 - x1, y2 - y1
            if box_w > 150 or box_h < 30: 
                continue
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self.heatmap_accumulator, (cx, cy), BLUR_RADIUS, 2.5, -1)
            
            if track_id not in self.track_start_times:
                self.track_start_times[track_id] = current_time
            else:
                self.dwell_times[track_id] = current_time - self.track_start_times[track_id]

        for old_id in list(self.track_start_times.keys()):
            if old_id not in current_frame_ids:
                self.track_start_times.pop(old_id, None)

        # Plot sampling control (every 0.5 seconds)
        elapsed_plot_time = current_time - self.plot_start_time
        if len(self.time_history) == 0 or (elapsed_plot_time - self.time_history[-1] >= 0.5):
            avg_dwell = sum(self.dwell_times.values()) / len(self.dwell_times) if self.dwell_times else 0
            
            self.time_history.append(elapsed_plot_time)
            self.crowd_history.append(len(track_ids))
            self.dwell_history.append(avg_dwell)
            
            if len(self.time_history) > 60:
                self.time_history.pop(0)
                self.crowd_history.pop(0)
                self.dwell_history.pop(0)

            self.render_live_plots()

    def render_live_plots(self):
        if not self.time_history:
            return
        self.line1.set_data(self.time_history, self.crowd_history)
        self.line2.set_data(self.time_history, self.dwell_history)
        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()
        margin = 5
        self.ax1.set_xlim(min(self.time_history), max(self.time_history) + margin)
        self.ax2.set_xlim(min(self.time_history), max(self.time_history) + margin)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def generate_heatmap_overlay(self, base_frame):
        if self.heatmap_accumulator is None:
            return base_frame
        blurred = cv2.GaussianBlur(self.heatmap_accumulator, (51, 51), 0)
        normalized = np.clip(blurred * 60, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay_frame = cv2.addWeighted(base_frame, 0.55, colored_heatmap, 0.45, 0)
        return overlay_frame

    def draw_dashboard(self, frame, current_crowd, line_y):
        h, w, _ = frame.shape
        
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        avg_dwell = sum(self.dwell_times.values()) / len(self.dwell_times) if self.dwell_times else 0

        # Draw a horizontal crossing line matching the new flow direction
        cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 255), 3)
        if current_crowd >= CROWD_THRESHOLD:
            # Emergency Situation (Red Alert)
            hud_color = (0, 0, 200)      # deep red
            text_status = "CRITICAL CROWD WARNING!"
            status_color = (0, 0, 255)   # Text in bright red for the system
        else:
            # Normal State (Elegant Gray Glass Panel)
            hud_color = (20, 20, 20)     
            text_status = "QUEUE ANALYTICS SYSTEM"
            status_color = (0, 255, 255)

        # Draw semi-transparent HUD panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (370, 160), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "QUEUE ANALYTICS SYSTEM", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Engine Speed: {self.fps:.1f} FPS (Threaded)", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Live Queue Count: {current_crowd}", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Avg Waiting Time: {avg_dwell:.1f} sec", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("[Vision Lab Engine] Processing High-Density Queue Video...")
    
    video_stream = ThreadedVideoStream().start()
    time.sleep(1.0) 
    
    vision_system = AdvancedVisionSystem()

    while not video_stream.stopped:
        # Optimized Frame Dropping to maintain real-time speed
        frame = None
        while not video_stream.queue.empty():
            frame = video_stream.read()
            
        if frame is None:
            continue

        # -----------------------------------------------------------------
        frame = cv2.resize(frame, (1024, 768))
        # -----------------------------------------------------------------

        h, w, _ = frame.shape
        line_y = int(h * 0.35) # Position the line near the entrance gates

        # Adjusted imgsz to 640 for hardware efficiency and smoother real-time playback
        results = vision_system.model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_LIMIT, imgsz=736, verbose=False, tracker="bytetrack.yaml")
        
        current_crowd_count = 0
        boxes = []
        track_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            current_crowd_count = len(track_ids)
            
            vision_system.update_analytics(frame.shape, boxes, track_ids)

        annotated_frame = vision_system.generate_heatmap_overlay(frame)
        
        if current_crowd_count > 0:
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                d_time = vision_system.dwell_times.get(tid, 0.0)
                cv2.putText(annotated_frame, f"ID:{tid} [{d_time:.1f}s]", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        vision_system.draw_dashboard(annotated_frame, current_crowd_count, line_y)

        cv2.imshow("Vision Advanced Analytics (Heatmap + Threading)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    plt.ioff(); plt.close('all')
    cv2.destroyAllWindows()
    print("[System Shutdown] Finished cleanly.")
