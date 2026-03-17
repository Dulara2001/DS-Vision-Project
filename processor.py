#processor.py

import cv2
import numpy as np
import os
import time
import datetime
import threading
import config
from config import CAMERA_MAP, LINE_CFGS, CAMERA_FLAGS  
from camera_stream import CameraStream
from kafka_producer import publish_detection

# SETTINGS: 0.5 is middle, 0.6 is slightly below middle.
# LINE_POSITION_PERCENT = 0.6

class CameraProcessor(threading.Thread):
    def __init__(self, cam_name, rtsp_url, logger, trackers):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.rtsp_url = rtsp_url
        self.logger = logger
        self.tracker = trackers[cam_name]
        self.tracker.camera_info = CAMERA_MAP.get(cam_name, {"id": 0, "desc": cam_name})
        self.line_cfg    = LINE_CFGS.get(cam_name, [0.2, 0.5, 1.0])  #  per-camera line
        self.stream = CameraStream(cam_name, rtsp_url).start()
        self.stopped = False
        self.final_frame = None
        self.frame_lock = threading.Lock()
        self.video_writer = None
        self.should_record = config.RECORD_VIDEO
        self.count = 0 # Count per camera

        self.tracker.camera_flags = CAMERA_FLAGS.get(cam_name, {
            "detect_gender":   True,
            "hardcode_gender": None,
            "detect_race":     True,
            "hardcode_race":   None,
        })

    def run(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            
            x_start_pct, y_pct, x_end_pct = self.line_cfg
            line_y = int(h * y_pct)
            x_start = int(w * x_start_pct)
            x_end = int(w * x_end_pct)

            # 1. Process Frame
            annotated_frame, new_counts = self.tracker.process_frame(frame, line_y)

            # 2. Log and Count
            if new_counts:
                for item in new_counts:
                    person_id, gender, age, race, global_id = item 
                    self.count += 1
                    cam_info = CAMERA_MAP.get(self.cam_name, {"id": 0, "desc": self.cam_name})
                    # self.logger.log_visitor(cam_info, person_id, gender, age, race, global_id, is_returning=False)  # Local logging

                    # only log locally if logger exists
                    if self.logger:
                        self.logger.log_visitor(cam_info, person_id, gender, age, race, global_id, is_returning=False)

                    publish_detection(cam_info["id"], global_id, gender, age, race, is_returning=False)  # Kafka publish

            # 3. Visualization (Using the dynamic config variables!)
            cv2.line(annotated_frame, (x_start, line_y), (x_end, line_y), (0, 255, 255), 3)
            cv2.putText(annotated_frame, "COUNTING LINE", (x_start, line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            final_frame = draw_dashboard(annotated_frame, self.cam_name, self.count)

            # 4. Record Frame
            if self.should_record:
                if self.video_writer is None:
                    if not os.path.exists("recordings"):
                        os.makedirs("recordings")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recordings/{self.cam_name}_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 20.0
                    self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                    print(f"[{self.cam_name}] Recording started: {filename}")
                self.video_writer.write(final_frame)

            # 5. Store latest frame for display
            with self.frame_lock:
                self.final_frame = final_frame

        # Cleanup
        self.stream.stop()
        if self.video_writer:
            self.video_writer.release()

    def get_frame(self):
        with self.frame_lock:
            return self.final_frame.copy() if self.final_frame is not None else None

    def stop(self):
        self.stopped = True

def draw_dashboard(frame, cam_name, count):
    # Simplified semi-transparent HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"CAM: {cam_name}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"CAM VISITORS: {count}", (20, 75), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    return frame
