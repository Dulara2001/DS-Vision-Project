# main.py

import cv2
import numpy as np
import time
import config
from config import CAMERAS, YOLO_MODEL, TARGET_CLASSES, WINDOW_MODE, CAMERA_MAP
from tracker import HumanTracker
from database import LocalJsonLogger 
from processor import CameraProcessor
from camera_stream import CameraStream
from milvus_db import get_or_create_collection  
from kafka_producer import flush_producer
from face_engine import set_logger

def main():
    # Initialize Local Logger
    # logger = LocalJsonLogger("detection_log.json")
    set_logger(None)  

    #  Connect to Milvus
    milvus_collection = get_or_create_collection()
    
    trackers = {}
    processors = {}
    
    # Enable recording if configured
    if config.RECORD_VIDEO:
        print("[System] Video Recording Enabled")

    # Start Processors
    for cam_name, rtsp_url in CAMERAS.items():
        trackers[cam_name] = HumanTracker(
            YOLO_MODEL,
            TARGET_CLASSES,
            milvus_collection=milvus_collection
        )
        processors[cam_name] = CameraProcessor(cam_name, rtsp_url, None, trackers)
        processors[cam_name].start()
        
    # print(f"AI Active. WINDOW Mode: {WINDOW_MODE}")
    print(f"[System] {len(CAMERAS)} camera(s) active. WINDOW Mode: {WINDOW_MODE}")
    print("Looking for crossings...")

    try:
        while True:
            if WINDOW_MODE:
                frames = []
                for processor in processors.values():
                    frame = processor.get_frame()
                    if frame is None:
                        # Placeholder for cameras not yet ready
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    else:
                        frame = cv2.resize(frame, (640, 360))
                    frames.append(frame)

                # Grid View Logic (e.g., 3x3 for 9 cameras)
                num_cams = len(processors)
                cols = 3
                rows = (num_cams + cols - 1) // cols
                
                # Pad with black frames if necessary
                while len(frames) < rows * cols:
                    frames.append(np.zeros((360, 640, 3), dtype=np.uint8))

                row_images = []
                for i in range(rows):
                    row_images.append(np.concatenate(frames[i*cols : (i+1)*cols], axis=1))
                
                final_grid = np.concatenate(row_images, axis=0)
                
                win_name = "Multi-Camera Grid View"
                cv2.imshow(win_name, final_grid)
            
            # Exit logic
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # If WINDOW, we need a small sleep to prevent 100% CPU usage
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Cleaning up threads and resources...")
        for processor in processors.values():
            processor.stop()
        
        for processor in processors.values():
            processor.join()

        flush_producer()    
                
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
