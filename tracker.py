#tracker.py

from ultralytics import YOLO
import torch
import cv2
from face_engine import update_all_face_buffers, capture_and_store, cleanup_buffer  
import threading

FACE_BUFFER_ZONE_PX = 300


class HumanTracker:
    def __init__(self, model_path, target_classes, milvus_collection=None):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        print(f"[Tracker] Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.target_classes = target_classes

        self.track_history = {}
        self.track_start_y = {}
        self.counted_ids = set()

        self.milvus_collection = milvus_collection
        self.active_track_ids = set()
        self.camera_info = None

        self.newly_counted_lock = threading.Lock()   
        self.newly_counted_buffer = []

        #  Prevent overlapping buffer update threads
        self.buffer_update_running = False
        self.buffer_update_lock    = threading.Lock()


        self.camera_flags = {
            "detect_gender":   True,
            "hardcode_gender": None,
            "detect_race":     True,
            "hardcode_race":   None,
        }

    def _handle_crossing(self, track_id):
        """ Runs in background thread — tracking loop never blocked."""
        result, gender, age, race, global_id  = capture_and_store(
            track_id,
            self.milvus_collection,
            self.camera_info,
            self.camera_flags
        )
        if result:
            with self.newly_counted_lock:
                self.newly_counted_buffer.append((track_id, gender, age, race, global_id))


    def _handle_buffer_update(self, track_boxes, frame):
        """
        Face buffer update (InsightFace + FairFace) — background thread.
        Skip if previous update still running to avoid GPU overload.
        """
        with self.buffer_update_lock:
            if self.buffer_update_running:
                return  # previous frame still processing, skip this one
            self.buffer_update_running = True

        try:
            update_all_face_buffers(track_boxes, frame)
        finally:
            with self.buffer_update_lock:
                self.buffer_update_running = False


    def process_frame(self, frame, line_y):
        results = self.model.track(
            frame,
            classes=self.target_classes,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False
        )

        #  Collect results from completed background threads
        newly_counted = []
        with self.newly_counted_lock:
            if self.newly_counted_buffer:
                newly_counted = self.newly_counted_buffer.copy()
                self.newly_counted_buffer.clear()

        current_frame_ids = set()
        buffer_zone_boxes = {}  #  collect all people near line

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                current_frame_ids.add(track_id)
                is_locked = track_id in self.counted_ids

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id not in self.track_start_y:
                    self.track_start_y[track_id] = cy

                #  Collect people in buffer zone instead of calling InsightFace per person
                if not is_locked and cy >= (line_y - FACE_BUFFER_ZONE_PX):
                    buffer_zone_boxes[int(track_id)] = box

                if not is_locked:
                    if track_id in self.track_history:
                        prev_cy = self.track_history[track_id]
                        start_cy = self.track_start_y[track_id]

                        if prev_cy < line_y and cy >= line_y and start_cy < line_y:
                            #  Lock immediately — box turns green right away
                            self.counted_ids.add(track_id)
                            is_locked = True
                            #  Heavy work runs in background — doesn't block next person
                            threading.Thread(
                                target=self._handle_crossing,
                                args=(track_id,),
                                daemon=True
                            ).start()

                self.track_history[track_id] = cy

                color = (0, 255, 0) if is_locked else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #  Run InsightFace ONCE for all people in buffer zone
        #  Run buffer update in background — never blocks tracking loop
        if buffer_zone_boxes:
            threading.Thread(
                target=self._handle_buffer_update,
                args=(buffer_zone_boxes, frame.copy()),  #  frame.copy() — thread gets its own copy
                daemon=True
            ).start()

        # Cleanup buffer for IDs that disappeared without crossing
        disappeared = self.active_track_ids - current_frame_ids
        for gone_id in disappeared:
            if gone_id not in self.counted_ids:
                cleanup_buffer(gone_id)
        self.active_track_ids = current_frame_ids

        return frame, newly_counted