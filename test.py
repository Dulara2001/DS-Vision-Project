# import cv2
# import os
# from config import CAMERAS, YOLO_MODEL, TARGET_CLASSES
# from tracker import HumanTracker
# from camera_stream import CameraStream

# # SETTINGS: 0.5 is middle, 0.6 is slightly below middle.
# LINE_POSITION_PERCENT = 0.7 

# def draw_dashboard(frame, cam_name, count):
#     # Simplified semi-transparent HUD
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
#     cv2.putText(frame, f"CAM: {cam_name}", (20, 40), 
#                 cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
#     cv2.putText(frame, f"TOTAL VISITORS: {count}", (20, 75), 
#                 cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
#     return frame

# def main():
#     # Removed: db = BigQueryLogger() 
#     trackers = {}
#     streams = {}
#     total_visitors = 0 
    
#     for cam_name, rtsp_url in CAMERAS.items():
#         trackers[cam_name] = HumanTracker(YOLO_MODEL, TARGET_CLASSES)
#         streams[cam_name] = CameraStream(cam_name, rtsp_url).start()
        
#     print("AI Active. Looking for crossings... (Logging to Console)")

#     while True:
#         for cam_name, stream in streams.items():
#             ret, frame = stream.read()
            
#             if ret:
#                 h, w, _ = frame.shape
#                 line_y = int(h * LINE_POSITION_PERCENT)
                
#                 # 1. Process Frame
#                 annotated_frame, new_counts = trackers[cam_name].process_frame(frame, line_y)
                
#                 # 2. Log and Count
#                 if new_counts:
#                     for person_id in new_counts:
#                         total_visitors += 1
#                         # REPLACED DATABASE LOG WITH PRINT LOG
#                         print(f"[LOG] Person Counted! Cam: {cam_name} | Person ID: {person_id} | Total Visitors: {total_visitors}")

#                 # 3. Draw the Line
#                 x_start, x_end = int(w * 0.1), int(w * 0.9)
#                 cv2.line(annotated_frame, (x_start, line_y), (x_end, line_y), (0, 255, 255), 3)
#                 cv2.putText(annotated_frame, "COUNTING LINE", (x_start, line_y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#                 # 4. Display
#                 final_frame = draw_dashboard(annotated_frame, cam_name, total_visitors)
#                 cv2.imshow(f"Feed: {cam_name}", final_frame)
                
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     for stream in streams.values():
#         stream.stop()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()











# #     from google.cloud import bigquery
# # from datetime import datetime
# # import threading
# # import os
# # import json

# # class BigQueryLogger:
# #     def __init__(self):
# #         self.client = bigquery.Client()
# #         self.table_id = os.getenv("BIGQUERY_TABLE_ID")
# #         # Name of the local backup file
# #         self.json_file = "detections_log.json" 
# #         self.lock = threading.Lock() # Prevents file corruption from multi-threading
# #         print(f"✅ Connected to BigQuery Table: {self.table_id}")

# #     def log_visitor(self, camera_id, track_id):
# #         """Starts a background thread to upload data and save JSON."""
# #         threading.Thread(target=self._upload, args=(camera_id, track_id)).start()

# #     def _save_to_local_json(self, data):
# #         """Appends the detection data to a local JSON list."""
# #         with self.lock: # Ensure only one thread writes to the file at a time
# #             logs = []
# #             # 1. Load existing logs if file exists
# #             if os.path.exists(self.json_file):
# #                 try:
# #                     with open(self.json_file, "r") as f:
# #                         logs = json.load(f)
# #                 except json.JSONDecodeError:
# #                     logs = []

# #             # 2. Append new data
# #             logs.append(data)

# #             # 3. Write back to file
# #             with open(self.json_file, "w") as f:
# #                 json.dump(logs, f, indent=4)

# #     def _upload(self, camera_id, track_id):
# #         """Uploads to BigQuery AND saves to JSON."""
# #         timestamp = datetime.now().isoformat()
        
# #         # This is the JSON object your senior wants
# #         row_to_insert = {
# #             "camera_id": camera_id,
# #             "id": str(track_id),
# #             "timestamp": timestamp,
# #             "gender": None,
# #             "race": None,
# #             "local_foreigner": None
# #         }

# #         # 1. Save to Local JSON first (Offline Backup)
# #         self._save_to_local_json(row_to_insert)

# #         # 2. Upload to BigQuery (Online Cloud)
# #         errors = self.client.insert_rows_json(self.table_id, [row_to_insert])
        
# #         if errors == []:
# #             print(f"🚀 [Cloud + JSON] Uploaded: ID {track_id} from {camera_id}")
# #         else:
# #             print(f"❌ BigQuery Error: {errors}")




# test_fairface.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

FAIRFACE_MODEL_PATH = "fairface_alldata_4race_20191111.pt"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 18)
state = torch.load(FAIRFACE_MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# ★ Use any female face image from captured_faces folder
img_path = "captured_faces/id_605_1771577355_score0.89.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

tensor = transform(img_rgb).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)

print(f"Full output shape: {outputs.shape}")
print(f"Full output values: {outputs[0].tolist()}")
print()

# FairFace output layout:
# indices 0-6:  race (7 classes)
# indices 7-8:  gender (2 classes) — Male, Female
# indices 9-17: age (9 classes)

race_logits   = outputs[0, 0:7]
gender_logits = outputs[0, 7:9]
age_logits    = outputs[0, 9:18]

print(f"Race logits:   {race_logits.tolist()}")
print(f"Gender logits: {gender_logits.tolist()}")
print(f"  index 0 (Male):   {gender_logits[0].item():.4f}")
print(f"  index 1 (Female): {gender_logits[1].item():.4f}")
print(f"  argmax → {'Male' if torch.argmax(gender_logits).item() == 0 else 'Female'}")
print()
print(f"Age logits:    {age_logits.tolist()}")







# # database.py


# # from google.cloud import bigquery
# # from datetime import datetime
# # import threading
# # import os

# # class BigQueryLogger:
# #     def __init__(self):
# #         # The library automatically looks for GOOGLE_APPLICATION_CREDENTIALS in env vars
# #         self.client = bigquery.Client()
# #         self.table_id = os.getenv("BIGQUERY_TABLE_ID")
# #         print(f"✅ Connected to BigQuery Table: {self.table_id}")

# #     def log_visitor(self, camera_id, track_id):
# #         """Starts a background thread to upload data."""
# #         threading.Thread(target=self._upload, args=(camera_id, track_id)).start()

# #     def _upload(self, camera_id, track_id):
# #         """The actual upload function running in the background."""
# #         timestamp = datetime.now().isoformat()
        
# #         row_to_insert = [
# #             {
# #                 "camera_id": camera_id,
# #                 "id": str(track_id),
# #                 "timestamp": timestamp,
# #                 "gender": None,
# #                 "race": None,
# #                 "local_foreigner": None
# #             }
# #         ]

# #         errors = self.client.insert_rows_json(self.table_id, row_to_insert)
        
# #         if errors == []:
# #             print(f"🚀 Uploaded: ID {track_id} from {camera_id}")
# #         else:
# #             print(f"❌ Error uploading: {errors}")