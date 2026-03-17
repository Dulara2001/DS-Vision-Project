#database.py

import json
import threading
from datetime import datetime
import os

class LocalJsonLogger:
    def __init__(self, filename="detection_log.json"):
        self.filename = filename
        self.lock = threading.Lock()
        
        # Initialize the file as an empty list if it doesn't exist
        if not os.path.exists(self.filename) or os.stat(self.filename).st_size == 0:
            with open(self.filename, 'w') as f:
                json.dump([], f)
        
        print(f"✅ Logging started.")
        print(f"File path: {os.path.abspath(self.filename)}")

    def log_visitor(self, camera_info, track_id, gender=None, age=None, race=None, global_id=None, is_returning=False):
        """Starts a background thread to save data."""
        threading.Thread(target=self._save_to_file, args=(camera_info, track_id, gender, age, race, global_id, is_returning)).start()

    def _save_to_file(self, camera_info, track_id, gender=None, age=None, race=None, global_id=None, is_returning=False):
        """Prepends the detection to the JSON array."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        
        # Construct the JSON object
        new_entry = {
            "global_id": global_id,
            "camera_id": camera_info['id'],
            "camera_id_desc": camera_info['desc'],
            "id": int(track_id),
            "timestamp": timestamp,
            "gender": gender,
            "race": race,
            "age": age,
            "is_returning": is_returning   # True or False
        }

        # Thread-safe read/write operation
        with self.lock:
            try:
                # 1. Read existing data
                with open(self.filename, 'r') as f:
                    try:
                        data_list = json.load(f)
                    except json.JSONDecodeError:
                        data_list = []

                # 2. Insert new entry at the START of the list (Index 0)
                data_list.insert(0, new_entry)

                # 3. Overwrite file with updated list
                with open(self.filename, 'w') as f:
                    json.dump(data_list, f, indent=4)


                status = "🔁 Returning" if is_returning else "🆕 New"
                print(f"{status} visitor logged: ID {track_id} | Global ID: {global_id} | {gender}, {age}, {race} from {camera_info['desc']}")
            except Exception as e:
                print(f"❌ Error updating log file: {e}")
                
            #     print(f"🚀 Logged: ID {track_id} | Global ID: {global_id} | {gender}, {age}, {race} from {camera_info['desc']}")
            # except Exception as e:
            #     print(f"❌ Error updating log file: {e}")


# # class for returning visitors
# class ReturningVisitorLogger:
#     def __init__(self, filename="return_visitors_log.json"):
#         self.filename = filename
#         self.lock = threading.Lock()
        
#         if not os.path.exists(self.filename) or os.stat(self.filename).st_size == 0:
#             with open(self.filename, 'w') as f:
#                 json.dump([], f)
        
#         print(f"✅ Returning visitor log started.")
#         print(f"File path: {os.path.abspath(self.filename)}")

#     def log_visitor(self, camera_info, track_id, gender=None, age=None, race=None, global_id=None):
#         threading.Thread(target=self._save_to_file, args=(camera_info, track_id, gender, age, race, global_id)).start()

#     def _save_to_file(self, camera_info, track_id, gender=None, age=None, race=None, global_id=None):
#         timestamp = datetime.now().isoformat()
        
#         new_entry = {
#             "global_id": global_id,
#             "camera_id": camera_info['id'],
#             "camera_id_desc": camera_info['desc'],
#             "id": int(track_id),
#             "timestamp": timestamp,
#             "gender": gender,
#             "race": race,
#             "age": age
#         }

#         with self.lock:
#             try:
#                 with open(self.filename, 'r') as f:
#                     try:
#                         data_list = json.load(f)
#                     except json.JSONDecodeError:
#                         data_list = []

#                 data_list.insert(0, new_entry)

#                 with open(self.filename, 'w') as f:
#                     json.dump(data_list, f, indent=4)
                
#                 print(f"🔁 Returning visitor logged: ID {track_id} | Global ID: {global_id} | {gender}, {age}, {race}")
#             except Exception as e:
#                 print(f"❌ Error updating returning visitor log: {e}")