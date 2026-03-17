# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── System ──────────────────────────────────────────
WINDOW_MODE  = os.getenv("WINDOW_MODE", "True").lower() == "true"
RECORD_VIDEO = os.getenv("RECORD_VIDEO", "False").lower() == "true"

FOREIGNER_CONFIDENCE = float(os.getenv("FOREIGNER_CONFIDENCE", "0.7"))

LOCAL_RACES = [r.strip() for r in os.getenv("LOCAL_RACES", "Indian,Black").split(",")]


# ── AI Models ───────────────────────────────────────
YOLO_MODEL         = os.getenv("YOLO_MODEL", "yolov8n.pt")
FAIRFACE_MODEL     = os.getenv("FAIRFACE_MODEL", "fairface_alldata_4race_20191111.pt")
REID_THRESHOLD     = float(os.getenv("REID_THRESHOLD", "0.6"))
FACE_QUALITY_THRESHOLD = float(os.getenv("FACE_QUALITY_THRESHOLD", "0.65"))

TARGET_CLASSES = [0]

# ── Milvus ──────────────────────────────────────────
MILVUS_HOST       = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT       = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "visitor_faces")


# ── Kafka ────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
KAFKA_SASL_MECHANISM    = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
KAFKA_USERNAME          = os.getenv("KAFKA_USERNAME", "")
KAFKA_PASSWORD          = os.getenv("KAFKA_PASSWORD", "")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "detection-data-topic")




# ── Cameras — auto-discover CAM1_ ... CAM9_ ─────────
def _load_cameras():
    cameras    = {}
    camera_map = {}
    line_cfgs  = {}
    camera_flags = {}

    for i in range(1, 10):  # CAM1 to CAM9
        name     = os.getenv(f"CAM{i}_NAME")
        ip       = os.getenv(f"CAM{i}_IP")
        username = os.getenv(f"CAM{i}_USERNAME")
        password = os.getenv(f"CAM{i}_PASSWORD")
        channel  = os.getenv(f"CAM{i}_CHANNEL", "101")
        line_str = os.getenv(f"CAM{i}_LINE", "0.2,0.5,1.0")

        # Skip if not configured
        if not name or not ip:
            continue

        rtsp_url = f"rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}"

        cameras[name]    = rtsp_url
        camera_map[name] = {"id": i, "desc": name}
        line_cfgs[name]  = [float(x) for x in line_str.split(",")]


        camera_flags[name] = {
            "detect_gender":   os.getenv(f"CAM{i}_DETECT_GENDER", "true").lower() == "true",
            "hardcode_gender": os.getenv(f"CAM{i}_HARDCODE_GENDER", None),
            "detect_race":     os.getenv(f"CAM{i}_DETECT_RACE", "true").lower() == "true",
            "hardcode_race":   os.getenv(f"CAM{i}_HARDCODE_RACE", None),
        }

        print(f"[Config] ✅ Camera {i} loaded: {name} @ {ip}")

    return cameras, camera_map, line_cfgs, camera_flags

CAMERAS, CAMERA_MAP, LINE_CFGS, CAMERA_FLAGS = _load_cameras()