# 🎥 Multi-Camera People Counting System

A real-time computer vision system that detects, tracks, and classifies people crossing a line across multiple camera feeds. Built with YOLO, ByteTrack, InsightFace, and FairFace. Results are stored in Milvus and published to Kafka.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Docker Deployment](#docker-deployment)
- [Data Collection & Training](#data-collection--training)
- [Kafka Payload](#kafka-payload)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system connects to up to **9 IP cameras** via RTSP and counts people crossing a virtual line. For each person detected, it:

- Assigns a **global identity** that works across all cameras
- Detects **gender**, **age**, and **ethnicity**
- Recognizes **returning visitors** using face embeddings stored in Milvus
- Publishes detection events to a **Kafka topic** in real time

---

## Features

- ✅ Up to 9 simultaneous camera feeds
- ✅ Cross-camera identity tracking (same person recognized on different cameras)
- ✅ Returning visitor detection
- ✅ Gender, age, and race classification
- ✅ Per-camera hardcoded gender/race override via `.env`
- ✅ Kafka integration with SASL/PLAIN authentication
- ✅ Milvus vector database for face embedding storage
- ✅ Fully containerized with Docker + GPU support
- ✅ Custom ethnicity model training pipeline (body crop based)

---

## System Architecture

```
RTSP Camera(s)
      │
      ▼
 CameraStream          ← Reads frames in background thread (TCP transport)
      │
      ▼
  HumanTracker         ← YOLO detection + ByteTrack tracking
      │
      ├── Person enters buffer zone near line
      │         └── InsightFace + FairFace → face embedding, gender, race, age
      │
      └── Person crosses line
                └── Search Milvus → New or Returning?
                          ├── New     → store embedding → publish to Kafka
                          └── Returning → load saved data → publish to Kafka
```

---

## Project Structure

```
ds-vision-project/
├── main.py               # Entry point
├── config.py             # All settings loaded from .env
├── processor.py          # Per-camera thread manager
├── tracker.py            # YOLO + ByteTrack tracking logic
├── face_engine.py        # InsightFace + FairFace inference, Milvus search/store
├── fairface.py           # FairFace ResNet34 model (gender + race)
├── camera_stream.py      # RTSP stream reader with TCP transport
├── milvus_db.py          # Milvus collection schema and connection
├── kafka_producer.py     # Confluent Kafka producer
├── database.py           # Local JSON logger (optional)
├── train_ethnicity.py    # Custom ethnicity model training script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                  # Your configuration (not committed to git)
```

---

## Requirements

### Software
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- NVIDIA Container Toolkit (for GPU support on Linux)
- Python 3.10+ (only needed if running without Docker)

### Hardware
- NVIDIA GPU recommended (runs on CPU but slower)
- IP cameras accessible via RTSP

---

## Installation

### Option 1 — Docker (Recommended)

**1. Clone the repository**
```bash
git clone https://github.com/yourname/ds-vision-project.git
cd ds-vision-project
```

**2. Create your `.env` file**
```bash
cp .env.example .env
# Edit .env with your camera and Kafka settings
```

**3. Build the Docker image**
```bash
docker build -t ncingapte/private:ds-vision-proccessor-v1.0 .
```

**4. Start everything**
```bash
docker compose -f docker-compose.yml up -d
```

**5. Check logs**
```bash
docker logs ds-vision-processor -f
```

---

### Option 2 — Local Python

**1. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start Milvus separately**
```bash
# Run only Milvus services (without the vision processor)
docker compose up etcd minio standalone -d
```

**4. Run the app**
```bash
python main.py
```

---

## Configuration

All settings are controlled via the `.env` file.

### System Settings

```env
WINDOW_MODE=False          # True = show live video window, False = headless
RECORD_VIDEO=False         # True = save recordings to /recordings folder
```

### AI Model Settings

```env
YOLO_MODEL=yolov8n.pt
FAIRFACE_MODEL=fairface_alldata_4race_20191111.pt
REID_THRESHOLD=0.6                # Face similarity threshold for returning visitor (0-1)
FACE_QUALITY_THRESHOLD=0.65       # Minimum face detection score to store
```

### Milvus Settings

```env
MILVUS_HOST=milvus-standalone     # Use container name when running via Docker
MILVUS_PORT=19530
MILVUS_COLLECTION=visitor_faces
```

### Kafka Settings

```env
KAFKA_BOOTSTRAP_SERVERS=your-kafka-server:
KAFKA_SECURITY_PROTOCOL=
KAFKA_SASL_MECHANISM=
KAFKA_USERNAME=your-username
KAFKA_PASSWORD=your-password
KAFKA_TOPIC=
```

### Camera Settings

Configure up to 9 cameras using `CAM1_` to `CAM9_` prefixes:

```env
CAM1_NAME=Entrance_Gate
CAM1_IP=
CAM1_USERNAME=admin
CAM1_PASSWORD=
CAM1_CHANNEL=101
CAM1_LINE=0.2,0.5,1.0        # x_start%, y_position%, x_end% of the counting line

CAM2_NAME=Exit_Gate
CAM2_IP=
CAM2_USERNAME=
CAM2_PASSWORD=
CAM2_CHANNEL=101
CAM2_LINE=0.0,0.6,1.0
```

### Per-Camera Gender/Race Hardcoding

If a camera covers an area where all visitors are known to be a specific group, you can skip AI detection and hardcode the values:

```env
# Camera 2 — always classify as Male Foreigner, skip AI
CAM2_DETECT_GENDER=false
CAM2_HARDCODE_GENDER=Male
CAM2_DETECT_RACE=false
CAM2_HARDCODE_RACE=Foreigner

# Camera 1 — use AI for everything (default)
CAM1_DETECT_GENDER=true
CAM1_DETECT_RACE=true
```

### Race Classification Settings

```env
LOCAL_RACES=Indian,Black          # FairFace races classified as "Local"
FOREIGNER_CONFIDENCE=0.7          # Minimum confidence to classify as "Foreigner"
                                  # Below this threshold → defaults to "Local"
```

---

## Running the System

### Start
```bash
docker compose -f docker-compose.yml up -d
```

### Stop
```bash
docker compose -f docker-compose.yml down
```

### View logs
```bash
docker logs ds-vision-processor -f
```

### Expected healthy output
```
[Config] ✅ Camera 1 loaded: Entrance_Gate @ 192.168.1.100
[FairFace] ✅ FairFace Model loaded successfully.
[Milvus] ✅ Loaded existing collection: visitor_faces
[Tracker] Using device: cuda:0
[Entrance_Gate] Connecting...
[System] 1 camera(s) active. WINDOW Mode: False
Looking for crossings...
[FaceCapture] ✅ High quality (score=0.82) | Male, Adult, Local
[Milvus] ✅ Embedding stored for global_id=123456789
[Kafka] ✅ Sent to partition 0 offset 42
```

---

## Docker Deployment

### Build and push image
```bash
docker build -t yourname/repo:tag .
docker login
docker push yourname/repo:tag
```

### Deploy on Linux server

**1. Create a folder on the server**
```bash
mkdir /root/ds-vision
cd /root/ds-vision
```

**2. Copy files to server**
```bash
scp docker-compose.yml root@<server-ip>:/root/ds-vision/
scp .env root@<server-ip>:/root/ds-vision/
```

**3. Install NVIDIA Container Toolkit (for GPU)**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**4. Start the system**
```bash
docker compose -f docker-compose.yml up -d
```

### Persist collected data (body crops) across restarts

Add this volume to `docker-compose.yml` under `ds-vision-processor`:
```yaml
volumes:
  - /root/ds-vision/body_dataset:/app/body_dataset
```

---

## Data Collection & Training

This system includes a pipeline to collect body crop images and train a custom ethnicity classifier.

### Step 1 — Enable data collection

In `.env`:
```env
COLLECT_DATA=true
COLLECT_DATA_DIR=body_dataset
```

Images will be saved to `body_dataset/Unlabeled/` automatically when people cross the line.

### Step 2 — Label images manually

Move images into the correct folders:
```
body_dataset/
├── Local/
│   ├── cam1_1000001.jpg
│   └── cam1_1000002.jpg
└── Foreigner/
    ├── cam1_1000003.jpg
    └── cam1_1000004.jpg
```

### Step 3 — Train the model

```bash
python train_ethnicity.py
```

This produces `ethnicity_model.pt`.

### Step 4 — Deploy the model

Copy `ethnicity_model.pt` to the server:
```bash
scp ethnicity_model.pt root@<server-ip>:/root/ds-vision/
```

Add volume mount in `docker-compose.yml`:
```yaml
volumes:
  - /root/ds-vision/ethnicity_model.pt:/app/ethnicity_model.pt
```

Restart the system:
```bash
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up -d
```

The system will automatically use your custom model instead of FairFace for race classification.

### Step 5 — Improve over time

Collect more data → label → retrain → redeploy. Each cycle improves accuracy.

---

## Kafka Payload

Every detection (new or returning) publishes this JSON to the Kafka topic:

```json
{
  "cameraId":    1,
  "id":          772783384049,
  "timestamp":   "2026-03-12T13:20:49.331",
  "isReturning": "false",
  "gender":      "Male",
  "race":        "Local",
  "age":         "Adult"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cameraId` | int | Camera number (1-9) |
| `id` | int | Global person identity (stable across cameras) |
| `timestamp` | string | ISO format with milliseconds |
| `isReturning` | string | `"true"` if seen before, `"false"` if new |
| `gender` | string | `"Male"` or `"Female"` |
| `race` | string | `"Local"` or `"Foreigner"` |
| `age` | string | `"Kid"` (under 15) or `"Adult"` |

---

## Troubleshooting

### Camera not connecting (RTSP timeout)
The system uses TCP transport for RTSP which works through Docker NAT. If still failing:
- Verify camera IP is reachable: `ping <camera-ip>`
- Test port 554: `docker exec -it ds-vision-processor python -c "import socket; print(socket.create_connection(('<camera-ip>', 554), 5))"`
- Check RTSP credentials in `.env`

### Milvus connection failed
- Make sure `MILVUS_HOST=milvus-standalone` in `.env` (not `localhost`)
- Make sure `env_file: - .env` is in `docker-compose.yml`
- Wait for Milvus to be healthy before the app starts (handled by `depends_on`)

### GPU not available
- Install NVIDIA Container Toolkit on the server
- Verify with: `nvidia-smi` and `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- System falls back to CPU automatically if GPU is unavailable

### App crashes with Qt/display error
- Make sure `WINDOW_MODE=False` in `.env`
- `WINDOW_MODE=True` requires a display and does not work in Docker

### InsightFace model downloading every restart
The model downloads to `/root/.insightface/` inside the container. Add a volume to persist it:
```yaml
volumes:
  - /root/ds-vision/insightface_models:/root/.insightface
```

---

## License

This project is proprietary. All rights reserved.
