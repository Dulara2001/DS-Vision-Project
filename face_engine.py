#face_engine.py
import cv2
import os
import time
import uuid
from kafka_producer import publish_detection
import numpy as np
import threading
from insightface.app import FaceAnalysis
from database import LocalJsonLogger
from fairface import get_gender_and_race 
from config import (
    REID_THRESHOLD,
    FACE_QUALITY_THRESHOLD,
)

#  returning visitor logger
# returning_logger = ReturningVisitorLogger("return_visitors_log.json")

logger_instance = None


def set_logger(logger):
    global logger_instance
    logger_instance = logger

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MAX_YAW   = 35
MAX_PITCH = 25

EMBEDDING_FRAMES = 10 
MIN_FACE_SIZE = 70    # minimum face size in pixels

# CAPTURE_FOLDER = "captured_faces"
# os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────
# InsightFace Setup
# ──────────────────────────────────────────────
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(960, 960))  

face_buffer = {}
face_buffer_lock = threading.Lock()
inference_lock = threading.Lock()

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def is_blurry(image, threshold=60):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def is_frontal(face):
    if not hasattr(face, 'pose') or face.pose is None:
        return True

    pitch, yaw, roll = face.pose
    yaw_ok   = abs(yaw)   <= MAX_YAW
    pitch_ok = abs(pitch) <= MAX_PITCH

    return yaw_ok and pitch_ok




def get_aligned_face_crop(frame, face, pad_ratio=0.15):
    """Aligns the face so eyes are perfectly horizontal, then crops."""
    h, w = frame.shape[:2]
    fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
    
    # 1. Calculate rotation angle using eyes
    if hasattr(face, 'kps') and face.kps is not None:
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        
        # Calculate angle between the two eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Only rotate if the head tilt is noticeable (saves CPU)
        if abs(angle) > 2.0:
            eye_center = (
                int((left_eye[0] + right_eye[0]) / 2),
                int((left_eye[1] + right_eye[1]) / 2)
            )
            # Rotate the frame around the center of the eyes
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # 2. Apply tight padding and crop from the (now aligned) frame
    face_w = fx2 - fx1
    face_h = fy2 - fy1
    pad_x = int(face_w * pad_ratio)
    pad_y = int(face_h * pad_ratio)

    fx1 = max(0, fx1 - pad_x)
    fy1 = max(0, fy1 - pad_y)
    fx2 = min(w, fx2 + pad_x)
    fy2 = min(h, fy2 + pad_y)

    return frame[fy1:fy2, fx1:fx2]





def update_all_face_buffers(track_boxes, frame):

    if not track_boxes:
        return

    with inference_lock:
        faces = face_app.get(frame)

    if not faces:
        return

    h, w = frame.shape[:2]

    for track_id, box in track_boxes.items():
        x1, y1, x2, y2 = [int(v) for v in box]

        matched_face = None
        for face in faces:
            fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
            face_cx = (fx1 + fx2) // 2
            face_cy = (fy1 + fy2) // 2

            if x1 < face_cx < x2 and y1 < face_cy < y2:
                if matched_face is None or face.det_score > matched_face.det_score:
                    matched_face = face

        if matched_face is None:
            continue

        if not is_frontal(matched_face):
            continue

        score = float(matched_face.det_score)

        fx1, fy1, fx2, fy2 = [int(v) for v in matched_face.bbox]
        face_w = fx2 - fx1
        face_h = fy2 - fy1

        #  Reject small faces
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            continue
        

        # Aligns the head perfectly upright, applies 15% padding, and crops
        face_crop = get_aligned_face_crop(frame, matched_face, pad_ratio=0.15)

        if face_crop is None or face_crop.size == 0:
            continue

        #  Reject blurry faces
        if is_blurry(face_crop):
            continue

        normed_embedding = normalize_embedding(matched_face.embedding)


        #  Run FairFace ONCE
        fairface_gender, fairface_race, fairface_conf = get_gender_and_race(face_crop)

        raw_age = matched_face.age
        raw_insightface_gender = matched_face.sex
        if raw_insightface_gender is not None:
            insightface_gender = "Male" if raw_insightface_gender == 1 else "Female"
        else:
            insightface_gender = None
        

        if fairface_gender is None or raw_age is None:
            continue

        # ---- MODEL LEVEL VOTING ----
        frame_gender_votes = []

        # InsightFace vote — only add if valid
        if insightface_gender is not None:
            frame_gender_votes.append(insightface_gender)

        # FairFace weighted vote
        if fairface_conf >= 0.85:
            frame_gender_votes.extend([fairface_gender, fairface_gender])
        elif fairface_conf >= 0.65:
            frame_gender_votes.append(fairface_gender)

        race    = fairface_race
        age_int = int(raw_age)
        age     = "Kid" if age_int < 15 else "Adult"

        with face_buffer_lock:
            existing = face_buffer.get(track_id)

            if existing is None:
                face_buffer[track_id] = {
                    "best_score": score,
                    "best_crop": face_crop,
                    "embeddings": [normed_embedding],
                    "mean_embedding": normed_embedding,
                    "age": age,
                    "gender_votes": frame_gender_votes.copy(),
                    "race_votes": [race] if race is not None else []
                }
            else:
                if score > existing["best_score"]:
                    existing["best_score"] = score
                    existing["best_crop"] = face_crop
                    existing["age"] = age

                # ---- FRAME LEVEL WEIGHTING ----
                if score > 0.7:
                    existing["gender_votes"].extend(frame_gender_votes * 2)
                    if race is not None:
                        existing["race_votes"].extend([race, race])
                else:
                    existing["gender_votes"].extend(frame_gender_votes)
                    if race is not None:
                        existing["race_votes"].append(race)

                if len(existing["embeddings"]) < EMBEDDING_FRAMES:
                    existing["embeddings"].append(normed_embedding)
                    mean_vec = np.mean(existing["embeddings"], axis=0)
                    existing["mean_embedding"] = normalize_embedding(mean_vec)

def _majority_vote(votes):
    """Return the most common value in a list, or None if empty."""
    if not votes:
        return None
    return max(set(votes), key=votes.count)

def capture_and_store(track_id, milvus_collection, camera_info=None, camera_flags=None):
    with face_buffer_lock:
        data = face_buffer.pop(track_id, None)

    if data is None:
        print(f"[FaceCapture] ⚠️ No good face buffered for ID {track_id}. Skipping.")
        return False, None, None, None, None

    score = data["best_score"]
    face_crop = data["best_crop"]
    embedding = data["mean_embedding"]
    #  Use majority vote for gender and age
    gender = _majority_vote(data.get("gender_votes", []))
    age    = data.get("age")
    race   = _majority_vote(data.get("race_votes", []))


    #  apply hardcoded values if detect is false
    if camera_flags:
        if not camera_flags.get("detect_gender") and camera_flags.get("hardcode_gender"):
            gender = camera_flags["hardcode_gender"]
            print(f"[FaceCapture] 🔒 Gender hardcoded: {gender}")
        if not camera_flags.get("detect_race") and camera_flags.get("hardcode_race"):
            race = camera_flags["hardcode_race"]
            print(f"[FaceCapture] 🔒 Race hardcoded: {race}")

    # print(f"[FaceCapture] Gender votes: {data.get('gender_votes', [])} → {gender}")

    if milvus_collection is not None:
        already_exists, global_id, saved_gender, saved_age, saved_race = search_existing_face(embedding, milvus_collection)
        if already_exists:
            print(f"[FaceCapture] 🔁 Returning visitor. global_id={global_id}")
            #  Log to returning visitors file
            if camera_info is not None:
                if logger_instance:
                    logger_instance.log_visitor(
                        camera_info, track_id,
                        saved_gender, saved_age, saved_race,
                        global_id, is_returning=True
                    )
                publish_detection(camera_info["id"], global_id, saved_gender, saved_age, saved_race, is_returning=True)
            return False, None, None, None, None


    # New person — generate global_id now
    global_id = _generate_global_id()

    # filename = f"{CAPTURE_FOLDER}/id_{track_id}_{int(time.time())}_score{score:.2f}.jpg"
    # if face_crop is not None and face_crop.size > 0:
    #     cv2.imwrite(filename, face_crop)
    #     print(f"[FaceCapture] 💾 Saved: {filename}")

    if score >= FACE_QUALITY_THRESHOLD:
        print(f"[FaceCapture] ✅ High quality (score={score:.2f}) | {gender}, {age}, {race}")
    else:
        print(f"[FaceCapture] ⚠️ Low quality (score={score:.2f}) | {gender}, {age}, {race}")

    if milvus_collection is not None:
        threading.Thread(
            target=_store_to_milvus,
            args=(milvus_collection, track_id, embedding, score, global_id, gender, age, race)
        ).start()

    return True, gender, age, race, global_id

def _store_to_milvus(collection, track_id, embedding, score, global_id, gender, age, race):
    try:
        entities = [
            [int(global_id)],       
            [int(track_id)],        
            [embedding.tolist()],  
            [float(score)],   
            [str(gender or "")],        
            [str(age    or "")],     
            [str(race   or "")],      
        ]
        collection.insert(entities)
        collection.flush()
        print(f"[Milvus] ✅ Embedding stored for global_id={global_id} | {gender}, {age}, {race}")
    except Exception as e:
        print(f"[Milvus] ❌ Failed to store embedding: {e}")




def _generate_global_id():
    # Simple unique integer ID using timestamp + random
    return int(time.time() * 1000) % (10**12)


def search_existing_face(embedding, milvus_collection, similarity_threshold=REID_THRESHOLD):
    try:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = milvus_collection.search(
            data=[normalize_embedding(embedding).tolist()],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["track_id", "quality_score", "global_id","gender", "age", "race"]
        )

        if results and len(results[0]) > 0:
            top_match = results[0][0]
            similarity = top_match.distance

            if similarity >= similarity_threshold:
                global_id = top_match.entity.get("global_id")
                gender    = top_match.entity.get("gender") or None
                age       = top_match.entity.get("age")    or None
                race      = top_match.entity.get("race")   or None
                print(f"[FaceSearch] 🔁 Known person! similarity={similarity:.2f} global_id={global_id} | {gender}, {age}, {race}")
                return True, global_id, gender, age, race 
            else:
                print(f"[FaceSearch] 🆕 New person. Best similarity={similarity:.2f}")
                return False, None, None, None, None

        return False, None, None, None, None

    except Exception as e:
        print(f"[FaceSearch] ❌ Search failed: {e}")
        return False, None, None, None, None

def cleanup_buffer(track_id):
    with face_buffer_lock:
        face_buffer.pop(track_id, None)