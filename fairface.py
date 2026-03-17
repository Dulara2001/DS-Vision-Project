# fairface.py

import cv2
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from config import FAIRFACE_MODEL, LOCAL_RACES, FOREIGNER_CONFIDENCE

# 1. Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[FairFace] Loading model on: {device}")

# 2. Initialize Model Architecture 
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 18)

# 3. Load Pre-trained Weights
try:
    model.load_state_dict(torch.load(FAIRFACE_MODEL, map_location=device))
    print("[FairFace] ✅ FairFace Model loaded successfully.")
except FileNotFoundError:
    print("[FairFace] ❌ Error: '{FAIRFACE_MODEL}' not found!")

model.to(device)
model.eval()

# 4. Image Transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FairFace 4-race label order
RACE_LABELS = ["White", "Black", "Asian", "Indian"]

#  Indices 0-3 = race, 7-8 = gender
def _run_model(face_crop_bgr):
    """Run model once, return full output tensor."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None
    try:
        face_rgb     = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img      = Image.fromarray(face_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(input_tensor)
    except Exception as e:
        print(f"[FairFace] ⚠️ Inference Error: {e}")
        return None


def get_gender(face_crop_bgr):
    """Returns 'Male' or 'Female', or None if failed."""
    outputs = _run_model(face_crop_bgr)
    if outputs is None:
        return None
    gender_preds = outputs[0, 7:9]
    gender_idx   = torch.argmax(gender_preds).item()
    return "Male" if gender_idx == 0 else "Female"


def get_race(face_crop_bgr):
    """
    Returns 'Local' or 'Foreigner' based on predicted race.
    Local    = races in LOCAL_RACES env var (default: Indian, Black)
    Foreigner = everything else (White, Asian)
    """
    outputs = _run_model(face_crop_bgr)
    if outputs is None:
        return None

    # Race logits are at index 0-3
    race_preds = outputs[0, 0:4]
    race_probs = torch.softmax(race_preds, dim=0)
    race_idx   = torch.argmax(race_probs).item()
    race_label = RACE_LABELS[race_idx]
    confidence = race_probs[race_idx].item()

    # Only classify as Foreigner if model is confident enough
    # If not confident, default to Local (safer for Sri Lanka deployment)
    if race_label not in LOCAL_RACES:
        if confidence >= FOREIGNER_CONFIDENCE:
            race = "Foreigner"
        else:
            race = "Local"  #  uncertain → assume Local
    else:
        race = "Local"

    print(f"[FairFace] Race: {race_label} ({confidence:.2f}) → {race}")
    return race


def get_gender_and_race(face_crop_bgr):
    """
    Run model ONCE and return both gender and race.
    """
    outputs = _run_model(face_crop_bgr)
    if outputs is None:
        return None, None

    # Gender
    gender_preds = outputs[0, 7:9]
    gender_probs = torch.softmax(gender_preds, dim=0)
    gender_idx = torch.argmax(gender_probs).item()
    gender_conf = gender_probs[gender_idx].item()

    gender = "Male" if gender_idx == 0 else "Female"


    # Race → Local or Foreigner
    race_preds = outputs[0, 0:4]
    race_probs = torch.softmax(race_preds, dim=0)
    race_idx   = torch.argmax(race_probs).item()
    race_label = RACE_LABELS[race_idx]
    race       = "Local" if race_label in LOCAL_RACES else "Foreigner"

    return gender, race, gender_conf