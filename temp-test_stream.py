import cv2
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# --- 1. Configuration ---
# Replace these with your actual login details
CAMERA_IP = "192.168.1.64"  # Update this if your IP is different
USERNAME = "admin"
PASSWORD = "Sdm@140927"  # The password you just used to log in


# Fetch and parse the counting line
# We split the string "0,1080,3840,1080" into a list of integers
line_data = os.getenv("CAM1_LINE", "0,0,0,0").split(",")
LINE_COORDS = [int(v) for v in line_data]

# Hikvision standard RTSP URL for the Main Stream (High Quality)
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/Streaming/Channels/102"

print(f"Attempting to connect to: {CAMERA_IP}...")

# --- 2. Initialize Video Capture ---
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open the video stream. Please check your IP and Password.")
    exit()

print("Success! Live stream started. Press 'q' to quit.")

# --- 3. The Video Loop ---
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab a frame. Reconnecting might be needed.")
        break
    
    # Resize the frame for your screen 
    # (Your camera is 4K, which is too big to display on most laptop screens without resizing)
    display_frame = cv2.resize(frame, (1280, 720))
    
    # Show the video window
    cv2.imshow('Hikvision Live Feed', display_frame)
    
    # Wait for 1 ms and check if the 'q' key was pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting stream...")
        break

# --- 4. Clean Up ---
cap.release()
cv2.destroyAllWindows()