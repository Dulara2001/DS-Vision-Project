#camera_stream.py
import cv2
import threading
import time
import os

class CameraStream:
    def __init__(self, name, rtsp_url):
        self.name = name
        self.rtsp_url = rtsp_url
        self.stopped = False
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None

        print(f"[{self.name}] Connecting...")
        self.cap = self._open_cap()
        self.ret, self.frame = self.cap.read()

    def _open_cap(self):
    # Force TCP transport for RTSP - works better through Docker NAT
        rtsp_tcp_url = self.rtsp_url
        if self.rtsp_url.startswith("rtsp://"):
            rtsp_tcp_url = self.rtsp_url.replace("rtsp://", "rtsp://", 1)
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_tcp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened() or not self.ret:
                print(f"[{self.name}] Signal lost. Reconnecting...")
                time.sleep(2)
                self.cap = self._open_cap()
                self.ret, _ = self.cap.read()
                continue

            ret, frame = self.cap.read()

            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame

        self.cap.release()

    def read(self):
        with self.lock:
            if self.ret:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True