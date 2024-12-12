import cv2
import numpy as np
import mediapipe as mp
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Streamlit page config
st.set_page_config(page_title="Hand Gesture Volume Control", layout="wide")

# Initialize COM library for pycaw
CoInitialize()

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Initialize audio control with error handling
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]
except Exception as e:
    st.error(f"Audio device error: {e}")
    st.stop()

# Streamlit sidebar info
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Ensure your webcam is enabled.
2. Show your hand to the camera.
3. Move your thumb and index finger closer or apart to adjust the volume.
4. Close the pinky to set the volume.
5. Press 'Stop' to end the video feed.
""")

# Helper function to calculate distance between landmarks
def find_distance(lmList, p1, p2, img):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    return length, img, (x1, y1, x2, y2, cx, cy)

# Video processing class for streamlit-webrtc
class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.prevVolPer = -1

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                lmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(handLms.landmark)]

            if lmList:
                length, img, lineInfo = find_distance(lmList, 4, 8, img)
                volPer = np.interp(length, [50, 200], [0, 100])

                # Set volume when pinky is down
                fingersUp = [lmList[i][2] < lmList[i - 3][2] for i in [8, 12, 16, 20]]
                if not fingersUp[3]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    self.prevVolPer = volPer

                # Display volume percentage
                cv2.putText(img, f'Volume: {int(volPer)}%', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam with streamlit-webrtc
webrtc_streamer(
    key="hand-gesture-control",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandGestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Clean up COM library on exit
CoUninitialize()
