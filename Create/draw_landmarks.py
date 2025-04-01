# Drawing Landmarks
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def draw_landmarks(frame : np.ndarray, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # frame.shape give y, x, z
    height, width, _ = frame.shape
    
    results = hands.process(frame_rgb)
    data_x = []
    data_y = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                data_x.append(landmark.x*width)
                data_y.append(landmark.y*height)
    
    return data_x, data_y