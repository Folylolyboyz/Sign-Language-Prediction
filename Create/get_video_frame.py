import cv2
import mediapipe as mp

from draw_landmarks import draw_landmarks
from get_cutout import cut_out_ratio_maintained


mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
hands_ratio = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

def get_video_frame(cam : cv2.VideoCapture):
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    # print(type(frame))
    frame_draw = frame.copy()
    data_x, data_y = draw_landmarks(frame_draw, hands)
    
    cv2.imshow("Image", frame_draw)
    
    if data_x and data_y:
        ratio_maintained_frame = cut_out_ratio_maintained(frame, data_x, data_y)
        cv2.imshow("White Frame", ratio_maintained_frame)
        
        draw_ratio_maintained_frame = ratio_maintained_frame.copy()
        data_x_ratio, data_y_ratio = draw_landmarks(draw_ratio_maintained_frame, hands_ratio)
        cv2.imshow("Draw Frame", draw_ratio_maintained_frame)
    
        return data_x_ratio, data_y_ratio
    return []