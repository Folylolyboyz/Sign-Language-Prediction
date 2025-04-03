import numpy as np
import pandas as pd

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from Create.draw_landmarks import draw_landmarks
from Create.camera_setup import cam_setup
from Create.get_cutout import cut_out_ratio_maintained

import onnxruntime as rt

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
hands_ratio = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

def get_frame(cam : cv2.VideoCapture):
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
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

def start_cam(cam : cv2.VideoCapture):
    while True:
        try:
            get_frame(cam)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("s"):
                data = []
                data_x_ratio, data_y_ratio = get_frame(cam)
                data_x_ratio = list(map(lambda x: x/400, data_x_ratio))
                data_y_ratio = list(map(lambda x: x/400, data_y_ratio))
                data.append(data_x_ratio+data_y_ratio)
                columns = [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)]
                df = pd.DataFrame(data, columns=columns)
                
                label = "ABCDEFGHIKLMNOPQRSTUVWXY"
                d = {i:j for i,j in enumerate(label)}
                
                sess = rt.InferenceSession("Train/ensemble_model.onnx", providers=["CPUExecutionProvider"])
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                hand_sign = df.values.astype(np.float32)
                
                prediction = sess.run([label_name], {input_name: hand_sign})
                prediction_label = np.argmax(prediction)

                predicted_label = d[prediction_label]
                print(f"Predicted label: {predicted_label}")
                    
            elif key == ord("q"):
                break
        
        except Exception as E:
            print(f"Error: {E}")
            continue

def main():
    cam = cam_setup()
    start_cam(cam)

if __name__ == "__main__":
    main()