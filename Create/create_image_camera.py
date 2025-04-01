import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
from math import ceil
import pandas as pd

from draw_landmarks import draw_landmarks

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
hands_ratio = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

def cam_setup(width=1280, height=720) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # print(type(cam))
    return cam

def cut_out_ratio_maintained(frame: np.ndarray, data_x, data_y):
    #Standardize or normalize whatever the image on size 400x400 while maintaining ratio
        
    #Cut out to basically take out a box of my hand
    x, y, _ = frame.shape
    cutout_padding = 25
    min_x = int(min(data_x))
    max_x = int(max(data_x))
    min_y = int(min(data_y))
    max_y = int(max(data_y))
    # print(min_x, min_y, max_x, max_y)
    
    padded_x_min, padded_x_max =  max(min_x-cutout_padding, 0), min(max_x+cutout_padding, y)
    padded_y_min, padded_y_max =  max(min_y-cutout_padding, 0), min(max_y+cutout_padding, x)
    cropped_frame = frame[padded_y_min:padded_y_max, padded_x_min:padded_x_max]
    # cropped_frame = frame[min_y-cutout_padding:max_y+cutout_padding, min_x-cutout_padding:max_x+cutout_padding]
    # cropped_frame = frame[min_y:max_y, min_x:max_x]
    
    
    # NOT NEEDED TO SHOW IN FINAL
    # cv2.imshow("Cut out",cropped_frame)
    
    #Size of final image
    resize_size = 400
    
    x, y, _ = cropped_frame.shape
    # print(x,y)
    
    white_frame = np.ones((resize_size, resize_size, 3), np.uint8)*255
    
    try:
        if x > y:
            y_cal = ceil(y * resize_size/x)
            resized_frame = cv2.resize(cropped_frame, (y_cal, resize_size))
            padding = (400 - y_cal)//2
            white_frame[:, padding:y_cal+padding] = resized_frame
                            
            # cv2.imshow("Resize", cv2.resize(cropped_frame, (y_cal, resize_size)))
        else:
            x_cal = ceil(x * resize_size/y)
            resized_frame = cv2.resize(cropped_frame, (resize_size, x_cal))
            
            padding = (400 - x_cal)//2
            white_frame[padding:x_cal+padding, :] = resized_frame
            
            # cv2.imshow("Resize", cv2.resize(cropped_frame, (resize_size, x_cal)))
    except:
        pass
    
    return white_frame

def get_frame(cam : cv2.VideoCapture):
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

def start_cam(cam : cv2.VideoCapture):
    if len(sys.argv) != 2:
        print("Argument error")
        return
    label = sys.argv[1]
    data = []
    
    while True:
        try:
            get_frame(cam)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("s"):
                for i in range(20):
                    data_x_ratio, data_y_ratio = get_frame(cam)
                    data_x_ratio = list(map(lambda x: x/400, data_x_ratio))
                    data_y_ratio = list(map(lambda x: x/400, data_y_ratio))
                    data.append([label]+data_x_ratio+data_y_ratio)
                    
            elif key == ord("q"):
                LOC = f"./Dataset"
                FILE = f"{LOC}/{label}.csv"
                os.makedirs(LOC, exist_ok=True)
                columns = ["label"] + [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)]
                df = pd.DataFrame(data, columns=columns)
                
                if os.path.exists(FILE):
                    if df is None or df.empty:
                        df = pd.read_csv(FILE)
                    else:
                        df = pd.concat([df, pd.read_csv(FILE)], ignore_index=True)
                df.to_csv(FILE, index=False)
                break
        
        except Exception as E:
            print(f"Error: {E}")
            continue

def main():
    cam = cam_setup()
    start_cam(cam)

if __name__ == "__main__":
    main()