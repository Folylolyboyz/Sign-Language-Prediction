import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from math import ceil

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

