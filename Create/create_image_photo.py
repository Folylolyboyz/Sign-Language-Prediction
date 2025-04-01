import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2

import pandas as pd

from camera_setup import cam_setup
from get_cutout import cut_out_ratio_maintained
from get_image_frame import get_image_frame

def start_cam():
    DATASET = "Dataset"
    if not os.path.exists(DATASET):
        return
    
    for folder in os.listdir(DATASET):
        data = []
        for picture in os.listdir(f"{DATASET}/{folder}"):
            frame = cv2.imread(f"{DATASET}/{folder}/{picture}")
            # cv2.imshow("Image", frame)
            # key = cv2.waitKey(1)
            try:
                data_x_ratio, data_y_ratio = get_image_frame(frame)
                data_x_ratio = list(map(lambda x: x/400, data_x_ratio))
                data_y_ratio = list(map(lambda x: x/400, data_y_ratio))
                data.append([folder]+data_x_ratio+data_y_ratio)
                # frame = cut_out_ratio_maintained(frame)
                # cv2.imshow("Image", frame)
                # key = cv2.waitKey(0)
            except:                
                continue
        LOC = f"DataCSV"
        FILE = f"{LOC}/{folder}.csv"
        os.makedirs(LOC, exist_ok=True)
        columns = ["label"] + [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)]
        df = pd.DataFrame(data, columns=columns)
        
        if os.path.exists(FILE):
            if df is None or df.empty:
                df = pd.read_csv(FILE)
            else:
                df = pd.concat([df, pd.read_csv(FILE)], ignore_index=True)
        df.to_csv(FILE, index=False)
            
    
    
def main():
    start_cam()

if __name__ == "__main__":
    main()