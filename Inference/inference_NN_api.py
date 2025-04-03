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
from Create.get_cutout import cut_out_ratio_maintained

import onnxruntime as rt

from fastapi import FastAPI,WebSocket
import uvicorn

from collections import Counter

app = FastAPI()

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
hands_ratio = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)


def get_frame(data):
    image_np = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # frame = cv2.imread(data)
    frame = cv2.flip(frame, 1)
    frame_draw = frame.copy()
    data_x, data_y = draw_landmarks(frame_draw, hands)
    # print(data_x)
    if data_x and data_y:
        draw_ratio_maintained_frame = cut_out_ratio_maintained(frame, data_x, data_y)
        data_x_ratio, data_y_ratio = draw_landmarks(draw_ratio_maintained_frame, hands_ratio)
    
        return data_x_ratio, data_y_ratio
    return False, False



def start_cam(data):
    data_x_ratio, data_y_ratio = get_frame(data)
    try:
        if data_x_ratio and data_y_ratio:
            data = []
            data_x_ratio = list(map(lambda x: x/400, data_x_ratio))
            data_y_ratio = list(map(lambda x: x/400, data_y_ratio))
            data.append(data_x_ratio+data_y_ratio)
            # print(data)
            columns = [f"x{i}" for i in range(1, 22)] + [f"y{i}" for i in range(1, 22)]
            df = pd.DataFrame(data, columns=columns)
            
            label = "ABCDEFGHIKLMNOPQRSTUVWXY"
            d = {i:j for i,j in enumerate(label)}
            
            sess = rt.InferenceSession("Train/ensemble_model.onnx", providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            hand_sign = df.values.astype(np.float32)
            
            prediction = sess.run([label_name], {input_name: hand_sign})
            predicted_label = np.argmax(prediction)
            predicted_label = d[predicted_label]
            # print(predicted_label)
            return predicted_label
        return " "
            
    except Exception as E:
        print(f"Error: {E}")
        # return "Error"
        return " "


@app.get("/")
def hello_world():
    return "Hello World"

@app.websocket("/predict")
async def predict(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected!")

    while True:
        try:
            pred = []
            for i in range(30):
                data = await websocket.receive_bytes()
                prediction_label = start_cam(data)
                pred.append(prediction_label)
            
            prediction_label = Counter(pred).most_common(1)[0][0]
            # image_np = np.frombuffer(data, np.uint8)
            # frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            await websocket.send_text(f"{prediction_label}")
            
            for i in range(15):
                data = await websocket.receive_bytes()
                prediction_label = start_cam(data)
                pred.append(prediction_label)
            # if frame is not None:
            #     cv2.imshow("WebP Stream", frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

        except Exception as e:
            print(f"Error: {e}")
            break

uvicorn.run(app, host="0.0.0.0", port=8010)