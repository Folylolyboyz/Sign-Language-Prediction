import cv2

def cam_setup(width=1280, height=720) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # print(type(cam))
    return cam