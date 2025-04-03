import os
import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.5)



#Directory of images
DIR = "./Dataset"

#Folder in data in which to store
folder_name = input("Folder: ")

#Makes NEWDIR using the DIR and the folder name
NEWDIR = os.path.join(DIR, folder_name)

#Create folder(s) if it doesn't exist
if not os.path.exists(NEWDIR):
    os.makedirs(NEWDIR)
else:
    print("Folder exists")


#Either specify the location or number for live capture
cam = cv.VideoCapture(0)

#Video goes on until the key q is pressed
while True:
    result, image = cam.read()
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    #To make a copy of the original image frame
    imageog = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    #After processing
    output = hands.process(image_rgb)
    data_coordinates = []
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    cv.imshow("Video", image)
    
    
    
    #waitKey(0) waits indifinitely so freezes the frame
    #waitKey(1) waits for 1ms and keeps showing the frames one after the other
    #Making the value > 1 makes it so it doesn't stress hardware out
    k = cv.waitKey(25)
    
    #k stores the value of the key pressed during 1ms or defined wait time
    if k == ord("q"):
        #Destroys the window when q is pressed
        cv.destroyWindow("Video")
        break
    if k == ord("s"):
        c=len(os.listdir(NEWDIR))
        cv.imwrite(os.path.join(NEWDIR, f"{c}.jpg"), imageog)

cam.release()
cv.destroyAllWindows()