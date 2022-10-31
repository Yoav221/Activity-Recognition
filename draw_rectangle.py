import cv2
import pandas as pd
from const import *

cap = cv2.VideoCapture(INPUT_VID)

df = pd.read_csv('Final_Prediction.csv')
frame_index = df['frame_index']
width_index = df['width_index']
height_index = df['height_index']

while cap.isOpended():
    ret, image = cap.read()
    for x in range(len(df)):
        cv2.rectangle(image, (height_index[x], width_index[x], height_index[x]+10, width_index[x]+10),
                      (255, 0, 0))
    cv2.imshow("Video", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
