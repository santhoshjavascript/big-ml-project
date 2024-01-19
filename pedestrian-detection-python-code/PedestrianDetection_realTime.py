# Pedestrian Detection using Opencv Python ProjectGurukul

import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression


# Histogram of Oriented Gradients Detector

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Create VideoCapture object

cap = cv2.VideoCapture("video.mp4")

def Detector(frame):
    width = frame.shape[1]
    max_width = 700

    # Resize the frame if the frame width is greater than defined max_width

    if width > max_width:
        frame = imutils.resize(frame, width=max_width)

    # Using Sliding window concept predict the detctions

    pedestrians, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

    # apply non-maxima suppression to remove overlapped bounding boxes

    pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)
    # print(pedestrians)

    count = 0

    #  Draw bounding box over detected pedestrians 

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 100), 2)
        cv2.rectangle(frame, (x, y - 20), (w,y), (0, 0, 255), -1)
        cv2.putText(frame, f'P{count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        count += 1

    cv2.putText(frame, f'Total Persons : {count}', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0,0), 2)

    return frame

while True:
    _, frame = cap.read()

    output = Detector(frame)

    cv2.imshow('output', output)

    # Loop breaks if key "q" is pressed

    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture object and destroy all the active windows after the loop breaks

cap.release()
cv2.destroyAllWindows()

