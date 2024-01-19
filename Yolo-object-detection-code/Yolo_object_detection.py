import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# load Yolo
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

labels = []

with open('coco.names', 'r') as f:
  labels = [line.strip() for line in f.readlines()]

len(labels)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size = (len(labels),3))

#loading Image:
image = cv2.imread("person.jpg")

plt.imshow(image)

height,width,channels = image.shape

# Detecting objects:

blob = cv2.dnn.blobFromImage(image, 0.00392,(416,416),(0,0,0),True,crop = False)

net.setInput(blob)
output = net.forward(output_layers)

# Showing Information On Screen:

boxes = []
confidences = []
class_ids = []

for out in output:
    for detection in out:
        scores= detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence>0.5:
            # Object Detected:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            # Rectangle Coordinates:
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(labels[class_ids[i]])
        color = colors[i]
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        cv2.putText(image,label,(x,y+30),font,2,color,3)


cv2.imshow("Project Gurukul",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./images/detected_img2.jpg" , image)
