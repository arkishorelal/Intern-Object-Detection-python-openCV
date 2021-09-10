#importing libraries
import cv2 as cv
import matplotlib.pyplot as plt

#creating model
configFile = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozenModel = r'frozen_inference_graph.pb'
model = cv.dnn_DetectionModel(frozenModel,configFile)

#creating class
classLabels = []
file_name = 'object.txt'  # coco dataset
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    
#printing coco dataset
print(classLabels)

#Setting up the configuration of the model
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#capturing video
cap = cv.VideoCapture('video path')
#cap = cv.VideoCapture(0) ==> webcam
if not cap.isOpened():
     raise IOError("Cannot Open Video")

#Setting the font scale and font style
font_scale = 1.1
font = cv.FONT_HERSHEY_COMPLEX

#Reading each frame and detecting the objects in it
while True:
    ret,frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame,confThreshold=0.62)
    if len(ClassIndex)!= 0:
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if ClassInd <= 80:
                cv.rectangle(frame,boxes,(255,0,0),2)
                cv.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
    cv.imshow("output path",frame)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break 
cap.release()
cv.destroyAllWindows()


#For Image (optional)

img = cv.imread(r'image path')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb);

ClassIndex, confidence, bbox = model.detect(rgb,confThreshold=0.5)

font_scale = 1.1
font = cv.FONT_HERSHEY_COMPLEX
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv.rectangle(img,boxes,(255,0,0),2)
    cv.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
cv.imshow('118940363-successful-handsome-man-near-the-car-.jpg',img);
cv.waitKey(0)
