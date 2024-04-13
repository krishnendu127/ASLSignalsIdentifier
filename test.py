import math
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier

cap= cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
classifier= Classifier("Model/keras_model.h5", "Model/labels.txt")

offset=20
imgSize=300

folder= "Data/C"
counter=0

labels=["A","B","C"]

while True:
    success,img = cap.read()
    hands,img= detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h= hand['bbox']

        imgWhite=np.ones((imgSize , imgSize, 3), np.uint8)*255
        imgCrop= img[y-offset: y+h+offset,x-offset:x+w+offset]

        aspectRatio=h/w

        if aspectRatio > 1:
            k = imgSize/h
            wcal= math.ceil(k*w)
            imgResize= cv2.resize(imgCrop,(wcal, imgSize))
            imgResizeShape= imgResize.shape
            wgap= math.ceil((imgSize- wcal)/2)

            imgWhite[: ,wgap: wcal+wgap]=imgResize
            prediction,index=classifier.getPrediction((imgWhite))
            print((prediction,index))
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hcal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize - hcal) / 2)
            imgWhite[hgap:hcal + hgap, :] = imgResize




        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)


    cv2.imshow("Image",img)
    cv2.waitKey(1)
