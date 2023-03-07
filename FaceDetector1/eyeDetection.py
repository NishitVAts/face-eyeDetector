from random import randrange

import cv2 as cv

data = cv.CascadeClassifier('eye.xml')
img = cv.VideoCapture(0)
key = cv.waitKey(1)
while True:
    successful_frame_read, frame = img.read()
    grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    eyeCords = data.detectMultiScale(grey)
    for (x,y,w,h) in eyeCords:
        cv.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
        cv.imshow('hello',frame)
        key = cv.waitKey(1)
    if key == 81 or key == 113:
        break
