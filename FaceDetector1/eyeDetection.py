from random import randrange

import cv2 as cv

faceData = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
data = cv.CascadeClassifier('eye.xml')
smile_data = cv.CascadeClassifier('smile.xml')
img = cv.VideoCapture(0)
cv.waitKey(1)

while True:
    successful_frame_read, frame = img.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eyeCords = data.detectMultiScale(grey)
    faceCords = faceData.detectMultiScale(grey)

    for (x, y, w, h) in faceCords:
        cv.rectangle(frame, (x, y), (x + w, y + h), (randrange(256),256,256), 2)
        for (ex, ey, ew, eh) in eyeCords:
            cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
        #     smileCords = smile_data.detectMultiScale(grey)
        #
        #     for (sx, sy, sw, sh) in smileCords:
        #         cv.rectangle(frame, (sx, sy),(sx + sw, sy + sh),(0, 255, 255 ),2)

    cv.imshow('hello', frame)
    key = cv.waitKey(1)
    # if key == 113,
