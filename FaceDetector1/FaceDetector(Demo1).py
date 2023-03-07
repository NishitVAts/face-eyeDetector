import cv2 as cv
from random import randrange
#importing the face data 
faceData = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# #choosing the image to detect
# img = cv.imread('istockphoto-858269070-170667a.jpg')
# webcam video camera detection of faces
webC = cv.VideoCapture(0)
key = cv.waitKey(1)
while True:
    successful_frame_read, frame = webC.read()
    greImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detecting faces
    faceCords = faceData.detectMultiScale(greImg)
    # Draw rectangles on faces:
    for (x, y, w, h) in faceCords:
        cv.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # showing the images
    cv.imshow('hello', frame)
    key = cv.waitKey(1)
    # stop if q is pressed 81 and 113 are ascii character of the letter Q
    if key==81 or key ==113:
        break

webC.release()