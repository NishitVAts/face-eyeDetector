import cv2 as cv
data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
demo = cv.imread('getty_98328574_200014081818843105215_52829.jpg')
grey = cv.cvtColor(demo,cv.COLOR_BGR2GRAY)
show = cv.imshow('img',grey)
faceCord = data.detectMultiScale(grey)
print(faceCord)
cv.waitKey()
