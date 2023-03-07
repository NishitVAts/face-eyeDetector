import cv2 as cv
# importing cascades files
data = cv.CascadeClassifier('cars.xml')

# taking an image as input here turning it  into greyscale
img = cv.imread('john-arano-LzxsSWAVMYs-unsplash.jpeg')
greImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detecting the cars
cars = data.detectMultiScale(greImg)
# for (x, y, w, h) in cars:
#     cv.rectangle(img , (x, y), (x+w, y+h), (0, 255, 0), 2)
(x, y, w, h) = cars
cv.rectangle(img , (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('hello', img)
print(cars)


# draw rectangles here
