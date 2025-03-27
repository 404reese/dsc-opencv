# processing

import cv2

img = cv2.imread('landscape.jpg')

# grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grey Image', grey)

# blur
blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
# cv2.imshow('Blur Image', blur)

# edge detection

edges = cv2.Canny(img, 20, 75)
edges100 = cv2.Canny(img, 100, 175)
cv2.imshow('Edges100', edges100)
cv2.imshow('Edges', edges)
# how it works? where there is a lot of contrast, it will detect edges. sharp value change of pixels


cv2.waitKey(0) 