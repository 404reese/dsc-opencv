import cv2

image_path = 'suki the pookie.webp'
image = cv2.imread(image_path)
print('Read image:', image_path)

cv2.imwrite(image_path + '_copy.jpg', image)

cv2.imshow('Image', image)
cv2.waitKey(0) 