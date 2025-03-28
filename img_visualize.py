import cv2

# read image
image_path = 'suki the pookie.webp'
image = cv2.imread(image_path)
print('Read image:', image_path)

# write image
cv2.imwrite(image_path + '_copy.jpg', image)

# visualize image
cv2.imshow('Image', image)
cv2.waitKey(0) # if not written window will open and close in fraction of second by default