import cv2

image = cv2.imread('suki the pookie.webp')
print(type(image)) # <class 'numpy.ndarray'>
print(image.shape) # height, width, channels