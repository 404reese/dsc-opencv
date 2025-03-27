# rescale : while the most cameras including your webcam do not support going higher than its maximum capability, you can use this to rescale the image to a lower resolution

# example
import cv2
video_path = 'video_3840x2160.mp4'

cv2.VideoCapture(video_path)
print('Read video:', video_path)

video = cv2.VideoCapture(video_path) # duplication
ret = True

while ret:
    ret, frame = video.read()
    cv2.imshow('frame', frame)

#rescale

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale) # frame.shape[1] is width of your image
    height = int(frame.shape[0] * scale) # frame.shape[0] is height of your image
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


cv2.waitKey(0)