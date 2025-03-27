import cv2

video_path = 'video.mp4'

cv2.VideoCapture(video_path)
print('Read video:', video_path)

video = cv2.VideoCapture(video_path) 

ret = True

while ret:
    ret, frame = video.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(40) 