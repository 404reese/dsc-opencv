import cv2

# Open video file
video_path = 'video.mp4'

# read
cv2.VideoCapture(video_path)
print('Read video:', video_path)

# visualize video
video = cv2.VideoCapture(video_path) # duplication

ret = True
# after end of all frames will not have anything to read so ret will be false

while ret:
    ret, frame = video.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(40) # if not written window will open and close in fraction of second by default