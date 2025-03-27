import cv2

video_path = 'video_3840x2160.mp4'
video = cv2.VideoCapture(video_path)

def rescale_frame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

while True:
    ret, frame = video.read()
    if not ret:  
        
        break

    resized_frame = rescale_frame(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('frame_resized', resized_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

video.release()
cv2.destroyAllWindows()
