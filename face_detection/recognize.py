import cv2
import numpy as np

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

# Load label mapping
label_map = {}
with open("labels.txt", "r") as f:
    for line in f.readlines():
        label_id, name = line.strip().split(',')
        label_map[int(label_id)] = name

# Initialize camera and detector
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (200, 200))
        
        # Predict
        label_id, confidence = recognizer.predict(resized_face)
        
        if confidence < 100:  # Lower confidence is better
            name = label_map.get(label_id, "Unknown")
            confidence_text = f"Confidence: {round(confidence, 2)}%"
        else:
            name = "Unknown"
            confidence_text = ""
        
        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()