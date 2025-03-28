import os
import cv2
import numpy as np

# Create face detector and recognizer
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []
label_ids = {}
current_label = 0

dataset_path = "Faces/"
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(("jpg", "png")):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
            # Assign numeric label
            if label not in label_ids:
                label_ids[label] = current_label
                current_label += 1
            
            # Read and process image
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (200, 200))
                faces.append(resized_face)
                labels.append(label_ids[label])

# Train the model
recognizer.train(faces, np.array(labels))

# Save the model
recognizer.save("face_recognizer.yml")
print("Training completed. Model saved!")

# Save label mapping
with open("labels.txt", "w") as f:
    for name, label_id in label_ids.items():
        f.write(f"{label_id},{name}\n")