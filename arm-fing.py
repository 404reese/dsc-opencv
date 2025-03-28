import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Finger names in order of MediaPipe hand landmarks
FINGER_NAMES = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky"
]

def detect_hand_and_arm():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:
        
        while cap.isOpened():
            # Read frame from webcam
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            
            # Process the image and detect hands
            results = hands.process(image)
            
            # Convert back to BGR for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Determine if it's left or right hand
                    hand_type = handedness.classification[0].label
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Label fingers
                    for i, finger_name in enumerate(FINGER_NAMES):
                        # Get tip landmark index (4, 8, 12, 16, 20 for each finger)
                        tip_idx = 4 * (i + 1)
                        tip = hand_landmarks.landmark[tip_idx]
                        
                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = image.shape
                        cx, cy = int(tip.x * w), int(tip.y * h)
                        
                        # Draw finger name
                        cv2.putText(
                            image, 
                            f"{hand_type} {finger_name}", 
                            (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            1
                        )
                    
                    # Estimate arm position (using wrist and elbow landmarks)
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    w_x, w_y = int(wrist.x * w), int(wrist.y * h)
                    
                    # Draw arm label
                    cv2.putText(
                        image, 
                        f"{hand_type} Arm", 
                        (w_x, w_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 0, 0), 
                        2
                    )
            
            # Display the image
            cv2.imshow('Hand and Arm Detection', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run the detection
if __name__ == "__main__":
    detect_hand_and_arm()