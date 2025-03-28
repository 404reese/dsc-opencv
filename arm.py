import cv2
import mediapipe as mp

def detect_arms():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # MediaPipe setup for hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)  # Detect up to 2 hands
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # Flip and convert color for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand type (Left or Right)
                hand_type = handedness.classification[0].label
                
                # Get wrist position (for arm)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w = frame.shape[:2]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                
                # Display arm label
                cv2.putText(frame, f"{hand_type} Arm", (wrist_x, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Arm Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_arms()