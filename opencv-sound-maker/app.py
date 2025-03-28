import cv2
import mediapipe as mp
import pygame
import time

# Initialize Pygame mixer for sound playback
pygame.mixer.init()

# Load sound files (replace with your own paths)
SOUND_FILES = {
    'thumb': 'guitar_sound_1.wav',
    'index': 'guitar_sound_2.wav',
    'middle': 'guitar_sound_3.wav',
    'ring': 'guitar_sound_4.wav',
    'pinky': 'guitar_sound_5.wav'
}

# Load sounds into Pygame
sounds = {
    'thumb': pygame.mixer.Sound(SOUND_FILES['thumb']),
    'index': pygame.mixer.Sound(SOUND_FILES['index']),
    'middle': pygame.mixer.Sound(SOUND_FILES['middle']),
    'ring': pygame.mixer.Sound(SOUND_FILES['ring']),
    'pinky': pygame.mixer.Sound(SOUND_FILES['pinky'])
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks indices
FINGER_TIPS = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

# Previous finger states
prev_finger_state = {finger: False for finger in FINGER_TIPS}

def is_finger_up(landmarks, finger):
    tip_id = FINGER_TIPS[finger]
    pip_id = tip_id - 2  # PIP joint is 2 positions back
    
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]
    
    # Check if finger tip is above PIP joint (y coordinate decreases upwards)
    return tip.y < pip.y

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip and convert image
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Draw hand landmarks
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        current_finger_state = {}
        for finger in FINGER_TIPS:
            raised = is_finger_up(landmarks, finger)
            current_finger_state[finger] = raised
            
            if raised:
                tip = landmarks[FINGER_TIPS[finger]]
                h, w, c = image.shape
                cx, cy = int(tip.x * w), int(tip.y * h)
                cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                
            if raised and not prev_finger_state[finger]:
                sounds[finger].play()
                print(f"Playing {finger} sound!")
        
        prev_finger_state = current_finger_state

    cv2.imshow('Guitar Hand', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()