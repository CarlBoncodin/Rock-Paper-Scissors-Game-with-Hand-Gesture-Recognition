import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random
import time
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import preprocess_input
# Load pre-trained MobileNetV2 model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Load the trained model weights
model.load_weights('gesture_model_weights_xc.h5')
# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def detect_gesture_with_model(frame, hand_landmarks):
    h, w, _ = frame.shape
    landmarks_x = [int(landmark.x * w) for landmark in hand_landmarks]
    landmarks_y = [int(landmark.y * h) for landmark in hand_landmarks]
    x1, y1 = min(landmarks_x), min(landmarks_y)
    x2, y2 = max(landmarks_x), max(landmarks_y)
    margin = 30
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    hand_roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    processed_roi = preprocess_image(hand_roi)
    if processed_roi is not None:
        predictions = model.predict(processed_roi)
        gesture_class = np.argmax(predictions)
        if gesture_class == 0:
            return "Rock"
        elif gesture_class == 1:
            return "Paper"
        elif gesture_class == 2:
            return "Scissors"
    return "Unknown"
def generateMove():
    cpuMoveInt = random.randint(0, 2)
    if cpuMoveInt == 0:
        cpuMove = "Rock"
    elif cpuMoveInt == 1:
        cpuMove = "Paper"
    else:
        cpuMove = "Scissors"
    return cpuMove
cap = cv2.VideoCapture(0)
start_time = None
delay_duration = 5  # in seconds
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        if start_time is None:
            start_time = time.time()  # Set the start time when hands are detected
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= delay_duration:
                start_time = None
            else:
                continue  # Skip processing frames during the delay
        gesture = detect_gesture_with_model(frame, hand_landmarks)
        cpuMove = generateMove()
        condition = "0 # 0 lose, 1 win, -1 tie"
        if gesture == "Rock" and cpuMove == "Rock":
            condition = -1
        elif gesture == "Rock" and cpuMove == "Paper":
            condition = 0
        elif gesture == "Rock" and cpuMove == "Scissors":
            condition = 1
        elif gesture == "Paper" and cpuMove == "Rock":
            condition = 1
        elif gesture == "Paper" and cpuMove == "Paper":
            condition = -1
        elif gesture == "Paper" and cpuMove == "Scissors":
            condition = 0
        elif gesture == "Scissors" and cpuMove == "Rock":
            condition = 0
        elif gesture == "Scissors" and cpuMove == "Paper":
            condition = 1
        elif gesture == "Scissors" and cpuMove == "Scissors":
            condition = -1
        cv2.putText(frame, f"Your move: {gesture}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"My move: {cpuMove}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if condition >= 0:
            state = ""
            if condition == 1:
                state = "win"
            else:
                state = "lose"
            cv2.putText(frame, f"You {state}!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"We tied!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Ready when you are!", (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Rock Paper Scissors Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()