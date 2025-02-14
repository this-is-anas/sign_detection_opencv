import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    if thumb_tip.y < thumb_ip.y and index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y:
        return "Thumbs Up"
    if index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y:
        return "Peace Sign"
    if all(landmarks[i].y < landmarks[0].y for i in [8, 12, 16, 20]):
        return "Hello"
    if all(landmarks[i].y > landmarks[5].y for i in [8, 12, 16, 20]):
        return "Fist"
    if index_tip.y < index_mcp.y and middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y < pinky_mcp.y:
        return "Rock Sign"
    
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = recognize_gesture(landmarks)
            if gesture:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
