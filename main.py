import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque

# Screen size
w_screen, h_screen = pyautogui.size()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

# Smoothing
plocX, plocY = 0, 0
smoothening = 5

# Double-tap gesture tracking
index_y_history = deque(maxlen=20)
tap_times = deque(maxlen=2)
double_tap_cooldown = 0.5  # seconds

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the webcam
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = img.shape

            # Index tip coordinates
            x1 = int(hand_landmarks.landmark[8].x * w)
            y1 = int(hand_landmarks.landmark[8].y * h)

            # Draw fingertip
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            # Map index finger to screen
            screen_x = int(hand_landmarks.landmark[8].x * w_screen)
            screen_y = int(hand_landmarks.landmark[8].y * h_screen)

            # Smooth cursor movement
            clocX = plocX + (screen_x - plocX) // smoothening
            clocY = plocY + (screen_y - plocY) // smoothening
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

            # === Double-tap gesture detection ===
            index_y = hand_landmarks.landmark[8].y
            index_y_history.append(index_y)

            if len(index_y_history) >= 5:
                dy1 = index_y_history[-5] - index_y_history[-3]
                dy2 = index_y_history[-3] - index_y_history[-1]

                if dy1 > 0.02 and dy2 < -0.02:  # Down and up pattern
                    now = time.time()
                    if not tap_times or now - tap_times[-1] > 0.15:
                        tap_times.append(now)

                    if len(tap_times) == 2 and (tap_times[1] - tap_times[0]) < double_tap_cooldown:
                        pyautogui.doubleClick()
                        print("💡 Double Tap Detected")
                        tap_times.clear()

            # Draw hand connections
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Ultron And Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
