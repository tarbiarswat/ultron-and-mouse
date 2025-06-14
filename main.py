import cv2
import mediapipe as mp
import pyautogui

w_screen, h_screen = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[8]  # Index fingertip
            h, w, _ = img.shape
            x, y = int(lm.x * w), int(lm.y * h)

            # Map to screen coordinates
            screen_x = int(lm.x * w_screen)
            screen_y = int(lm.y * h_screen)
            pyautogui.moveTo(screen_x, screen_y)

            # Draw on camera frame
            cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Ultron And Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
