import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
camara = cv2.VideoCapture(0)
fingertip = [8, 12, 16, 20]
thimbtip = 4


def detect(img, hand_landmarks):
    if(hand_landmarks):
        for hand_landmark in hand_landmarks:
            lmlist = []
            for id, lm in enumerate(hand_landmark.landmark):
                lmlist.append(lm)
            fingerstatus = []
            for tip in fingertip:
                if lmlist[tip].x < lmlist[tip-3].x:
                    fingerstatus.append(True)
                else:
                    fingerstatus.append(False)
            if all(fingerstatus):
                if lmlist[thimbtip].y < lmlist[thimbtip-1].y < lmlist[thimbtip-2].y:
                    cv2.putText(img, "LIKE", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                if lmlist[thimbtip].y > lmlist[thimbtip-1].y < lmlist[thimbtip-2].y:
                    cv2.putText(img, "disLIKE", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(
                (0, 0, 255), 2, 2), mp_draw.DrawingSpec((0, 255, 0), 4, 2))


while True:
    ret, img = camara.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)
    hand_landmarks = results.multi_hand_landmarks
    detect(img, hand_landmarks)
    cv2.imshow("LikeDislike",img)
    if cv2.waitKey(1)==32:
        break