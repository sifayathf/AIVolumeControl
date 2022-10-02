import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture (0)

mpHands = mp.solutions.hands
hands = mpHands.Hands (False)
mpDraw = mp.solutions.drawing_utils
pTime = time.time ()
while True:
    success, img = cap.read ()
    imgRGB = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    results = hands.process (imgRGB)
    # print(results.multi_hand_landmarks)
    image_height, image_width, _ = imgRGB.shape
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate (handLMS.landmark):
                print (id, lm)
                print (
                    f'Index finger tip coordinates: (',
                    f'{handLMS.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{handLMS.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                x1, y1 = handLMS.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * image_width, handLMS.landmark[
                    mpHands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                x2, y2 = handLMS.landmark[mpHands.HandLandmark.THUMB_TIP].x * image_width, handLMS.landmark[
                    mpHands.HandLandmark.THUMB_TIP].y * image_height
                print(x1, y1,int(x2), int(y2))
                cv2.circle(img, (int(x1),int(y1)), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (int(x2),int(y2)), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (int((x1+x2)/2),int((y1+y2)/2)), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks (img, handLMS, mpHands.HAND_CONNECTIONS)

    cTime = time.time ()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText (img, str (int (fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                 (255, 0, 255), 3)

    cv2.imshow ("Image", img)
    cv2.waitKey (1)
