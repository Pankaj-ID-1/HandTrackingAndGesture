import cv2
import mediapipe as mp
import time
import math
import numpy as np
class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            model_complexity=self.modelComp
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 0, 255), 2)
        return lmList, bbox
    tipIds = [4, 8, 12, 16, 20]
    def fingersUp(self, lmList):
        fingers = []
        if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, lmlist, p1, p2, img, draw=True):
        index_x, index_y = lmlist[p2][1], lmlist[p2][2]
        thumb_x, thumb_y = lmlist[p1][1], lmlist[p1][2]

        # Highlight the thumb and index
        cv2.circle(img, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)

        # Drawing line b/w index and thumb and find midpoint and highlight it
        cv2.line(img, (index_x, index_y), (thumb_x, thumb_y), (0, 0, 255), 3)
        cx, cy = (index_x + thumb_x) // 2, (index_y + thumb_y) // 2
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(index_x - thumb_x, index_y - thumb_y)
        return length, img, [thumb_x, thumb_y, index_x, index_y, cx, cy]
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (246, 196, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()