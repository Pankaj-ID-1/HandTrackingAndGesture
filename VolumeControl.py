import cv2
import time
import numpy as np
import handTrackingProjectmodule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevtime = 0

detector = htm.handDetector(detectionCon=0.7, maxHands=1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)
minvol = volrange[0]
maxvol = volrange[1]
vol=0
volbar=400
volper = 0
area = 0
lmlist = []
colorVol = [255, 0, 0]
while True:
    success, img = cap.read()

    #find Hand
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=True)
    if len(lmlist) != 0:
        area = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))//100
        if 250 < area < 1000:
            print("yes")
            length, img, lineinfo = detector.findDistance(lmlist, 4, 8, img)
            volbar = np.interp(length, (50, 200), (400, 150))
            volper = np.interp(length, [50, 200], [0, 100])
            smoothness = 10
            volper = smoothness*round(volper/smoothness)
            fingers = detector.fingersUp(lmlist)

            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volper/100, None)
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
    cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f'{int(volper)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, colorVol, 3)
    currtime = time.time()
    fps = 1 / (currtime - prevtime)
    prevtime = currtime

    cv2.putText(img, "FPS: {:.2f}".format(int(fps)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Img', img)
    cv2.waitKey(1)