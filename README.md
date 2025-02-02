
<div align="center">
  <h1>Gesture Volume Control Using OpenCV and MediaPipe</h1>
 </div>

> This Project uses OpenCV and MediaPipe to Control system volume

Using the HandTrackingModule Feature are limitless
<br/>
Pyautogui library can be used to in scrolling
<br/>
(https://github.com/Crozzers/screen_brightness_control) can be used to control the brightness through hand gesture
> 
## Features

- **Hand Gesture Recognition**: Leverages the MediaPipe library for accurate hand tracking and gesture recognition.
- **Volume Control**: Adjust the volume of your primary audio output device by simply moving your hand up or down.
- **Real-Time Feedback**: Provides immediate response to hand movements, ensuring a seamless user experience.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or above installed on your computer.
- A webcam or any compatible camera module for hand tracking.

## REQUIREMENTS
+ opencv-python
+ mediapipe
+ comtypes
+ numpy
+ pycaw

```bash
pip install -r requirements.txt
```
***
### MEDIAPIPE
<div align="center">
  <img alt="mediapipeLogo" src="images/mediapipe.png" />
</div>

> MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.

#### Hand Landmark Model
After the palm detection over the whole image our subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.

To obtain ground truth data, we have manually annotated ~30K real-world images with 21 3D coordinates, as shown below (we take Z-value from image depth map, if it exists per corresponding coordinate). To better cover the possible hand poses and provide additional supervision on the nature of hand geometry, we also render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates.<br>

#### Solution APIs
##### Configuration Options
> Naming style and availability may differ slightly across platforms/languages.

+ <b>STATIC_IMAGE_MODE</b><br>
If set to false, the solution treats the input images as a video stream. It will try to detect hands in the first input images, and upon a successful detection further localizes the hand landmarks. In subsequent images, once all max_num_hands hands are detected and the corresponding hand landmarks are localized, it simply tracks those landmarks without invoking another detection until it loses track of any of the hands. This reduces latency and is ideal for processing video frames. If set to true, hand detection runs on every input image, ideal for processing a batch of static, possibly unrelated, images. Default to false.

+ <b>MAX_NUM_HANDS</b><br>
Maximum number of hands to detect. Default to 2. But for changing the Volume it is set 1 as default so that stability is high and error is less

+ <b>MODEL_COMPLEXITY</b><br>
Complexity of the hand landmark model: 0 or 1. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.

+ <b>MIN_DETECTION_CONFIDENCE</b><br>
Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5 and for volume control default is 0.7

+ <b>MIN_TRACKING_CONFIDENCE:</b><br>
Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. Ignored if static_image_mode is true, where hand detection simply runs on every image. Default to 0.7.

<br>

Source: [MediaPipe Hands Solutions](https://google.github.io/mediapipe/solutions/hands#python-solution-api)

<div align="center">
    <img alt="mediapipeLogo" src="images/hand_landmarks_docs.png" height="200 x    " />
    
</div>


## CODE EXPLANATION
<b>Importing Libraries</b>
```py
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
```
***
Solution APIs 
```py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```
***

Volume Control Library Usage 
```py
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
```
***
Getting Volume Range using `volume.GetVolumeRange()` Method
```py
volRange = volume.GetVolumeRange()
minVol , maxVol , volBar, volPer= volRange[0] , volRange[1], 400, 0
```
***
Setting up webCam using OpenCV
```py
wCam, hCam = 1280, 720
cam = cv2.VideoCapture(0)
#Use 0 for deafult web cam and increase value based on connected web cam to its index
```
***
Using MediaPipe Hand Landmark Model for identifying Hands 
```py
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cam.isOpened():
    success, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )
```
***
Using multi_hand_landmarks method for Finding postion of Hand landmarks
```py
lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])    
```
***
Assigning variables for Thumb and Index finger position
```py
if len(lmList) != 0:
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]
```
***
Marking Thumb and Index finger using `cv2.circle()` and Drawing a line between them using `cv2.line()`
```py
cv2.circle(image, (x1,y1),15,(255,255,255))  
cv2.circle(image, (x2,y2),15,(255,255,255))  
cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
length = math.hypot(x2-x1,y2-y1)
if length < 50:
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
```
***
Converting Length range into Volume range using `numpy.interp()`
```py
vol = np.interp(length, [50, 220], [minVol, maxVol])
```
***
Changing System Volume using `volume.SetMasterVolumeLevel()` method
```py
volume.SetMasterVolumeLevel(vol, None)
volBar = np.interp(length, [50, 220], [400, 150])
volPer = np.interp(length, [50, 220], [0, 100])
```
***
Drawing Volume Bar using `cv2.rectangle()` method
```py
cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
        1, (0, 0, 0), 3)}

```
***
Displaying Output using `cv2.imshow` method
```py
cv2.imshow('handDetector', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
```
***
## Acknowledgments

- This project is built using the [MediaPipe](https://google.github.io/mediapipe/) framework for hand tracking and pose estimation.
- Special thanks to the [pycaw](https://github.com/AndreMiras/pycaw) library for controlling the system volume.

Closing webCam
```py
cam.release()
```
***
