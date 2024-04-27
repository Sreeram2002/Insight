from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
from gtts import gTTS
import os
from playsound import playsound

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("/home/itsbighero6/text_detector_using-EAST-master/frozen_east_text_detection.pb")

time.sleep(3)  # WAIT FOR 3 SECONDS TO CAPTURE FRAME !!!!

hasFrame, image = cap.read()
if not hasFrame:
    print("Failed to capture frame")
    exit()

orig = image
(H, W) = image.shape[:2]

(newW, newH) = (640, 320)
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        if scoresData[x] < 0.5:
            continue

        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

boxes = non_max_suppression(np.array(rects), probs=confidences)

for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Extract the ROI (Region of Interest) from the original image
    roi = orig[startY:endY, startX:endX]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')

    # Print the detected text
    print("Detected text:", text)

    # Convert the detected text to speech
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")


    # Draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)

cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
