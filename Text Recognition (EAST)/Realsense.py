import numpy as np
import cv2
import pytesseract
from gtts import gTTS
import os
from playsound import playsound
import pyrealsense2 as rs

# Configure depth and color streams of RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load the EAST text detection model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to OpenCV format
        image = np.asanyarray(color_frame.get_data())

        orig = image.copy()
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

        boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.3)

        for i in boxes:
            i = i[0]
            (startX, startY, endX, endY) = rects[i]

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

