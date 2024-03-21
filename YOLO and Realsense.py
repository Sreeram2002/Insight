# This code takes a frame, runs YOLOV5 Object Detection Model and provides voice feedback


import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
import torch 
from pathlib import Path
import os
from gtts import gTTS
import playsound

pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

pipe.start(cfg)

# Wait for the first frame
frames = pipe.wait_for_frames()

depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

if not depth_frame or not color_frame:
    print("Error: Could not get frames.")
    exit()

depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

contrast = 2  # adjust contrast value
brightness = 75  # adjust brightness value
frame = cv2.addWeighted(color_image, contrast, np.zeros(color_image.shape, color_image.dtype), 0, brightness)

depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5), cv2.COLORMAP_JET)

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)




# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)

# Run YOLOv5 detection on the frame
results = model(frame)

# Process the results (e.g., display bounding boxes and labels)
results.render()  # Renders detections (bounding boxes and labels)

# Display the resulting frame (optional)
cv2.imshow('YOLOv5 Webcam Detection (Single Frame)', frame)

results.print()

preds = results.pred  # Get predictions

text_to_speak = ''

for pred in preds:
    for *box, conf, cls in pred:
        label = results.names[int(cls)]  # Get the label of the class
        
        if text_to_speak != '':
            text_to_speak += ','
        text_to_speak += f'detected {label}'
            
    print(text_to_speak)
    
    speech = gTTS(text=text_to_speak, lang ='en')
    speech.save("objects_detected.mp3")
    playsound.playsound("objects_detected.mp3", block=True)  # Block execution until audio finishes
    #os.remove("objects_detected.mp3")





cv2.waitKey(0)  # Wait for any key press before closing
cv2.destroyAllWindows()
pipe.stop()
