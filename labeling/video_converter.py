#!/usr/bin/python

import cv2
import numpy as np

"""
Training:
-   DJI_0007 - 02.11.2021 - Keterminde havn - Mother-calf resting (close) (2).MOV   Frame rate: 0.33 fps
-   20190407 - Male - Five porpoises foraging.MOV                                   Frame rate: 0.33 fps
-   20190319 - Male - Group of porpoises and a calf (3).MOV                         Frame rate: 0.33 fps
-   20190629 - Kerteminde bay - Three porpoises Dennis data.MOV                     Frame rate: 0.33 fps
-   DJI_0001 - 29.05.2021 - Thyborøn - Group of four resting (+ rainblows).MOV      Frame rate: 0.33 fps

Validation/test:
-   DJI_0014.MOV                                                                                            Frame rate: 0.33 fps
-   DJI_0013 - 31.08.2021 - Fyns Hoved - Mother-calf foraging and traveling + huge Liions Maine jellies.MOV Frame rate: 0.33 fps
"""

try:
    video = cv2.VideoCapture("data/videos/4096x2160/DJI_0013 - 31.08.2021 - Fyns Hoved - Mother-calf foraging and traveling + huge Liions Maine jellies.MOV")
except:
    print("Invalid video")

# Calculate the duration, fps, frames as well as frame height and width
fps = video.get(cv2.CAP_PROP_FPS)
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frames / fps

print(f"Framerate: {fps}")
print(f"Frame count: {frames}")
print(f"Duration: {duration} sec.")
print(f"Duration: {int(duration/60)} min. {int(duration%60)} sec.")

height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f"Frame height: {int(height)}")
print(f"Frame width: {int(width)}")

sec = 3.0
count = 0
while video.isOpened():
    count += 1
    if count <= int(frames/(sec * fps)):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(sec * count * fps))
        ret, frame = video.read()
        if ret == True:
            cv2.imwrite("labeling/video_conversion/images/20210831_image"+str(count)+".png", frame)
    else:
        break
