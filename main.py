import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolodetect import YoloDetect


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame

# from yolodetect import YoloDetect
url = "rtsp://admin:td123456@192.168.55.21/cam/realmonitor?channel=1&subtype=00"
# Initializing video object
video = cv2.VideoCapture(url)
points = []
detect = False

#Video information
# fps = video.get(cv2.CAP_PROP_FPS)
# w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f"FPS: {fps}");

model = YoloDetect();
# Initialzing object for writing video output
torch.cuda.empty_cache()
# Initializing model and setting it for inference
with torch.no_grad():
    n = 0
    while(video.isOpened()):
        ret, img0 = video.read()
        
        # if ret:
        n = n + 1
        if n == 4:
            if(detect):
                img0,num = model.detect(img0)
            
            # Ve ploygon
            img0 = draw_polygon(img0, points)
            cv2.imshow('Intrusion Warning',img0);
            cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)    
            n = 0
            
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            points.append(points[0])
            detect = True
    

