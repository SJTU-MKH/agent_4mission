import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

def crop(img,xyxy):
    x1, y1 ,x2, y2 = map(int,xyxy)
    return img[y1:y2 ,x1:x2, :]

def main(img_path, label_path):
    global count
    img = cv2.imread(img_path)
    h, w, _ = img.shape 
    with open(label_path) as file:
        content=file.readlines()
    for i in range(len(content)):
        c = content[i].split(' ')[0]
        xywh = content[i].split(' ')[1:]
        xywh = [xywh[0], xywh[1], xywh[2], xywh[3][:-2]]
        xywh = [float(j) for j in xywh]
        xywh = [xywh[0]*w, xywh[1]*h, xywh[2]*w, xywh[3]*h]
        xyxy = [xywh[0]-xywh[2]/2, xywh[1]-xywh[3]/2, xywh[0]+xywh[2]/2, xywh[1]+xywh[3]/2]
        crop_img = crop(img, xyxy)
        count += 1
        cv2.imwrite(f'/home3/HWGroup/liujy/agent_4mission_detection/resnet18/images/crop/{count}.jpg',crop_img)

count = 0
img_path = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/images/bigshape.jpg'
label_path = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/images/bigshape.txt'
main(img_path, label_path)