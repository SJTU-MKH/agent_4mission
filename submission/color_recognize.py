# 颜色识别
"""
input:
    - *.jpg  # 原始图像
    - *.txt  # 检测的roi框
output:
    - 图片中每个ROI区域的色相值H, 颜色类别
    
标准颜色表:
    RED = { 255, 0, 0 };
    GREEN = { 0, 255, 0 };
    BLUE = { 0, 0, 255 }; 
    YELLOW = { 255, 255, 0 }; 
    PIN = { 255, 0, 255 }; 
    QING = { 0, 255, 255 }; 
    BLACK = { 0, 0, 0 };
    WHITE = { 255, 255, 255 };

"""
import cv2
import numpy as np
from pyparsing import col

def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

# hsv range:[h, s, v]
color = ["blue", "red", "red1", "yellow", "green", "pin", "qing", 'black', 'white' ]
color_range = {
    "blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},
    "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
    "red1": {"color_lower": np.array([0, 43, 46]), "color_upper": np.array([10, 255, 255])},
    "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
    "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
    "pin": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
    "qing": {"color_lower": np.array([78, 43, 46]), "color_upper": np.array([99, 255, 255])},
    'black': {"color_lower": np.array([0, 0, 0]), "color_upper": np.array([180, 255, 46])},
    'white': {"color_lower": np.array([0, 0, 221]), "color_upper": np.array([180, 30, 255])},
         }
shape = {0:"circle", 1: "rhombus", 2:"fivestar", 3:"triangle", 4:"rectangle"}

img_path = '/Users/liujiyao/Documents/安卓识别项目@马可汇/agent_4mission/submission/7.jpg'
label_path = '/Users/liujiyao/Documents/安卓识别项目@马可汇/agent_4mission/submission/7.txt'

img = cv2.imread(img_path)
img = equalize_hist_color(img)
img = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波降噪，模糊图片
h, w, _ = img.shape
# bbox
with open(label_path) as file:
    content=file.readlines()

# 获取多个bbox
boxs = []
for line in content:
    box = line[:-2].split(' ')
    box = [float(i) for i in box[1:]]
    bbox = [(box[0]-box[2]/2)*w, (box[1]-box[3]/2)*h, (box[0]+box[2]/2)*w, (box[1]+box[3]/2)*h]
    bbox = [int(i) for i in bbox]
    boxs.append(bbox)

for box in boxs:
    crop = img[box[1]:box[3],box[0]:box[2],:]
    # cv2.imshow('crop', crop)
    # cv2.waitKey(0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 统计主要元素的hsv值
    squ = hsv[:,:,0].reshape(-1)
    count = np.bincount(squ)
    h = np.argmax(count)  # 占比最多的h值
    squ = hsv[:,:,1].reshape(-1)
    count = np.bincount(squ)
    s = np.argmax(count)  # 占比最多的s值
    squ = hsv[:,:,2].reshape(-1)
    count = np.bincount(squ)
    v = np.argmax(count)  # 占比最多的v值
    hsv = np.array([h,s,v])

    # 遍历颜色判断
    for c in color:
        upper = (np.maximum(hsv,color_range[c]["color_lower"])==hsv)
        lower = (np.minimum(hsv,color_range[c]["color_upper"])==hsv)
        if np.all(upper) and np.all(lower):
            print(f"hsv:{hsv}, result:{c}")
    # break