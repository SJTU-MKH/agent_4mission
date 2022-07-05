# 颜色识别
"""
input:
    - *.jpg  # 原始图像
    - *.txt  # 检测结果 yolov5格式
output:
    - 图片中每个ROI区域的主要hsv值, 颜色类别
    
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
from PIL import Image, ImageDraw, ImageFont

def balance(img):
    # 调整白平衡 
    b, g, r = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])

    return balance_img


def equalize_hist_color(img):
    # 彩色图直方图均衡化
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image


def DrawBox(im, box):
    # 绘制Bounding Box
    draw = ImageDraw.Draw(im)
    draw.rectangle(box,  outline="#FFFFFF", width=1)


def DrawLabel(im, loc, label, color):
    # 绘制label
    draw = ImageDraw.Draw(im)
    # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    # font = ImageFont.truetype('/System/Library/Fonts/Times.ttc', 16)
    # draw.text(loc, color, font=font)
    draw.text(loc, color)

def ImgShow(imgpath, boxs, cs, label="n"):
    # 展示图片
    im = Image.open(imgpath)
    for i in range(len(cs)):
        DrawBox(im, [tuple(boxs[i][0:2]),tuple(boxs[i][2:])])
        DrawLabel(im, boxs[i][0:2], label, cs[i])
    # 显示图片
    # im.show()
    im.save('/home3/HWGroup/liujy/agent_4mission_detection/submission/result_color.jpg')


color = ["red1", "yellow", "green", "qing", "blue", "pin", "red", 'black', 'white' ]
# color_range = {
#     "red1": {"color_lower": np.array([0, 35, 46]), "color_upper": np.array([15, 255, 255])},
#     "yellow": {"color_lower": np.array([15, 35, 46]), "color_upper": np.array([34, 255, 255])},
#     "green": {"color_lower": np.array([35, 35, 46]), "color_upper": np.array([77, 255, 255])},
#     "qing": {"color_lower": np.array([78, 35, 46]), "color_upper": np.array([99, 255, 255])},
#     "blue": {"color_lower": np.array([100, 35, 46]), "color_upper": np.array([124, 255, 255])},
#     "pin": {"color_lower": np.array([125, 35, 46]), "color_upper": np.array([155, 255, 255])},
#     "red": {"color_lower": np.array([156, 35, 46]), "color_upper": np.array([180, 255, 255])},
#     'black': {"color_lower": np.array([0, 0, 0]), "color_upper": np.array([180, 255, 35])},
#     'white': {"color_lower": np.array([0, 0, 221]), "color_upper": np.array([180, 30, 255])},
#          }


# 根据实际情况修正
color_range = {
    "red1": {"color_lower": np.array([0, 35, 46]), "color_upper": np.array([15, 255, 255])},
    "yellow": {"color_lower": np.array([15, 35, 46]), "color_upper": np.array([47, 255, 255])},
    "green": {"color_lower": np.array([48, 35, 46]), "color_upper": np.array([77, 255, 255])},
    "qing": {"color_lower": np.array([78, 35, 46]), "color_upper": np.array([88, 255, 255])},
    "blue": {"color_lower": np.array([89, 35, 46]), "color_upper": np.array([124, 255, 255])},
    "pin": {"color_lower": np.array([125, 35, 46]), "color_upper": np.array([155, 255, 255])},
    "red": {"color_lower": np.array([156, 35, 46]), "color_upper": np.array([180, 255, 255])},
    'black': {"color_lower": np.array([0, 0, 0]), "color_upper": np.array([180, 255, 35])},
    'white': {"color_lower": np.array([0, 0, 221]), "color_upper": np.array([180, 30, 255])},
         }
shape = {0:"circle", 1: "rhombus", 2:"fivestar", 3:"triangle", 4:"rectangle"}

def main(img_path, label_path):
    img = cv2.imread(img_path)
    img = balance(img)
    # img = equalize_hist_color(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波降噪，模糊图片
    H, W, _ = img.shape

    # 获取bbox
    with open(label_path) as file:
        content=file.readlines()

    # 遍历boxes，识别每个目标图形的颜色
    boxs = []
    cs = []
    for line in content:
        box = line[:-2].split(' ')
        box = [float(i) for i in box[1:]]
        bbox = [(box[0]-box[2]/2)*W, (box[1]-box[3]/2)*H, (box[0]+box[2]/2)*W, (box[1]+box[3]/2)*H]
        bbox = [int(i) for i in bbox]
        boxs.append(bbox)

        crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        y, x, z = crop.shape
        if x>4 and y>4:
            crop = crop[3*y//8 : 5*y//8 , 3*x//8 : 5*x//8 ,:]
        # im = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))  
        # im.show()

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

        # hsv = hsv[y//2,x//2,:]

        res = "none"
        # 遍历颜色判断
        for c in color:
            upper = (np.maximum(hsv,color_range[c]["color_lower"])==hsv)
            lower = (np.minimum(hsv,color_range[c]["color_upper"])==hsv)
            if np.all(upper) and np.all(lower):
                res = c
                break
        print(f"hsv:{hsv}, result:{res}")
        cs.append(res)

    # 展示颜色识别结果
    ImgShow(img_path, boxs, cs)
    

img_path = '/home3/HWGroup/liujy/agent_4mission_detection/submission/test.jpg'
label_path = '/home3/HWGroup/liujy/agent_4mission_detection/submission/test.txt'
# img_path = '/home3/HWGroup/liujy/agent_4mission_detection/submission/bigshape.jpg'
# label_path = '/home3/HWGroup/liujy/agent_4mission_detection/submission/bigshape.txt'
main(img_path, label_path)