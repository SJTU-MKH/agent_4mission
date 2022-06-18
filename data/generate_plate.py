# 生成带标签的车牌数据集

# TODO: 1.逐张读取图片—>获取尺寸->标签转化->修改图片标签文件名->保存

import glob
import os
import cv2
import random
import numpy as np
import pdb

def generate_label(img_path, cls):
    img = cv2.imread(img_path)
    x1y1, x2y2 = img_path.split('-')[2].split('_')
    xmin, ymin = map(int,x1y1.split("&"))
    xmax, ymax = map(int,x2y2.split("&"))
    h, w, _ = img.shape
    # pdb.set_trace()
    box = dict(cls=cls, xywh=[(xmin+xmax)/2/w, (ymin+ymax)/2/h, (xmax-xmin)/w, (ymax-ymin)/h])
    return img, box

clss = [7, 8]
for cls in clss:
    if cls == 7:
        pic_path = "/Users/liujiyao/Documents/安卓识别项目@马可汇/plate_number_detect/unlabeled_graph/blue"
    elif cls == 8:
        pic_path = "/Users/liujiyao/Documents/安卓识别项目@马可汇/plate_number_detect/unlabeled_graph/green"
    save_path = "mission/plate"

    if not os.path.exists(save_path + "/pic"):
        os.makedirs(save_path + "/pic")
    if not os.path.exists(save_path + "/annotation"):
        os.makedirs(save_path + "/annotation")

    pic_paths = glob.glob(os.path.join(pic_path,'*'))
    print(len(pic_paths))
    index = glob.glob(os.path.join(save_path,'pic/*'))
    num = len(index)

    for i in range(len(pic_paths)):
        img, bbox = generate_label(pic_paths[i],cls)
        img_save_path = save_path+"/pic/"+str(i+num)+".jpg"
        txt_save_path = img_save_path.replace(".jpg", ".txt").replace("pic", "annotation")
        cv2.imwrite(img_save_path, img)
        f_txt = open(txt_save_path, "w")
        f_txt.write(str(bbox["cls"])+" "+" ".join([str(x) for x in bbox["xywh"]])+"\n")
        f_txt.close()
