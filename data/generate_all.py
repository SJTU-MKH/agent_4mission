# 整合所有数据

# 标签：0直行、1左转、2右转、3掉头、4禁止直行、5禁止通行、6图形、7蓝色车牌、8新能源车牌、9二维码
# TODO: 整合数据，修改标签，修改图片和txt序号

# from __future__ import annotations
import glob
import os
import cv2
import random
import numpy as np
import pdb
import shutil

def label_change(label_path, cls):
    labels = []
    with open(label_path) as file:
        content=file.readlines()
    for line in content:
        # print(line)
        spt = line[:-1].split(' ')
        spt[0] = str(cls)
        labels.append(' '.join(spt))
    return labels

root = '/Users/liujiyao/Documents/安卓识别项目@马可汇/agent_4mission/data/mission'
save_path = '/Users/liujiyao/Documents/安卓识别项目@马可汇/agent_4mission/data/mission/all'
if not os.path.exists( os.path.join(root, "all",'images')):
    os.makedirs(os.path.join(root, "all",'images'))
if not os.path.exists(os.path.join(root, "all",'labels')):
    os.makedirs(os.path.join(root, "all",'labels'))

# 1. 交通标志 clss: 0, 1, 2, 3, 4, 5
pic_path = os.path.join(root, "traffic_sign",'labeled/pic')
pic_paths = glob.glob(os.path.join(pic_path,'*'))
annotation_path = os.path.join(root, "traffic_sign",'labeled/annotation')
annotation_paths = glob.glob(os.path.join(annotation_path,'*'))
print(len(pic_paths))
index = glob.glob(os.path.join(save_path,'images/*'))
num = len(index)

for i in range(len(pic_paths)):
    shutil.copy(pic_paths[i],save_path+"/images/"+str(i+num)+".jpg")
    shutil.copy(annotation_paths[i],save_path+"/labels/"+str(i+num)+".txt")
    # break

# 2. 图形 clss: 6
cls = 6
pic_path = os.path.join(root, "biggraph",'labeled/pic')
pic_paths = sorted(glob.glob(os.path.join(pic_path,'*')))
annotation_path = os.path.join(root, "biggraph",'labeled/annotation')
annotation_paths = sorted(glob.glob(os.path.join(annotation_path,'*')))
print(len(pic_paths))
index = glob.glob(os.path.join(save_path,'images/*'))
num = len(index)

for i in range(len(pic_paths)):
    shutil.copy(pic_paths[i],save_path+"/images/"+str(i+num)+".jpg")
    labels = label_change(annotation_paths[i],cls)
    f_txt = open(save_path+"/labels/"+str(i+num)+".txt", "w")
    for label in labels:
        f_txt.write(label + "\n")
    f_txt.close()
    # break


# 3. 车牌 clss: 7,8
pic_path = os.path.join(root, "plate",'labeled/pic')
pic_paths = glob.glob(os.path.join(pic_path,'*'))
random.shuffle(pic_paths)
annotation_path = os.path.join(root, "plate",'labeled/annotation')
annotation_paths = glob.glob(os.path.join(annotation_path,'*'))
random.shuffle(annotation_paths)
print(len(pic_paths[:220]))
index = glob.glob(os.path.join(save_path,'images/*'))
num = len(index)

for i in range(len(pic_paths[:220])):
    shutil.copy(pic_paths[i],save_path+"/images/"+str(i+num)+".jpg")
    shutil.copy(annotation_paths[i],save_path+"/labels/"+str(i+num)+".txt")
    # break


# 4. 二维码 clss: 9
cls = 9
pic_path = os.path.join(root, "Qrcode",'labeled/pic')
pic_paths = sorted(glob.glob(os.path.join(pic_path,'*')))
annotation_path = os.path.join(root, "Qrcode",'labeled/annotation')
annotation_paths = sorted(glob.glob(os.path.join(annotation_path,'*')))
print(len(pic_paths))
index = glob.glob(os.path.join(save_path,'images/*'))
num = len(index)

for i in range(len(pic_paths)):
    shutil.copy(pic_paths[i],save_path+"/images/"+str(i+num)+".jpg")
    labels = label_change(annotation_paths[i],cls)
    f_txt = open(save_path+"/labels/"+str(i+num)+".txt", "w")
    for label in labels:
        f_txt.write(label + "\n")
    f_txt.close()
    # break


    # img, bbox = generate_label(pic_paths[i],cls)
    # img_save_path = save_path+"/pic/"+str(i+num)+".jpg"
    # txt_save_path = img_save_path.replace(".jpg", ".txt").replace("pic", "annotation")
    # cv2.imwrite(img_save_path, img)
    # f_txt = open(txt_save_path, "w")
    # f_txt.write(str(bbox["cls"])+" "+" ".join([str(x) for x in bbox["xywh"]])+"\n")
    # f_txt.close()
