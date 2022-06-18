from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
import random


def random_onetransform(num):
    "单个变换"
    T = dict([('0', RandomHorizontalFlip(1)),
                       ('1', RandomScale(0.3, diff=True)),
                       ('2', RandomTranslate(0.3, diff=True)),
                       ('3', RandomRotate(20)),
                       ('4', RandomShear(0.2))])
    return T[str(num)]


def xyxyc2cxywh_n(x, w, h):
    y = np.zeros_like(x)
    y[:, 1] = (x[:, 0] + x[:, 2])/2/w
    y[:, 2] = (x[:, 1] + x[:, 3])/2/h
    y[:, 3] = (x[:, 2] - x[:, 0])/w
    y[:, 4] = (x[:, 3] - x[:, 1])/h
    y[:, 0] = x[:, 4]
    return y


mission = ["Qrcode", "biggraph"]

data_path = "mission/"
pic_path = "/labeled/pic"
for m in mission:
    img_folder = data_path + m + pic_path
    imgs_path = glob.glob(img_folder+"/*")
    num = len(imgs_path)
    for index, img_path in enumerate(imgs_path):
        words = img_path.split(".")
        txt_path = words[0].replace("pic", "txt") + ".txt"
        if not os.path.exists(txt_path):
            continue
        img_ = cv2.imread(img_path)[:, :, ::-1]
        bboxes_ = np.loadtxt(txt_path, dtype=np.float32)
        bboxes_ = bboxes_.reshape([-1, 5])

        ## xml->txt to annotation
        # h, w, _ = img_.shape
        # bboxes_ = xyxyc2cxywh_n(bboxes_, w, h)
        # words = txt_path.split('/')
        # words[-2] = "annotation"
        # new_txt_path = "/".join(words)
        # np.savetxt(new_txt_path, bboxes_, "%0.8f")

        transform_num = int(random.random()*5)
        transforms = random_onetransform(transform_num)
        img, bboxes = transforms(img_, bboxes_)
        h, w, _ = img.shape
        bboxes = xyxyc2cxywh_n(bboxes, w, h)
        words = img_path.split('/')[:-1]
        words.append(str(index+num))
        new_img_path = '/'.join(words)+".jpg"
        new_txt_path = '/'.join(words).replace("pic", "annotation")+".txt"
        np.savetxt(new_txt_path, bboxes, "%0.8f")
        cv2.imwrite(new_img_path, img)
        ## 多变换
        # transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
