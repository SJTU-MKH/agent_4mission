import cv2
import numpy as np
import glob
import os


def cxywh_n2xyxyc(x, w, h):
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 1] - x[:, 3]/2)*w
    y[:, 1] = (x[:, 2] - x[:, 4]/2)*h
    y[:, 2] = (x[:, 1] + x[:, 3]/2)*w
    y[:, 3] = (x[:, 2] + x[:, 4]/2)*h
    y[:, 4] = x[:, 0]
    return y


mission = ["Qrcode", "biggraph"]
data_path = "mission/"
pic_path = "/labeled/pic"
for m in mission:
    img_folder = data_path + m + pic_path
    imgs_path = glob.glob(img_folder+"/*")
    for index, img_path in enumerate(imgs_path):
        words = img_path.split(".")
        txt_path = words[0].replace("pic", "annotation") + ".txt"
        if not os.path.exists(txt_path):
            continue
        img_ = cv2.imread(img_path)
        bboxes_ = np.loadtxt(txt_path, dtype=np.float32)
        bboxes_ = bboxes_.reshape([-1, 5])
        h, w, _ = img_.shape
        bboxes_ = cxywh_n2xyxyc(bboxes_, w, h)
        for i in range(len(bboxes_)):
            box = bboxes_[i, :]
            cv2.line(img_, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), [0, 255, 0])
            cv2.line(img_, (int(box[0]), int(box[3])), (int(box[2]), int(box[3])), [0, 255, 0])
            cv2.line(img_, (int(box[2]), int(box[3])), (int(box[2]), int(box[1])), [0, 255, 0])
            cv2.line(img_, (int(box[2]), int(box[1])), (int(box[0]), int(box[1])), [0, 255, 0])
        cv2.imshow("win", img_)
        cv2.waitKey(0)