import sys
sys.path.append('../')
import utils.dataMake.geometry as dmg
import utils.dataMake.labelWrite as lw
import glob
import os
import cv2
import random
import numpy as np

def add_graph(img_path,add_img_path):
    bboxs = []
    colors_name = list(dmg.color.keys())
    img = cv2.imread(img_path)
    img_h, img_w, channel = img.shape
    step_w = (img_w-40)/4
    step_h = img_h/2
    # 颜色与周围的不同
    last_index = -1
    ll_index = -1
    for i in range(4):
        center_w = (i+0.5)*step_w+20
        for j in range(2):
            center_h = (j+0.5)*step_h
            while True:
                color_index = int(random.random()*8)
                if color_index != last_index and color_index != ll_index:
                    ll_index = last_index
                    last_index = color_index
                    break

            color = dmg.color[colors_name[color_index]]
            graph_index = int(random.random()*5)
            center = (int(center_w), int(center_h))
            # print(center)
            rotation = random.random()
            if graph_index == 0:
                graph_obj = 0  # "circle"
                R = int((0.5+random.random()/2)*step_w/2)
                cv2.circle(img, center, R, color, -1)
                xmin = center[0]-R
                xmax = center[0]+R
                ymin = center[1]-R
                ymax = center[1]+R
            elif graph_index == 1:
                graph_obj = 1  # "rhombus"
                w = (0.5+random.random()/2)*step_w/2
                h = (0.5+random.random()/2)*step_h/2
                vertexs = dmg.compute_rhombus(center, w, h, rotation)
            elif graph_index == 2:
                graph_obj = 2  # "fivestar"
                R = (0.5+random.random()/2)*step_w/2
                vertexs = dmg.compute_fivestar(center, R, rotation)
            elif graph_index == 3:
                graph_obj = 3  # "triangle"
                w = (0.5+random.random()/2)*step_w/2
                h = (0.5+random.random()/2)*step_h/2
                vertexs = dmg.compute_triangle(center, w, h, rotation)
            elif graph_index == 4:
                graph_obj = 4  # "rectangle"
                w = (0.5+random.random()/2)*step_w/2
                h = (0.5+random.random()/2)*step_h/2
                vertexs = dmg.compute_rectangle(center, w, h, rotation)

            if graph_index > 0:
                pts = np.array(vertexs, dtype=np.int32)

                xmin = np.min(pts[:, 0])
                xmax = np.max(pts[:, 0])
                ymin = np.min(pts[:, 1])
                ymax = np.max(pts[:, 1])
                if xmin<0 or xmax>img_w or ymin<0 or ymax>img_h:
                    continue
                cv2.drawContours(img, [pts], -1, color, -1)
                # cv2.line(img, (xmin, ymin), (xmin, ymax), (0,255,0), 3)
                # cv2.line(img, (xmin, ymax), (xmax, ymax), (0,255,0), 3)
            # box归一化
            # dict{"cls":"", "xywh":[c_x, c_y, w, h], "color":[bgr], "rotation":0~1}
            box = dict(cls=graph_obj, xywh=[(xmin+xmax)/2/img_w, (ymin+ymax)/2/img_h, (xmax-xmin)/img_w, (ymax-ymin)/img_h], color=color, rotation=rotation)
            # print(box)
            bboxs.append(box)
    # 边缘模糊
    dst = cv2.blur(img, (5, 5))
    # gamma
    # gamma = random.uniform(0.2,2)
    # dst = gamma_transform(dst,1)
    # cv2.imshow("poyline", dst)
    # cv2.waitKey(0)

    # 图像混叠
    add_img = cv2.imread(add_img_path)
    add_img = cv2.resize(add_img, (img_w, img_h))
    rst = cv2.addWeighted(dst, 0.6, add_img, 0.4, 0)
    if random.random()>0.7:
        blank = 255 * np.ones(rst.shape, rst.dtype)
        rst = cv2.addWeighted(rst, 0.7, blank, 1-0.7, 0)
    return rst, bboxs


def gamma_transform(image, gamma=1.6):

    max_value = np.max(image)
    min_value = np.min(image)
    value_l = max_value - min_value
    image = (image - min_value)/value_l
    image = np.power(image, gamma)
    image = image * 255
    return image.astype(np.uint8)


save_path = "mission/graph/labeled/"

if not os.path.exists(save_path + "pic"):
    os.makedirs(save_path + "pic")
if not os.path.exists(save_path + "annotation"):
    os.makedirs(save_path + "annotation")

pic_paths = glob.glob("background_graph/*")
save_path = "./mission/graph/labeled/"
index = glob.glob("./mission/graph/labeled/pic/*")
num = len(index)
# f_txt = open("mission_graph.txt", "w")

import random
shuffle_pic_paths = pic_paths.copy()
random.shuffle(shuffle_pic_paths)
for i in range(len(pic_paths)):
    img, bboxes = add_graph(pic_paths[i],shuffle_pic_paths[i])
    img_save_path = save_path+"pic/"+str(i+num)+".jpg"
    txt_save_path = img_save_path.replace(".jpg", ".txt").replace("pic", "annotation")
    cv2.imwrite(img_save_path, img)
    f_txt = open(txt_save_path, "w")
    for bbox in bboxes:
        f_txt.write(str(bbox["cls"])+" "+" ".join([str(x) for x in bbox["xywh"]])+"\n")
    f_txt.close()