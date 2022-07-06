import os
import time
import cv2
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import  DataLoader
import torchsummary
from torch.autograd import Variable
import copy
import numpy as np

CLASSES = ['red', 'green', 'blue', 'yellow', 'pin', 'qing', 'black', 'white']
SHAPE = ['circle', 'rhombus', 'fivestar', 'triangle', 'rectangle']

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

def get_model(m_path=None, vis_model=False):

    resnet18 = models.resnet18(pretrained=False)
    # torchsummary.summary(resnet18, (3,224,224))
    # 修改全连接层的输出
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 8)

    # 加载模型参数
    if m_path:
        checkpoint = torch.load(m_path)
        resnet18.load_state_dict(checkpoint['model_state_dict'])

    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device=device)

    return resnet18

model = get_model()
model.load_state_dict(torch.load('/home3/HWGroup/liujy/agent_4mission_detection/resnet18/best_model.pth'))
model = model.to(device)
model.eval()


def crop(img,xyxy):
    x1, y1 ,x2, y2 = map(int,xyxy)
    return img[y1:y2 ,x1:x2, :]

def main(img_path, label_path):
    # img = cv2.imread(img_path)
    IMG = Image.open(img_path).convert('RGB')
    w, h = IMG.size
    # h, w, _ = img.shape 
    with open(label_path) as file:
        content=file.readlines()
    for i in range(len(content)):
        c = content[i].split(' ')[0]
        xywh = content[i].split(' ')[1:]
        xywh = [xywh[0], xywh[1], xywh[2], xywh[3][:-2]]
        xywh = [float(j) for j in xywh]
        xywh = [xywh[0]*w, xywh[1]*h, xywh[2]*w, xywh[3]*h]
        xyxy = [xywh[0]-xywh[2]/2, xywh[1]-xywh[3]/2, xywh[0]+xywh[2]/2, xywh[1]+xywh[3]/2]
        # img = Image.open(img_path).convert('RGB')
        img = IMG.crop(xyxy)
    
        img = val_transforms(img)
        img  = np.transpose(np.expand_dims(np.array(img, np.float32), 0), (0, 1, 3, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(img)
            photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   获得所属种类
        #---------------------------------------------------#
        class_name  = np.argmax(preds)
        probability = np.max(preds)

        print(CLASSES[class_name],probability,SHAPE[int(c)])


img_path = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/images/bigshape.jpg'
label_path = '/home3/HWGroup/liujy/agent_4mission_detection/resnet18/images/bigshape.txt'
main(img_path, label_path)


