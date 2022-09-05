# 基于yolov5的显示器内容检测

+ 主任务：十类目标检测

+ 子任务：色块形状检测+颜色识别

## 1. 显示器内容检测

![9yCUOB](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/9yCUOB.jpg)

![train_batch1](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/train_batch1.jpg)

### 算法详情

+ 标签：0直行、1左转、2右转、3掉头、4禁止直行、5禁止通行、6图+ 形、7蓝色车牌、8新能源车牌、9二维码
+ 算法：yolov5s
+ 输入图片的尺寸：320
+ 训练数据：./datasets/mainmission
+ 训练好的模型： [yolov5/runs/train/mainmission3.0/weights](https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/mainmission3.0/weights)

## 2.色块形状检测及颜色识别：

![train_batch1](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/train_batch0.jpg)

### 形状检测

+ 训练好的模型权重：[yolov5/runs/train/yolov5s2](https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/yolov5s2)

+ 训练数据：/home3/HWGroup/liujy/agent_4mission_detection/datasets/realgraph5.0

### 颜色识别

#### a.传统方法


+ 根据检测框中心点像素HSV范围确定颜色，需要根据实际使用情况调整颜色阈值范围，代码：[submission/color_recognize.py](https://github.com/huihui500/agent_4mission/blob/liujy/submission/color_recognize.py)

+ 测试代码：[查看图像形状检测框坐标](https://github.com/huihui500/agent_4mission/blob/liujy/yolov5/location.ipynb)

#### b.深度学习方法

使用resnet18实现颜色分类：0-red,1-green,2-blue,3-yellow,4-pin,5-qing,6-black,7-white.

+ 推理代码：[resnet18/inference.py](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/inference.py)
+ 测试样例：[resnet18/images](https://github.com/huihui500/agent_4mission/tree/liujy/resnet18/images)
+ 训练好的模型：[resnet18/runs/model2.0/best_model.pth](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/runs/model2.0/best_model.pth)

## 3.检测模型推理代码（1/2）

代码：https://github.com/huihui500/agent_4mission/blob/liujy/yolov5/detect.py

使用方法:

```bash
python detect.py --weight '权重路径' --source '图片/图片文件夹/视频路径' --save-txt --save-conf --img-size 320
```


## utils

1. bin2img and img2bin: ./bin2img


