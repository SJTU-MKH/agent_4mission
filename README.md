# 基于yolov5的显示器内容检测

算法流程图：

![2L4MrT](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/2L4MrT.png)

算法基于yolov5实现，更多教程参考官方文档：[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## 数据集

链接: https://pan.baidu.com/s/1i0W2EGbbcpH35lIyMluLEw 提取码: 9vle 

从百度网盘下载数据，并放至./yolov5/dataset/目录下

## 1. 显示器内容检测

![XnZPH3](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/XnZPH3.png)

### 基本设置

+ 标签：0直行、1左转、2右转、3掉头、4禁止直行、5禁止通行、6图+ 形、7蓝色车牌、8新能源车牌、9二维码

+ 算法：yolov5m
+ 输入图片的尺寸：320
+ 训练数据：./datasets/mainmission2/

### 使用方法 

**训练**

不加载预训练参数训练：
```
python train.py --batch-size 64 --epochs 300 --data data/mainmission2.yaml --name 'mainmission2' --cfg yolov5m.yaml --weights ''
```

**测试**

训练好的模型位置：`yolov5/runs/train/mainmission4.0_yolov5m/weights/best.pt`

推理保存带框图和label的txt
```
python detect.py --weight ./runs/train/mainmission4.0_yolov5m/weights/best.pt --source ./datasets/test_main/ --save-txt --save-conf
```

### 算法优化：

优化策略参考：[https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)

## 2.色块形状检测及颜色识别：

![lorU2b](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/lorU2b.png)

+ 标签：0-圆形、1-菱形、2-五角星、3-三角形、4-矩形
+ 算法：yolov5l
+ 训练数据：./datasets/realgraph6.0/

色块形状检测：

**训练**


不加载预训练参数训练：
```
python train.py --batch-size 64 --epochs 300 --data data/realgraph6.0.yaml --name 'realgraph6.0' --cfg yolov5m.yaml --weights ''
```


**测试**

训练好的模型位置：`./yolov5/runs/train/graph_yolov5l/`

推理保存带框图和label的txt
```
python detect.py --weight ./runs/train/graph_yolov5l//weights/best.pt --source ./datasets/test_shape/ --save-txt --save-conf
```

**色块颜色识别：**

使用resnet18实现颜色分类：0-red,1-green,2-blue,3-yellow,4-pin,5-qing,6-black,7-white.

+ 训练代码：[resnet18/train.py](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/train.py)
+ 推理代码：[resnet18/inference.py](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/inference.py)
+ 测试样例：[resnet18/images](https://github.com/huihui500/agent_4mission/tree/liujy/resnet18/images)
+ 训练好的模型：[resnet18/runs/model2.0/best_model.pth](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/runs/model2.0/best_model.pth)

## 3.数据制作

参考：[roboflow数据标注工具教学](https://www.bilibili.com/video/BV1LD4y1672d/)


