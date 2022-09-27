# 基于yolov5的显示器内容检测

算法流程图：

![2L4MrT](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/2L4MrT.png)

## 1. 显示器内容检测

![9yCUOB](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/9yCUOB.jpg)

### 基本设置

+ 标签：0直行、1左转、2右转、3掉头、4禁止直行、5禁止通行、6图+ 形、7蓝色车牌、8新能源车牌、9二维码

+ 算法：yolov5m
+ 输入图片的尺寸：320
+ 训练数据：./datasets/mainmission

### 使用方法 

**训练**

不加载预训练参数训练：
```
python train.py --batch-size 64 --epochs 300 --data data/mainmission.yaml --name 'mainmission' --cfg yolov5m.yaml --weights ''
```

其他参数：
+ `--img 320`：resize尺寸  默认640
+ `--hyp "/home3/HWGroup/liujy/agent_4mission_detection/yolov5/data/hyps/hyp.scratch-low-finetune.yaml"` ： Finetune 学习率调整

**测试**

训练好的模型位置：`yolov5/runs/train/mainmission4.0_yolov5m/weights/best.pt`进行推理。

查看命令行参数

`python detect.py  -h`

推理保存带框图和label的txt
`python detect.py --weight ./runs/train/mainmission4.0_yolov5m/weights/best.pt --source ./datasets/test_main/ --save-txt --save-conf --img-size 320`

### 算法优化：

优化策略参考：[https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)

## 2.色块形状检测及颜色识别：

![train_batch1](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/train_batch0.jpg)

色块形状检测：

模型位置：/home3/HWGroup/liujy/agent_4mission_detection/yolov5/runs/train/graph_yolov5l/

**色块颜色识别：**

使用resnet18实现颜色分类：0-red,1-green,2-blue,3-yellow,4-pin,5-qing,6-black,7-white.

+ 推理代码：[resnet18/inference.py](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/inference.py)
+ 测试样例：[resnet18/images](https://github.com/huihui500/agent_4mission/tree/liujy/resnet18/images)
+ 训练好的模型：[resnet18/runs/model2.0/best_model.pth](https://github.com/huihui500/agent_4mission/blob/liujy/resnet18/runs/model2.0/best_model.pth)

## 3.数据制作

参考：[roboflow数据标注工具教学](https://www.bilibili.com/video/BV1LD4y1672d/)


