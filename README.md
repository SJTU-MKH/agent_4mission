# yolov5s实现检测（主任务+色块检测）

## pipline

主任务：十类目标检测->色块检测：截取主任务检测到的显示器区域，进行下一阶段的色块检测。


## train

1. 模型选择：yolov5s

2. 输入图片的尺寸：320

3. 权重文件的位置

主任务：模型权重 [yolov5/runs/train/main_mission/weights](https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/main_mission/weights)

色块检测：模型权重 [yolov5/runs/train/realgraph-320/weights](https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/realgraph-320/weights)

## inference demo

代码：https://github.com/huihui500/agent_4mission/blob/liujy/yolov5/detect.py

使用:

```bash
python detect.py --weight '权重路径' --source '图片/图片文件夹/视频路径' --save-txt --save-conf --img-size 320
```

# 颜色识别

更新使用resnet18实现颜色分类：0-red,1-green,2-blue,3-yellow,4-pin,5-qing,6-black,7-white.

需要根据实际使用情况调整颜色阈值范围，代码：[submission/color_recognize.py](https://github.com/huihui500/agent_4mission/blob/liujy/submission/color_recognize.py)

测试代码：[查看图像形状检测框坐标](https://github.com/huihui500/agent_4mission/blob/liujy/yolov5/location.ipynb)