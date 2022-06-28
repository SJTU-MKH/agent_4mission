# yolov5s实现检测（主任务+色块检测）

## pipline

主任务：十类目标检测->色块检测：截取主任务检测到的显示器区域，进行下一阶段的色块检测。

## 设置

1. 模型选择：yolov5s


## result

主任务：模型训练结果https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/main_mission/weights

色块检测：模型训练结果https://github.com/huihui500/agent_4mission/tree/liujy/yolov5/runs/train/realgraph-320/weights