# agent_4mission

## 类别:
10类:
- 交通标志6个 
- 图形
- 蓝车牌
- 绿车牌
- 二维码

## 数据生成
### 小图形块生成(data/graph)
> circle rectangle triangle fivestar rhombus菱形
- 收集背景图放入　"data/background_graph"
- 运行 generate_graph.py即可
- 新找到的图片直接按以上操作即可补充进mission中（*注意：**标签顺序为0-circle, 1-rhombus, 2-fivestar, 3-triangle, 4-rectangle***）

### qrcode与biggraph(带显示器)
- 对Qrcode和biggraph都增广:aug.py


## 任务方案
### 主任务：

十分类目标检测：

1. 标签：0直行、1左转、2右转、3掉头、4禁止直行、5禁止通行、6图形、7蓝色车牌、8新能源车牌、9二维码
2. 算法：yolov5s

### 子任务

#### 路标检测

#### 图形分类

> 颜色:color = {"RED": [0, 0, 255], "GREEN": [0, 255, 0], "BLUE": [255, 0, 0], "YELLOW": [0, 255, 255],
         "PIN": [255, 0, 255], "QING": [255, 255, 0], "BLACK": [0, 0, 0], "WHITE": [255, 255, 255]}
- 传统： Hu 矩: cv2.matchShape()
- 网络直接检测后提取颜色，代码位置：[color_recognize.py](https://github.com/huihui500/agent_4mission/blob/main/submission/color_recognize.py)

#### 车牌识别

#### 二维码检测识别
