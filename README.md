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
- 新找到的图片直接按以上操作即可补充进mission中

### qrcode与biggraph(带显示器)
- 对Qrcode和biggraph都增广:aug.py


## 任务方案
### 图形分类
> 颜色:color = {"RED": [0, 0, 255], "GREEN": [0, 255, 0], "BLUE": [255, 0, 0], "YELLOW": [0, 255, 255],
         "PIN": [255, 0, 255], "QING": [255, 255, 0], "BLACK": [0, 0, 0], "WHITE": [255, 255, 255]}
- 传统： Hu 矩: cv2.matchShape()
- 网络直接检测后提取颜色

### 车牌识别
