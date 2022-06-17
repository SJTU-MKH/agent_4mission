import numpy as np
import random

# 矩形:rectangle
# 五角星:fivestar
# 三角形:triangle
# 菱形:rhombus
# 绘制后需要进行平滑边缘

# demo
# vs = compute_triangle([100, 100], 60, 60, 0.5)
# vs = compute_fivestar([100, 100], 30, 0.5)
# vs = compute_rhombus([100, 100], 60, 120, 0.25)
# print(vs)
# canvas = np.zeros((512, 512, 3), dtype=np.uint8)
# pts = np.array(vs, dtype=np.int32)  # 五个点
# # cv.fillPoly(canvas,[pts],(255,0,255),8,0)#填充多边形
# # cv.polylines(canvas,[pts],True,(0,0,255),2,8,0);#绘制多边形
# cv2.drawContours(canvas, [pts], -1, (255, 0, 0), -1)  # 万能操作 集合上个两个API优点 填充＋绘制
# dst = cv2.blur(canvas, (5, 5))
# cv2.imshow("poyline", dst)
# c = cv2.waitKey(0)  # 设置关闭窗口
# cv2.destroyAllWindows()
###
# RGB format;for cv BGR, the channels need to transform.
color = {"RED": [0, 0, 255], "GREEN": [0, 255, 0], "BLUE": [255, 0, 0], "YELLOW": [0, 255, 255],
         "PIN": [255, 0, 255], "QING": [255, 255, 0], "BLACK": [0, 0, 0], "WHITE": [255, 255, 255]}


def compute_rectangle(center, w, h, angle):
    """
    利用Polor-coordinates旋转, vertexs are anticlockwise
    :param center:中心坐标
    :param w:宽
    :param h:高
    :param angle:角度映射关系　[0, 1] -> [0, pi/2]
    :return:
    """
    vertexs = []
    ws = [+w / 2, -w / 2, -w / 2, +w / 2]
    hs = [-h / 2, -h / 2, +h / 2, +h / 2]
    R = np.sqrt(w * w + h * h) / 2
    angle = angle * np.pi / 2
    for i in range(4):
        # 夹角表示极线与y轴的夹角(正角0~2pi)
        theta = (np.arctan2(ws[i], hs[i]) + 2 * np.pi) % (2 * np.pi)
        # print((theta + angle) / np.pi)
        # print([np.sin(theta + angle), np.cos(theta + angle)])
        vertex = [center[0] + R * np.sin(theta + angle), center[1] + R * np.cos(theta + angle)]
        vertexs.append(vertex)
    return vertexs


def compute_fivestar(center, r, angle):
    """

    :param center: 中心坐标
    :param r: 五角星边长
    :param angle:角度映射关系 [0, 1] -> [0, pi]
    :return:
    """
    vertexs = []

    A = [0, 0]
    B = [A[0] + r, A[1]]
    C = [B[0] + r * np.cos(72 / 180 * np.pi), B[1] - r * np.sin(72 / 180 * np.pi)]
    D = [B[0] + r * np.cos(72 / 180 * np.pi) * 2, B[1]]
    E = [D[0] + r, D[1]]
    F = [E[0] - r * np.cos(36 / 180 * np.pi), E[1] + r * np.sin(36 / 180 * np.pi)]

    G = [F[0] + r * np.cos(72 / 180 * np.pi), F[1] + r * np.sin(72 / 180 * np.pi)]
    # 中
    H = [C[0], A[1] + C[0] * np.tan(36 / 180 * np.pi)]

    J = [A[0] + r * np.cos(36 / 180 * np.pi), F[1]]
    I = [J[0] - r * np.cos(72 / 180 * np.pi), G[1]]

    vertexs.append(A)
    vertexs.append(B)
    vertexs.append(C)
    vertexs.append(D)
    vertexs.append(E)
    vertexs.append(F)
    vertexs.append(G)
    vertexs.append(H)
    vertexs.append(I)
    vertexs.append(J)

    mass = [0, 0]
    for i in range(10):
        mass[0] += vertexs[i][0]
        mass[1] += vertexs[i][1]
    mass[0] /= 10
    mass[1] /= 10

    for i in range(10):
        vertexs[i][0] += (-mass[0] + center[0])
        vertexs[i][1] += (-mass[1] + center[1])

    # rotate
    angle = angle * np.pi
    R = [np.sqrt((vertexs[0][0] - center[0]) ** 2 + (vertexs[0][1] - center[1]) ** 2),
         np.sqrt((vertexs[1][0] - center[0]) ** 2 + (vertexs[1][1] - center[1]) ** 2)]
    for i in range(10):
        # 夹角表示极线与y轴的夹角(正角0~2pi)
        theta = (np.arctan2(vertexs[i][0] - center[0], vertexs[i][1] - center[1]) + 2 * np.pi) % (2 * np.pi)
        # print((theta + angle) / np.pi)
        # print([np.sin(theta + angle), np.cos(theta + angle)])
        vertexs[i][0] = center[0] + R[i % 2] * np.sin(theta + angle)
        vertexs[i][1] = center[1] + R[i % 2] * np.cos(theta + angle)
    return vertexs


def compute_triangle(center, w, h, angle):
    """
    绕底边中心旋转
    :param center:中心坐标
    :param w:底边长
    :param h:高
    :param angle: 角度映射关系　[0, 1] -> [0, 2*pi]
    :return:
    """
    vertexs = []
    r_w = random.random()
    # 左直角三角形,等腰三角形,右直角三角形
    if r_w < 0.2:
        ws = [w / 2, -w / 2, -w / 2]
    elif r_w < 0.8:
        ws = [w / 2, 0, -w / 2]
    else:
        ws = [w / 2, w / 2, -w / 2]
    hs = [0, h, 0]

    angle = angle * np.pi * 2
    for i in range(3):
        # 夹角表示极线与y轴的夹角(正角0~2pi)
        theta = (np.arctan2(ws[i], hs[i]) + 2 * np.pi) % (2 * np.pi)
        R = np.sqrt(ws[i] * ws[i] + hs[i] * hs[i])
        # print((theta + angle) / np.pi)
        # print([np.sin(theta + angle), np.cos(theta + angle)])
        vertex = [center[0] + R * np.sin(theta + angle), center[1] + R * np.cos(theta + angle)]
        vertexs.append(vertex)
    return vertexs


def compute_rhombus(center, w, h, angle):
    """

    :param center:中心坐标
    :param w:对角线宽
    :param h:对角线长
    :param angle:角度映射关系　[0, 1] -> [0, pi]
    :return:
    """
    if h/w < 1.4:
        h = w*1.5
    vertexs = []
    ws = [+w / 2, 0, -w / 2, 0]
    hs = [0, -h / 2, 0, +h / 2]
    R = [w / 2, h / 2]
    angle = angle * np.pi
    theta = [np.pi / 2, np.pi, np.pi * 3 / 2, 0]
    for i in range(4):
        # 夹角表示极线与y轴的夹角(正角0~2pi)
        # print((theta[i] + angle) / np.pi)
        # print([np.sin(theta[i] + angle), np.cos(theta[i] + angle)])
        vertex = [center[0] + R[i % 2] * np.sin(theta[i] + angle), center[1] + R[i % 2] * np.cos(theta[i] + angle)]
        vertexs.append(vertex)
    return vertexs
