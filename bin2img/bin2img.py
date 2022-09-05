# coding=utf-8
"""
    func: 1.bin2png 2.jpg2bin
    file:
        bin2img/bin/bigshape.bin: 软件生成的rgb555bin
        bin2img/bin/bigshape.jpg: 原图，size-(128,230)
"""

import cv2
import numpy as np
import os


def convert_bin2png(path,height, width,channel):
    """把单通道的bin文件转换为png"""
    input_name = path
    image = np.fromfile(input_name, dtype=np.uint16)
    image = np.reshape(image, (height, width))
    # //获取高字节的5个bit
    R = (image & 0x7C00) / 0x0400
    # //获取中间5个bit
    G = (image & 0x03E0) / 0x0020
    # //获取低字节5个bit
    B = image & 0x001F
    image = np.array([R,G,B]).transpose(1,2,0)
    image_gray = np.reshape(image, (height, width, channel))
    output_name = "result.png"  # 需在输入文件夹中新建子文件夹png，作为输出图片路径
    out_path_name = output_name  # 输出图片路径
    cv2.imwrite(out_path_name, image_gray)
    # print("already wrote", i + 1, "/", pic_num, "pictures")  # 浏览进度



import os,sys,struct
from PIL import Image

# 图片jpg转RGB555
def main1():
	infile = "./bin/bigshape.jpg"
	outfile = "./bin/res.bin"
	im=Image.open(infile)
	im.show()
	print("read %s\nImage Width:%d Height:%d" % (infile, im.size[0], im.size[1]))

	f = open(outfile, "wb")
	pix = im.load()  #load pixel array
	for h in range(im.size[1]):
		for w in range(im.size[0]):
			R = pix[w, h][0] 
			G = pix[w, h][1] 
			B = pix[w, h][2] 
			rgb = (R << 10) | (G << 5) | B
			# 转换的图是小端的，所以先后半字节，再前半字节
			f.write(struct.pack('B', rgb & 255))
			f.write(struct.pack('B', (rgb >> 8) & 255))

	f.close()
	print("write to %s" % outfile)

# bin转png
def main2():
    path = "./bin/bigshape.bin"  # 输入bin文件的路径
    width =  230 # 图片宽度
    height = 128  # 图片高度
    # width = 512  # 图片宽度
    # height = 500  # 图片高度  
    channel = 3  # 通道数
    fps = 30  # 保存视频的FPS，30：正常速度， 60:2倍速
    # 转换成图片帧
    convert_bin2png(path, height, width, channel)
    # 转换成视频
    # convert_bin2video()


if __name__ == '__main__':
    main2()




