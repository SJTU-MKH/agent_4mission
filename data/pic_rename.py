import os
import glob

pics_path = glob.glob("background_graph/*")


for i in range(len(pics_path)):
    pic_path = pics_path[i]
    r_pic_path = pic_path.split('/')[0] + "/" + str(i) + ".jpg"
    os.rename(pic_path, r_pic_path)