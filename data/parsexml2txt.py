import xml.etree.ElementTree as et
import numpy as np
import glob


def xyxy2xywh_normal(x, w, h):
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2])/2
    y[:, 1] = (x[:, 1] + x[:, 3])/2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    y[:, 4] = x[:, 4]
    return y


def xml2txt_1cls(xml):
    bboxes = np.zeros([0, 5])
    txt_path = xml.replace("xml", "txt")
    tree = et.parse(xml)
    root = tree.getroot()

    node_obj = root.findall("object")
    for node in node_obj:
        for item in node:
            if item.tag == "bndbox":
                xyxy = np.zeros([1, 5])
                for i, v in enumerate(item):
                    xyxy[0, i] = v.text
                bboxes = np.vstack([bboxes, xyxy])
    np.savetxt(txt_path, bboxes, fmt="%0.8f")
    return bboxes


mission = ["biggraph", "Qrcode"]
data_path = "mission/"
pic_path = "/labeled/pic/*"

for m in mission:
    img_folder = data_path + m + pic_path
    imgs_path = glob.glob(img_folder+"*")
    for img_path in imgs_path:
        words = img_path.split(".")
        xml_path = words[0].replace("pic", "xml") + ".xml"
        xml2txt_1cls(xml_path)