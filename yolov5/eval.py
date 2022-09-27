# 筛选bad case

"""
TODO:
    1.读取GT和推理txt
    2.保存不匹配的txt名

"""
import operator

# 判断两个yolov5标签文件的标签是否相同
def match(train_txt, pre_txt):
    file1 = open(train_txt, 'r')
    txt1 = file1.readlines()
    label1 = sorted([i[0] for i in txt1])
    file1.close()

    file2 = open(train_txt, 'r')
    txt2 = file2.readlines()
    label2 = sorted([i[0] for i in txt2])
    file2.close()

    if not operator.eq(label1,label2):
        print("not match")
        return train_txt
    return None

train_root = ''
pred_root = ''



